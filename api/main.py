"""Backend FastAPI de MaintiDoc.

Lancement : uvicorn api.main:app --reload   (documentation interactive : http://localhost:8000/docs)

Sécurité (section 9 du sujet) :
- authentification par jeton JWT, groupes relus côté serveur à chaque requête ;
- filtrage par groupe avant la récupération (recherche), puis à l'ouverture des fichiers ;
- un fichier interdit et un fichier inexistant donnent la même réponse (404) : on ne révèle pas son existence ;
- journalisation de métadonnées seulement (route, utilisateur, statut, durée), jamais du texte des questions.
"""

import io
import logging
import time
import uuid
from collections.abc import Callable
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import Response
from fastapi.security import OAuth2PasswordRequestForm
from PIL import Image, UnidentifiedImageError

from api.auth import create_token, current_user
from api.schemas import AgentResponse, ApprovalRequest, Evidence, Me, SearchResponse, Token
from common.access import User, authenticate, can_access
from common.config import get_settings
from agent.graph import CHECKPOINTER, build_agent
from agent.mcp_client import MCPClient, StdioMCPClient
from generation.answer import answer
from generation.schema import Answer
from generation.vlm import VLM, RemoteVLM, VLMUnavailable, default_vlm
from retrieval.fusion import Hit
from retrieval.search import SearchResult, search

MAX_QUESTION_CHARS = 1000
log = logging.getLogger("maintidoc.api")
app = FastAPI(title="MaintiDoc", version="0.1.0",
              description="Assistant documentaire multimodal pour la maintenance SAN (prototype, données fictives).")


# --------------------------------------------------------------------------- journalisation minimale

@app.middleware("http")
async def access_log(request: Request, call_next):
    request.state.request_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()
    response = await call_next(request)
    log.info("requete=%s route=%s %s utilisateur=%s statut=%s duree_ms=%d", request.state.request_id,
             request.method, request.url.path, getattr(request.state, "user_id", "-"), response.status_code,
             (time.perf_counter() - start) * 1000)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


def authenticated(request: Request, user: User = Depends(current_user)) -> User:
    request.state.user_id = user.user_id   # pour le journal, sans aucune donnée de contenu
    return user


CurrentUser = Annotated[User, Depends(authenticated)]


# --------------------------------------------------------------------------- dépendances remplaçables (tests)

def get_search_fn() -> Callable[..., SearchResult]:
    return search


@lru_cache
def _remote_vlm() -> RemoteVLM:
    s = get_settings()
    return RemoteVLM(s.vlm_remote_url, s.vlm_remote_token.get_secret_value(), timeout=s.vlm_remote_timeout)


def get_vlm() -> VLM:
    s = get_settings()
    if s.vlm_backend == "qwen":
        return default_vlm()
    if s.vlm_backend == "remote" and s.vlm_remote_url and s.vlm_remote_token.get_secret_value():
        return _remote_vlm()
    raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                        "VLM indisponible : utiliser /search, ou VLM_BACKEND=qwen sur un PC avec GPU NVIDIA, ou VLM_BACKEND=remote avec "
                        "VLM_REMOTE_URL et VLM_REMOTE_TOKEN), ou VLM_BACKEND=qwen sur une machine avec GPU.")


def get_optional_vlm() -> VLM | None:
    """Pour reprendre un agent après approbation : le VLM n'est plus nécessaire (création du ticket seulement)."""
    try:
        return get_vlm()
    except HTTPException:
        return None


@lru_cache
def get_mcp() -> MCPClient:
    return StdioMCPClient()


# --------------------------------------------------------------------------- validation des entrées

def check_question(question: str) -> str:
    question = question.strip()
    if not question or len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"La question doit contenir entre 1 et {MAX_QUESTION_CHARS} caractères.")
    return question


async def read_photo(photo: UploadFile | None) -> Image.Image | None:
    """Photo facultative : taille bornée, format image vérifié par Pillow (pas seulement l'extension)."""
    if photo is None or not photo.filename:
        return None
    limit = get_settings().max_upload_mb * 1024 * 1024
    data = await photo.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"Photo trop volumineuse ({get_settings().max_upload_mb} Mo maximum).")
    try:
        Image.open(io.BytesIO(data)).verify()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Le fichier envoyé n'est pas une image valide.")


def to_evidence(hit: Hit) -> Evidence:
    p = hit.payload
    url = f"/elements/{hit.uid}/image" if p["type"] in ("figure", "page") else None
    return Evidence(uid=hit.uid, document_id=p["document_id"], titre=p["titre"], page=p["page"],
                    element_id=p["element_id"], type=p["type"], section=p.get("section"), text=p.get("text"),
                    rerank_score=hit.rerank_score, image_url=url)


# --------------------------------------------------------------------------- routes

@app.get("/health")
def health() -> dict:
    s = get_settings()
    return {"status": "ok", "index_version": s.index_version, "vlm_backend": s.vlm_backend,
            "prompt": s.answer_prompt}


@app.post("/auth/token", response_model=Token)
def login(form: Annotated[OAuth2PasswordRequestForm, Depends()]) -> Token:
    user = authenticate(form.username, form.password)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Identifiant ou mot de passe incorrect",
                            headers={"WWW-Authenticate": "Bearer"})
    token, ttl = create_token(user)
    return Token(access_token=token, expires_in=ttl)


@app.get("/me", response_model=Me)
def me(user: CurrentUser) -> Me:
    return Me(user_id=user.user_id, nom=user.nom, groupes=sorted(user.groupes))


@app.post("/search", response_model=SearchResponse)
async def search_route(user: CurrentUser, question: Annotated[str, Form()],
                       photo: Annotated[UploadFile | None, File()] = None,
                       search_fn: Callable[..., SearchResult] = Depends(get_search_fn)) -> SearchResponse:
    """RAG multimodal : preuves citables (passages, tableaux, figures, pages) accessibles à l'utilisateur."""
    question = check_question(question)
    result = search_fn(user, question, await read_photo(photo))
    return SearchResponse(question=question, passages=[to_evidence(h) for h in result.passages],
                          images=[to_evidence(h) for h in result.images], index_version=get_settings().index_version)


@app.post("/ask", response_model=Answer)
async def ask_route(user: CurrentUser, question: Annotated[str, Form()],
                    photo: Annotated[UploadFile | None, File()] = None,
                    vlm: VLM = Depends(get_vlm),
                    search_fn: Callable[..., SearchResult] = Depends(get_search_fn)) -> Answer:
    """Réponse sourcée complète : recherche, paquet de preuves, VLM, JSON validé, vérification des citations."""
    try:
        return answer(user, check_question(question), await read_photo(photo), vlm=vlm, search_fn=search_fn)
    except VLMUnavailable as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from None


NOT_FOUND = HTTPException(status.HTTP_404_NOT_FOUND, "Introuvable")


def _locate_document(document_id: str) -> tuple[list[str], Callable[[], bytes]] | None:
    """(groupes autorisés, lecture du PDF) selon le profil de stockage, ou None si le document n'existe pas."""
    if get_settings().storage_backend == "local":
        from storage import local
        doc = local.get_document(document_id)
        return (doc["groupes_autorises"], doc["path"].read_bytes) if doc and doc["path"].exists() else None
    from storage.metadata_db import connect, get_document
    from storage.s3 import read_file
    with connect() as con:
        doc = get_document(con, document_id)
    return (doc["groupes_autorises"], lambda: read_file(doc["s3_key"])) if doc and doc["s3_key"] else None


def _locate_element(uid: str) -> tuple[list[str], Callable[[], bytes], str] | None:
    """(groupes autorisés, lecture de l'image, nom du fichier) selon le profil, ou None si l'élément n'a pas d'image."""
    if get_settings().storage_backend == "local":
        from storage import local
        el = local.get_element(uid)
        ok = el and el["path"] is not None and el["path"].exists()
        return (el["groupes_autorises"], el["path"].read_bytes, el["path"].name) if ok else None
    from storage.metadata_db import connect, get_element
    from storage.s3 import read_file
    with connect() as con:
        el = get_element(con, uid)
    return (el["groupes_autorises"], lambda: read_file(el["s3_key"]), el["s3_key"]) if el and el["s3_key"] else None


@app.get("/documents/{document_id}/pdf")
def document_pdf(document_id: str, user: CurrentUser) -> Response:
    """PDF source, après contrôle d'accès (exigence du sujet : contrôle aussi à l'ouverture des documents)."""
    found = _locate_document(document_id)
    if found is None or not can_access(user, found[0]):   # interdit et inexistant : même réponse
        raise NOT_FOUND
    return Response(found[1](), media_type="application/pdf")


@app.get("/elements/{uid}/image")
def element_image(uid: str, user: CurrentUser) -> Response:
    """Figure ou page d'un document, après contrôle d'accès."""
    found = _locate_element(uid)
    if found is None or not can_access(user, found[0]):
        raise NOT_FOUND
    media = "image/jpeg" if found[2].lower().endswith((".jpg", ".jpeg")) else "image/png"
    return Response(found[1](), media_type=media)


# --------------------------------------------------------------------------- agent (LangGraph + MCP)

def _agent_response(thread_id: str, values: dict, pending: bool) -> AgentResponse:
    return AgentResponse(thread_id=thread_id, status="en_attente_approbation" if pending else values.get("status", "termine"),
                         answer=values.get("answer"), clarification_question=values.get("clarification_question"),
                         historique=values.get("history"), tickets_ouverts=values.get("open_tickets"),
                         ticket_draft=values.get("ticket_draft"), ticket=values.get("ticket"),
                         errors=values.get("errors", []), mcp_calls=values.get("mcp_calls", 0),
                         model_calls=values.get("model_calls", 0),
                         search_attempts=values.get("search_attempts", []), photo_analysis=values.get("photo_analysis"))


def _photo_b64(photo: UploadFile | None) -> str | None:
    if photo is None or not photo.filename:
        return None
    import base64
    limit = get_settings().max_upload_mb * 1024 * 1024
    data = photo.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"Photo trop volumineuse ({get_settings().max_upload_mb} Mo maximum).")
    try:
        Image.open(io.BytesIO(data)).verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Le fichier envoyé n'est pas une image valide.")
    return base64.b64encode(data).decode()


@app.post("/agent", response_model=AgentResponse)
def agent_route(request: Request, user: CurrentUser, question: Annotated[str, Form()],
                photo: Annotated[UploadFile | None, File()] = None, vlm: VLM = Depends(get_vlm),
                mcp: MCPClient = Depends(get_mcp),
                search_fn: Callable[..., SearchResult] = Depends(get_search_fn)) -> AgentResponse:
    """Agent complet : analyse, historique MCP si besoin, réponse sourcée, brouillon de ticket à approuver.

    Route synchrone : le client MCP (stdio) gère sa propre boucle asynchrone dans le fil d'exécution de la requête.
    """
    thread_id = uuid.uuid4().hex
    token = request.headers["Authorization"].split(" ", 1)[1]
    config = {"configurable": {"thread_id": thread_id, "auth_token": token}}
    agent = build_agent(vlm, mcp, search_fn, CHECKPOINTER)
    try:
        values = agent.invoke({"user_id": user.user_id, "question": check_question(question),
                               "photo_b64": _photo_b64(photo)}, config)
    except VLMUnavailable as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from None
    return _agent_response(thread_id, values, pending=bool(agent.get_state(config).next))


@app.post("/agent/{thread_id}/approval", response_model=AgentResponse)
def approval_route(thread_id: str, decision: ApprovalRequest, request: Request, user: CurrentUser,
                   vlm: VLM | None = Depends(get_optional_vlm), mcp: MCPClient = Depends(get_mcp)) -> AgentResponse:
    """Approbation humaine du brouillon (éventuellement modifié) ou refus. Seul l'auteur de la demande peut décider."""
    from langgraph.types import Command
    token = request.headers["Authorization"].split(" ", 1)[1]
    config = {"configurable": {"thread_id": thread_id, "auth_token": token}}
    agent = build_agent(vlm, mcp, get_search_fn(), CHECKPOINTER)
    state = agent.get_state(config)
    if not state.values or state.values.get("user_id") != user.user_id:   # demande d'un autre : même 404
        raise NOT_FOUND
    if not state.next:
        raise HTTPException(status.HTTP_409_CONFLICT, "Aucune approbation en attente pour cette demande.")
    draft = state.values["ticket_draft"]
    resume = {"approved": decision.approved,
              "title": (decision.title if decision.title is not None else draft["title"]).strip(),
              "description": (decision.description if decision.description is not None else draft["description"]).strip()}
    values = agent.invoke(Command(resume=resume), config)
    return _agent_response(thread_id, values, pending=bool(agent.get_state(config).next))
