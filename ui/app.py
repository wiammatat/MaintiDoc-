"""Application Gradio de MaintiDoc (étape 7 de la feuille de route) : interface du technicien.

    python -m ui.app                 (l'API doit tourner : uvicorn api.main:app) -> http://127.0.0.1:7860

- Assistant : question et photo facultative -> agent (analyse de la photo, historique MCP, réponse sourcée,
  vérification). Si un ticket est demandé, le brouillon s'affiche : le technicien le relit, le modifie, puis
  l'approuve ou le refuse. Rien n'est créé sans ce clic.
- Recherche documentaire : preuves brutes (passages, tableaux, figures), utilisable même sans VLM.

L'interface n'a aucun droit propre : chaque appel passe par l'API avec le jeton de l'utilisateur connecté, conservé
dans l'état de session Gradio (côté serveur). Les fonctions de rappel (on_login, on_ask, ...) ne dépendent pas de
l'affichage : les tests les appellent directement.
"""

import io
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")   # aucune donnée d'usage envoyée à l'extérieur

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from common.config import get_settings  # noqa: E402
from ui.client import ApiClient, ApiError, Photo  # noqa: E402
from ui.render import agent_md, esc, search_md, trace_md  # noqa: E402

MAX_PDFS = 3


def make_client(token: str | None = None) -> ApiClient:
    """Remplacée dans les tests par un client branché sur l'API en mémoire."""
    s = get_settings()
    return ApiClient(s.ui_api_url, token=token, timeout=s.ui_timeout)


def new_session() -> dict:
    return {"token": None, "user": None, "thread_id": None}


def _photo(image: Image.Image | None) -> Photo | None:
    if image is None:
        return None
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return Photo("photo.jpg", buf.getvalue(), "image/jpeg")


def _error(e: ApiError, session: dict) -> str:
    if e.status == 401:
        session.update(new_session())
        return "⚠️ Session expirée : reconnectez-vous."
    if e.status == 503:
        return f"⚠️ {esc(e)} L'onglet « Recherche documentaire » reste utilisable."
    return f"⚠️ {esc(e)}"


def _empty() -> pd.DataFrame:
    return pd.DataFrame()


def _pdfs(client: ApiClient, r: dict) -> list[str]:
    """PDF des documents cités, récupérés par l'API (accès revérifié), dans un dossier temporaire de la session."""
    ids = list(dict.fromkeys(c["document_id"] for c in ((r.get("answer") or {}).get("citations") or [])))[:MAX_PDFS]
    folder = Path(tempfile.mkdtemp(prefix="maintidoc_"))
    paths = []
    for doc_id in ids:
        if not re.fullmatch(r"[\w.-]+", doc_id):
            continue
        try:
            (folder / f"{doc_id}.pdf").write_bytes(client.document_pdf(doc_id))
            paths.append(str(folder / f"{doc_id}.pdf"))
        except ApiError:
            pass
    return paths


# --------------------------------------------------------------------------- rappels

def on_health() -> str:
    try:
        h = make_client().health()
        vlm = "désactivé" if h.get("vlm_backend") == "disabled" else h.get("vlm_backend")
        return f"API connectée · index {esc(h.get('index_version'))} · modèle : {esc(vlm)}"
    except ApiError as e:
        return f"⚠️ {esc(e)}"


def on_login(user_id: str, password: str, session: dict):
    """-> (session, message, zone principale visible)"""
    client = make_client()
    try:
        user = client.login((user_id or "").strip(), password or "")
    except ApiError as e:
        msg = "⚠️ Identifiant ou mot de passe incorrect." if e.status == 401 else f"⚠️ {esc(e)}"
        return session, msg, gr.update(visible=False)
    session = new_session() | {"token": client.token, "user": user}
    msg = f"Connecté : **{esc(user['nom'])}** ({esc(user['user_id'])}) · groupes : {esc(', '.join(user['groupes']))}"
    return session, msg, gr.update(visible=True)


def on_logout(session: dict):
    return new_session(), "Déconnecté.", gr.update(visible=False)


def _agent_outputs(client: ApiClient, r: dict, session: dict) -> tuple:
    """-> (session, résultat, traçabilité, historique, tickets ouverts, zone ticket, titre, description, PDF)"""
    session = session | {"thread_id": r["thread_id"]}
    pending = r["status"] == "en_attente_approbation" and r.get("ticket_draft")
    draft = r.get("ticket_draft") or {}
    hist = r.get("historique") or {}
    tickets = (r.get("tickets_ouverts") or {}).get("tickets", [])
    return (session, agent_md(r), trace_md(r), pd.DataFrame(hist.get("interventions", [])), pd.DataFrame(tickets),
            gr.update(visible=bool(pending)), draft.get("title", ""), draft.get("description", ""), _pdfs(client, r))


def _failed(message: str, session: dict) -> tuple:
    return session, message, "", _empty(), _empty(), gr.update(visible=False), "", "", []


def on_ask(question: str, image: Image.Image | None, session: dict) -> tuple:
    if not session.get("token"):
        return _failed("⚠️ Connectez-vous d'abord.", session)
    if not (question or "").strip():
        return _failed("⚠️ Écrivez une question.", session)
    client = make_client(session["token"])
    try:
        return _agent_outputs(client, client.agent(question.strip(), _photo(image)), session)
    except ApiError as e:
        return _failed(_error(e, session), session)


def on_decide(approved: bool, title: str, description: str, session: dict) -> tuple:
    if not session.get("token") or not session.get("thread_id"):
        return _failed("⚠️ Aucune demande en cours.", session)
    client = make_client(session["token"])
    try:
        # Le contenu envoyé est celui affiché (éventuellement corrigé) : c'est lui qui est approuvé puis créé.
        r = client.approve(session["thread_id"], approved, title=title, description=description)
        return _agent_outputs(client, r, session)
    except ApiError as e:
        return _failed(_error(e, session), session)


def on_search(question: str, image: Image.Image | None, session: dict) -> tuple:
    """-> (passages, galerie d'images)"""
    if not session.get("token"):
        return "⚠️ Connectez-vous d'abord.", []
    if not (question or "").strip():
        return "⚠️ Écrivez une question.", []
    client = make_client(session["token"])
    try:
        res = client.search(question.strip(), _photo(image))
    except ApiError as e:
        return _error(e, session), []
    gallery = []
    for im in res["images"]:
        if im.get("image_url"):
            try:
                gallery.append((Image.open(io.BytesIO(client.element_image(im["image_url"]))),
                                f"{im['titre']}, page {im['page']}"))
            except (ApiError, OSError):
                pass
    return search_md(res), gallery


# --------------------------------------------------------------------------- interface

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="MaintiDoc") as demo:
        session = gr.State(new_session())
        gr.Markdown("# MaintiDoc\nAssistant de maintenance SAN : réponses sourcées, avec validation humaine des actions.")
        health = gr.Markdown()
        with gr.Row():
            user_id = gr.Textbox(label="Identifiant", scale=2)
            password = gr.Textbox(label="Mot de passe", type="password", scale=2)
            login_btn = gr.Button("Se connecter", variant="primary", scale=1)
            logout_btn = gr.Button("Se déconnecter", scale=1)
        login_msg = gr.Markdown()

        with gr.Column(visible=False) as main:
            with gr.Tab("Assistant"):
                with gr.Row():
                    with gr.Column(scale=3):
                        question = gr.Textbox(label="Votre question", lines=3, max_length=1000,
                                              placeholder="Ex. : Voyant L2 orange sur BAIE-SN400-002, que faire ? "
                                                          "Prépare un ticket.")
                        ask_btn = gr.Button("Envoyer", variant="primary")
                    photo = gr.Image(label="Photo de l'équipement (facultative)", type="pil", sources=["upload"],
                                     scale=2, height=260)
                result = gr.Markdown()
                with gr.Group(visible=False) as ticket_box:
                    gr.Markdown("### Brouillon de ticket\nRelisez et corrigez : c'est le contenu exact affiché ici qui "
                                "sera créé, et seulement après votre approbation.")
                    title = gr.Textbox(label="Titre", max_length=120)
                    description = gr.Textbox(label="Description", lines=6, max_length=4000)
                    with gr.Row():
                        approve_btn = gr.Button("Approuver et créer", variant="primary")
                        refuse_btn = gr.Button("Refuser", variant="stop")
                pdfs = gr.File(label="Documents cités (PDF)", file_count="multiple", interactive=False)
                with gr.Accordion("Historique de l'équipement", open=False):
                    history = gr.Dataframe(label="Interventions", interactive=False)
                    open_tickets = gr.Dataframe(label="Tickets ouverts", interactive=False)
                with gr.Accordion("Traçabilité", open=False):
                    trace = gr.Markdown()

            with gr.Tab("Recherche documentaire"):
                with gr.Row():
                    s_question = gr.Textbox(label="Rechercher dans la documentation", max_length=1000, scale=3)
                    s_photo = gr.Image(label="Photo (facultative)", type="pil", sources=["upload"], scale=1, height=160)
                s_btn = gr.Button("Rechercher")
                s_result = gr.Markdown()
                s_gallery = gr.Gallery(label="Figures et pages", columns=3, height="auto")

        agent_out = [session, result, trace, history, open_tickets, ticket_box, title, description, pdfs]
        demo.load(on_health, outputs=health)
        login_btn.click(on_login, [user_id, password, session], [session, login_msg, main])
        password.submit(on_login, [user_id, password, session], [session, login_msg, main])
        logout_btn.click(on_logout, [session], [session, login_msg, main])
        ask_btn.click(on_ask, [question, photo, session], agent_out)
        approve_btn.click(lambda t, d, s: on_decide(True, t, d, s), [title, description, session], agent_out)
        refuse_btn.click(lambda t, d, s: on_decide(False, t, d, s), [title, description, session], agent_out)
        s_btn.click(on_search, [s_question, s_photo, session], [s_result, s_gallery])
    return demo


if __name__ == "__main__":
    # 127.0.0.1 : pas d'exposition sur le réseau ; share=False : pas de lien public Gradio.
    build_ui().queue().launch(server_name="127.0.0.1", server_port=7860, share=False)
