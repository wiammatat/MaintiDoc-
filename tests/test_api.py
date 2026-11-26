"""Vérifie le backend FastAPI : authentification, validation, recherche, réponse, contrôle d'accès aux fichiers."""

import io
import json
import logging
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app, get_search_fn, get_vlm
from common.config import get_settings
from generation.vlm import ScriptedVLM
from retrieval.fusion import Hit
from retrieval.search import SearchResult

S = get_settings()
client = TestClient(app)


def token(user_id: str) -> str:
    r = client.post("/auth/token", data={"username": user_id, "password": f"{user_id}-demo"})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def auth(user_id: str) -> dict:
    return {"Authorization": f"Bearer {token(user_id)}"}


def hit(doc, element, text, type_="passage", famille="baie"):
    return Hit(uid=f"{doc}:v1:{element}", payload={
        "document_id": doc, "version": "v1", "page": 2, "element_id": element, "type": type_, "famille": famille,
        "titre": f"Titre {doc}", "section": "3. Panneau de contrôle", "text": text,
        "image_path": f"processed/images/{doc}/{element}.png" if type_ != "passage" else None,
        "groupes_autorises": ["tech_stockage", "responsable"]})


FAKE = SearchResult(passages=[hit("manuel_baie_sn-400", "table_2_1_1", "| L2 | orange | Disque défaillant |")],
                    images=[hit("manuel_baie_sn-400", "figure_2_1", "Figure 3.1", "figure")])


@pytest.fixture
def fake_search():
    app.dependency_overrides[get_search_fn] = lambda: (lambda user, question, photo=None: FAKE)
    yield
    app.dependency_overrides.clear()


def png_bytes(size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 120, 20)).save(buf, "PNG")
    return buf.getvalue()


# --- Authentification

def test_health_public():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_connexion_et_profil():
    r = client.get("/me", headers=auth("tech_s01"))
    assert r.status_code == 200
    assert r.json() == {"user_id": "tech_s01", "nom": "Technicien Stockage 1", "groupes": ["tech_stockage"]}


@pytest.mark.parametrize("username,password", [("tech_s01", "mauvais"), ("inconnu", "inconnu-demo"), ("tech_s01", "")])
def test_connexion_refusee_sans_reveler_la_cause(username, password):
    r = client.post("/auth/token", data={"username": username, "password": password})
    assert r.status_code in (401, 422)
    if r.status_code == 401:
        assert r.json()["detail"] == "Identifiant ou mot de passe incorrect"


def test_route_protegee_sans_jeton():
    assert client.get("/me").status_code == 401
    assert client.post("/search", data={"question": "voyant"}).status_code == 401


def test_jeton_falsifie_refuse():
    forged = jwt.encode({"sub": "resp_01", "exp": datetime.now(UTC) + timedelta(hours=1)}, "x" * 40, algorithm="HS256")
    assert client.get("/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401


def test_jeton_expire_refuse():
    secret = S.jwt_secret.get_secret_value()
    expired = jwt.encode({"sub": "tech_s01", "exp": datetime.now(UTC) - timedelta(minutes=1)}, secret, algorithm="HS256")
    assert client.get("/me", headers={"Authorization": f"Bearer {expired}"}).status_code == 401


def test_groupes_relus_cote_serveur():
    # Un jeton valide qui prétendrait d'autres groupes n'a aucun effet : seuls le « sub » et users.json comptent.
    secret = S.jwt_secret.get_secret_value()
    tok = jwt.encode({"sub": "tech_r01", "groupes": ["responsable"], "exp": datetime.now(UTC) + timedelta(hours=1)},
                     secret, algorithm="HS256")
    assert client.get("/me", headers={"Authorization": f"Bearer {tok}"}).json()["groupes"] == ["tech_reseau"]


# --- Validation des entrées

def test_question_vide_ou_trop_longue(fake_search):
    h = auth("tech_s01")
    assert client.post("/search", headers=h, data={"question": "   "}).status_code == 422
    assert client.post("/search", headers=h, data={"question": "x" * 1001}).status_code == 422


def test_fichier_qui_n_est_pas_une_image(fake_search):
    r = client.post("/search", headers=auth("tech_s01"), data={"question": "voyant"},
                    files={"photo": ("photo.jpg", b"pas une image", "image/jpeg")})
    assert r.status_code == 415


def test_photo_trop_volumineuse(fake_search, monkeypatch):
    monkeypatch.setattr(S, "max_upload_mb", 0)
    r = client.post("/search", headers=auth("tech_s01"), data={"question": "voyant"},
                    files={"photo": ("photo.png", png_bytes(), "image/png")})
    assert r.status_code == 413


# --- Recherche et réponse

def test_recherche_renvoie_des_preuves_citables(fake_search):
    r = client.post("/search", headers=auth("tech_s01"), data={"question": "Que signifie le voyant orange ?"},
                    files={"photo": ("photo.png", png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["passages"][0]["element_id"] == "table_2_1_1" and body["passages"][0]["image_url"] is None
    assert body["images"][0]["image_url"] == "/elements/manuel_baie_sn-400:v1:figure_2_1/image"


def test_ask_indisponible_sans_gpu(fake_search, monkeypatch):
    monkeypatch.setattr(S, "vlm_backend", "disabled")   # indépendant du .env local
    r = client.post("/ask", headers=auth("tech_s01"), data={"question": "Que signifie le voyant orange ?"})
    assert r.status_code == 503 and "VLM indisponible" in r.json()["detail"]


def test_ask_avec_vlm(fake_search):
    reply = json.dumps({"status": "answered", "answer": "Disque défaillant [S1].", "citations": ["S1"],
                        "uncertainties": [], "clarification_question": None, "proposed_action": None})
    checks = json.dumps({"checks": [{"id": 1, "verdict": "soutenu", "raison": "ok"}]})
    app.dependency_overrides[get_vlm] = lambda: ScriptedVLM(reply, checks)
    r = client.post("/ask", headers=auth("tech_s01"), data={"question": "Que signifie le voyant orange ?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered" and body["citations"][0]["element_id"] == "table_2_1_1"
    assert body["versions"]["prompt"] == S.answer_prompt


def test_ask_garde_fou_equipement(fake_search):
    app.dependency_overrides[get_vlm] = lambda: ScriptedVLM("ne doit pas être appelé")
    r = client.post("/ask", headers=auth("tech_r01"), data={"question": "Voyant orange de la baie SN-400 ?"})
    assert r.status_code == 200 and r.json()["status"] == "abstained" and "SN-400" in r.json()["answer"]


def test_journal_sans_texte_de_question(fake_search, caplog):
    with caplog.at_level(logging.INFO, logger="maintidoc.api"):
        r = client.post("/search", headers=auth("tech_s01"), data={"question": "QUESTION-SECRETE-123"})
    assert "X-Request-ID" in r.headers
    assert any("utilisateur=tech_s01" in m for m in caplog.messages)
    assert not any("QUESTION-SECRETE-123" in m for m in caplog.messages)


# --- Contrôle d'accès à l'ouverture des fichiers (nécessite MinIO et PostgreSQL)

def _storage_up() -> bool:
    try:
        from storage.metadata_db import connect, get_document
        with connect() as con:
            return get_document(con, "manuel_baie_sn-400") is not None
    except Exception:
        return False


needs_storage = pytest.mark.skipif(not _storage_up(), reason="MinIO / PostgreSQL indisponibles ou corpus non ingéré")


@needs_storage
def test_pdf_autorise():
    r = client.get("/documents/manuel_baie_sn-400/pdf", headers=auth("tech_s01"))
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf" and r.content.startswith(b"%PDF")


@needs_storage
def test_pdf_interdit_et_inexistant_meme_reponse():
    interdit = client.get("/documents/manuel_baie_sn-400/pdf", headers=auth("tech_r01"))
    inexistant = client.get("/documents/n_existe_pas/pdf", headers=auth("tech_r01"))
    assert interdit.status_code == inexistant.status_code == 404
    assert interdit.json() == inexistant.json()


@needs_storage
def test_note_confidentielle_reservee_aux_responsables():
    assert client.get("/documents/note_service_baie_02/pdf", headers=auth("tech_s01")).status_code == 404
    assert client.get("/documents/note_service_baie_02/pdf", headers=auth("resp_01")).status_code == 200


@needs_storage
def test_image_avec_controle_d_acces():
    url = "/elements/manuel_baie_sn-400:v1:figure_2_1/image"
    ok = client.get(url, headers=auth("tech_s01"))
    assert ok.status_code == 200 and ok.headers["content-type"] == "image/png"
    assert client.get(url, headers=auth("tech_r01")).status_code == 404


# --- Même contrôle d'accès en profil léger (STORAGE_BACKEND=local, sans Docker : Lightning AI Studio)

@pytest.fixture
def local_storage(monkeypatch):
    if not (S.data_dir / "processed" / "elements.jsonl").exists():
        pytest.skip("corpus non ingéré : lancer python -m pipeline.ingest --no-store")
    monkeypatch.setattr(S, "storage_backend", "local")


def test_local_pdf_autorise(local_storage):
    r = client.get("/documents/manuel_baie_sn-400/pdf", headers=auth("tech_s01"))
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf" and r.content.startswith(b"%PDF")


def test_local_pdf_interdit_et_inexistant_meme_reponse(local_storage):
    interdit = client.get("/documents/manuel_baie_sn-400/pdf", headers=auth("tech_r01"))
    inexistant = client.get("/documents/n_existe_pas/pdf", headers=auth("tech_r01"))
    assert interdit.status_code == inexistant.status_code == 404 and interdit.json() == inexistant.json()


def test_local_note_confidentielle(local_storage):
    assert client.get("/documents/note_service_baie_02/pdf", headers=auth("tech_s01")).status_code == 404
    assert client.get("/documents/note_service_baie_02/pdf", headers=auth("resp_01")).status_code == 200


def test_local_image_avec_controle_d_acces(local_storage):
    url = "/elements/manuel_baie_sn-400:v1:figure_2_1/image"
    ok = client.get(url, headers=auth("tech_s01"))
    assert ok.status_code == 200 and ok.headers["content-type"] == "image/png" and ok.content.startswith(b"\x89PNG")
    assert client.get(url, headers=auth("tech_r01")).status_code == 404
    # Un passage (sans image) et un chemin malveillant ne renvoient rien.
    assert client.get("/elements/manuel_baie_sn-400:v1:table_2_1_1/image", headers=auth("tech_s01")).status_code == 404
    assert client.get("/elements/..%2F..%2F.env/image", headers=auth("resp_01")).status_code == 404
