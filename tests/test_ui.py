"""Vérifie l'application Gradio : client HTTP contre la vraie API (en mémoire) et parcours de l'interface."""

import shutil
import sqlite3

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import mcp_server.server as server
from agent.mcp_client import InMemoryMCPClient
from api.main import app, get_mcp, get_search_fn, get_vlm
from common.config import get_settings
from generation.vlm import ScriptedVLM
from tests.test_agent import ANSWER, CHECK, DB, DRAFT, FAKE
from ui.client import ApiClient, ApiError

pytestmark = pytest.mark.skipif(not DB.exists(), reason="base MCP absente : python scripts/generate_corpus.py --mcp-only")
QUESTION = "Voyant orange sur BAIE-SN400-002, prépare un ticket."


@pytest.fixture
def api(tmp_path, monkeypatch):
    copy = tmp_path / "maintenance.db"
    shutil.copy(DB, copy)

    def db():
        con = sqlite3.connect(copy)
        con.row_factory = sqlite3.Row
        return con
    monkeypatch.setattr(server, "_db", db)
    monkeypatch.setattr(api_main, "_locate_document", lambda _id: None)   # sans Docker ni profil local
    app.dependency_overrides[get_vlm] = lambda: ScriptedVLM(ANSWER, CHECK, DRAFT)
    app.dependency_overrides[get_search_fn] = lambda: (lambda *a, **k: FAKE)
    app.dependency_overrides[get_mcp] = lambda: InMemoryMCPClient()
    yield copy
    app.dependency_overrides.clear()


def api_client() -> ApiClient:
    return ApiClient(http=TestClient(app))


# --- Client HTTP

def test_client_connexion_et_erreurs(api):
    c = api_client()
    with pytest.raises(ApiError) as e:
        c.login("tech_s01", "mauvais")
    assert e.value.status == 401 and c.token is None
    me = c.login("tech_s01", "tech_s01-demo")
    assert me["user_id"] == "tech_s01" and c.token
    with pytest.raises(ApiError) as e:
        c.document_pdf("document_inexistant")
    assert e.value.status == 404
    with pytest.raises(ApiError):
        c.element_image("https://ailleurs.example/x.png")   # seules les adresses fournies par l'API


def test_client_agent_puis_approbation(api):
    c = api_client()
    c.login("tech_s01", "tech_s01-demo")
    r = c.agent(QUESTION)
    assert r["status"] == "en_attente_approbation"
    r = c.approve(r["thread_id"], True, title="Titre relu", description=r["ticket_draft"]["description"])
    assert r["status"] == "ticket_cree"
    titre = sqlite3.connect(api).execute("SELECT titre FROM tickets WHERE ticket_id = ?",
                                         (r["ticket"]["ticket_id"],)).fetchone()[0]
    assert titre == "Titre relu"


def test_client_api_injoignable():
    c = ApiClient("http://127.0.0.1:9", timeout=2)
    with pytest.raises(ApiError) as e:
        c.health()
    assert e.value.status in (0, 504)   # refus de connexion ou délai dépassé selon le système


# --- Échappement et mise en forme

def test_echappement_du_texte_non_fiable():
    from ui.render import h
    out = h('<script>alert(1)</script> <a href="http://pirate.example">x</a>')
    assert "<script>" not in out and "<a " not in out and "&lt;script&gt;" in out and "&quot;" in out


def test_reponse_piegee_affichee_comme_du_texte():
    from ui.render import agent_html, search_html
    piege = 'Voir <a href="http://pirate.example/x.exe">le correctif</a><img src=x onerror=alert(1)>'
    r = {"thread_id": "t", "status": "termine", "model_calls": 1, "mcp_calls": 0,
         "answer": {"status": "answered", "citations": [{"document_id": "<b>doc</b>", "page": 1, "element_id": "e"}],
                    "answer": piege, "uncertainties": [piege]}}
    html = agent_html(r)
    assert "<a href" not in html and "<img" not in html and "<b>doc</b>" not in html and "&lt;a href" in html
    res = {"index_version": "v1", "images": [], "passages": [
        {"titre": piege, "page": 1, "type": "text", "text": piege, "section": piege}]}
    assert "<img" not in search_html(res) and "<a href" not in search_html(res)


# --- Interface Gradio (fonctions de rappel appelées directement)

@pytest.fixture
def ui(api, monkeypatch):
    import ui.app as ui_app
    monkeypatch.setattr(ui_app, "make_client", lambda token=None: ApiClient(token=token, http=TestClient(app)))
    return ui_app


def login(ui, user="tech_s01"):
    session, msg, login_view, app_view, card = ui.on_login(user, f"{user}-demo", ui.new_session())
    assert app_view["visible"] is True and login_view["visible"] is False, msg
    assert user in card
    return session


def test_interface_construite():
    import ui.app as ui_app
    assert ui_app.build_ui() is not None


def test_connexion(ui):
    session, msg, login_view, app_view, _ = ui.on_login("tech_s01", "faux", ui.new_session())
    assert "incorrect" in msg and app_view["visible"] is False and session["token"] is None
    session = login(ui)
    assert session["user"]["user_id"] == "tech_s01" and session["token"]
    session, _, login_view, app_view, card = ui.on_logout(session)
    assert session["token"] is None and app_view["visible"] is False and login_view["visible"] is True and card == ""


def test_question_sans_connexion(ui):
    out = ui.on_ask(QUESTION, None, ui.new_session())
    assert "Connectez-vous" in out[1]


def test_parcours_ticket_approuve_apres_modification(ui, api):
    session = login(ui)
    session, result, trace, history, tickets, box, title, desc, pdfs = ui.on_ask(QUESTION, None, session)
    assert box["visible"] is True and title.startswith("BAIE-SN400-002") and "en attente" in result
    assert not history.empty and "Date" in history.columns and session["thread_id"] in trace
    session, result, *_ , box, _, _, _ = ui.on_decide(True, "Titre corrigé dans l'interface", desc, session)
    assert box["visible"] is False and "Ticket créé" in result
    titre = sqlite3.connect(api).execute("SELECT titre FROM tickets ORDER BY rowid DESC LIMIT 1").fetchone()[0]
    assert titre == "Titre corrigé dans l'interface"


def test_refus(ui, api):
    session = login(ui)
    session, *_ = ui.on_ask(QUESTION, None, session)
    session, result, *_ = ui.on_decide(False, "x", "y", session)
    assert "Ticket refusé" in result
    assert sqlite3.connect(api).execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 3


def test_photo_envoyee_a_l_agent(ui):
    from PIL import Image
    session = login(ui)
    out = ui.on_ask("Que signifie ce voyant ?", Image.new("RGB", (64, 64), "gray"), session)
    assert "Analyse de la photo" in out[1]   # la photo est bien transmise et analysée par l'agent


def test_modele_indisponible(ui, monkeypatch):
    monkeypatch.setattr(get_settings(), "vlm_backend", "disabled")   # indépendant du .env local
    app.dependency_overrides.pop(get_vlm)
    session = login(ui)
    out = ui.on_ask(QUESTION, None, session)
    assert "Recherche documentaire" in out[1] and out[5]["visible"] is False


def test_recherche(ui):
    session = login(ui)
    md, gallery = ui.on_search("voyant orange", None, session)
    assert "Manuel d" in md and "page 2" in md and gallery == []
