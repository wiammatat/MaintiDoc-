"""Vérifie les routes de l'agent : POST /agent et POST /agent/{id}/approval (validation humaine)."""

import shutil
import sqlite3

import pytest
from fastapi.testclient import TestClient

import mcp_server.server as server
from agent.mcp_client import InMemoryMCPClient
from api.main import app, get_mcp, get_search_fn, get_vlm
from common.config import get_settings
from generation.vlm import ScriptedVLM
from tests.test_agent import ANSWER, CHECK, DRAFT, FAKE
from tests.test_api import auth

S = get_settings()
DB = S.data_dir / "mcp" / "maintenance.db"
pytestmark = pytest.mark.skipif(not DB.exists(), reason="base MCP absente : python scripts/generate_corpus.py --mcp-only")
client = TestClient(app)
QUESTION = "Voyant orange sur BAIE-SN400-002, prépare un ticket."


@pytest.fixture
def agent_deps(tmp_path, monkeypatch):
    copy = tmp_path / "maintenance.db"
    shutil.copy(DB, copy)

    def db():
        con = sqlite3.connect(copy)
        con.row_factory = sqlite3.Row
        return con
    monkeypatch.setattr(server, "_db", db)
    app.dependency_overrides[get_vlm] = lambda: ScriptedVLM(ANSWER, CHECK, DRAFT)
    app.dependency_overrides[get_search_fn] = lambda: (lambda *a, **k: FAKE)
    app.dependency_overrides[get_mcp] = lambda: InMemoryMCPClient()
    yield copy
    app.dependency_overrides.clear()


def ask(user="tech_s01", question=QUESTION):
    r = client.post("/agent", headers=auth(user), data={"question": question})
    assert r.status_code == 200, r.text
    return r.json()


def test_brouillon_en_attente_puis_creation(agent_deps):
    body = ask()
    assert body["status"] == "en_attente_approbation" and body["ticket_draft"]["equipment_id"] == "BAIE-SN400-002"
    assert body["historique"]["modele"] == "SN-400" and body["answer"]["status"] == "answered"
    r = client.post(f"/agent/{body['thread_id']}/approval", headers=auth("tech_s01"), json={"approved": True})
    assert r.status_code == 200 and r.json()["status"] == "ticket_cree" and r.json()["ticket"]["created"] is True
    # Plus rien en attente : une seconde approbation est refusée (et ne crée pas de doublon).
    assert client.post(f"/agent/{body['thread_id']}/approval", headers=auth("tech_s01"),
                       json={"approved": True}).status_code == 409


def test_seul_l_auteur_peut_approuver(agent_deps):
    body = ask()
    for other in ("tech_s02", "resp_01"):
        r = client.post(f"/agent/{body['thread_id']}/approval", headers=auth(other), json={"approved": True})
        assert r.status_code == 404
    assert client.post("/agent/inexistant/approval", headers=auth("tech_s01"), json={"approved": True}).status_code == 404


def test_refus(agent_deps):
    body = ask()
    r = client.post(f"/agent/{body['thread_id']}/approval", headers=auth("tech_s01"), json={"approved": False})
    assert r.json()["status"] == "ticket_refuse" and r.json()["ticket"] is None
    n = sqlite3.connect(agent_deps).execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
    assert n == 3   # aucun ticket ajouté


def test_contenu_modifie_par_l_humain(agent_deps):
    body = ask()
    r = client.post(f"/agent/{body['thread_id']}/approval", headers=auth("tech_s01"),
                    json={"approved": True, "title": "Titre relu et corrigé"})
    ticket_id = r.json()["ticket"]["ticket_id"]
    titre = sqlite3.connect(agent_deps).execute("SELECT titre FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()[0]
    assert titre == "Titre relu et corrigé"


def test_question_simple_sans_ticket(agent_deps):
    body = ask(question="Que signifie le voyant orange ?")
    assert body["status"] == "termine" and body["ticket_draft"] is None and body["mcp_calls"] == 0


def test_agent_sans_vlm(agent_deps, monkeypatch):
    del app.dependency_overrides[get_vlm]
    monkeypatch.setattr(S, "vlm_backend", "disabled")
    assert client.post("/agent", headers=auth("tech_s01"), data={"question": QUESTION}).status_code == 503
