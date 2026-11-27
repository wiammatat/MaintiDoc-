"""Vérifie l'agent LangGraph : routage, historique MCP, brouillon, approbation humaine, création, budgets."""

import json
import shutil
import sqlite3
import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import mcp_server.server as server
from agent.graph import build_agent
from agent.intent import analyze
from agent.mcp_client import InMemoryMCPClient
from common.access import get_user
from common.config import get_settings
from common.tokens import create_token
from generation.vlm import ScriptedVLM
from retrieval.fusion import Hit
from retrieval.search import SearchResult

S = get_settings()
DB = S.data_dir / "mcp" / "maintenance.db"
pytestmark = pytest.mark.skipif(not DB.exists(), reason="base MCP absente : python scripts/generate_corpus.py --mcp-only")

PASSAGE = Hit(uid="manuel_baie_sn-400:v1:table_2_1_1", payload={
    "document_id": "manuel_baie_sn-400", "version": "v1", "page": 2, "element_id": "table_2_1_1", "type": "table",
    "famille": "baie", "titre": "Manuel d'utilisation SN-400", "section": "3. Panneau de contrôle",
    "text": "| L2 | orange | Disque défaillant ou reconstruction RAID |", "groupes_autorises": ["tech_stockage", "responsable"]})
FAKE = SearchResult(passages=[PASSAGE], images=[])

ANSWER = json.dumps({"status": "answered", "answer": "Le voyant orange signale un disque défaillant [S1].",
                     "citations": ["S1"], "uncertainties": [], "clarification_question": None, "proposed_action": None})
CHECK = json.dumps({"checks": [{"id": 1, "verdict": "soutenu", "raison": "ok"}]})
DRAFT = json.dumps({"title": "BAIE-SN400-002 : disque défaillant (voyant orange)",
                    "description": "Voyant L2 orange : disque défaillant (manuel SN-400, page 2). Remplacer le disque (PRC-B01)."})


@pytest.fixture
def db_copy(tmp_path, monkeypatch):
    copy = tmp_path / "maintenance.db"
    shutil.copy(DB, copy)

    def db():
        con = sqlite3.connect(copy)
        con.row_factory = sqlite3.Row
        return con
    monkeypatch.setattr(server, "_db", db)
    return copy


def make(*replies):
    mcp = InMemoryMCPClient()
    agent = build_agent(ScriptedVLM(*replies), mcp, search_fn=lambda *a: FAKE, checkpointer=InMemorySaver())
    return agent, mcp


def run(agent, user_id, question):
    config = {"configurable": {"thread_id": uuid.uuid4().hex, "auth_token": create_token(get_user(user_id))[0]}}
    return agent.invoke({"user_id": user_id, "question": question, "photo_b64": None}, config), config


# --- Analyse de la demande

def test_analyse():
    i = analyze("Voyant orange sur la baie-sn400-002 : peux-tu préparer un ticket ?")
    assert i.equipment_id == "BAIE-SN400-002" and i.wants_ticket and not i.wants_history
    i = analyze("Quelles sont les dernières interventions sur SW-FL48-002 ?")
    assert i.equipment_id == "SW-FL48-002" and i.wants_history and not i.wants_ticket
    i = analyze("Que signifie le voyant orange ?")
    assert i.equipment_id is None and not i.wants_history and not i.wants_ticket


# --- Parcours

def test_question_documentaire_sans_mcp(db_copy):
    agent, mcp = make(ANSWER, CHECK)
    out, _ = run(agent, "tech_s01", "Que signifie le voyant orange ?")
    assert out["status"] == "termine" and out["answer"]["status"] == "answered"
    assert mcp.calls == [] and out.get("ticket_draft") is None


def test_historique_via_mcp(db_copy):
    agent, mcp = make(ANSWER, CHECK)
    out, _ = run(agent, "tech_s01", "Historique des interventions sur BAIE-SN400-002 ?")
    assert mcp.calls == ["get_equipment_history", "get_open_tickets"]
    assert out["history"]["modele"] == "SN-400" and out["open_tickets"]["tickets"][0]["ticket_id"] == "TCK-0001"
    assert out["mcp_calls"] == 2 and out.get("ticket_draft") is None


def test_ticket_approuve_puis_cree(db_copy):
    agent, mcp = make(ANSWER, CHECK, DRAFT)
    out, config = run(agent, "tech_s01", "Voyant orange sur BAIE-SN400-002, prépare un ticket.")
    assert "__interrupt__" in out and out["status"] == "en_attente_approbation"
    draft = out["ticket_draft"]
    assert draft["equipment_id"] == "BAIE-SN400-002" and out["draft_origin"] == "modele"
    assert "create_ticket" not in mcp.calls                   # rien n'est créé avant l'approbation
    out = agent.invoke(Command(resume={"approved": True, "title": draft["title"], "description": draft["description"]}),
                       config)
    assert out["status"] == "ticket_cree" and out["ticket"]["created"] is True
    assert mcp.calls == ["get_equipment_history", "get_open_tickets", "create_ticket"] and out["mcp_calls"] == 3
    row = sqlite3.connect(db_copy).execute("SELECT titre, cree_par FROM tickets WHERE ticket_id = ?",
                                           (out["ticket"]["ticket_id"],)).fetchone()
    assert row == (draft["title"], "tech_s01")


def test_ticket_modifie_par_l_humain(db_copy):
    agent, _ = make(ANSWER, CHECK, DRAFT)
    out, config = run(agent, "tech_s01", "Voyant orange sur BAIE-SN400-002, ouvre un ticket")
    out = agent.invoke(Command(resume={"approved": True, "title": "Titre corrigé par le technicien",
                                       "description": out["ticket_draft"]["description"]}), config)
    assert out["status"] == "ticket_cree"
    titre = sqlite3.connect(db_copy).execute("SELECT titre FROM tickets WHERE ticket_id = ?",
                                             (out["ticket"]["ticket_id"],)).fetchone()[0]
    assert titre == "Titre corrigé par le technicien"       # c'est le contenu approuvé qui est créé


def test_ticket_refuse(db_copy):
    agent, mcp = make(ANSWER, CHECK, DRAFT)
    out, config = run(agent, "tech_s01", "Voyant orange sur BAIE-SN400-002, prépare un ticket.")
    out = agent.invoke(Command(resume={"approved": False}), config)
    assert out["status"] == "ticket_refuse" and out.get("ticket") is None and "create_ticket" not in mcp.calls


def test_ticket_sans_equipement_demande_une_clarification(db_copy):
    agent, mcp = make(ANSWER, CHECK)
    out, _ = run(agent, "tech_s01", "Le voyant est orange, prépare un ticket.")
    assert out["clarification_question"].startswith("Quel est l'identifiant") and mcp.calls == []
    assert out.get("ticket_draft") is None


def test_pas_de_ticket_sur_un_equipement_non_autorise(db_copy):
    agent, mcp = make(ANSWER, CHECK, DRAFT)
    out, _ = run(agent, "tech_r01", "Voyant orange sur BAIE-SN400-002, prépare un ticket.")
    assert out.get("ticket_draft") is None and out["status"] == "termine"
    assert any("non autorisé" in e for e in out["errors"]) and "create_ticket" not in mcp.calls


def test_budget_mcp(db_copy, monkeypatch):
    monkeypatch.setattr(S, "max_mcp_calls", 1)
    agent, mcp = make(ANSWER, CHECK)
    out, _ = run(agent, "tech_s01", "Historique de BAIE-SN400-002 ?")
    assert mcp.calls == ["get_equipment_history"] and out["mcp_calls"] == 1
    assert any("Budget MCP atteint" in e for e in out["errors"])


def test_brouillon_de_secours_si_sortie_invalide(db_copy):
    agent, _ = make(ANSWER, CHECK, "pas du JSON")
    out, _ = run(agent, "tech_s01", "Voyant orange sur BAIE-SN400-002, prépare un ticket.")
    assert out["draft_origin"] == "modele_de_texte"
    assert out["ticket_draft"]["title"].startswith("BAIE-SN400-002")
    assert "manuel_baie_sn-400, page 2" in out["ticket_draft"]["description"]
