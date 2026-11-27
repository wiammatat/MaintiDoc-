"""Vérifie le serveur MCP (protocole MCP réel, en mémoire) : droits à chaque appel, approbation, idempotence."""

import shutil
import sqlite3
import time

import pytest

import mcp_server.server as server
from agent.mcp_client import InMemoryMCPClient, MCPError
from common.access import get_user
from common.config import get_settings
from common.tokens import APPROVAL_TTL_SECONDS, create_approval, create_token, ticket_hash

DB = get_settings().data_dir / "mcp" / "maintenance.db"
pytestmark = pytest.mark.skipif(not DB.exists(), reason="base MCP absente : python scripts/generate_corpus.py --mcp-only")


@pytest.fixture
def mcp(tmp_path, monkeypatch):
    """Copie de la base : les tickets créés par les tests ne touchent pas la vraie base."""
    copy = tmp_path / "maintenance.db"
    shutil.copy(DB, copy)

    def db():
        con = sqlite3.connect(copy)
        con.row_factory = sqlite3.Row
        return con
    monkeypatch.setattr(server, "_db", db)
    return InMemoryMCPClient()


def tok(user_id: str) -> str:
    return create_token(get_user(user_id))[0]


def approve(user_id, eq, title, desc, now=None):
    return create_approval(user_id, ticket_hash(eq, title, desc), now=now)


def ticket(eq="BAIE-SN400-002", title="Disque défaillant tiroir 3", desc="Alarme E01, voyant orange."):
    return {"equipment_id": eq, "title": title, "description": desc}


# --- Lecture

def test_historique_et_tickets_ouverts(mcp):
    h = mcp.call("get_equipment_history", {"equipment_id": "BAIE-SN400-002", "auth_token": tok("tech_s01")})
    assert h["modele"] == "SN-400" and h["interventions"]
    dates = [i["date"] for i in h["interventions"]]
    assert dates == sorted(dates, reverse=True)
    t = mcp.call("get_open_tickets", {"equipment_id": "BAIE-SN400-002", "auth_token": tok("tech_s01")})
    assert [x["ticket_id"] for x in t["tickets"]] == ["TCK-0001"]


def test_droits_verifies_a_chaque_appel(mcp):
    with pytest.raises(MCPError, match="non autorisé"):
        mcp.call("get_equipment_history", {"equipment_id": "BAIE-SN400-002", "auth_token": tok("tech_r01")})
    with pytest.raises(MCPError, match="jeton"):
        mcp.call("get_open_tickets", {"equipment_id": "BAIE-SN400-002", "auth_token": "faux"})
    # Le responsable voit les deux familles.
    for eq in ("BAIE-SN400-002", "SW-FL48-002"):
        assert mcp.call("get_open_tickets", {"equipment_id": eq, "auth_token": tok("resp_01")})["tickets"]


def test_equipement_inexistant_et_interdit_meme_message(mcp):
    msgs = []
    for eq, user in (("BAIE-INCONNU-999", "tech_s01"), ("BAIE-SN400-002", "tech_r01")):
        with pytest.raises(MCPError) as e:
            mcp.call("get_equipment_history", {"equipment_id": eq, "auth_token": tok(user)})
        msgs.append(str(e.value).replace(eq, "X"))
    assert msgs[0] == msgs[1]


# --- Création de ticket

def test_creation_sans_approbation_refusee(mcp):
    with pytest.raises(MCPError, match="approbation"):
        mcp.call("create_ticket", ticket() | {"approval_token": "123.abc", "auth_token": tok("tech_s01")})


def test_creation_approuvee_puis_idempotente(mcp):
    t = ticket()
    args = t | {"approval_token": approve("tech_s01", **_k(t)), "auth_token": tok("tech_s01")}
    first = mcp.call("create_ticket", args)
    assert first["created"] is True and first["ticket_id"].startswith("TCK-")
    second = mcp.call("create_ticket", args)            # même envoi : pas de doublon
    assert second["created"] is False and second["ticket_id"] == first["ticket_id"]
    t2 = mcp.call("get_open_tickets", {"equipment_id": "BAIE-SN400-002", "auth_token": tok("tech_s01")})
    assert sum(x["ticket_id"] == first["ticket_id"] for x in t2["tickets"]) == 1


def test_contenu_modifie_apres_approbation_refuse(mcp):
    t = ticket()
    token = approve("tech_s01", **_k(t))
    with pytest.raises(MCPError, match="ne correspond pas"):
        mcp.call("create_ticket", t | {"description": "Contenu modifié après l'approbation",
                                       "approval_token": token, "auth_token": tok("tech_s01")})


def test_approbation_d_un_autre_utilisateur_refusee(mcp):
    t = ticket()
    token = approve("resp_01", **_k(t))                 # approuvé par quelqu'un d'autre
    with pytest.raises(MCPError, match="ne correspond pas"):
        mcp.call("create_ticket", t | {"approval_token": token, "auth_token": tok("tech_s01")})


def test_approbation_expiree_refusee(mcp):
    t = ticket()
    token = approve("tech_s01", **_k(t), now=time.time() - APPROVAL_TTL_SECONDS - 5)
    with pytest.raises(MCPError, match="expirée"):
        mcp.call("create_ticket", t | {"approval_token": token, "auth_token": tok("tech_s01")})


def test_creation_sur_equipement_interdit_refusee(mcp):
    t = ticket()
    token = approve("tech_r01", **_k(t))                # approbation valide, mais pas de droit sur la baie
    with pytest.raises(MCPError, match="non autorisé"):
        mcp.call("create_ticket", t | {"approval_token": token, "auth_token": tok("tech_r01")})


def test_contenu_vide_ou_trop_long_refuse(mcp):
    for t in (ticket(title=""), ticket(title="x" * 121)):
        with pytest.raises(MCPError):
            mcp.call("create_ticket", t | {"approval_token": approve("tech_s01", **_k(t)), "auth_token": tok("tech_s01")})


def _k(t: dict) -> dict:
    return {"eq": t["equipment_id"], "title": t["title"], "desc": t["description"]}
