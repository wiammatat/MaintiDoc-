"""Serveur MCP de démonstration : historique des interventions et tickets de maintenance (données fictives).

Outils (section 6 du sujet) :
    get_equipment_history(equipment_id)
    get_open_tickets(equipment_id)
    create_ticket(equipment_id, title, description, approval_token)

Chaque outil reçoit aussi auth_token, le jeton de connexion de l'utilisateur. Le serveur vérifie à CHAQUE appel
l'identité et les droits sur l'équipement : il ne fait pas confiance à l'agent qui l'appelle.
create_ticket exige en plus un jeton d'approbation lié à l'utilisateur et au contenu exact du ticket,
et une clé d'idempotence empêche les doublons.

Lancement (transport stdio, utilisé par l'agent) : python -m mcp_server.server
"""

import sqlite3
from contextlib import closing
from datetime import date

from mcp.server.fastmcp import FastMCP

from common.access import User
from common.config import get_settings
from common.tokens import InvalidToken, idempotency_key, ticket_hash, user_from_token, verify_approval

MAX_TITLE = 120
MAX_DESCRIPTION = 4000
HISTORY_LIMIT = 10

mcp = FastMCP("maintidoc-maintenance")


def _db() -> sqlite3.Connection:
    path = get_settings().data_dir / "mcp" / "maintenance.db"
    if not path.exists():
        raise RuntimeError("Base MCP absente : lancer python scripts/generate_corpus.py --mcp-only")
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def _authorize(auth_token: str, equipment_id: str, con: sqlite3.Connection) -> tuple[User, sqlite3.Row]:
    """Vérifie l'utilisateur et son droit sur l'équipement. Équipement inexistant ou interdit : même message."""
    try:
        user = user_from_token(auth_token)
    except InvalidToken as e:
        raise PermissionError(f"Accès refusé : {e}") from None
    equipment = con.execute("SELECT * FROM equipements WHERE equipment_id = ?", (equipment_id,)).fetchone()
    if equipment is None or not (user.groupes & {equipment["groupe"], "responsable"}):
        raise PermissionError(f"Équipement {equipment_id} introuvable ou non autorisé")
    return user, equipment


@mcp.tool()
def get_equipment_history(equipment_id: str, auth_token: str) -> dict:
    """Dernières interventions sur un équipement (les plus récentes d'abord)."""
    with closing(_db()) as con:
        _, equipment = _authorize(auth_token, equipment_id, con)
        rows = con.execute("SELECT date, type, code_alarme, description, technicien, duree_h FROM interventions "
                           "WHERE equipment_id = ? ORDER BY date DESC LIMIT ?", (equipment_id, HISTORY_LIMIT)).fetchall()
    return {"equipment_id": equipment_id, "modele": equipment["modele"], "site": equipment["site"],
            "interventions": [dict(r) for r in rows]}


@mcp.tool()
def get_open_tickets(equipment_id: str, auth_token: str) -> dict:
    """Tickets encore ouverts sur un équipement."""
    with closing(_db()) as con:
        _authorize(auth_token, equipment_id, con)
        rows = con.execute("SELECT ticket_id, titre, description, cree_par, cree_le FROM tickets "
                           "WHERE equipment_id = ? AND statut = 'ouvert' ORDER BY cree_le DESC", (equipment_id,)).fetchall()
    return {"equipment_id": equipment_id, "tickets": [dict(r) for r in rows]}


@mcp.tool()
def create_ticket(equipment_id: str, title: str, description: str, approval_token: str, auth_token: str) -> dict:
    """Crée un ticket APRÈS approbation humaine du contenu exact. Un second envoi identique ne crée pas de doublon."""
    title, description = title.strip(), description.strip()
    if not title or len(title) > MAX_TITLE or not description or len(description) > MAX_DESCRIPTION:
        raise ValueError(f"Titre (1 à {MAX_TITLE} caractères) et description (1 à {MAX_DESCRIPTION}) obligatoires")
    with closing(_db()) as con:
        user, _ = _authorize(auth_token, equipment_id, con)
        content = ticket_hash(equipment_id, title, description)
        try:
            verify_approval(approval_token, user.user_id, content)
        except InvalidToken as e:
            raise PermissionError(f"Création refusée : {e}") from None
        key = idempotency_key(user.user_id, content)
        existing = con.execute("SELECT ticket_id FROM tickets WHERE idempotency_key = ?", (key,)).fetchone()
        if existing:
            return {"ticket_id": existing["ticket_id"], "created": False, "message": "Ticket déjà créé (aucun doublon)"}
        with con:   # transaction
            n = con.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] + 1
            ticket_id = f"TCK-{n:04d}"
            con.execute("INSERT INTO tickets VALUES (?, ?, ?, ?, 'ouvert', ?, ?, ?)",
                        (ticket_id, equipment_id, title, description, user.user_id, date.today().isoformat(), key))
    return {"ticket_id": ticket_id, "created": True, "message": "Ticket créé"}


if __name__ == "__main__":
    mcp.run(transport="stdio")
