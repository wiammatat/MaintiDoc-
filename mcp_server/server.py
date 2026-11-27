"""Serveur MCP de démonstration : historique des interventions et tickets de maintenance (données fictives).

Outils (section 6 du sujet) :
    get_equipment_history(equipment_id)
    get_open_tickets(equipment_id)
    create_ticket(equipment_id, title, description, approval_token)

Chaque outil reçoit aussi auth_token, le jeton de connexion de l'utilisateur. Le serveur vérifie à CHAQUE appel
l'identité et les droits sur l'équipement : il ne fait pas confiance à l'agent qui l'appelle.
create_ticket exige en plus un jeton d'approbation lié à l'utilisateur et au contenu exact du ticket,
et une clé d'idempotence empêche les doublons.

Journal d'audit (table audit_mcp) : chaque appel, accepté OU refusé, est enregistré côté serveur avec l'horodatage,
l'identifiant de la demande (thread de l'agent), l'outil, l'utilisateur, l'équipement, le résultat et le motif d'un
refus. Ni jeton ni contenu de ticket : seulement l'empreinte SHA-256 du contenu. Les lignes sont chaînées par hachage
(chaque ligne contient le hachage de la précédente) : une modification ou une suppression est détectable (verify_audit).

Lancement (transport stdio, utilisé par l'agent) : python -m mcp_server.server
"""

import hashlib
import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import date, datetime, timezone

from mcp.server.fastmcp import FastMCP

from common.access import User
from common.config import get_settings
from common.tokens import InvalidToken, idempotency_key, ticket_hash, user_from_token, verify_approval

MAX_TITLE = 120
MAX_DESCRIPTION = 4000
HISTORY_LIMIT = 10
MAX_REASON = 200
GENESIS = "0" * 64

AUDIT_SCHEMA = """CREATE TABLE IF NOT EXISTS audit_mcp (
    id INTEGER PRIMARY KEY AUTOINCREMENT, horodatage TEXT NOT NULL, request_id TEXT, outil TEXT NOT NULL,
    user_id TEXT, equipment_id TEXT, resultat TEXT NOT NULL, motif TEXT, empreinte_contenu TEXT, ticket_id TEXT,
    hash_precedent TEXT NOT NULL, hash TEXT NOT NULL)"""
AUDIT_FIELDS = ("horodatage", "request_id", "outil", "user_id", "equipment_id", "resultat", "motif",
                "empreinte_contenu", "ticket_id")

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


# --------------------------------------------------------------------------- journal d'audit

def _row_hash(previous: str, entry: dict) -> str:
    payload = json.dumps([previous] + [entry.get(f) for f in AUDIT_FIELDS], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _audit(entry: dict) -> None:
    """Ajoute une ligne au journal, dans sa propre transaction (indépendante de l'opération journalisée)."""
    entry = {f: entry.get(f) for f in AUDIT_FIELDS} | {"horodatage": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with closing(_db()) as con, con:
        con.execute(AUDIT_SCHEMA)
        last = con.execute("SELECT hash FROM audit_mcp ORDER BY id DESC LIMIT 1").fetchone()
        previous = last[0] if last else GENESIS
        con.execute(f"INSERT INTO audit_mcp ({', '.join(AUDIT_FIELDS)}, hash_precedent, hash) "
                    f"VALUES ({', '.join('?' * len(AUDIT_FIELDS))}, ?, ?)",
                    [entry[f] for f in AUDIT_FIELDS] + [previous, _row_hash(previous, entry)])


def verify_audit(con: sqlite3.Connection) -> list[int]:
    """Identifiants des lignes dont la chaîne de hachage est rompue (liste vide : journal intact)."""
    con.execute(AUDIT_SCHEMA)
    broken, previous = [], GENESIS
    for row in con.execute("SELECT * FROM audit_mcp ORDER BY id"):
        entry = {f: row[f] for f in AUDIT_FIELDS}
        if row["hash_precedent"] != previous or row["hash"] != _row_hash(previous, entry):
            broken.append(row["id"])
        previous = row["hash"]
    return broken


def _user_id(auth_token: str) -> str | None:
    try:
        return user_from_token(auth_token).user_id
    except InvalidToken:
        return None   # jeton invalide : l'appel est journalisé sans utilisateur


def _audited(tool: str, auth_token: str, equipment_id: str, request_id: str, run: Callable[[], dict],
             content_hash: str | None = None) -> dict:
    """Exécute l'outil et journalise le résultat, y compris les refus et les erreurs."""
    entry = {"outil": tool, "user_id": _user_id(auth_token), "equipment_id": equipment_id,
             "request_id": (request_id or "")[:64] or None, "empreinte_contenu": content_hash}
    try:
        result = run()
    except PermissionError as e:
        _audit(entry | {"resultat": "refuse", "motif": str(e)[:MAX_REASON]})
        raise
    except ValueError as e:
        _audit(entry | {"resultat": "invalide", "motif": str(e)[:MAX_REASON]})
        raise
    except Exception as e:
        _audit(entry | {"resultat": "erreur", "motif": type(e).__name__})
        raise
    status = "ok" if result.get("created", True) else "doublon"
    _audit(entry | {"resultat": status, "ticket_id": result.get("ticket_id")})
    return result


# --------------------------------------------------------------------------- outils

@mcp.tool()
def get_equipment_history(equipment_id: str, auth_token: str, request_id: str = "") -> dict:
    """Dernières interventions sur un équipement (les plus récentes d'abord)."""
    return _audited("get_equipment_history", auth_token, equipment_id, request_id,
                    lambda: _history(equipment_id, auth_token))


def _history(equipment_id: str, auth_token: str) -> dict:
    with closing(_db()) as con:
        _, equipment = _authorize(auth_token, equipment_id, con)
        rows = con.execute("SELECT date, type, code_alarme, description, technicien, duree_h FROM interventions "
                           "WHERE equipment_id = ? ORDER BY date DESC LIMIT ?", (equipment_id, HISTORY_LIMIT)).fetchall()
    return {"equipment_id": equipment_id, "modele": equipment["modele"], "site": equipment["site"],
            "interventions": [dict(r) for r in rows]}


@mcp.tool()
def get_open_tickets(equipment_id: str, auth_token: str, request_id: str = "") -> dict:
    """Tickets encore ouverts sur un équipement."""
    return _audited("get_open_tickets", auth_token, equipment_id, request_id,
                    lambda: _open_tickets(equipment_id, auth_token))


def _open_tickets(equipment_id: str, auth_token: str) -> dict:
    with closing(_db()) as con:
        _authorize(auth_token, equipment_id, con)
        rows = con.execute("SELECT ticket_id, titre, description, cree_par, cree_le FROM tickets "
                           "WHERE equipment_id = ? AND statut = 'ouvert' ORDER BY cree_le DESC", (equipment_id,)).fetchall()
    return {"equipment_id": equipment_id, "tickets": [dict(r) for r in rows]}


@mcp.tool()
def create_ticket(equipment_id: str, title: str, description: str, approval_token: str, auth_token: str,
                  request_id: str = "") -> dict:
    """Crée un ticket APRÈS approbation humaine du contenu exact. Un second envoi identique ne crée pas de doublon."""
    return _audited("create_ticket", auth_token, equipment_id, request_id,
                    lambda: _create_ticket(equipment_id, title, description, approval_token, auth_token),
                    content_hash=ticket_hash(equipment_id, title.strip(), description.strip()))


def _create_ticket(equipment_id: str, title: str, description: str, approval_token: str, auth_token: str) -> dict:
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
