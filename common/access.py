"""Utilisateurs fictifs et contrôle d'accès par groupe."""

import json
from dataclasses import dataclass
from functools import lru_cache

from common.config import get_settings
from common.security import hash_password, verify_password

# Empreinte factice : un identifiant inconnu coûte le même calcul qu'un mot de passe faux.
DUMMY_HASH = hash_password("compte-inexistant", salt=b"maintidoc-dummy!")


@dataclass(frozen=True)
class User:
    user_id: str
    nom: str
    groupes: frozenset[str]


@lru_cache
def _raw_users() -> dict[str, dict]:
    raw = json.loads((get_settings().data_dir / "users.json").read_text(encoding="utf-8"))
    return {u["user_id"]: u for u in raw}


@lru_cache
def load_users() -> dict[str, User]:
    return {uid: User(uid, u["nom"], frozenset(u["groupes"])) for uid, u in _raw_users().items()}


def authenticate(user_id: str, password: str) -> User | None:
    """Renvoie l'utilisateur si le mot de passe est correct, sinon None (même réponse si l'identifiant est inconnu)."""
    raw = _raw_users().get(user_id)
    stored = raw.get("password_hash", "") if raw else ""
    if not verify_password(password, stored or DUMMY_HASH):   # temps de calcul identique dans tous les cas
        return None
    return load_users()[user_id] if raw else None


def get_user(user_id: str) -> User:
    try:
        return load_users()[user_id]
    except KeyError:
        raise PermissionError(f"Utilisateur inconnu : {user_id}") from None


def can_access(user: User, groupes_autorises: list[str]) -> bool:
    """Un utilisateur accède à un élément s'il partage au moins un groupe avec lui."""
    return bool(user.groupes & set(groupes_autorises))
