"""Utilisateurs fictifs et contrôle d'accès par groupe."""

import json
from dataclasses import dataclass
from functools import lru_cache

from common.config import get_settings


@dataclass(frozen=True)
class User:
    user_id: str
    nom: str
    groupes: frozenset[str]


@lru_cache
def load_users() -> dict[str, User]:
    raw = json.loads((get_settings().data_dir / "users.json").read_text(encoding="utf-8"))
    return {u["user_id"]: User(u["user_id"], u["nom"], frozenset(u["groupes"])) for u in raw}


def get_user(user_id: str) -> User:
    try:
        return load_users()[user_id]
    except KeyError:
        raise PermissionError(f"Utilisateur inconnu : {user_id}") from None


def can_access(user: User, groupes_autorises: list[str]) -> bool:
    """Un utilisateur accède à un élément s'il partage au moins un groupe avec lui."""
    return bool(user.groupes & set(groupes_autorises))
