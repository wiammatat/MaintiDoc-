"""Garde-fous déterministes, appliqués avant d'appeler le VLM.

Équipement sans source accessible : les baies et les switchs ont des faces avant identiques mais des voyants
de sens différent. Sur Kaggle, un technicien réseau a demandé le sens du voyant orange d'une baie SN-400 ;
sans aucun document de baie autorisé pour lui, le modèle a répondu avec le sens du voyant d'un switch.
Si la question cite un modèle pour lequel l'utilisateur n'a accès à aucun document, on s'abstient.

La règle porte sur les droits d'accès, pas sur les premiers résultats de la recherche : une question
qui compare deux modèles ne doit pas être bloquée parce que l'un des deux est moins bien classé.
"""

import json
import re
from collections import defaultdict
from functools import lru_cache

from common.access import User, can_access
from common.config import get_settings


@lru_cache
def _metadata() -> list[dict]:
    return json.loads((get_settings().corpus_dir / "metadata.json").read_text(encoding="utf-8"))


@lru_cache
def model_groups() -> dict[str, list[str]]:
    """Modèle -> groupes ayant accès à au moins un document propre à ce modèle (manuel, alarmes, fiche...)."""
    groups: dict[str, set[str]] = defaultdict(set)
    for m in _metadata():
        if m.get("modele"):
            groups[m["modele"]].update(m["groupes_autorises"])
    return {k: sorted(v) for k, v in groups.items()}


def models_in_question(question: str) -> list[str]:
    """Modèles cités dans la question, avec ou sans tiret (« SN-400 », « sn400 »)."""
    found = []
    for model in model_groups():
        pattern = r"\b" + r"-?".join(re.escape(part) for part in model.split("-")) + r"\b"
        if re.search(pattern, question, re.IGNORECASE):
            found.append(model)
    return found


def models_without_access(question: str, user: User) -> list[str]:
    """Modèles cités pour lesquels l'utilisateur n'a accès à aucun document propre au modèle."""
    return [m for m in models_in_question(question) if not can_access(user, model_groups()[m])]
