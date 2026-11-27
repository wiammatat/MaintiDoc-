"""Reformulation de la question pour la boucle de recherche contrôlée de l'agent.

La reformulation est produite par le VLM (prompt query_rewrite_v1, validée par LangChain), puis contrôlée :
les identifiants d'équipement, les modèles et les codes de la question d'origine doivent y figurer. S'ils
manquent, ils sont rajoutés : une reformulation ne doit jamais faire glisser la recherche vers un autre équipement.
"""

import re

from pydantic import BaseModel, Field

from agent.intent import EQUIPMENT_RE
from generation.guards import models_in_question
from generation.prompting import parse, render
from generation.vlm import VLM, VLMUnavailable

CODE_RE = re.compile(r"\b(?:[EF]\d{2}|PRC-[A-Z]\d{2}|SEC-[A-Z]\d{2})\b", re.IGNORECASE)


class QueryRewrite(BaseModel):
    query: str = Field(min_length=3, max_length=300)


def identifiers(text: str) -> list[str]:
    """Identifiants à préserver : équipements, modèles, codes d'alarme et de procédure."""
    found = [m.group(0).upper() for m in EQUIPMENT_RE.finditer(text)] + models_in_question(text)
    found += [m.group(0).upper() for m in CODE_RE.finditer(text)]
    return list(dict.fromkeys(found))


def keep_identifiers(original: str, rewritten: str) -> str:
    missing = [i for i in identifiers(original) if i.lower() not in rewritten.lower()]
    return f"{rewritten} {' '.join(missing)}".strip() if missing else rewritten


def rewrite_query(vlm: VLM, question: str, previous_query: str, outcome: str) -> str | None:
    """Nouvelle requête de recherche, ou None si le modèle n'a pas produit de reformulation valide."""
    system, user = render("query_rewrite_v1", question=question, previous_query=previous_query, outcome=outcome)
    try:
        out = parse(QueryRewrite, vlm.generate(system, user, []))
    except (ValueError, VLMUnavailable):
        return None
    query = keep_identifiers(question, out.query.strip())
    return query if query.lower() != previous_query.lower() else None
