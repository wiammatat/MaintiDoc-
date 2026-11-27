"""Brouillon de ticket : rédigé par le VLM (prompt ticket_draft_v1), sinon par un modèle de texte déterministe.

Le brouillon n'est qu'une proposition : sa création exige l'approbation humaine du contenu exact.
"""

import re

from pydantic import BaseModel, Field, ValidationError

from generation.prompting import parse, render
from generation.vlm import VLM, VLMUnavailable
from retrieval.sanitize import neutralize

MAX_TITLE = 120
MAX_DESCRIPTION = 4000


class DraftOutput(BaseModel):
    """Ce que le VLM doit produire."""
    title: str = Field(min_length=1, max_length=MAX_TITLE)
    description: str = Field(min_length=1, max_length=MAX_DESCRIPTION)


class TicketDraft(BaseModel):
    equipment_id: str
    title: str = Field(min_length=1, max_length=MAX_TITLE)
    description: str = Field(min_length=1, max_length=MAX_DESCRIPTION)


def _history_lines(history: dict | None, limit: int = 3) -> str:
    rows = (history or {}).get("interventions", [])[:limit]
    # L'historique vient d'un outil : c'est une donnée non fiable, nettoyée comme les documents.
    return "\n".join(f"- {r['date']} ({r['type']}) : {neutralize(r['description'])[0]}" for r in rows) or "aucune"


def _open_lines(open_tickets: dict | None) -> str:
    rows = (open_tickets or {}).get("tickets", [])
    return "\n".join(f"- {t['ticket_id']} : {neutralize(t['titre'])[0]}" for t in rows) or "aucun"


def _citation_lines(answer: dict) -> str:
    return "\n".join(f"- {c['document_id']}, page {c['page']} ({c['element_id']})"
                     for c in answer.get("citations", [])) or "aucune"


def template_draft(equipment_id: str, question: str, answer: dict, history: dict | None,
                   open_tickets: dict | None) -> TicketDraft:
    """Brouillon sans modèle : sûr et reproductible."""
    first = re.split(r"(?<=[.?!])\s", question.strip())[0]
    title = f"{equipment_id} - {first}"[:MAX_TITLE]
    description = (f"Demande : {question.strip()}\n\nCe que dit la documentation : {answer.get('answer', '')}\n\n"
                   f"Références :\n{_citation_lines(answer)}\n\nInterventions récentes :\n{_history_lines(history)}\n\n"
                   f"Tickets ouverts :\n{_open_lines(open_tickets)}")
    return TicketDraft(equipment_id=equipment_id, title=title, description=description[:MAX_DESCRIPTION])


def draft_ticket(vlm: VLM | None, equipment_id: str, question: str, answer: dict, history: dict | None,
                 open_tickets: dict | None, max_calls: int) -> tuple[TicketDraft, int, str]:
    """Renvoie (brouillon, appels au modèle, origine : « modele » ou « modele_de_texte »)."""
    if vlm is None or max_calls <= 0:
        return template_draft(equipment_id, question, answer, history, open_tickets), 0, "modele_de_texte"
    system, user = render("ticket_draft_v1", equipment_id=equipment_id, question=question,
                          answer=answer.get("answer", ""), citations=_citation_lines(answer),
                          history=_history_lines(history, limit=5), open_tickets=_open_lines(open_tickets))
    try:
        out = parse(DraftOutput, vlm.generate(system, user, []))   # validation par PydanticOutputParser (LangChain)
        return TicketDraft(equipment_id=equipment_id, title=out.title, description=out.description), 1, "modele"
    except (VLMUnavailable, ValueError, TypeError, AttributeError, ValidationError):
        # Sortie invalide ou VLM indisponible : on retombe sur le modèle de texte plutôt que d'échouer.
        return template_draft(equipment_id, question, answer, history, open_tickets), 1, "modele_de_texte"
