"""Analyse de la demande : équipement concerné, besoin d'historique, demande de ticket.

Routage déterministe plutôt que décidé par le VLM : il déclenche des outils qui agissent (MCP, création de
ticket) et doit donc être fiable, reproductible et testable. Un modèle de 7 milliards de paramètres ne l'est pas
assez (voir docs/choix_techniques.md, essais Kaggle). Le VLM reste utilisé pour répondre et rédiger le ticket.
"""

import re
from dataclasses import asdict, dataclass

EQUIPMENT_RE = re.compile(r"\b(?:BAIE|SW)-[A-Z]{2}\d{2,3}-\d{3}\b", re.IGNORECASE)
HISTORY_RE = re.compile(r"\b(historique|interventions?|derni[eè]res?\s+(pannes?|alarmes?|interventions?)|"
                        r"d[ée]j[àa]\s+(eu|arriv[ée]|vu)|ant[ée]c[ée]dents?)\b", re.IGNORECASE)
TICKET_RE = re.compile(r"\b(tickets?|signaler|d[ée]clarer\s+(une|la)\s+panne|ouvr(?:e|ez|ir)\s+une\s+(demande|intervention))\b",
                       re.IGNORECASE)


@dataclass
class Intent:
    equipment_id: str | None
    wants_history: bool
    wants_ticket: bool

    def to_dict(self) -> dict:
        return asdict(self)


def analyze(question: str) -> Intent:
    match = EQUIPMENT_RE.search(question)
    return Intent(equipment_id=match.group(0).upper() if match else None,
                  wants_history=bool(HISTORY_RE.search(question)),
                  wants_ticket=bool(TICKET_RE.search(question)))
