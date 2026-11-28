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
# Demande de ticket : il faut un verbe d'action qui porte sur le ticket. Le mot « ticket » seul ne suffit pas :
# « Que vérifier avant d'ouvrir un ticket ? » ou « Qui approuve un ticket ? » sont des questions d'information
# (questions Q07, Q08, Q12 du prototype du professeur, scénarios HUM-009, HUM-010, HUM-014).
_TICKET = r"(?:tickets?|demandes?\s+d['’]intervention)"
_GAP = r"[\s\-]+(?:\S+\s+){0,3}?"                                   # au plus 3 mots : « crée directement un ticket »
_IMPERATIF = r"(?:cr[ée]e|cr[ée]ez|ouvre|ouvrez|pr[ée]pare|pr[ée]parez|fais|faites|r[ée]dige|r[ée]digez|lance|lancez)"
_INFINITIF = r"(?:cr[ée]er|ouvrir|pr[ée]parer|faire|r[ée]diger|lancer)"
_DEMANDE = r"(?:peux[\s\-]tu|pouvez[\s\-]vous|tu\s+peux|vous\s+pouvez|je\s+veux|je\s+voudrais|j['’]aimerais|merci\s+de|il\s+faut)"
TICKET_RE = re.compile(
    rf"\b{_IMPERATIF}{_GAP}{_TICKET}\b"                             # « ouvre un ticket », « prépare-moi un ticket »
    rf"|\b{_DEMANDE}\s+(?:\S+\s+)?{_INFINITIF}{_GAP}{_TICKET}\b"    # « peux-tu préparer un ticket »
    rf"|(?:^|[.!?;:,]\s*){_INFINITIF}{_GAP}{_TICKET}\b"             # « Créer un ticket pour … » en début de phrase
    r"|\b(?:signaler|signale|d[ée]clarer|d[ée]clare)\s+(?:une|un|la|le)\s+(?:panne|incident|d[ée]faut|probl[eè]me)\b",
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
