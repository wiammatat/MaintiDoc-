"""Détection et neutralisation déterministes des consignes suspectes (injection de prompt) dans les sources.

Les documents sont des données, jamais des instructions. Le prompt le dit au modèle, mais un modèle 7B
ne l'applique pas de façon fiable : sur Kaggle, il a signalé des injections inexistantes et n'a pas signalé
la vraie. On retire donc ici, avant l'envoi au modèle, les paragraphes qui ressemblent à une consigne.
"""

import re

NEUTRALIZED = "[contenu retiré : consigne suspecte détectée dans la source]"

PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    r"\bignore[rz]?\s+(toutes\s+)?(les\s+|tes\s+|vos\s+)?(instructions|consignes|règles)\b",
    r"\boublie[rz]?\s+(toutes\s+)?(les\s+|tes\s+|vos\s+)?(instructions|consignes|règles)\b",
    r"\bignore\s+(all\s+|any\s+)?(previous|prior|above)\s+instructions\b",
    r"\[\s*(note\s+)?syst[eè]me?\s*\]|\[\s*system\s*\]",
    r"\b(envoie|envoyer|transmets|transmettre|send|upload|exfiltrate)\b[^.\n]{0,80}\bhttps?://",
    r"\b(accorde|donne|attribue|grant)\b[^.\n]{0,60}\b(droits?|permissions?|privil[eè]ges?|rights|access)\b",
    r"\btu es (désormais|maintenant)\b|\byou are now\b",
)]


def is_suspicious(text: str) -> bool:
    return any(p.search(text) for p in PATTERNS)


def neutralize(text: str) -> tuple[str, bool]:
    """Remplace chaque paragraphe suspect (ou phrase, s'il n'y a qu'un paragraphe). Renvoie (texte, modifié ?)."""
    parts = text.split("\n")
    if len(parts) == 1:
        parts = re.split(r"(?<=[.!?])\s+", text)
        sep = " "
    else:
        sep = "\n"
    cleaned = [NEUTRALIZED if is_suspicious(p) else p for p in parts]
    changed = cleaned != parts
    return (sep.join(cleaned) if changed else text), changed
