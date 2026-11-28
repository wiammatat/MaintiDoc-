"""Métriques d'évaluation, sans dépendance au modèle ni à l'index (testables seules).

Recherche (pertinence binaire au niveau du document) :
    Recall@k  part des documents pertinents présents dans les k premiers documents distincts ;
    nDCG@k    gain actualisé normalisé : un document pertinent compte plus s'il est classé haut ;
    MRR@k     1 / rang du premier document pertinent (0 s'il n'est pas dans les k premiers).

Le classement est celui des documents distincts, dans l'ordre de leur premier passage : cinq passages du même
tableau d'alarmes ne comptent qu'une fois.
"""

import math
import re
import statistics
import unicodedata


def ranked_documents(document_ids: list[str]) -> list[str]:
    """Documents distincts, dans l'ordre de leur première apparition."""
    seen, out = set(), []
    for d in document_ids:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal


def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return next((1 / (i + 1) for i, d in enumerate(ranked[:k]) if d in relevant), 0.0)


def retrieval_scores(ranked: list[str], relevant: list[str]) -> dict:
    rel = set(relevant)
    return {"recall@5": recall_at_k(ranked, rel, 5), "ndcg@5": ndcg_at_k(ranked, rel, 5),
            "mrr@10": mrr_at_k(ranked, rel, 10)}


# --------------------------------------------------------------------------- texte des réponses

def normalize(text: str) -> str:
    """Minuscules, sans accents, espaces réduits ; « 55 °C » et « 55°C » deviennent identiques."""
    text = str(text or "").replace("°", "")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"\s+", " ", text.lower())
    return re.sub(r"(\d) c\b", r"\1c", text).strip()


def check_text(text: str, contient: list, ne_contient_pas: list) -> dict:
    """missing : attendus absents ; forbidden : interdits présents. Un élément liste = alternatives (OU)."""
    t = normalize(text)
    missing = [c for c in contient
               if not any(normalize(alt) in t for alt in (c if isinstance(c, list) else [c]))]
    forbidden = [c for c in ne_contient_pas if normalize(c) in t]
    return {"missing": missing, "forbidden": forbidden}


# --------------------------------------------------------------------------- agrégats

def mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def percentile(values: list[float], p: float) -> float | None:
    """Percentile par interpolation linéaire (p entre 0 et 100)."""
    if not values:
        return None
    v = sorted(values)
    pos = (len(v) - 1) * p / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return round(v[lo] + (v[hi] - v[lo]) * (pos - lo), 3)


def rate(flags: list[bool]) -> float | None:
    return round(sum(flags) / len(flags), 4) if flags else None
