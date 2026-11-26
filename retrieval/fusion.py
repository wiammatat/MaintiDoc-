"""Fusion des résultats texte + image par Reciprocal Rank Fusion (RRF)."""

from dataclasses import dataclass, field

RRF_K = 60
# Poids des listes dans la fusion. CLIP rapproche mal une question générale en français
# (« Que signifie ce voyant ? ») des pages de texte : ce signal est plus faible qu'une photo.
RRF_WEIGHTS = {"texte": 1.0, "image_par_photo": 1.0, "image_par_question": 0.5}


@dataclass
class Hit:
    uid: str
    payload: dict
    rrf_score: float = 0.0
    rerank_score: float | None = None
    ranks: dict[str, int] = field(default_factory=dict)   # rang dans chaque liste d'origine

    @property
    def citation(self) -> dict:
        return {k: self.payload[k] for k in ("document_id", "page", "element_id")}


def reciprocal_rank_fusion(ranked_lists: dict[str, list[tuple[str, dict]]], k: int = RRF_K,
                           weights: dict[str, float] | None = None) -> list[Hit]:
    """Fusionne des listes classées : score(d) = somme sur les listes de poids / (k + rang)."""
    hits: dict[str, Hit] = {}
    for name, ranked in ranked_lists.items():
        weight = (weights or {}).get(name, 1.0)
        for rank, (uid, payload) in enumerate(ranked, 1):
            hit = hits.setdefault(uid, Hit(uid=uid, payload=payload))
            hit.rrf_score += weight / (k + rank)
            hit.ranks[name] = rank
    return sorted(hits.values(), key=lambda h: h.rrf_score, reverse=True)
