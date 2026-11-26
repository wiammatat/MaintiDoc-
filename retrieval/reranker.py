"""Reranker multilingue (BGE-reranker-v2-m3) : relit la question avec chaque candidat textuel."""

from functools import lru_cache

from sentence_transformers import CrossEncoder

from common.config import get_settings
from indexing.embeddings import model_kwargs
from indexing.text_index import context_text
from retrieval.fusion import Hit


@lru_cache
def get_reranker(name: str) -> CrossEncoder:
    model = CrossEncoder(name, device=get_settings().device, model_kwargs=model_kwargs())
    # Garde-fou : un modèle mal chargé (ex. tokenizer absent du cache) donne le même score à tout.
    good, bad = model.predict([("alarme de température", "Alarme : température du contrôleur trop élevée."),
                               ("alarme de température", "Recette de la tarte aux pommes.")])
    if not good > bad:
        raise RuntimeError(f"Reranker {name} incohérent (scores {good:.3g} / {bad:.3g}) : vider son cache et le "
                           "retélécharger")
    return model


def rerank(question: str, hits: list[Hit], model_name: str) -> list[Hit]:
    """Reclasse les candidats ; le texte comparé est le même que celui encodé dans l'index (titre + section)."""
    if not hits:
        return hits
    scores = get_reranker(model_name).predict([(question, context_text(h.payload)) for h in hits], batch_size=16)
    for hit, score in zip(hits, scores):
        hit.rerank_score = float(score)
    return sorted(hits, key=lambda h: h.rerank_score, reverse=True)
