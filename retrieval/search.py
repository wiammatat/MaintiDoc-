"""Étape 4 : RAG multimodal, recherche texte + image.

1. vérifier l'utilisateur et ses groupes ;
2. rechercher dans l'index textuel (LlamaIndex + BGE-M3) et dans l'index visuel (CLIP, par la question
   et par la photo), en filtrant par groupe AVANT la récupération ;
3. fusionner les listes par Reciprocal Rank Fusion ;
4. reclasser les candidats textuels (passages, tableaux, OCR, légendes) avec le reranker ;
5. contrôler à nouveau l'accès sur chaque résultat (défense en profondeur).
"""

from dataclasses import dataclass

from PIL import Image

from common.access import User, can_access
from common.config import get_settings
from indexing.embeddings import embed_images, embed_texts_for_images
from indexing.text_index import search_text
from indexing.visual_index import search_visual
from retrieval.fusion import RRF_WEIGHTS, Hit, reciprocal_rank_fusion
from retrieval.reranker import rerank


@dataclass
class SearchResult:
    passages: list[Hit]   # éléments textuels reclassés
    images: list[Hit]     # figures et pages les plus pertinentes


def search(user: User, question: str, image: Image.Image | None = None, *, filters: dict | None = None,
           k_text: int = 20, k_image: int = 10, top_passages: int = 5, top_images: int = 3,
           reranker: str | None = "default", rerank_candidates: int | None = None) -> SearchResult:
    """Recherche les éléments pertinents accessibles à l'utilisateur.

    filters : contraintes supplémentaires, ex. {"famille": "baie", "modele": "SN-400"}.
    reranker : "default" (modèle de la configuration), un nom ou chemin de modèle, ou None (sans reranking).
    rerank_candidates : nombre de passages relus par le reranker, pris en tête de la fusion (défaut :
        RERANK_CANDIDATES ; 0 = tous). Le reranker coûte environ 1 s par candidat sur CPU.
    """
    if not user.groupes:
        raise PermissionError(f"L'utilisateur {user.user_id} n'appartient à aucun groupe")
    s = get_settings()
    groups = sorted(user.groupes)

    ranked = {
        "texte": search_text(question, groups, filters, k_text),
        "image_par_question": search_visual(embed_texts_for_images([question])[0], groups, filters, k_image),
    }
    if image is not None:
        ranked["image_par_photo"] = search_visual(embed_images([image])[0], groups, filters, k_image)

    # Défense en profondeur : on ne garde que ce que l'utilisateur a le droit de voir.
    fused = [h for h in reciprocal_rank_fusion(ranked, weights=RRF_WEIGHTS)
             if can_access(user, h.payload["groupes_autorises"])]

    passages = [h for h in fused if h.payload.get("text")]
    if reranker:
        n = s.rerank_candidates if rerank_candidates is None else rerank_candidates
        passages = rerank(question, passages[:n] if n else passages,
                          s.reranker_model if reranker == "default" else reranker)
    images = [h for h in fused if h.payload["type"] in ("figure", "page")]
    return SearchResult(passages=passages[:top_passages], images=images[:top_images])
