"""Modèles d'embeddings, chargés une seule fois à la première utilisation.

- Texte : BGE-M3 (dense, multilingue).
- Images : CLIP ViT-B/32 pour les images et sa version multilingue pour le texte,
  dans le même espace vectoriel (recherche d'images par une question ou par une photo).
"""

from functools import lru_cache

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from common.config import get_settings

BATCH_SIZE = 16


@lru_cache
def _model(name: str) -> SentenceTransformer:
    return SentenceTransformer(name, device="cpu")


def text_dim() -> int:
    return _model(get_settings().text_embedding_model).get_sentence_embedding_dimension()


def image_dim() -> int:
    return _model(get_settings().image_encoder_model).get_sentence_embedding_dimension()


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embeddings BGE-M3 normalisés, pour les passages comme pour les questions."""
    return _model(get_settings().text_embedding_model).encode(
        texts, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=len(texts) > BATCH_SIZE)


def embed_images(images: list[Image.Image]) -> np.ndarray:
    """Embeddings CLIP normalisés d'images."""
    return _model(get_settings().image_encoder_model).encode(
        [im.convert("RGB") for im in images], batch_size=BATCH_SIZE, normalize_embeddings=True,
        show_progress_bar=len(images) > BATCH_SIZE)


def embed_texts_for_images(texts: list[str]) -> np.ndarray:
    """Embeddings CLIP multilingues d'un texte, comparables aux embeddings d'images."""
    return _model(get_settings().image_embedding_model).encode(texts, normalize_embeddings=True)
