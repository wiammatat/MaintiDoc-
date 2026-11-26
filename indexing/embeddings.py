"""Modèles d'embeddings, chargés une seule fois à la première utilisation.

- Texte : BGE-M3 (dense, multilingue).
- Images : CLIP ViT-B/32 pour les images et sa version multilingue pour le texte,
  dans le même espace vectoriel (recherche d'images par une question ou par une photo).
"""

import os
from functools import lru_cache

# Empêche transformers de télécharger en arrière-plan une copie convertie (2 Go) des poids de BGE-M3.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from common.config import get_settings  # noqa: E402

BATCH_SIZE = 16


def model_kwargs() -> dict:
    """Arguments de chargement communs : précision définie par MODEL_DTYPE."""
    return {"dtype": getattr(torch, get_settings().model_dtype)}


@lru_cache
def _model(name: str) -> SentenceTransformer:
    return SentenceTransformer(name, device=get_settings().device, model_kwargs=model_kwargs())


def text_dim() -> int:
    return _model(get_settings().text_embedding_model).get_embedding_dimension()


def image_dim() -> int:
    return _model(get_settings().image_encoder_model).get_embedding_dimension()


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
