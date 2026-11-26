"""Étape 3b : index visuel avec CLIP, stocké dans Qdrant.

Figures et pages sont encodées par CLIP ViT-B/32. On peut y chercher avec une photo
(encodeur d'images) ou avec une question en français (encodeur de texte multilingue, même espace).
"""

from PIL import Image
from qdrant_client import models

from common.config import get_settings
from indexing.embeddings import embed_images, image_dim
from indexing.qdrant import access_filter, client, create_payload_indexes, delete_documents, payload, point_id
from preparation.schema import Element

IMAGE_TYPES = {"figure", "page"}


def build_visual_index(elements: list[Element], recreate: bool = False) -> int:
    """(Ré)indexe les figures et les pages. Renvoie le nombre d'images indexées."""
    s = get_settings()
    name = s.image_collection
    if recreate and client().collection_exists(name):
        client().delete_collection(name)
    if not client().collection_exists(name):
        client().create_collection(name, vectors_config=models.VectorParams(size=image_dim(),
                                                                            distance=models.Distance.COSINE))
        create_payload_indexes(name)
    else:
        delete_documents(name, sorted({e.document_id for e in elements}))

    images = [e for e in elements if e.type in IMAGE_TYPES and e.image_path]
    if images:
        vectors = embed_images([Image.open(s.data_dir / e.image_path) for e in images])
        client().upsert(name, wait=True, points=[
            models.PointStruct(id=point_id(e.uid), vector=v.tolist(), payload=payload(e))
            for e, v in zip(images, vectors)])
    return len(images)


def search_visual(vector, groups: list[str], filters: dict | None = None, k: int = 10) -> list[tuple[str, dict]]:
    """Recherche par similarité CLIP, filtrée par groupe AVANT la récupération. Renvoie [(uid, payload)]."""
    points = client().query_points(get_settings().image_collection, query=vector.tolist(),
                                   query_filter=access_filter(groups, filters), limit=k, with_payload=True).points
    return [(p.payload["uid"], p.payload) for p in points]
