"""Indexation des éléments dans Qdrant : une collection texte et une collection images.

Usage :
    python -m ingestion.index              # indexe data/processed/elements.jsonl
    python -m ingestion.index --recreate   # supprime et recrée les collections

Le contrôle d'accès repose sur le champ groupes_autorises de chaque point,
filtré au moment de la recherche (avant la récupération, jamais après).
"""

import argparse
import json
import time
import uuid

from PIL import Image
from qdrant_client import QdrantClient, models

from common.config import get_settings
from ingestion.schema import Element
from retrieval.embeddings import embed_images, embed_texts, image_dim, text_dim

TEXT_TYPES = {"passage", "table", "figure"}   # figure : sa légende
IMAGE_TYPES = {"figure", "page"}
KEYWORD_FIELDS = ["groupes_autorises", "document_id", "famille", "modele", "doc_type", "type"]


def client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url, timeout=30)


def point_id(el: Element) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, el.uid))


def text_for_embedding(el: Element) -> str:
    """Le titre du document et la section donnent le contexte au passage (modèle, type de document)."""
    return "\n".join(p for p in (el.titre, el.section, el.text) if p)


def payload(el: Element) -> dict:
    return el.model_dump(exclude={"bbox"}) | {"uid": el.uid, "index_version": get_settings().index_version}


def ensure_collections(qc: QdrantClient, recreate: bool = False) -> None:
    s = get_settings()
    for name, dim in ((s.text_collection, text_dim()), (s.image_collection, image_dim())):
        if recreate and qc.collection_exists(name):
            qc.delete_collection(name)
        if not qc.collection_exists(name):
            qc.create_collection(name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
            for field in KEYWORD_FIELDS:
                qc.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)


def index_elements(qc: QdrantClient, elements: list[Element]) -> tuple[int, int]:
    """Remplace dans l'index tous les points des documents concernés."""
    s = get_settings()
    doc_ids = sorted({e.document_id for e in elements})
    doc_filter = models.Filter(must=[models.FieldCondition(key="document_id", match=models.MatchAny(any=doc_ids))])
    for name in (s.text_collection, s.image_collection):
        qc.delete(name, points_selector=models.FilterSelector(filter=doc_filter), wait=True)

    texts = [e for e in elements if e.type in TEXT_TYPES and e.text]
    if texts:
        vectors = embed_texts([text_for_embedding(e) for e in texts])
        qc.upsert(s.text_collection, wait=True, points=[
            models.PointStruct(id=point_id(e), vector=v.tolist(), payload=payload(e)) for e, v in zip(texts, vectors)])

    images = [e for e in elements if e.type in IMAGE_TYPES and e.image_path]
    if images:
        vectors = embed_images([Image.open(s.data_dir / e.image_path) for e in images])
        qc.upsert(s.image_collection, wait=True, points=[
            models.PointStruct(id=point_id(e), vector=v.tolist(), payload=payload(e)) for e, v in zip(images, vectors)])
    return len(texts), len(images)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recreate", action="store_true", help="supprimer et recréer les collections")
    args = parser.parse_args()

    s = get_settings()
    path = s.data_dir / "processed" / "elements.jsonl"
    elements = [Element.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    start = time.perf_counter()
    qc = client()
    ensure_collections(qc, recreate=args.recreate)
    n_text, n_img = index_elements(qc, elements)
    print(f"{n_text} points dans {s.text_collection}, {n_img} points dans {s.image_collection} "
          f"({time.perf_counter() - start:.1f} s)")


if __name__ == "__main__":
    main()
