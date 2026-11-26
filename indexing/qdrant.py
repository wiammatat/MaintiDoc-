"""Accès commun à Qdrant pour les deux index (textuel et visuel)."""

import uuid
from functools import lru_cache

from qdrant_client import QdrantClient, models

from common.config import get_settings
from preparation.schema import Element

# Champs filtrables. groupes_autorises porte le contrôle d'accès : filtré à la recherche, avant la récupération.
KEYWORD_FIELDS = ["groupes_autorises", "document_id", "famille", "modele", "doc_type", "type"]


@lru_cache
def client() -> QdrantClient:
    """Client partagé. Avec QDRANT_PATH, Qdrant tourne en local dans le processus (une seule instance permise)."""
    s = get_settings()
    if s.qdrant_path:
        return QdrantClient(path=str(s.qdrant_path))
    return QdrantClient(url=s.qdrant_url, timeout=30)


def point_id(uid: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, uid))


def payload(el: Element) -> dict:
    """Métadonnées stockées avec chaque vecteur : provenance complète pour la citation et le filtrage."""
    return el.model_dump(exclude={"bbox"}) | {"uid": el.uid, "index_version": get_settings().index_version}


def create_payload_indexes(collection: str) -> None:
    for field in KEYWORD_FIELDS:
        client().create_payload_index(collection, field, models.PayloadSchemaType.KEYWORD)


def delete_documents(collection: str, document_ids: list[str]) -> None:
    """Supprime les points de ces documents (ré-indexation sans doublons)."""
    if not client().collection_exists(collection):
        return
    flt = models.Filter(must=[models.FieldCondition(key="document_id", match=models.MatchAny(any=document_ids))])
    client().delete(collection, points_selector=models.FilterSelector(filter=flt), wait=True)


def access_filter(groups: list[str], filters: dict | None = None) -> models.Filter:
    """Filtre Qdrant : au moins un groupe en commun avec l'élément, plus des contraintes optionnelles."""
    must = [models.FieldCondition(key="groupes_autorises", match=models.MatchAny(any=sorted(groups)))]
    for key, value in (filters or {}).items():
        must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
    return models.Filter(must=must)
