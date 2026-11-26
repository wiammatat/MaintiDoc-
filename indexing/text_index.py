"""Étape 3a : index textuel avec LlamaIndex + BGE-M3, stocké dans Qdrant.

Chaque passage, tableau, légende ou texte OCR devient un nœud LlamaIndex. Le titre du document
et la section sont ajoutés au texte encodé pour donner le contexte (modèle, type de document) ;
les autres métadonnées (provenance, groupes) sont stockées sans être encodées.
"""

from functools import lru_cache

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.vector_stores.qdrant import QdrantVectorStore

from common.config import get_settings
from indexing.embeddings import embed_texts
from indexing.qdrant import client, create_payload_indexes, delete_documents, payload, point_id
from preparation.schema import Element

TEXT_TYPES = {"passage", "table", "figure"}   # figure : sa légende
EMBEDDED_METADATA = ("titre", "section")       # métadonnées ajoutées au texte encodé


class BGEM3Embedding(BaseEmbedding):
    """Branche LlamaIndex sur le BGE-M3 du projet : un seul modèle chargé, mêmes réglages (précision, CPU/GPU)."""

    model_name: str = "BAAI/bge-m3"
    embed_batch_size: int = 16

    def _get_query_embedding(self, query: str) -> list[float]:
        return embed_texts([query])[0].tolist()

    def _get_text_embedding(self, text: str) -> list[float]:
        return embed_texts([text])[0].tolist()

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts).tolist()

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)


def to_node(el: Element) -> TextNode:
    # Les valeurs absentes sont omises : sinon LlamaIndex encoderait littéralement « section: None ».
    metadata = {k: v for k, v in payload(el).items() if k != "text" and v is not None}
    hidden = [k for k in metadata if k not in EMBEDDED_METADATA]
    return TextNode(id_=point_id(el.uid), text=el.text, metadata=metadata,
                    excluded_embed_metadata_keys=hidden, excluded_llm_metadata_keys=hidden)


def context_text(p: dict) -> str:
    """Texte tel qu'encodé (titre + section + contenu, même ordre) : réutilisé par le reranker pour rester cohérent."""
    metadata = {k: v for k, v in p.items() if k in EMBEDDED_METADATA and v is not None}
    return TextNode(text=p.get("text") or "", metadata=metadata).get_content(metadata_mode=MetadataMode.EMBED)


def _vector_store() -> QdrantVectorStore:
    return QdrantVectorStore(client=client(), collection_name=get_settings().text_collection)


@lru_cache
def get_index() -> VectorStoreIndex:
    return VectorStoreIndex.from_vector_store(_vector_store(), embed_model=BGEM3Embedding())


def build_text_index(elements: list[Element], recreate: bool = False) -> int:
    """(Ré)indexe les éléments textuels. Renvoie le nombre de nœuds indexés."""
    name = get_settings().text_collection
    nodes = [to_node(e) for e in elements if e.type in TEXT_TYPES and e.text]
    if recreate and client().collection_exists(name):
        client().delete_collection(name)
    else:
        delete_documents(name, sorted({e.document_id for e in elements}))
    if nodes:
        VectorStoreIndex(nodes, storage_context=StorageContext.from_defaults(vector_store=_vector_store()),
                         embed_model=BGEM3Embedding())
        create_payload_indexes(name)
    get_index.cache_clear()
    return len(nodes)


def search_text(question: str, groups: list[str], filters: dict | None = None,
                k: int = 20) -> list[tuple[str, dict]]:
    """Recherche sémantique filtrée par groupe AVANT la récupération. Renvoie [(uid, payload)]."""
    conditions = [MetadataFilter(key="groupes_autorises", value=sorted(groups), operator=FilterOperator.IN)]
    conditions += [MetadataFilter(key=key, value=value) for key, value in (filters or {}).items()]
    retriever = get_index().as_retriever(similarity_top_k=k, filters=MetadataFilters(filters=conditions))
    return [(n.node.metadata["uid"], n.node.metadata | {"text": n.node.get_content()})
            for n in retriever.retrieve(question)]
