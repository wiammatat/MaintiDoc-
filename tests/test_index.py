"""Vérifie l'index Qdrant (étape 1.3). Nécessite docker compose up et python -m ingestion.index."""

import pytest
from qdrant_client import models

from common.config import get_settings
from ingestion.index import client
from retrieval.embeddings import embed_texts, embed_texts_for_images

S = get_settings()


def _index_ready() -> bool:
    try:
        qc = client()
        return all(qc.collection_exists(c) and qc.count(c).count > 0
                   for c in (S.text_collection, S.image_collection))
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _index_ready(), reason="index Qdrant absent : lancer python -m ingestion.index")


def groups_filter(*groups: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="groupes_autorises", match=models.MatchAny(any=list(groups)))])


def search_text(query: str, *groups: str, limit: int = 5):
    return client().query_points(S.text_collection, query=embed_texts([query])[0].tolist(),
                                 query_filter=groups_filter(*groups), limit=limit, with_payload=True).points


def test_payload_complet():
    point = client().scroll(S.text_collection, limit=1, with_payload=True)[0][0]
    for key in ("document_id", "version", "page", "element_id", "groupes_autorises", "text", "index_version"):
        assert key in point.payload


def test_recherche_texte_pertinente():
    hits = search_text("Que faire en cas d'alarme E05 fuite de garniture ?", "tech_pompes")
    assert any("E05" in h.payload["text"] for h in hits[:3])


def test_filtre_groupes_avant_recuperation():
    # Un technicien compresseurs ne doit jamais récupérer un document réservé aux pompes.
    for hit in search_text("seuil d'alarme E01 surchauffe moteur pompe HF-350", "tech_compresseurs", limit=20):
        assert "tech_compresseurs" in hit.payload["groupes_autorises"]
        assert hit.payload["famille"] != "pompe" or hit.payload["doc_type"] == "securite"


def test_note_confidentielle_reservee_aux_responsables():
    query = "retour d'expérience garnitures mécaniques lot défaillant"
    assert all(h.payload["document_id"] != "note_service_pompe_02" for h in search_text(query, "tech_pompes", limit=50))
    assert any(h.payload["document_id"] == "note_service_pompe_02" for h in search_text(query, "responsable"))


def test_recherche_image_par_texte():
    vector = embed_texts_for_images(["panneau de commande avec voyants"])[0].tolist()
    hits = client().query_points(S.image_collection, query=vector, query_filter=groups_filter("responsable"),
                                 limit=5, with_payload=True).points
    assert hits and all(h.payload["type"] in ("figure", "page") for h in hits)
