"""Vérifie les index (étape 3) : textuel (LlamaIndex + BGE-M3) et visuel (CLIP).

Les tests sur Qdrant nécessitent docker compose up et python -m pipeline.index.
"""

import pytest
from llama_index.core.schema import MetadataMode

from common.config import get_settings
from indexing.embeddings import embed_texts_for_images
from indexing.qdrant import client
from indexing.text_index import context_text, search_text, to_node
from indexing.visual_index import search_visual
from preparation.schema import Element

S = get_settings()

ELEMENT = Element(document_id="manuel_baie_sn-400", version="v1", page=2, element_id="table_2_1_1", type="table",
                  section="3. Panneau de contrôle", text="| L2 | orange | Disque défaillant |",
                  titre="Manuel d'utilisation SN-400", famille="baie", modele="SN-400", doc_type="manuel",
                  groupes_autorises=["tech_stockage", "responsable"])


# --- Nœuds LlamaIndex (sans service)

def test_noeud_encode_titre_et_section_mais_pas_les_droits():
    encoded = to_node(ELEMENT).get_content(metadata_mode=MetadataMode.EMBED)
    assert "Manuel d'utilisation SN-400" in encoded and "3. Panneau de contrôle" in encoded
    assert "| L2 | orange |" in encoded
    assert "tech_stockage" not in encoded and "groupes_autorises" not in encoded


def test_noeud_garde_la_provenance():
    meta = to_node(ELEMENT).metadata
    assert meta["uid"] == ELEMENT.uid and meta["page"] == 2 and meta["groupes_autorises"] == ELEMENT.groupes_autorises
    assert "text" not in meta   # le texte est porté par le nœud, pas dupliqué en métadonnée


def test_context_text_identique_au_texte_encode():
    p = ELEMENT.model_dump()
    assert context_text(p) == to_node(ELEMENT).get_content(metadata_mode=MetadataMode.EMBED)


# --- Index réels

def _index_ready() -> bool:
    try:
        return all(client().collection_exists(c) and client().count(c).count > 0
                   for c in (S.text_collection, S.image_collection))
    except Exception:
        return False


needs_index = pytest.mark.skipif(not _index_ready(), reason="index absent : lancer python -m pipeline.index")


@needs_index
def test_payload_complet():
    point = client().scroll(S.text_collection, limit=1, with_payload=True)[0][0]
    for key in ("document_id", "version", "page", "element_id", "groupes_autorises", "uid", "index_version",
                "_node_content"):
        assert key in point.payload


@needs_index
def test_recherche_texte_pertinente():
    hits = search_text("Que faire en cas d'alarme E05 batterie de cache faible ?", ["tech_stockage"])
    assert any("E05" in p["text"] for _, p in hits[:3])


@needs_index
def test_filtre_groupes_avant_recuperation():
    # Un technicien réseau ne récupère jamais un document réservé aux baies.
    for _, p in search_text("seuil d'alarme E03 température contrôleur SN-400", ["tech_reseau"], k=50):
        assert "tech_reseau" in p["groupes_autorises"]


@needs_index
def test_note_confidentielle_reservee_aux_responsables():
    q = "retour d'expérience lot de disques défaillants"
    assert all(p["document_id"] != "note_service_baie_02" for _, p in search_text(q, ["tech_stockage"], k=50))
    assert any(p["document_id"] == "note_service_baie_02" for _, p in search_text(q, ["responsable"]))


@needs_index
def test_filtre_par_modele():
    hits = search_text("voyant orange", ["tech_stockage"], filters={"modele": "SN-400"})
    assert hits and all(p["modele"] == "SN-400" for _, p in hits)


@needs_index
def test_recherche_visuelle_par_texte():
    hits = search_visual(embed_texts_for_images(["schéma d'architecture d'un réseau de stockage"])[0], ["responsable"])
    assert hits and all(p["type"] in ("figure", "page") for _, p in hits)


def test_section_absente_non_encodee():
    sans_section = ELEMENT.model_copy(update={"section": None})
    encoded = to_node(sans_section).get_content(metadata_mode=MetadataMode.EMBED)
    assert "None" not in encoded and "section" not in encoded
    assert context_text(sans_section.model_dump()) == encoded
