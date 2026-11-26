"""Vérifie le RAG multimodal (étape 4) : recherche texte + image, fusion RRF, reranker, contrôle d'accès."""

import pytest
from PIL import Image

from common.access import User, get_user
from common.config import get_settings
from indexing.qdrant import client
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.search import search

S = get_settings()


# --- Fusion RRF (sans service)

def test_rrf_favorise_les_elements_presents_dans_plusieurs_listes():
    fused = reciprocal_rank_fusion({
        "texte": [("a", {}), ("b", {}), ("c", {})],
        "image": [("c", {}), ("d", {})],
    }, k=60)
    assert fused[0].uid == "c"                       # 1/63 + 1/61 > 1/61
    assert fused[0].ranks == {"texte": 3, "image": 1}
    assert [h.uid for h in fused[1:]] == ["a", "b", "d"]


def test_rrf_score():
    (hit,) = reciprocal_rank_fusion({"l": [("x", {})]}, k=60)
    assert hit.rrf_score == pytest.approx(1 / 61)


def test_rrf_pondere():
    fused = reciprocal_rank_fusion({"fort": [("a", {})], "faible": [("b", {})]}, k=60,
                                   weights={"fort": 1.0, "faible": 0.5})
    assert [h.uid for h in fused] == ["a", "b"]
    assert fused[1].rrf_score == pytest.approx(0.5 / 61)


def test_utilisateur_sans_groupe_refuse():
    with pytest.raises(PermissionError):
        search(User("intrus", "Intrus", frozenset()), "voyant orange")


# --- Recherche réelle (nécessite les index)

def _index_ready() -> bool:
    try:
        return client().count(S.text_collection).count > 0
    except Exception:
        return False


needs_index = pytest.mark.skipif(not _index_ready(), reason="index absent : lancer python -m pipeline.index")


@needs_index
def test_resultats_citables():
    res = search(get_user("tech_s01"), "Que faire en cas d'alarme E05 ?", reranker=None)
    assert res.passages
    assert set(res.passages[0].citation) == {"document_id", "page", "element_id"}


@needs_index
def test_reranker_place_le_bon_modele_en_tete():
    res = search(get_user("tech_s01"), "Que signifie le voyant orange sur la baie SN-400 ?")
    top = res.passages[0].payload
    assert top["document_id"] == "manuel_baie_sn-400" and "| L2 | orange |" in top["text"]
    assert res.passages[0].rerank_score is not None


@needs_index
def test_contradiction_remontee_avec_le_document_d_origine():
    docs = [h.payload["document_id"] for h in search(get_user("tech_r01"),
            "À quelle fréquence sauvegarder la configuration du FL-48 ?").passages]
    assert "note_service_switch_01" in docs and "procedure_switch_prc-s02" in docs


@needs_index
def test_cloisonnement_par_groupe():
    res = search(get_user("tech_r01"), "voyant orange baie SN-400 disque défaillant", reranker=None,
                 top_passages=50, top_images=50)
    for hit in res.passages + res.images:
        assert "tech_reseau" in hit.payload["groupes_autorises"]


@needs_index
def test_manuel_pedagogique_accessible_a_tous():
    for user in ("tech_s01", "tech_r01"):
        res = search(get_user(user), "Comment fonctionne le multipath avec deux fabrics ?")
        assert res.passages[0].payload["document_id"] == "manuel_san_pedagogique"


@needs_index
def test_recherche_par_photo():
    photo = Image.open(S.data_dir / "photos" / "photo_01.jpg")
    res = search(get_user("tech_s01"), "Que signifie ce voyant ?", photo, reranker=None)
    assert res.images[0].payload["type"] == "figure"
    assert "image_par_photo" in res.images[0].ranks
