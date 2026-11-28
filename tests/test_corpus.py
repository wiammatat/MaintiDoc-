"""Vérifie la cohérence du corpus SAN généré par scripts/generate_corpus.py."""

import json

import pytest

from common.access import can_access, get_user
from common.config import get_settings

DATA = get_settings().data_dir
GROUPES = {"tech_stockage", "tech_reseau", "responsable"}


@pytest.fixture(scope="module")
def metadata():
    return json.loads((DATA / "corpus" / "metadata.json").read_text(encoding="utf-8"))


def by_id(metadata, doc_id):
    return next(d for d in metadata if d["document_id"] == doc_id)


def test_taille_corpus(metadata):
    assert 30 <= len(metadata) <= 50


def test_deux_familles_et_manuel_commun(metadata):
    assert {d["famille"] for d in metadata} == {"baie", "switch", "san"}
    assert by_id(metadata, "manuel_san_pedagogique")["type"] == "manuel_pedagogique"


def test_fichiers_presents(metadata):
    for d in metadata:
        assert (DATA / d["fichier"]).exists(), d["fichier"]


def test_ids_uniques(metadata):
    ids = [d["document_id"] for d in metadata]
    assert len(ids) == len(set(ids))


def test_groupes_valides(metadata):
    for d in metadata:
        assert d["groupes_autorises"] and set(d["groupes_autorises"]) <= GROUPES


def test_manuel_pedagogique_et_securite_visibles_par_tous(metadata):
    for doc_id in ("manuel_san_pedagogique", "securite_baie_sec-b01", "securite_switch_sec-s01"):
        for user in ("tech_s01", "tech_r01", "resp_01"):
            assert can_access(get_user(user), by_id(metadata, doc_id)["groupes_autorises"])


def test_note_confidentielle_invisible_pour_techniciens(metadata):
    note = by_id(metadata, "note_service_baie_02")
    assert not can_access(get_user("tech_s01"), note["groupes_autorises"])
    assert can_access(get_user("resp_01"), note["groupes_autorises"])


def test_cloisonnement_familles(metadata):
    assert not can_access(get_user("tech_r01"), by_id(metadata, "manuel_baie_sn-400")["groupes_autorises"])
    assert not can_access(get_user("tech_s01"), by_id(metadata, "manuel_switch_fl-48")["groupes_autorises"])


def test_pieges_presents(metadata):
    assert any(d.get("test_injection") for d in metadata)
    assert by_id(metadata, "note_service_baie_01")["contredit"]
    assert by_id(metadata, "note_service_switch_01")["contredit"] == ["procedure_switch_prc-s02"]
    assert sum(bool(d.get("scanne")) for d in metadata) == 4


def test_utilisateur_inconnu_refuse():
    with pytest.raises(PermissionError):
        get_user("intrus")


def test_photos_verite_terrain():
    photos = json.loads((DATA / "photos" / "photos.json").read_text(encoding="utf-8"))
    assert len(photos) == 30
    assert {p["famille"] for p in photos} == {"baie", "switch"}
    assert any(p["ambigu"] for p in photos)
    for p in photos:
        assert (DATA / "photos" / p["fichier"]).exists()


# --- Guide d'utilisation (outils MCP, validation humaine, tickets)

def _guide_text() -> str:
    import sys
    sys.path.insert(0, str(DATA.parent / "scripts"))
    from corpus_content import GUIDE_MCP
    parts = []
    for titre, avant, tableau, apres in GUIDE_MCP["sections"]:
        parts += [titre, *avant, *apres] + [" | ".join(r) for r in tableau or []]
    return " ".join(parts)


def test_guide_mcp_visible_par_tous(metadata):
    guide = by_id(metadata, "guide_maintidoc_mcp")
    assert guide["type"] == "guide"
    for user in ("tech_s01", "tech_r01", "resp_01"):
        assert can_access(get_user(user), guide["groupes_autorises"])


def test_guide_mcp_aligne_sur_le_code():
    """Le guide décrit le vrai système : mêmes outils, même délai d'approbation, mêmes limites."""
    import asyncio

    from common.tokens import APPROVAL_TTL_SECONDS
    from mcp_server.server import MAX_DESCRIPTION, MAX_TITLE, mcp

    text = _guide_text()
    tools = {t.name for t in asyncio.run(mcp.list_tools())}
    assert tools == {"get_equipment_history", "get_open_tickets", "create_ticket"}
    assert all(t in text for t in tools) and "seulement ces trois outils" in text
    assert f"{APPROVAL_TTL_SECONDS // 60} minutes" in text
    assert f"Au plus {get_settings().max_mcp_calls} appels MCP" in text
    assert f"{MAX_TITLE} caractères" in text
    assert f"{MAX_DESCRIPTION:,} caractères".replace(",", " ") in text
