"""Vérifie la cohérence du corpus généré par scripts/generate_corpus.py."""

import json

import pytest

from common.access import can_access, get_user
from common.config import get_settings

DATA = get_settings().data_dir
GROUPES = {"tech_pompes", "tech_compresseurs", "responsable"}


@pytest.fixture(scope="module")
def metadata():
    return json.loads((DATA / "corpus" / "metadata.json").read_text(encoding="utf-8"))


def test_taille_corpus(metadata):
    assert 30 <= len(metadata) <= 50


def test_deux_familles(metadata):
    assert {d["famille"] for d in metadata} == {"pompe", "compresseur"}


def test_fichiers_presents(metadata):
    for d in metadata:
        assert (DATA / d["fichier"]).exists(), d["fichier"]


def test_ids_uniques(metadata):
    ids = [d["document_id"] for d in metadata]
    assert len(ids) == len(set(ids))


def test_groupes_valides(metadata):
    for d in metadata:
        assert d["groupes_autorises"] and set(d["groupes_autorises"]) <= GROUPES


def test_note_confidentielle_invisible_pour_techniciens(metadata):
    note = next(d for d in metadata if d["document_id"] == "note_service_pompe_02")
    assert not can_access(get_user("tech_p01"), note["groupes_autorises"])
    assert can_access(get_user("resp_01"), note["groupes_autorises"])


def test_cloisonnement_familles(metadata):
    tech_c = get_user("tech_c01")
    manuel_pompe = next(d for d in metadata if d["document_id"] == "manuel_pompe_hf-350")
    assert not can_access(tech_c, manuel_pompe["groupes_autorises"])


def test_pieges_presents(metadata):
    assert any(d.get("test_injection") for d in metadata)
    assert any(d.get("contredit") for d in metadata)
    assert any(d.get("scanne") for d in metadata)


def test_utilisateur_inconnu_refuse():
    with pytest.raises(PermissionError):
        get_user("intrus")


def test_photos_verite_terrain():
    photos = json.loads((DATA / "photos" / "photos.json").read_text(encoding="utf-8"))
    assert len(photos) == 30
    assert any(p["ambigu"] for p in photos)
    for p in photos:
        assert (DATA / "photos" / p["fichier"]).exists()
