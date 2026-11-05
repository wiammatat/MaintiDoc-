"""Vérifie l'extraction des documents (étape 1.1) et leur stockage."""

import json

import pymupdf
import pytest

from common.config import get_settings
from ingestion.extract import _chunk, _table_chunks, extract_document
from ingestion.ocr import ocr_available

S = get_settings()
META = {m["document_id"]: m for m in json.loads((S.corpus_dir / "metadata.json").read_text(encoding="utf-8"))}


@pytest.fixture(scope="module")
def extract(tmp_path_factory):
    cache = {}
    image_dir = tmp_path_factory.mktemp("images")

    def _extract(doc_id):
        if doc_id not in cache:
            cache[doc_id] = extract_document(S.data_dir / META[doc_id]["fichier"], META[doc_id], image_dir)
        return cache[doc_id]
    return _extract


def by_id(elements, element_id):
    return next(e for e in elements if e.element_id == element_id)


# --- Découpage

def test_chunk_respecte_la_taille():
    chunks = _chunk(["a" * 400] * 5, max_chars=1000)
    assert len(chunks) == 3 and all(len(c) <= 1000 for c in chunks)


def test_table_chunks_repetent_l_entete():
    rows = [["Code", "Action"]] + [[f"E{i:02d}", "x" * 200] for i in range(10)]
    chunks = _table_chunks(rows, max_chars=600)
    assert len(chunks) > 1
    assert all(c.startswith("| Code | Action |") for c in chunks)


# --- Manuel : texte, figure, tableau, sections

def test_une_page_par_page_du_pdf(extract):
    elements = extract("manuel_pompe_hf-350")
    n_pages = pymupdf.open(S.data_dir / META["manuel_pompe_hf-350"]["fichier"]).page_count
    assert sum(e.type == "page" for e in elements) == n_pages


def test_figure_avec_legende(extract):
    fig = by_id(extract("manuel_pompe_hf-350"), "figure_2_1")
    assert fig.type == "figure" and fig.page == 2
    assert fig.text.startswith("Figure 3.1")
    assert fig.section == "3. Panneau de commande"


def test_tableau_dans_la_bonne_section(extract):
    table = by_id(extract("manuel_pompe_hf-350"), "table_2_1_1")
    assert table.section == "3. Panneau de commande"
    assert "| L2 | orange |" in table.text


def test_seuil_specifique_au_modele(extract):
    assert any("85 °C" in (e.text or "") for e in extract("manuel_pompe_hf-350") if e.type == "passage")


def test_pied_de_page_supprime(extract):
    for e in extract("manuel_pompe_hf-350"):
        assert "document fictif MaintiDoc" not in (e.text or "")


def test_metadonnees_heritees(extract):
    for e in extract("manuel_pompe_hf-350"):
        assert e.version == "v1" and e.famille == "pompe" and e.modele == "HF-350"
        assert e.groupes_autorises == ["tech_pompes", "responsable"]


def test_identifiants_uniques(extract):
    for doc_id in ("manuel_pompe_hf-350", "alarmes_pompe_hf-350"):
        uids = [e.uid for e in extract(doc_id)]
        assert len(uids) == len(set(uids))


# --- Tableau des alarmes

def test_cellules_avec_espaces(extract):
    text = "\n".join(e.text for e in extract("alarmes_pompe_hf-350") if e.type == "table")
    assert "| E01 | Surchauffe moteur (seuil 85 °C) | L2 orange |" in text
    assert all(f"| E{i:02d} |" in text for i in range(1, 11))


# --- Scan

@pytest.mark.skipif(not ocr_available(), reason="Tesseract ou le pack français est absent")
def test_scan_passe_par_l_ocr(extract):
    passages = [e for e in extract("scan_compresseur_02") if e.type == "passage"]
    assert passages and all(e.source == "ocr" for e in passages)
    assert "soupape" in " ".join(e.text for e in passages).lower()


# --- Stockage (nécessite docker compose up)

def _services_up() -> bool:
    try:
        from ingestion.storage import connect, s3_client
        connect().close()
        s3_client().list_buckets()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _services_up(), reason="PostgreSQL ou le stockage S3 est indisponible")
def test_stockage_idempotent(extract):
    from ingestion.storage import connect, ensure_bucket, save_document, s3_client, upload_files

    doc_id = "note_service_pompe_02"
    meta, pdf = META[doc_id], S.data_dir / META[doc_id]["fichier"]
    ensure_bucket()
    with connect() as con:
        for _ in range(2):  # deux ingestions successives ne doivent pas dupliquer
            elements = extract_document(pdf, meta)
            save_document(con, meta, elements, upload_files(pdf, meta, elements))
        n = con.execute("SELECT count(*) FROM elements WHERE document_id = %s", (doc_id,)).fetchone()[0]
        groupes = con.execute("SELECT groupes_autorises FROM documents WHERE document_id = %s", (doc_id,)).fetchone()[0]
    assert n == len(elements)
    assert groupes == ["responsable"]
    s3_client().head_object(Bucket=S.s3_bucket, Key=f"{doc_id}/v1/source.pdf")
