"""Vérifie l'extraction PyMuPDF (étape 1), la préparation textuelle et visuelle (étape 2) et le stockage."""

import json

import pymupdf
import pytest

from common.config import get_settings
from extraction.pdf import extract_pdf
from preparation.document import prepare_document
from preparation.ocr import ocr_available
from preparation.text import chunk, table_chunks

S = get_settings()
META = {m["document_id"]: m for m in json.loads((S.corpus_dir / "metadata.json").read_text(encoding="utf-8"))}


def pdf(doc_id):
    return S.data_dir / META[doc_id]["fichier"]


@pytest.fixture(scope="module")
def prepare(tmp_path_factory):
    cache = {}
    image_dir = tmp_path_factory.mktemp("images")

    def _prepare(doc_id):
        if doc_id not in cache:
            cache[doc_id] = prepare_document(pdf(doc_id), META[doc_id], image_dir)
        return cache[doc_id]
    return _prepare


def by_id(elements, element_id):
    return next(e for e in elements if e.element_id == element_id)


# --- Étape 1 : extraction (blocs bruts)

def test_extraction_blocs_dans_l_ordre_de_lecture():
    page2 = extract_pdf(pdf("manuel_baie_sn-400"))[1]
    kinds = [b.kind for b in page2.blocks]
    assert kinds[0] == "heading" and "image" in kinds and "table" in kinds
    assert kinds.index("image") < kinds.index("table")       # la figure précède le tableau des voyants
    assert all(b.section == "3. Panneau de contrôle" for b in page2.blocks)


def test_extraction_legende_rattachee_a_l_image():
    image = next(b for b in extract_pdf(pdf("manuel_baie_sn-400"))[1].blocks if b.kind == "image")
    assert image.caption == "Figure 3.1 - Face avant du SN-400" and image.image


def test_extraction_detecte_les_scans():
    assert all(p.scanned for p in extract_pdf(pdf("scan_switch_02")))
    assert not any(p.scanned for p in extract_pdf(pdf("manuel_baie_sn-400")))


# --- Étape 2a : préparation textuelle

def test_chunk_respecte_la_taille():
    chunks = chunk(["a" * 400] * 5, max_chars=1000)
    assert len(chunks) == 3 and all(len(c) <= 1000 for c in chunks)


def test_table_chunks_repetent_l_entete():
    rows = [["Code", "Action"]] + [[f"E{i:02d}", "x" * 200] for i in range(10)]
    chunks = table_chunks(rows, max_chars=600)
    assert len(chunks) > 1 and all(c.startswith("| Code | Action |") for c in chunks)


def test_tableau_dans_la_bonne_section(prepare):
    table = by_id(prepare("manuel_baie_sn-400"), "table_2_1_1")
    assert table.section == "3. Panneau de contrôle" and "| L2 | orange |" in table.text


def test_seuil_specifique_au_modele(prepare):
    assert any("60 °C" in (e.text or "") for e in prepare("manuel_baie_sn-400") if e.type == "passage")


def test_cellules_avec_espaces(prepare):
    text = "\n".join(e.text for e in prepare("alarmes_baie_sn-400") if e.type == "table")
    assert "| E03 | Température contrôleur élevée (seuil 60 °C) | L2 orange |" in text
    assert all(f"| E{i:02d} |" in text for i in range(1, 11))


def test_pied_de_page_supprime(prepare):
    for e in prepare("manuel_baie_sn-400"):
        assert "document fictif MaintiDoc" not in (e.text or "")


@pytest.mark.skipif(not ocr_available(), reason="Tesseract ou le pack français est absent")
def test_scan_passe_par_l_ocr(prepare):
    passages = [e for e in prepare("scan_switch_02") if e.type == "passage"]
    assert passages and all(e.source == "ocr" for e in passages)
    assert "fabric" in " ".join(e.text for e in passages).lower()


# --- Étape 2b : préparation visuelle

def test_une_page_par_page_du_pdf(prepare):
    n_pages = pymupdf.open(pdf("manuel_baie_sn-400")).page_count
    assert sum(e.type == "page" for e in prepare("manuel_baie_sn-400")) == n_pages


def test_figure_avec_legende_et_image_originale(prepare):
    fig = by_id(prepare("manuel_baie_sn-400"), "figure_2_1")
    assert fig.type == "figure" and fig.page == 2 and fig.text.startswith("Figure 3.1")
    assert fig.section == "3. Panneau de contrôle" and fig.image_path.endswith(".png")


def test_manuel_pedagogique_couverture_et_5_schemas(prepare):
    figures = [e for e in prepare("manuel_san_pedagogique") if e.type == "figure"]
    assert len(figures) == 6 and all(f.text.startswith("Figure") for f in figures)
    assert figures[0].text.startswith("Figure 0") and figures[0].page == 1


# --- Provenance

def test_metadonnees_heritees(prepare):
    for e in prepare("manuel_baie_sn-400"):
        assert e.version == "v1" and e.famille == "baie" and e.modele == "SN-400"
        assert e.groupes_autorises == ["tech_stockage", "responsable"]


def test_identifiants_uniques(prepare):
    for doc_id in ("manuel_baie_sn-400", "alarmes_baie_sn-400", "manuel_san_pedagogique"):
        uids = [e.uid for e in prepare(doc_id)]
        assert len(uids) == len(set(uids))


# --- Stockage (nécessite docker compose up)

def _services_up() -> bool:
    try:
        from storage.metadata_db import connect
        from storage.s3 import s3_client
        connect().close()
        s3_client().list_buckets()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _services_up(), reason="PostgreSQL ou MinIO est indisponible")
def test_stockage_idempotent():
    from storage.metadata_db import connect, save_document
    from storage.s3 import ensure_bucket, s3_client, upload_files

    doc_id = "note_service_baie_02"
    meta = META[doc_id]
    ensure_bucket()
    with connect() as con:
        for _ in range(2):  # deux ingestions successives ne doivent pas dupliquer
            elements = prepare_document(pdf(doc_id), meta)
            save_document(con, meta, elements, upload_files(pdf(doc_id), meta, elements))
        n = con.execute("SELECT count(*) FROM elements WHERE document_id = %s", (doc_id,)).fetchone()[0]
        groupes = con.execute("SELECT groupes_autorises FROM documents WHERE document_id = %s", (doc_id,)).fetchone()[0]
    assert n == len(elements)
    assert groupes == ["responsable"]
    s3_client().head_object(Bucket=S.s3_bucket, Key=f"{doc_id}/v1/source.pdf")
