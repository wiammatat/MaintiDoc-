"""Enchaîne extraction et préparation pour un document : PDF -> éléments (passages, tableaux, figures, pages)."""

from pathlib import Path

from common.config import get_settings
from extraction.pdf import extract_pdf
from preparation.schema import Element, provenance
from preparation.text import prepare_text
from preparation.visual import prepare_visual


def prepare_document(pdf_path: Path, meta: dict, image_dir: Path | None = None) -> list[Element]:
    """meta : entrée de data/corpus/metadata.json. Les images vont dans image_dir/<document_id>/."""
    image_dir = (image_dir or get_settings().data_dir / "processed" / "images") / meta["document_id"]
    pages = extract_pdf(pdf_path)
    prov = provenance(meta)
    return prepare_visual(pdf_path, pages, prov, image_dir) + prepare_text(pdf_path, pages, prov)
