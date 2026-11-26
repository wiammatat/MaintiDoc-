"""Vérifie que l'OCR français lit les documents scannés du corpus."""

import pymupdf
import pytest
from PIL import Image

from common.config import get_settings
from preparation.ocr import ocr_available, ocr_image

pytestmark = pytest.mark.skipif(not ocr_available(), reason="Tesseract ou le pack français est absent")


def _render(name: str) -> Image.Image:
    pix = pymupdf.open(get_settings().corpus_dir / f"{name}.pdf")[0].get_pixmap(dpi=300)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def test_scan_sans_couche_texte():
    assert pymupdf.open(get_settings().corpus_dir / "scan_switch_02.pdf")[0].get_text().strip() == ""


def test_ocr_lit_le_scan():
    texte = ocr_image(_render("scan_switch_02")).lower()
    assert "basculement" in texte
    assert "fabric b" in texte
