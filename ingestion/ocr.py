"""OCR local avec Tesseract (français)."""

import os
from functools import lru_cache

import pytesseract
from PIL import Image

from common.config import get_settings


@lru_cache
def _configure() -> str:
    s = get_settings()
    pytesseract.pytesseract.tesseract_cmd = s.tesseract_cmd
    # Variable d'environnement plutôt que --tessdata-dir : pytesseract découpe mal les chemins sous Windows.
    os.environ["TESSDATA_PREFIX"] = str(s.tessdata_dir)
    return s.ocr_lang


def ocr_image(img: Image.Image) -> str:
    """Retourne le texte reconnu dans l'image."""
    return pytesseract.image_to_string(img, lang=_configure()).strip()


def ocr_available() -> bool:
    try:
        return _configure() in pytesseract.get_languages()
    except (pytesseract.TesseractNotFoundError, OSError):
        return False
