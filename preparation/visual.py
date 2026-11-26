"""Étape 2b : préparation visuelle (schémas PNG, image originale, métadonnées).

Enregistre l'image originale de chaque figure (avec sa légende) et le rendu de chaque page,
pour l'index visuel CLIP et pour le VLM.
"""

from pathlib import Path

from common.config import get_settings
from extraction.pdf import Page, render_page
from preparation.schema import Element

PAGE_DPI = 130   # rendu des pages pour le VLM et la recherche visuelle


def _relative(path: Path) -> str:
    data_dir = get_settings().data_dir
    try:
        return path.relative_to(data_dir).as_posix()
    except ValueError:
        return path.as_posix()


def prepare_visual(pdf_path: Path, pages: list[Page], prov: dict, image_dir: Path) -> list[Element]:
    """Figures et pages d'un document. Les images sont écrites dans image_dir."""
    image_dir.mkdir(parents=True, exist_ok=True)
    elements: list[Element] = []
    for page in pages:
        pno = page.number
        page_png = image_dir / f"page_{pno}.png"
        render_page(pdf_path, pno, PAGE_DPI).save(page_png)
        elements.append(Element(**prov, page=pno, element_id=f"page_{pno}", type="page",
                                image_path=_relative(page_png)))
        for n, b in enumerate((b for b in page.blocks if b.kind == "image"), 1):
            fig = image_dir / f"figure_{pno}_{n}.{b.ext}"
            fig.write_bytes(b.image)   # image originale, sans recompression
            elements.append(Element(**prov, page=pno, element_id=f"figure_{pno}_{n}", type="figure",
                                    section=b.section, text=b.caption, image_path=_relative(fig),
                                    bbox=list(b.bbox)))
    return elements
