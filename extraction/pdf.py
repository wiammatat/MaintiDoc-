"""Étape 1 du pipeline : extraction automatique d'un PDF avec PyMuPDF.

Chaque page est découpée en blocs bruts (titres, texte, tableaux, images), dans l'ordre de lecture,
avec leurs métadonnées : page, position, section courante et légende des images.
Aucun découpage en passages ici : c'est le rôle de la préparation textuelle et visuelle.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pymupdf
from PIL import Image

HEADING_MIN_SIZE = 12.5    # taille de police à partir de laquelle un bloc est un titre
OCR_MIN_CHARS = 20         # en dessous, la page est considérée comme scannée
CAPTION_MAX_GAP = 30       # écart vertical maximal (points) entre une image et sa légende
CAPTION_RE = re.compile(r"^(Figure|Fig\.)\s", re.IGNORECASE)
# En-têtes et pieds de page répétés : bruit pour la recherche.
FOOTER_PATTERNS = [re.compile(r"^\S+ - version \S+ - document fictif MaintiDoc( page \d+)?$"),
                   re.compile(r"^page \d+$")]


@dataclass
class Block:
    kind: Literal["heading", "text", "table", "image"]
    page: int
    bbox: tuple[float, float, float, float]
    section: str | None                 # titre de la section qui contient le bloc
    text: str | None = None             # titre ou paragraphe
    rows: list[list[str]] | None = None  # cellules d'un tableau
    image: bytes | None = None          # octets de l'image originale
    ext: str | None = None
    caption: str | None = None          # légende d'une image (« Figure 3.1 - ... »)


@dataclass
class Page:
    number: int
    scanned: bool                        # aucune couche texte : à passer à l'OCR
    blocks: list[Block] = field(default_factory=list)
    section_at_start: str | None = None  # section en cours au début de la page


def clean(text: str) -> str:
    return " ".join(text.split())


def is_footer(text: str) -> bool:
    return any(p.match(text) for p in FOOTER_PATTERNS)


def render_page(pdf_path: Path, number: int, dpi: int) -> Image.Image:
    """Rendu image d'une page (numérotée à partir de 1)."""
    with pymupdf.open(pdf_path) as doc:
        pix = doc[number - 1].get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _table_rows(page: pymupdf.Page, table) -> list[list[str]]:
    # Lecture cellule par cellule : extract() perd les espaces sur certains tableaux.
    return [[clean(page.get_textbox(pymupdf.Rect(c))) if c else "" for c in row.cells] for row in table.rows]


def _block_text(block: dict) -> tuple[str, float]:
    spans = [s for line in block["lines"] for s in line["spans"]]
    return clean(" ".join(s["text"] for s in spans)), max((s["size"] for s in spans), default=0)


def extract_pdf(pdf_path: Path) -> list[Page]:
    pages: list[Page] = []
    section: str | None = None
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            number = page.number + 1
            current = Page(number=number, scanned=len(page.get_text().strip()) < OCR_MIN_CHARS,
                           section_at_start=section)
            pages.append(current)
            if current.scanned:
                continue

            tables = page.find_tables().tables
            table_rects = [pymupdf.Rect(t.bbox) for t in tables]
            raw = page.get_text("dict")["blocks"] + [{"type": "table", "bbox": t.bbox, "table": t} for t in tables]
            raw.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))   # ordre de lecture
            captions: set[int] = set()

            for i, b in enumerate(raw):
                bbox = tuple(b["bbox"])
                if b["type"] == "table":
                    current.blocks.append(Block("table", number, bbox, section, rows=_table_rows(page, b["table"])))
                elif b["type"] == 1:   # image, avec sa légende si le bloc suivant en est une
                    caption = None
                    nxt = raw[i + 1] if i + 1 < len(raw) else None
                    if nxt and nxt["type"] == 0 and nxt["bbox"][1] - b["bbox"][3] < CAPTION_MAX_GAP:
                        txt, _ = _block_text(nxt)
                        if CAPTION_RE.match(txt):
                            caption = txt
                            captions.add(i + 1)
                    current.blocks.append(Block("image", number, bbox, section, image=b["image"], ext=b["ext"],
                                                caption=caption))
                elif i not in captions:
                    text, size = _block_text(b)
                    center = pymupdf.Rect(bbox).tl + (pymupdf.Rect(bbox).br - pymupdf.Rect(bbox).tl) * 0.5
                    if not text or is_footer(text) or any(center in r for r in table_rects):
                        continue
                    if size >= HEADING_MIN_SIZE:
                        section = text
                        current.blocks.append(Block("heading", number, bbox, section, text=text))
                    else:
                        current.blocks.append(Block("text", number, bbox, section, text=text))
    return pages
