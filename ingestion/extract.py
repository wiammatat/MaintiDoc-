"""Extraction d'un PDF en éléments : passages, tableaux, figures et pages.

Les pages sans couche texte (scans) passent par l'OCR.
"""

import re
from pathlib import Path

import pymupdf
from PIL import Image

from common.config import get_settings
from ingestion.ocr import ocr_image
from ingestion.schema import Element

MAX_CHARS = 1000           # taille maximale d'un passage ou d'un morceau de tableau
HEADING_MIN_SIZE = 12.5    # taille de police à partir de laquelle un bloc est un titre
OCR_MIN_CHARS = 20         # en dessous, la page est considérée comme scannée
PAGE_DPI = 130             # rendu des pages pour le VLM et la recherche visuelle
OCR_DPI = 300
CAPTION_RE = re.compile(r"^(Figure|Fig\.)\s", re.IGNORECASE)
# En-têtes et pieds de page répétés : bruit pour la recherche.
FOOTER_PATTERNS = [re.compile(r"^\S+ - version \S+ - document fictif MaintiDoc( page \d+)?$"),
                   re.compile(r"^page \d+$")]


def _clean(text: str) -> str:
    return " ".join(text.split())


def _is_footer(text: str) -> bool:
    return any(p.match(text) for p in FOOTER_PATTERNS)


def _render(page: pymupdf.Page, dpi: int) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _chunk(paragraphs: list[str], max_chars: int = MAX_CHARS) -> list[str]:
    """Regroupe des paragraphes consécutifs en passages d'au plus max_chars caractères."""
    chunks, current = [], ""
    for p in paragraphs:
        if current and len(current) + len(p) + 1 > max_chars:
            chunks.append(current)
            current = ""
        current = f"{current}\n{p}" if current else p
    if current:
        chunks.append(current)
    return chunks


def _table_rows(page: pymupdf.Page, table) -> list[list[str]]:
    # Lecture cellule par cellule : extract() perd les espaces sur certains tableaux.
    return [[_clean(page.get_textbox(pymupdf.Rect(c))) if c else "" for c in row.cells] for row in table.rows]


def _table_chunks(rows: list[list[str]], max_chars: int = MAX_CHARS) -> list[str]:
    """Découpe un tableau en morceaux Markdown qui répètent tous la ligne d'en-tête."""
    def md(cells):
        return "| " + " | ".join(cells) + " |"

    header = md(rows[0]) + "\n|" + "---|" * len(rows[0])
    chunks, current = [], []
    for row in rows[1:]:
        line = md(row)
        if current and len(header) + sum(len(r) + 1 for r in current) + len(line) > max_chars:
            chunks.append(header + "\n" + "\n".join(current))
            current = []
        current.append(line)
    if current or not chunks:
        chunks.append(header + ("\n" + "\n".join(current) if current else ""))
    return chunks


def extract_document(pdf_path: Path, meta: dict, image_dir: Path | None = None) -> list[Element]:
    """Extrait les éléments d'un document. Les images sont écrites dans image_dir/<document_id>/."""
    data_dir = get_settings().data_dir
    image_dir = (image_dir or data_dir / "processed" / "images") / meta["document_id"]
    image_dir.mkdir(parents=True, exist_ok=True)

    base = dict(document_id=meta["document_id"], version=meta["version"], titre=meta["titre"],
                famille=meta["famille"], modele=meta.get("modele"), doc_type=meta["type"],
                groupes_autorises=meta["groupes_autorises"])

    def rel(path: Path) -> str:
        try:
            return path.relative_to(data_dir).as_posix()
        except ValueError:
            return path.as_posix()

    elements: list[Element] = []
    section: str | None = None

    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            pno = page.number + 1

            # Rendu de la page entière : élément visuel pour le VLM et la recherche par image.
            page_png = image_dir / f"page_{pno}.png"
            _render(page, PAGE_DPI).save(page_png)
            elements.append(Element(**base, page=pno, element_id=f"page_{pno}", type="page",
                                    image_path=rel(page_png)))

            # Page scannée : OCR puis découpage en passages.
            if len(page.get_text().strip()) < OCR_MIN_CHARS:
                text = ocr_image(_render(page, OCR_DPI))
                paragraphs = [_clean(p) for p in re.split(r"\n\s*\n", text) if _clean(p)]
                for k, chunk in enumerate(_chunk(paragraphs), 1):
                    elements.append(Element(**base, page=pno, element_id=f"passage_{pno}_{k}", type="passage",
                                            source="ocr", section=section, text=chunk))
                continue

            # Texte, figures et tableaux dans l'ordre de lecture, pour que chacun hérite de la bonne section.
            tables = page.find_tables().tables
            table_rects = [pymupdf.Rect(t.bbox) for t in tables]
            blocks = page.get_text("dict")["blocks"] + [{"type": "table", "bbox": t.bbox, "table": t} for t in tables]
            blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
            used_captions: set[int] = set()
            paragraphs: list[str] = []
            n_fig = n_pass = n_tab = 0

            def flush():
                nonlocal n_pass, paragraphs
                for chunk in _chunk(paragraphs):
                    n_pass += 1
                    elements.append(Element(**base, page=pno, element_id=f"passage_{pno}_{n_pass}",
                                            type="passage", section=section, text=chunk))
                paragraphs = []

            for i, b in enumerate(blocks):
                if b["type"] == "table":
                    flush()
                    n_tab += 1
                    for part, chunk in enumerate(_table_chunks(_table_rows(page, b["table"])), 1):
                        elements.append(Element(**base, page=pno, element_id=f"table_{pno}_{n_tab}_{part}",
                                                type="table", section=section, text=chunk, bbox=list(b["bbox"])))
                    continue
                if b["type"] == 1:  # image
                    n_fig += 1
                    caption = None
                    nxt = blocks[i + 1] if i + 1 < len(blocks) else None
                    if nxt and nxt["type"] == 0 and nxt["bbox"][1] - b["bbox"][3] < 30:
                        txt = _clean(" ".join(s["text"] for l in nxt["lines"] for s in l["spans"]))
                        if CAPTION_RE.match(txt):
                            caption = txt
                            used_captions.add(i + 1)
                    fig_path = image_dir / f"figure_{pno}_{n_fig}.{b['ext']}"
                    fig_path.write_bytes(b["image"])
                    elements.append(Element(**base, page=pno, element_id=f"figure_{pno}_{n_fig}", type="figure",
                                            section=section, text=caption, image_path=rel(fig_path),
                                            bbox=list(b["bbox"])))
                    continue

                if i in used_captions:
                    continue
                spans = [s for l in b["lines"] for s in l["spans"]]
                text = _clean(" ".join(s["text"] for s in spans))
                if not text or _is_footer(text):
                    continue
                center = pymupdf.Rect(b["bbox"]).tl + (pymupdf.Rect(b["bbox"]).br - pymupdf.Rect(b["bbox"]).tl) * 0.5
                if any(center in r for r in table_rects):
                    continue
                if max(s["size"] for s in spans) >= HEADING_MIN_SIZE:
                    flush()
                    section = text
                    continue
                paragraphs.append(text)
            flush()

    return elements
