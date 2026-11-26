"""Étape 2a : préparation textuelle (découpage, OCR si besoin, provenance).

Transforme les blocs bruts de l'extraction en passages et morceaux de tableaux, chacun rattaché
à sa page, sa section et son document. Les pages scannées passent par l'OCR.
"""

import re
from pathlib import Path

from extraction.pdf import Page, clean, render_page
from preparation.ocr import ocr_image
from preparation.schema import Element

MAX_CHARS = 1000   # taille maximale d'un passage ou d'un morceau de tableau
OCR_DPI = 300


def chunk(paragraphs: list[str], max_chars: int = MAX_CHARS) -> list[str]:
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


def table_chunks(rows: list[list[str]], max_chars: int = MAX_CHARS) -> list[str]:
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


def prepare_text(pdf_path: Path, pages: list[Page], prov: dict) -> list[Element]:
    """Passages et tableaux d'un document. prov : champs de provenance (preparation.schema.provenance)."""
    elements: list[Element] = []
    for page in pages:
        pno = page.number

        if page.scanned:
            text = ocr_image(render_page(pdf_path, pno, OCR_DPI))
            paragraphs = [clean(p) for p in re.split(r"\n\s*\n", text) if clean(p)]
            for k, c in enumerate(chunk(paragraphs), 1):
                elements.append(Element(**prov, page=pno, element_id=f"passage_{pno}_{k}", type="passage",
                                        source="ocr", section=page.section_at_start, text=c))
            continue

        section, paragraphs, n_pass, n_tab = page.section_at_start, [], 0, 0

        def flush():
            nonlocal n_pass, paragraphs
            for c in chunk(paragraphs):
                n_pass += 1
                elements.append(Element(**prov, page=pno, element_id=f"passage_{pno}_{n_pass}", type="passage",
                                        section=section, text=c))
            paragraphs = []

        for b in page.blocks:
            if b.kind == "heading":       # un nouveau titre ferme le passage en cours
                flush()
                section = b.section
            elif b.kind == "table":
                flush()
                n_tab += 1
                for part, c in enumerate(table_chunks(b.rows), 1):
                    elements.append(Element(**prov, page=pno, element_id=f"table_{pno}_{n_tab}_{part}",
                                            type="table", section=b.section, text=c, bbox=list(b.bbox)))
            elif b.kind == "text":
                paragraphs.append(b.text)
        flush()
    return elements
