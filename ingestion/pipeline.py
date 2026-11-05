"""Ingestion du corpus : extraction, découpage, stockage des fichiers et des métadonnées.

Usage :
    python -m ingestion.pipeline                 # tout le corpus, avec stockage S3 + PostgreSQL
    python -m ingestion.pipeline --no-store      # extraction seule (fichier JSONL local)
    python -m ingestion.pipeline --only manuel_pompe_hf-350 scan_pompe_01
"""

import argparse
import json
import time
from collections import Counter

from common.config import get_settings
from ingestion.extract import extract_document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-store", action="store_true", help="ne pas écrire dans S3 ni PostgreSQL")
    parser.add_argument("--only", nargs="+", metavar="DOC_ID", help="ne traiter que ces documents")
    args = parser.parse_args()

    s = get_settings()
    metadata = json.loads((s.corpus_dir / "metadata.json").read_text(encoding="utf-8"))
    if args.only:
        metadata = [m for m in metadata if m["document_id"] in args.only]

    con = None
    if not args.no_store:
        from ingestion.storage import connect, ensure_bucket, save_document, upload_files
        ensure_bucket()
        con = connect()

    out_dir = s.data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    start = time.perf_counter()

    with open(out_dir / "elements.jsonl", "w", encoding="utf-8") as out:
        for meta in metadata:
            pdf_path = s.data_dir / meta["fichier"]
            elements = extract_document(pdf_path, meta)
            if con is not None:
                pdf_key = upload_files(pdf_path, meta, elements)
                save_document(con, meta, elements, pdf_key)
            for el in elements:
                out.write(el.model_dump_json() + "\n")
                counts[el.type if el.source == "pdf" else f"{el.type} (OCR)"] += 1
            print(f"  {meta['document_id']:<34} {len(elements):>3} éléments")

    if con is not None:
        con.close()
    print(f"\n{len(metadata)} documents en {time.perf_counter() - start:.1f} s : "
          + ", ".join(f"{n} {t}" for t, n in sorted(counts.items())))
    print(f"Éléments : {out_dir / 'elements.jsonl'}" + ("" if con is None else "  |  stockés dans S3 et PostgreSQL"))


if __name__ == "__main__":
    main()
