"""Étapes 1-2 du pipeline : extraction PyMuPDF, préparation textuelle et visuelle, stockage.

Usage :
    python -m pipeline.ingest                 # tout le corpus, avec stockage MinIO + PostgreSQL
    python -m pipeline.ingest --no-store      # sans stockage (fichier JSONL local seulement)
    python -m pipeline.ingest --reset         # vide MinIO et PostgreSQL avant (changement de corpus)
    python -m pipeline.ingest --only manuel_baie_sn-400 scan_baie_01

Sortie : data/processed/elements.jsonl (un élément par ligne), utilisé par python -m pipeline.index.
"""

import argparse
import json
import time
from collections import Counter

from common.config import get_settings
from preparation.document import prepare_document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-store", action="store_true", help="ne pas écrire dans MinIO ni PostgreSQL")
    parser.add_argument("--only", nargs="+", metavar="DOC_ID", help="ne traiter que ces documents")
    parser.add_argument("--reset", action="store_true", help="vider MinIO et PostgreSQL avant l'ingestion")
    args = parser.parse_args()

    s = get_settings()
    metadata = json.loads((s.corpus_dir / "metadata.json").read_text(encoding="utf-8"))
    if args.only:
        metadata = [m for m in metadata if m["document_id"] in args.only]

    con = None
    # Profil léger (STORAGE_BACKEND=local) : pas de MinIO ni de PostgreSQL, l'API lit le disque.
    if not args.no_store and s.storage_backend == "docker":
        from storage.metadata_db import connect, reset, save_document
        from storage.s3 import clear_bucket, ensure_bucket, upload_files
        ensure_bucket()
        con = connect()
        if args.reset:
            reset(con)
            print(f"Réinitialisation : base vidée, {clear_bucket()} fichiers supprimés du bucket")

    out_dir = s.data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    start = time.perf_counter()

    with open(out_dir / "elements.jsonl", "w", encoding="utf-8") as out:
        for meta in metadata:
            pdf_path = s.data_dir / meta["fichier"]
            elements = prepare_document(pdf_path, meta)
            if con is not None:
                save_document(con, meta, elements, upload_files(pdf_path, meta, elements))
            for el in elements:
                out.write(el.model_dump_json() + "\n")
                counts[el.type if el.source == "pdf" else f"{el.type} (OCR)"] += 1
            print(f"  {meta['document_id']:<34} {len(elements):>3} éléments")

    if con is not None:
        con.close()
    print(f"\n{len(metadata)} documents en {time.perf_counter() - start:.1f} s : "
          + ", ".join(f"{n} {t}" for t, n in sorted(counts.items())))
    print(f"Éléments : {out_dir / 'elements.jsonl'}" + ("" if con is None else "  |  stockés dans MinIO et PostgreSQL"))


if __name__ == "__main__":
    main()
