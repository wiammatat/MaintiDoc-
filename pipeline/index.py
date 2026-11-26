"""Étape 3 du pipeline : index textuel (LlamaIndex + BGE-M3) et index visuel (CLIP) dans Qdrant.

Usage :
    python -m pipeline.index              # indexe data/processed/elements.jsonl
    python -m pipeline.index --recreate   # supprime et recrée les deux index
"""

import argparse
import time

from common.config import get_settings
from indexing.qdrant import client
from indexing.text_index import build_text_index
from indexing.visual_index import build_visual_index
from preparation.schema import Element


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recreate", action="store_true", help="supprimer et recréer les index")
    args = parser.parse_args()

    s = get_settings()
    path = s.data_dir / "processed" / "elements.jsonl"
    elements = [Element.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    start = time.perf_counter()
    n_text = build_text_index(elements, recreate=args.recreate)
    n_img = build_visual_index(elements, recreate=args.recreate)
    print(f"Index textuel {s.text_collection} : {n_text} nœuds (LlamaIndex + BGE-M3)")
    print(f"Index visuel  {s.image_collection} : {n_img} images (CLIP)  [{time.perf_counter() - start:.1f} s]")
    client().close()   # fermeture explicite : évite l'erreur de fermeture tardive du Qdrant embarqué


if __name__ == "__main__":
    main()
