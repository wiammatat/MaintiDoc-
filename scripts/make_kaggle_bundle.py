"""Construit dist/maintidoc_kaggle.zip : code + données utiles pour exécuter la génération sur Kaggle.

Usage : python scripts/make_kaggle_bundle.py
Prérequis : python -m pipeline.ingest (pour data/processed).
Le zip est à téléverser comme dataset Kaggle privé nommé « maintidoc-bundle ».
Aucun secret n'est inclus : ni .env, ni identifiants.
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ["common", "extraction", "preparation", "indexing", "retrieval", "generation", "pipeline", "prompts"]
FILES = ["requirements-kaggle.txt", "data/users.json", "data/corpus/metadata.json"]
DATA_DIRS = ["data/photos", "data/processed"]
EXCLUDE = {"__pycache__", ".env"}


def iter_files():
    for d in CODE + DATA_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and not EXCLUDE & set(p.parts) and p.name != ".env":
                yield p
    for f in FILES:
        yield ROOT / f


def main() -> None:
    if not (ROOT / "data/processed/elements.jsonl").exists():
        raise SystemExit("data/processed absent : lancer d'abord python -m pipeline.ingest")
    out = ROOT / "dist" / "maintidoc_kaggle.zip"
    out.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in iter_files():
            z.write(p, p.relative_to(ROOT).as_posix())
            n += 1
    print(f"{n} fichiers -> {out} ({out.stat().st_size / 1e6:.1f} Mo)")


if __name__ == "__main__":
    main()
