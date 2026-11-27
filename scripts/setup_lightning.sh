#!/usr/bin/env bash
# Installation de MaintiDoc dans un Lightning AI Studio (profil léger, sans Docker).
#
# Usage, depuis le dossier du projet cloné :   bash scripts/setup_lightning.sh
#
# - dépendances Python allégées (requirements-lightning.txt) et Tesseract français ;
# - fichier .env créé une seule fois, avec des secrets aléatoires ;
# - modèles dans un cache temporaire (HF_HOME) : ils ne consomment pas les 10 Go persistants ;
# - préparation du corpus et des index (Qdrant embarqué dans data/qdrant).
# Le script peut être relancé : il ne refait que ce qui manque.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

echo "== 1/5 Cache des modèles hors du disque persistant"
HF_CACHE="${MAINTIDOC_HF_HOME:-/tmp/maintidoc_hf}"
mkdir -p "$HF_CACHE"
export HF_HOME="$HF_CACHE"
grep -q "MAINTIDOC_HF" ~/.bashrc 2>/dev/null || echo "export HF_HOME=\"$HF_CACHE\"   # MAINTIDOC_HF" >> ~/.bashrc
FREE_GB=$(df -BG --output=avail "$HF_CACHE" | tail -1 | tr -dc '0-9')
echo "   cache : $HF_CACHE ($FREE_GB Go libres ; il en faut environ 25)"
[ "$FREE_GB" -ge 25 ] || echo "   ATTENTION : espace insuffisant pour tous les modèles (définir MAINTIDOC_HF_HOME ailleurs)"

echo "== 2/5 Dépendances"
pip install -q -r requirements-lightning.txt
if ! command -v tesseract >/dev/null || ! tesseract --list-langs 2>/dev/null | grep -qx fra; then
  sudo apt-get update -qq && sudo apt-get install -y -qq tesseract-ocr tesseract-ocr-fra
fi
TESSDATA_DIR="$(dirname "$(find /usr/share -name fra.traineddata 2>/dev/null | head -1)")"
echo "   tesseract : $(tesseract --version 2>&1 | head -1) | tessdata : $TESSDATA_DIR"

echo "== 3/5 Configuration (.env)"
if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  DEVICE=cuda; DTYPE=float16; VLM=qwen     # T4 : pas de bfloat16 natif
  echo "   GPU : $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
else
  DEVICE=cpu; DTYPE=float32; VLM=disabled
  echo "   pas de GPU : recherche seulement (POST /ask indisponible) ; passer le Studio sur GPU pour Qwen"
fi
if [ ! -f .env ]; then
  python - "$DEVICE" "$DTYPE" "$VLM" "$TESSDATA_DIR" "$ROOT" <<'PY'
import secrets, sys
device, dtype, vlm, tessdata, root = sys.argv[1:]
open(".env", "w", encoding="utf-8").write(f"""# Profil léger Lightning AI Studio (généré par scripts/setup_lightning.sh). Ne pas versionner.
STORAGE_BACKEND=local
QDRANT_PATH={root}/data/qdrant
DEVICE={device}
MODEL_DTYPE={dtype}
VLM_BACKEND={vlm}
VLM_QUANTIZATION=4bit
TESSERACT_CMD=tesseract
TESSDATA_PREFIX={tessdata}
JWT_SECRET={secrets.token_urlsafe(48)}
APPROVAL_SECRET={secrets.token_urlsafe(48)}
""")
PY
  echo "   .env créé (secrets aléatoires)"
else
  echo "   .env existant conservé (le supprimer pour le régénérer)"
fi

echo "== 4/5 Préparation du corpus (extraction, OCR) et base MCP"
python -m pipeline.ingest
[ -f data/mcp/maintenance.db ] || python scripts/generate_corpus.py --mcp-only

echo "== 5/5 Index textuel (LlamaIndex + BGE-M3) et visuel (CLIP)"
python -m pipeline.index --recreate

echo
echo "Installation terminée. Lancer l'API avec :   bash scripts/start_lightning.sh"
