#!/usr/bin/env bash
# Lance l'API MaintiDoc dans un Lightning AI Studio (profil léger).
#
# Usage : bash scripts/start_lightning.sh
# Puis, dans le Studio : panneau de droite -> Port Viewer -> ajouter le port 8000 pour obtenir l'adresse publique.
# La documentation interactive est à l'adresse obtenue suivie de /docs.
#
# Qdrant embarqué n'accepte qu'un seul processus : arrêter l'API avant de relancer python -m pipeline.index.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${MAINTIDOC_HF_HOME:-/tmp/maintidoc_hf}"
[ -f .env ] || { echo "Fichier .env absent : lancer d'abord bash scripts/setup_lightning.sh"; exit 1; }
# Les modèles sont retéléchargés au premier appel de chaque session (cache temporaire) : la première
# requête est plus lente.
exec uvicorn api.main:app --host 0.0.0.0 --port 8000
