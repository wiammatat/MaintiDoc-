# MaintiDoc

Assistant documentaire multimodal et agentique pour techniciens de maintenance : recherche dans les manuels (texte et images), analyse de photos, consultation de l'historique via MCP et préparation de tickets avec validation humaine.

> Prototype pédagogique. Données fictives. L'assistant ne pilote aucun équipement et ne remplace ni les procédures de sécurité ni un technicien habilité.

## Structure

```
common/       configuration, utilisateurs, contrôle d'accès
ingestion/    extraction PDF, OCR, découpage, indexation
retrieval/    recherche texte/image, fusion RRF, reranking
agent/        workflow LangGraph
mcp_server/   serveur MCP de démonstration (historique, tickets)
api/          API FastAPI
ui/           interface Streamlit
prompts/      prompts versionnés (YAML)
finetuning/   données et entraînement du reranker
eval/         jeu de test et scripts d'évaluation
tests/        tests automatisés
scripts/      génération du corpus et utilitaires
data/         corpus, photos, base MCP, utilisateurs (voir data/README.md)
docs/         sujet, cadrage, étapes, licences
```

## Installation

Prérequis : Python 3.11, Docker, Tesseract OCR (avec le pack de langue française).

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate            # Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # puis remplir les secrets
```

Pack de langue français pour l'OCR (non versionné, placé dans `models/tessdata/`) :

```bash
mkdir models\tessdata
copy "C:\Program Files\Tesseract-OCR\tessdata\eng.traineddata" models\tessdata\
copy "C:\Program Files\Tesseract-OCR\tessdata\osd.traineddata" models\tessdata\
curl -L -o models\tessdata\fra.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/fra.traineddata
```

Le chemin de `tesseract.exe` se règle avec `TESSERACT_CMD` dans `.env`.

## Démarrage

```bash
python scripts/generate_corpus.py   # génère data/ (déterministe)
docker compose up -d                # Qdrant, PostgreSQL, SeaweedFS (S3), MLflow
python scripts/check_services.py    # vérifie la connexion à chaque service
python -m ingestion.pipeline        # extraction + OCR + stockage S3 / PostgreSQL (~1 min 30)
pytest                              # vérifie le corpus, l'OCR et le contrôle d'accès
```

| Service | URL |
|---|---|
| Qdrant | http://localhost:6333/dashboard |
| PostgreSQL | localhost:5433 (5433 pour ne pas entrer en conflit avec un PostgreSQL local) |
| SeaweedFS (S3) | http://localhost:8333 |
| MLflow | http://localhost:5000 |

## Documentation
- [Sujet du projet](docs/sujet_projet.pdf)
- [Note de cadrage et risques](docs/cadrage.md)
- [Étapes détaillées](docs/MaintiDoc_Etapes.md)
- [Licences des composants](docs/licences.md)
- [Corpus](data/README.md)
