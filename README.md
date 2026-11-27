# MaintiDoc

Assistant documentaire multimodal et agentique pour la maintenance d'un **SAN** (baies de stockage et switchs Fibre Channel) : recherche dans les manuels (texte et images), analyse de photos, consultation de l'historique via MCP et préparation de tickets avec validation humaine.

> Prototype pédagogique. Données fictives. L'assistant ne pilote aucun équipement et ne remplace ni les procédures de sécurité ni un technicien habilité.

## Structure

Le code suit le pipeline RAG multimodal du prototype de référence (`Data_du_prof/`) :

```
Manuel PDF -> extraction -> préparation textuelle / visuelle -> index textuel / visuel
           -> RAG multimodal (recherche + fusion RRF) -> paquet de preuves -> VLM -> réponse JSON sourcée
```

```
common/       configuration, utilisateurs, contrôle d'accès
extraction/   1. PyMuPDF : blocs bruts (texte, tableaux, images) et métadonnées (page, section, légende)
preparation/  2. préparation textuelle (découpage, OCR, provenance) et visuelle (schémas, pages)
storage/         fichiers dans MinIO, métadonnées dans PostgreSQL
indexing/     3. index textuel (LlamaIndex + BGE-M3) et index visuel (CLIP), dans Qdrant
retrieval/    4. recherche texte + image, fusion RRF, reranker, paquet de preuves
generation/   5. VLM (Qwen2.5-VL), réponse JSON sourcée, vérification des citations
pipeline/        commandes : ingest (étapes 1-2) et index (étape 3)
agent/           6. agent LangGraph : analyse, historique MCP, réponse, ticket, validation humaine
mcp_server/      serveur MCP de maintenance (historique, tickets) avec contrôle des droits à chaque appel
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
docker compose up -d                # Qdrant, PostgreSQL, MinIO (S3), MLflow (1er lancement : compilation de MinIO)
python scripts/check_services.py    # vérifie la connexion à chaque service
python -m pipeline.ingest --reset   # extraction + préparation + stockage MinIO / PostgreSQL (~1 min 40)
python -m pipeline.index --recreate # index LlamaIndex + BGE-M3 et CLIP -> Qdrant (1er lancement : ~3,5 Go de modèles)
pytest                              # vérifie tout le pipeline (corpus, extraction, index, recherche, génération)
```

| Service | URL |
|---|---|
| Qdrant | http://localhost:6333/dashboard |
| PostgreSQL | localhost:5433 (5433 pour ne pas entrer en conflit avec un PostgreSQL local) |
| MinIO (API S3) | http://localhost:9000 (console : http://localhost:9001) |
| MLflow | http://localhost:5000 |

## Génération des réponses : PC avec carte NVIDIA

Qwen2.5-VL-7B nécessite un GPU NVIDIA. Avec une carte de **8 Go**, tout ne tient pas en même temps :

| Configuration | Réglages `.env` | Remarque |
|---|---|---|
| **A** (recommandée) : Qwen 7B en 4 bits sur la carte, recherche sur le processeur | `VLM_BACKEND=qwen`, `VLM_QUANTIZATION=4bit`, `DEVICE=cpu`, `MODEL_DTYPE=float32` | Modèle du sujet ; mémoire juste, reranker plus lent sur processeur |
| **B** (si A manque de mémoire) : Qwen 3B en 4 bits, tout sur la carte | `VLM_MODEL=Qwen/Qwen2.5-VL-3B-Instruct`, `VLM_QUANTIZATION=4bit`, `DEVICE=cuda`, `MODEL_DTYPE=float16` | Plus rapide, modèle plus petit (licence différente, voir docs/licences.md) |

Installer PyTorch avec CUDA (et non la version CPU) sur ce PC, ainsi que `bitsandbytes`.
Sans GPU, `POST /ask` et `POST /agent` renvoient 503 ; `POST /search` fonctionne. Les tests utilisent un faux VLM.

## API FastAPI

```bash
uvicorn api.main:app --reload       # documentation interactive : http://localhost:8000/docs
```

Comptes de démonstration (fictifs) : `tech_s01`, `tech_s02` (stockage), `tech_r01`, `tech_r02` (réseau), `resp_01`
(responsable) ; mot de passe `<identifiant>-demo`, par exemple `tech_s01-demo`.

| Route | Rôle |
|---|---|
| `POST /auth/token` | Connexion -> jeton JWT |
| `GET /me` | Profil et groupes |
| `POST /search` | Recherche multimodale (question + photo facultative) -> preuves citables |
| `POST /ask` | Réponse sourcée par le VLM (nécessite `VLM_BACKEND=qwen` ou `remote`) |
| `GET /documents/{id}/pdf`, `GET /elements/{uid}/image` | Fichiers, après contrôle d'accès |
| `POST /agent` | Agent LangGraph : réponse sourcée, historique MCP, brouillon de ticket à approuver |
| `POST /agent/{id}/approval` | Approbation humaine (contenu éventuellement modifié) ou refus ; création via MCP |

### Agent et serveur MCP

L'agent (`agent/graph.py`, LangGraph) appelle le serveur MCP de maintenance (`mcp_server/server.py`) en transport
stdio : `get_equipment_history`, `get_open_tickets`, `create_ticket`. La base fictive se crée avec
`python scripts/generate_corpus.py --mcp-only`. Exemple : « Voyant orange sur BAIE-SN400-002, prépare un ticket »
renvoie la réponse, l'historique et un brouillon ; le ticket n'est créé qu'après `POST /agent/{id}/approval`.

### Qwen sur un autre PC du réseau local

Si l'API tourne sur un PC sans GPU et Qwen sur le PC équipé de la carte NVIDIA :

1. Sur le PC avec GPU : `python -m generation.vlm_server --host 0.0.0.0 --port 8001` (affiche un jeton).
2. Dans le `.env` du PC qui fait tourner l'API : `VLM_BACKEND=remote`, `VLM_REMOTE_URL=http://<IP du PC GPU>:8001`,
   `VLM_REMOTE_TOKEN=<jeton affiché>`.

Le jeton change à chaque lancement du serveur ; sans lui, le serveur refuse toute requête.

## Lightning AI Studio (profil léger, sans Docker)

Tout tourne dans un seul Studio : recherche, Qwen2.5-VL-7B (4 bits sur un GPU T4) et API.
Profil léger : Qdrant embarqué, fichiers servis depuis le disque, pas de PostgreSQL ni de MinIO ;
même contrôle d'accès que le profil Docker.

1. Créer un Studio, ouvrir un terminal et cloner le dépôt : `git clone https://github.com/wiammatat/MaintiDoc-.git && cd MaintiDoc-`
2. Choisir une machine **GPU T4** (ou CPU pour la recherche seule), puis : `bash scripts/setup_lightning.sh`
3. Lancer l'API : `bash scripts/start_lightning.sh`
4. Panneau de droite -> **Port Viewer** -> ajouter le port **8000** : l'adresse obtenue suivie de `/docs` ouvre l'API.

Les modèles (~22 Go) sont placés dans un cache temporaire (`/tmp/maintidoc_hf`) pour ne pas consommer les 10 Go
persistants : ils sont retéléchargés à chaque session (premier appel plus lent). Repasser le Studio sur CPU
quand Qwen n'est pas utilisé, pour économiser les crédits GPU.

## Documentation
- [Sujet du projet](docs/sujet_projet.pdf)
- [Note de cadrage et risques](docs/cadrage.md)
- [Étapes détaillées](docs/MaintiDoc_Etapes.md)
- [Licences des composants](docs/licences.md)
- [Choix techniques et écarts au sujet](docs/choix_techniques.md)
- [Corpus](data/README.md)
