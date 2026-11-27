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
notebooks/       notebook Kaggle (partie GPU)
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

## Génération des réponses (VLM sur Kaggle)

Le VLM Qwen2.5-VL-7B nécessite un GPU : il tourne sur Kaggle.

1. En local : `python scripts/make_kaggle_bundle.py` crée `dist/maintidoc_kaggle.zip` (code + données, sans secret).
2. Sur Kaggle : créer un dataset privé **maintidoc-bundle** à partir de ce zip.
3. Importer `notebooks/kaggle_maintidoc.ipynb`, activer **GPU T4 x2** et **Internet**, ajouter le dataset, tout exécuter.
4. Télécharger `resultats_vlm.json` (onglet *Output*) pour la démonstration locale.

En local, les tests de génération utilisent un faux VLM (`ScriptedVLM`) : ni GPU ni téléchargement.

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

### Brancher l'API sur Qwen qui tourne sur Kaggle

1. Sur Kaggle : exécuter les sections 1 à 4 du notebook, puis la section **7. Serveur VLM pour le PC**.
2. Copier les trois lignes affichées (`VLM_BACKEND=remote`, `VLM_REMOTE_URL=...`, `VLM_REMOTE_TOKEN=...`) dans `.env`.
3. Relancer l'API : `POST /ask` utilise alors Qwen sur le GPU de Kaggle.

L'adresse et le jeton changent à chaque session Kaggle. Si la session s'arrête, `POST /ask` renvoie 503.

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
