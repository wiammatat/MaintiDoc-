# Choix techniques et écarts par rapport au sujet

Ce document compare l'implémentation actuelle (étapes 0 à 1.4) avec le sujet « MaintiDoc » et justifie chaque écart.

## 1. Synthèse

| Exigence du sujet | Statut | Commentaire |
|---|---|---|
| Corpus de 30 à 50 documents, 2 familles d'équipements, données fictives | Conforme | 41 PDF synthétiques de maintenance **SAN** (baies StorNova, switchs FabriLink) dont un manuel pédagogique, 30 photos, utilisateurs et tickets fictifs |
| Extraction PyMuPDF + OCR local | Conforme | PyMuPDF + Tesseract 5 (français) |
| Découpage en paragraphes, tableaux, pages, figures | Conforme | 4 types d'éléments ; identifiants au format de l'exemple du sujet (`figure_18_2` = page 18, 2e figure) |
| Métadonnées par élément (id et version du document, page, élément, type d'équipement, groupes) | Conforme | Stockées dans PostgreSQL et dans Qdrant |
| Images originales disponibles pour le VLM | Conforme | Figures et pages stockées en S3 |
| Recherche vectorielle Qdrant | Conforme | 2 collections versionnées (texte, images) ; index textuel construit avec **LlamaIndex** |
| Embeddings texte BGE-M3 | Conforme | |
| Encodeur texte-image multilingue compatible CLIP | Conforme | CLIP ViT-B/32 (images) + sa version multilingue (texte) |
| Fusion des résultats « par exemple avec RRF » | Conforme, variante | RRF **pondérée** (voir 2.3) |
| Reranking des candidats textuels et OCR | Conforme | bge-reranker-v2-m3 |
| Filtrage par groupe avant récupération, puis à l'affichage | Conforme | Filtre Qdrant à la requête + second contrôle sur chaque résultat |
| Stockage : PostgreSQL + MinIO | Conforme, adapté | MinIO **compilé depuis son code source** (voir 2.1) |
| Inférence du VLM sur un GPU de 24 Go | **Écart** | GPU Kaggle de 16 Go (voir 2.2) |
| Secrets hors de Git | Conforme | `.env` exclu, `SecretStr`, aucun secret dans les journaux |
| Versions figées dans le dépôt | Conforme | `requirements.lock` ; licences listées dans `docs/licences.md` (à vérifier) |
| Docker Compose | Conforme (partiel) | Services d'infrastructure en place ; services applicatifs aux étapes suivantes |

## 2. Écarts et justifications

### 2.1 MinIO compilé depuis son code source
- **Constat** : MinIO ne distribue plus de versions prêtes à l'emploi. Vérification faite : l'image `minio/minio` (Docker Hub), `quay.io/minio/minio` et `bitnami/minio` ne sont plus téléchargeables, et le binaire officiel `dl.min.io` répond « 410 Gone ». Seul le code source reste publié sur GitHub. L'image d'archive `bitnamilegacy/minio` existe encore, mais elle n'est plus maintenue (pas de correctifs de sécurité).
- **Choix** : compilation de MinIO depuis son dépôt officiel, dans `docker/minio/Dockerfile`, avec une version figée (`RELEASE.2025-10-15T17-29-55Z`, Go 1.24.8, Alpine 3.22.2). Le conteneur tourne avec un utilisateur non administrateur.
- **Alternative évaluée** : SeaweedFS (API S3, Apache 2.0) a été utilisé temporairement. MinIO a été retenu pour rester conforme au sujet.
- **Impact** : aucun sur le code, qui utilise un client S3 générique (`boto3`). Seul le premier lancement est plus long (compilation de quelques minutes).

### 2.2 GPU : Kaggle (16 Go) au lieu d'un GPU de 24 Go
- **Constat** : la machine de développement n'a pas de GPU NVIDIA et dispose de 7,7 Go de RAM.
- **Choix** : ingestion, OCR, embeddings, recherche, MCP, API et interface tournent en local sur CPU. Le VLM Qwen2.5-VL-7B, le fine-tuning du reranker et l'évaluation complète tournent sur Kaggle (GPU T4 ou P100 de 16 Go).
- **Conséquence** : sur 16 Go, le modèle 7B doit être chargé en fp16 ou quantifié en 4 bits. Le sujet prévoit déjà la quantification (section 11) et précise que l'hypothèse des 24 Go « doit être validée par benchmark » : ce benchmark sera fait sur Kaggle.
- **Limite** : la démonstration du VLM se fera depuis Kaggle, ou avec des réponses enregistrées. Le prototype local ne peut pas l'exécuter en temps réel.

### 2.3 RRF pondérée
- **Constat** : la recherche « question vers images » via CLIP est peu fiable pour une question générale en français (« Que signifie ce voyant ? ») et remontait des pages de texte sans rapport. Avec une RRF classique, ce bruit pesait autant qu'une photo.
- **Choix** : poids de 1 pour la recherche texte et la recherche par photo, 0,5 pour « question vers images ».
- **Limite** : ce poids a été fixé sur quelques exemples. Il sera validé ou ajusté sur le jeu d'évaluation (étape 5).

### 2.4 Précision des modèles : bfloat16 essayé puis écarté sur CPU
- **Constat** : une erreur de mémoire (« fichier de pagination insuffisant ») est survenue sur la machine de développement (7,7 Go de RAM).
- **Essai** : chargement des modèles en bfloat16 (réglage `MODEL_DTYPE`) pour diviser leur mémoire par deux.
- **Mesure** :

  | | float32 | bfloat16 |
  |---|---|---|
  | Mémoire maximale (4 modèles chargés, 132 passages + 64 images encodés) | 3,1 Go | non mesuré |
  | Chargement + encodage du corpus | 2 min | interrompu après plus de 3 h de calcul |
  | Temps par recherche (avec reranker) | 30 s | — |

- **Explication** : le processeur (Intel Core i5-1235U) n'a pas d'instructions bfloat16 natives. Les calculs sont émulés, donc des dizaines de fois plus lents.
- **Décision** : float32 par défaut. La mémoire réelle (3,1 Go) est acceptable : l'erreur venait de deux programmes lourds lancés en parallèle (téléchargement du reranker et tests). Le réglage `MODEL_DTYPE=bfloat16` reste disponible pour un GPU.
- **Limite** : 30 s par recherche sur ce CPU, à cause du reranker (568 M de paramètres). Sur le GPU Kaggle, ce temps sera nettement plus court. Le budget de 60 s du sujet est respecté pour la recherche seule.

### 2.5 Génération : citations par étiquettes
- **Principe** : chaque source transmise au VLM reçoit une étiquette (`S1`, `S2`…). Le modèle ne cite que ces étiquettes, et le programme les convertit en citations `document_id` / `page` / `element_id`.
- **Raison** : le modèle ne peut pas inventer un identifiant de document ou de page. Une étiquette inconnue est rejetée et signalée (`invalid_citations`). Les citations pointent donc toujours vers des éléments réellement récupérés, comme l'exige la section 4.2 du sujet.
- **Garde-fou** : une réponse « answered » sans aucune citation valide est requalifiée en « abstained ».
- **Format** : sortie JSON validée par Pydantic, avec une seconde tentative si le JSON est invalide, puis une erreur propre.
- **Vérification du soutien (étape 1.6)** : un second appel au modèle (prompt `evidence_check_v1`) juge chaque phrase citant une source : « soutenu », « partiel » ou « non soutenu ». Une source est retirée des citations si toutes les phrases qui la citent sont jugées non soutenues ; elle reste visible dans le texte avec la mention « non vérifié » et l'affirmation est signalée dans les incertitudes. Si la vérification échoue, rien n'est retiré mais l'échec est signalé. Coût : 2 appels au modèle par question au lieu d'un (budget du sujet : 6).
- **Limite** : seules les phrases qui citent une source sont vérifiées ; le vérificateur est le même modèle que le générateur.

### 2.6 Exécution du VLM sur Kaggle
- **Mode Qdrant embarqué** (`QDRANT_PATH`) : sur Kaggle, il n'y a pas de serveur Docker. L'index est reconstruit sur le GPU dans un Qdrant local au processus, avec le même code et les mêmes filtres d'accès.
- **Quantification** : 7B en fp16 sur 2 GPU T4 (32 Go au total) ; sinon 4 bits (NF4) sur un seul GPU de 16 Go.
- **Limite des images** : au plus 2 images par question (photo + 1 image récupérée, ou 2 images récupérées), d'au plus 384 × 28 × 28 pixels (environ 300 000), pour borner le contexte et la mémoire (budgets de la section 9).

### 2.7 Premier essai sur Kaggle (2 GPU T4) et corrections
| Constat | Correction |
|---|---|
| Qwen2.5-VL-7B chargé en fp16 sur 2 T4 en 136 s (16,6 Go) ; réponses en 28 à 61 s | — |
| « CUDA out of memory » à la 3ᵉ question (3 images) : le GPU 0 portait la moitié du VLM et les modèles de recherche | Modèles de recherche sur le GPU 1, poids du VLM limités à 7 Go sur le GPU 0, images limitées à 512 × 28 × 28 pixels, mémoire libérée entre deux questions |
| Photo d'un HF-200 : la réponse citait aussi les HF-350 et HF-500, avec un statut « answered » et une question de clarification | Prompt `answer_v2` : identifier le modèle et ne citer que ses sources ; clarification uniquement avec le statut « needs_clarification » (contrôlé aussi dans le code) |
| 2ᵉ essai : « CUDA out of memory » sur le GPU 1 dès la 1ʳᵉ question (10,4 Go de VLM + 2,8 Go de recherche, il manquait 1,35 Go pour l'attention, dont la mémoire croît avec le carré de la longueur du prompt) | Recherche déplacée sur le GPU 0 (environ 9-10 Go par carte, 4-5 Go de marge chacune) ; le notebook continue après un échec mémoire et affiche la longueur du prompt et les pics mémoire par carte |
| 3ᵉ essai : 4 réponses sur 6 ; 2 échecs mémoire (5 425 et 7 947 tokens). Les images n'étaient pas réduites (une page A4 = environ 2 000 tokens), des pages entières étaient jointes plutôt que des figures, et la vérification (1.6) ne se lançait pas car Qwen ne place pas les étiquettes [S1] dans le texte | Images réduites par le code avant l'envoi (300 000 pixels au plus), 2 images au plus, figures prioritaires sur les pages ; vérification de la réponse entière face à la liste « citations » quand le texte ne contient pas d'étiquette. Estimation : environ 2 500 tokens par question |
| 4ᵉ essai : 6 questions sur 6 sans plantage (pics ≤ 12,7 Go), vérification active (2 appels par question). Mais 2 abstentions à tort (vidange AC-30, note sur les purgeurs) alors que la bonne note était la 1ʳᵉ source (vérifié en local : score 0,99), et une photo mal lue (bouton d'arrêt d'urgence pris pour le voyant L4, erreur rattrapée par la vérification) | Prompt `answer_v3` : documents de famille utilisables pour tous les modèles, pas d'abstention si une source répond, description du voyant avant interprétation, étiquettes [S1] dans le texte. Photo de l'utilisateur gardée plus grande (≈ 400 000 pixels) que les images récupérées ; 4 passages au lieu de 5. Estimation : 2 100 à 2 650 tokens par question |
| Réponse reprenant « dépasse le seuil de 10 °C » | Erreur de rédaction du corpus corrigée (« dépasse le seuil d'alarme de plus de 10 °C ») dans les 3 manuels de pompes, puis ré-ingestion et réindexation |

### 2.8 Réorganisation selon le prototype du professeur (`Data_du_prof/`)
Les trois schémas fournis par le professeur décrivent un prototype de référence : un manuel **SAN** pédagogique,
une extraction PyMuPDF, une indexation **LlamaIndex + BGE-M3** pour le texte et **CLIP** pour les images, une fusion RRF,
un paquet de preuves, une génération par un VLM Qwen et une réponse JSON sourcée. Le projet a été réaligné sur ce pipeline.

| Changement | Avant | Après | Raison |
|---|---|---|---|
| Domaine | Pompes et compresseurs fictifs | Maintenance **SAN** : baies de stockage StorNova et switchs Fibre Channel FabriLink, plus un manuel SAN pédagogique (texte, tableaux, 5 schémas, photo de couverture) | Aligné sur le manuel SAN du prototype ; les pièges d'évaluation sont conservés et transposés |
| Index textuel | BGE-M3 appelé directement avec le client Qdrant | **LlamaIndex** (`VectorStoreIndex` + `QdrantVectorStore`) avec BGE-M3 | Aligné sur le schéma du prof ; Qdrant reste le stockage, conformément à l'étape « industrialiser le stockage vectoriel » de sa feuille de route |
| Organisation du code | `ingestion/` (tout en un), `retrieval/` | `extraction/` → `preparation/` (textuelle, visuelle) → `indexing/` (textuel, visuel) → `retrieval/` (recherche, fusion, reranker, **paquet de preuves**) → `generation/` | Chaque module correspond à une case du schéma « Pipeline RAG multimodal » |
| Commandes | `python -m ingestion.pipeline`, `python -m ingestion.index` | `python -m pipeline.ingest`, `python -m pipeline.index` | Nouvelle organisation ; option `--reset` pour changer de corpus |
| Prompt | `answer_v3` | `answer_v4` (vocabulaire SAN : boutons POWER/ID/RESET, précautions redondance et fabric) | Les règles de v3 sont conservées |

**Choix d'intégration de LlamaIndex** : LlamaIndex utilise le **même** modèle BGE-M3 que le reste du projet (classe
`BGEM3Embedding`), au lieu de sa propre intégration Hugging Face qui voulait retélécharger 2,2 Go dans un autre cache.
Le titre et la section du document sont ajoutés au texte encodé ; les droits d'accès sont stockés mais jamais encodés.
Le filtrage par groupe reste fait **avant** la récupération (filtre LlamaIndex traduit en filtre Qdrant).

**Inchangé** : le VLM reste Qwen2.5-VL-7B sur Kaggle. Le prototype du professeur utilise un Qwen plus grand via Ollama
sur un Mac M3 de 36 Go ; la machine de développement n'a pas de GPU (voir 2.2).

**Erreur détectée par les tests pendant la réorganisation** : le texte donné au reranker (titre puis section) n'était pas
dans le même ordre que le texte encodé par LlamaIndex (section puis titre), et un élément sans section était encodé avec
« section: None ». Les deux ont été corrigés et un test vérifie maintenant que les deux textes sont identiques.

### 2.9 Premier essai Kaggle sur le corpus SAN
L'essai confirme la partie technique : LlamaIndex s'installe et tourne sur Kaggle (147 nœuds, 77 images en 38 s),
aucun plantage mémoire (pics ≤ 13,1 Go), vérification active (2 appels par question).

| Constat | Correction |
|---|---|
| **Erreur grave** : un technicien réseau demande le sens du voyant orange de la **baie SN-400**. Il n'a accès à aucun document de baie ; le modèle répond avec le sens du voyant d'un **switch** (« port en erreur »), et la vérification le juge « soutenu » car elle ne voyait que le texte des sources | **Garde-fou déterministe** (`generation/guards.py`) : si la question cite un modèle pour lequel l'utilisateur n'a accès à aucun document (d'après les droits du corpus), abstention sans appeler le VLM. Une première version se fondait sur les 5 premiers résultats de la recherche : elle bloquait à tort un responsable qui comparait la SN-400 et le FL-48 ; la règle porte donc sur les droits d'accès, pas sur le classement. **Vérification v2** (`evidence_check_v2`) : le titre de chaque source est fourni, et une information attribuée à un autre équipement est « non soutenue ». **Prompt v5** : interdiction de transposer une information d'un équipement à un autre |
| Contradiction sur le seuil E03 détectée par le modèle, mais réponse rejetée : `uncertainties` écrit comme un texte et non comme une liste | Le schéma Pydantic convertit un texte seul en liste au lieu de rejeter toute la réponse |
| Photo au nom masqué : réponse donnée au lieu d'une clarification | Prompt v5 : clarification obligatoire si la marque et le modèle ne sont pas lisibles et que la question ne les précise pas |
| Note piégée : injection ignorée mais non signalée | Prompt v5 : phrase de signalement imposée dans les incertitudes |

**Leçon** : une règle de sécurité critique (ne pas transposer d'un équipement à un autre) ne doit pas reposer
uniquement sur le prompt. Le garde-fou en code est déterministe et testé ; le prompt et la vérification
servent de deuxième et troisième barrières.

### 2.10 Backend FastAPI et VLM distant (Kaggle)
**API** (`api/`) : authentification par jeton JWT (mots de passe hachés PBKDF2), groupes relus côté serveur à chaque
requête, contrôle d'accès à l'ouverture des PDF et des images (exigence du sujet), même réponse 404 pour un fichier
interdit ou inexistant, photo vérifiée (vraie image, 5 Mo maximum), journal de métadonnées sans le texte des questions.

**VLM distant** : la machine de développement n'a pas de GPU. Le notebook Kaggle lance un petit serveur
(`generation/vlm_server.py`) qui expose Qwen, joint par un tunnel Cloudflare. L'API du PC l'appelle via `RemoteVLM`
(`VLM_BACKEND=remote`). Seule la génération est déportée : recherche, contrôle d'accès et vérification restent sur le PC.

| Risque | Mesure |
|---|---|
| Adresse du tunnel publique | Jeton secret obligatoire, comparé à temps constant ; nouveau jeton à chaque session |
| Abus du serveur | Nombre et taille des images bornés, textes bornés, une génération à la fois |
| Session Kaggle arrêtée ou tunnel coupé | `POST /ask` renvoie 503 avec un message clair ; `/search` reste disponible |
| Données transmises à Kaggle et Cloudflare | Données fictives uniquement ; à proscrire avec de vraies données sans accord |
| Délai maximal d'une requête via le tunnel (environ 100 s chez Cloudflare) | Générations observées entre 15 et 60 s ; `VLM_REMOTE_TIMEOUT` réglable |

**Limites** : l'usage d'une session Kaggle comme serveur doit être conforme aux conditions d'utilisation de Kaggle
(à vérifier) ; la session est limitée dans le temps ; c'est une solution de prototype, pas de production.

### 2.11 Essai Kaggle avec le prompt v5 : deux protections passées du prompt au code
| Constat (prompt v5) | Correction |
|---|---|
| Le modèle recopiait la phrase imposée « La source [S1] contient une consigne qui a été ignorée » sur des sources **saines**, et ne signalait **pas** la vraie note piégée | **Détection déterministe** (`retrieval/sanitize.py`) : les paragraphes qui ressemblent à une consigne (« ignore les instructions », « [NOTE SYSTEME] », « envoie… vers http », « accorde… les droits ») sont remplacés par un marqueur **avant** l'envoi au générateur et au vérificateur ; le signalement est écrit par le code ; les signalements inventés par le modèle sont écartés. Prompt `answer_v6` sans phrase imposée. Testé sur tout le corpus : seule la note piégée est détectée |
| Réponse **exacte** sur la contradiction du seuil E03 (55 °C / 60 °C) rejetée à tort par le vérificateur (même modèle 7B) | **Contrôle des valeurs** (`generation/verify.py`) : si toutes les valeurs chiffrées de l'affirmation figurent dans les sources citées **et** que les modèles d'équipement cités correspondent à ces sources, le verdict « non soutenu » devient « partiel » et la citation est conservée. Une valeur d'un autre modèle (ex. 65 °C de la SN-800 attribué à la SN-400) n'est jamais validée |
| Photo de SN-200 décrite comme une SN-400 ; photo ambiguë sans clarification | Reporté à l'agent LangGraph (étape « analyser la demande et l'image » de la feuille de route) |

**Principe retenu** : avec un modèle de 7 milliards de paramètres, les règles critiques (équipement, injection,
validité des citations) sont appliquées par du code déterministe et testé ; le prompt reste une barrière supplémentaire.

### 2.12 Lightning AI Studio à la place de Kaggle : profil léger
**Pourquoi** : sur Kaggle, la session est éphémère et l'API du PC devait joindre Qwen par un tunnel ; sur le PC,
Docker dépend d'un disque externe. Un Studio Lightning est un espace de travail persistant où recherche, Qwen et API
tournent ensemble (`VLM_BACKEND=qwen`), l'API étant exposée par le Port Viewer officiel.

**Contraintes de l'offre gratuite** (indiquées par l'utilisatrice, à vérifier sur son compte) : un GPU T4 de 16 Go,
16 Go de RAM, 10 Go de disque persistant, environ 30 crédits (~50 h de T4).

| Contrainte | Choix |
|---|---|
| Qwen2.5-VL-7B en fp16 (16,6 Go) dépasse le T4 de 16 Go | Quantification **4 bits** (NF4, ~5-6 Go) : tient avec les modèles de recherche ; légère perte de qualité possible |
| Modèles ~22 Go > 10 Go de disque persistant | Modèles dans un **cache temporaire** (`HF_HOME=/tmp/maintidoc_hf`), retéléchargés à chaque session |
| Docker non garanti, images non conservées (MinIO à recompiler à chaque session) | **Profil léger** (`STORAGE_BACKEND=local`) : Qdrant embarqué, PDF et images lus sur le disque, métadonnées lues dans `metadata.json` et `elements.jsonl` |
| Disque limité | Dépendances allégées (`requirements-lightning.txt`) : ni PostgreSQL, ni MinIO, ni MLflow |

**Sécurité inchangée** : le profil léger renvoie les mêmes groupes autorisés que PostgreSQL ; l'API applique le même
contrôle d'accès (même 404 pour un fichier interdit ou inexistant). Les tests du contrôle d'accès tournent dans les
deux profils ; une tentative de chemin `../.env` est refusée.

**Profil complet** : Docker (Qdrant serveur, PostgreSQL, MinIO, MLflow) reste le profil de référence sur le PC ;
le choix se fait dans `.env` (`STORAGE_BACKEND`, `QDRANT_PATH`).

## 3. Ajouts non demandés par le sujet

| Ajout | Raison |
|---|---|
| `scripts/check_services.py` | Vérifier en une commande que les 4 services répondent avec les identifiants du `.env` |
| Vérification de cohérence du reranker au chargement | Un téléchargement incomplet (tokenizer absent) donnait le même score à tout, sans erreur. Le programme refuse maintenant un modèle incohérent |
| Désactivation de la conversion automatique des poids par transformers | transformers téléchargeait en arrière-plan une copie de 2 Go de BGE-M3 jamais utilisée |
| PostgreSQL sur le port 5433 | Le port 5432 est occupé par un PostgreSQL déjà installé sur la machine |
| Pièges dans le corpus (panneaux identiques, contradictions, document confidentiel, injection, scans) | Alimenter directement les catégories du jeu d'évaluation (section 10 du sujet) |

## 4. Points restant à traiter

- **MLflow** : le serveur Docker est en 2.19 et le client Python en 3.18. La compatibilité sera vérifiée, et les versions alignées si besoin, à l'étape de suivi des expériences.
- **Contrôle d'accès à l'ouverture des fichiers** : le sujet demande que l'accès soit aussi vérifié à l'ouverture des documents et des images. Ce sera fait dans l'API (étape suivante), qui servira les fichiers S3 après contrôle des groupes.
- **Licences** : à vérifier sur les pages officielles (`docs/licences.md`).
