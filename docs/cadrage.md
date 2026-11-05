# Note de cadrage : MaintiDoc

## 1. Objectif
Aider les techniciens de maintenance à trouver rapidement une information fiable dans la documentation technique, à partir d'une question en français et éventuellement d'une photo. Chaque réponse cite ses sources (document, page, figure). L'assistant peut consulter l'historique d'un équipement et préparer un ticket, dont la création exige une validation humaine.

## 2. Périmètre

| Élément | Retenu |
|---|---|
| Utilisateurs | Techniciens (pompes, compresseurs) et responsables de maintenance |
| Corpus | 40 documents synthétiques, 2 familles : pompes HydroFlux (HF-200, HF-350, HF-500) et compresseurs AeroComp (AC-15, AC-30, AC-55) |
| Entrées | Questions en français, photos (panneaux, plaques), PDF |
| Sorties | Réponse sourcée, demande de clarification, abstention, brouillon de ticket |
| Outil externe | Serveur MCP de démonstration : historique, tickets ouverts, création de ticket |
| Action sensible | Création de ticket, seulement après approbation explicite |
| Hors périmètre | Pilotage de machines, diagnostic garanti, interventions autonomes |

## 3. Utilisateurs et droits

| Utilisateur | Groupe | Accès |
|---|---|---|
| tech_p01, tech_p02 | tech_pompes | Documents pompes + sécurité |
| tech_c01, tech_c02 | tech_compresseurs | Documents compresseurs + sécurité |
| resp_01 | responsable | Tout, y compris les documents confidentiels |

## 4. Flux de données

```
Utilisateur (question + photo)
  -> API : authentification, récupération des groupes
  -> Agent : analyse de la demande et de l'image
  -> Recherche filtrée par groupes (Qdrant) -> fusion RRF -> reranker
  -> VLM : réponse avec citations -> vérification des citations
  -> (optionnel) MCP : historique / tickets
  -> Réponse, clarification ou brouillon de ticket -> approbation -> création via MCP
```

Données stockées : fichiers dans SeaweedFS (stockage compatible S3, en remplacement de MinIO), métadonnées dans PostgreSQL, vecteurs dans Qdrant, expériences dans MLflow. Les journaux contiennent des métadonnées techniques, sans contenu brut.

## 5. Analyse des risques

| # | Risque | Probabilité | Impact | Mesure prévue |
|---|---|---|---|---|
| R1 | Réponse fausse ou inventée (hallucination) | Élevée | Élevé | Citations obligatoires et vérifiées, abstention si aucune source |
| R2 | Confusion entre équipements au panneau identique | Élevée | Élevé | Identifier le modèle avant de répondre, sinon demander une clarification |
| R3 | Sources contradictoires (note de service et manuel) | Moyenne | Moyen | Signaler la contradiction, citer les deux sources avec leur date |
| R4 | Fuite d'un document vers un groupe non autorisé | Moyenne | Élevé | Filtrage par groupe avant la recherche et à l'affichage, tests dédiés |
| R5 | Injection de prompt dans un document, une image ou une réponse MCP | Moyenne | Élevé | Contenus traités comme des données, outils restreints, 20 scénarios de test |
| R6 | Ticket créé sans accord ou en double | Faible | Moyen | Jeton d'approbation lié au contenu, contrôle serveur, clé d'idempotence |
| R7 | Conseil dangereux (intervention sans consignation) | Faible | Très élevé | Rappel systématique des procédures de sécurité, refus hors rôle documentaire |
| R8 | Fuite de secrets | Faible | Élevé | Fichier .env exclu de Git, SecretStr, aucun secret dans les prompts ni les journaux |
| R9 | Latence ou mémoire GPU insuffisantes | Moyenne | Moyen | Quantification, limites sur les images et le contexte, budgets d'appels |
| R10 | Fine-tuning sans gain mesurable | Moyenne | Faible | Comparaison honnête des 3 configurations, résultat rapporté tel quel |

## 6. Critères de réussite (objectifs, non acquis)
- Au moins 90 % des citations soutiennent l'affirmation associée.
- Aucun ticket créé sans approbation et aucune fuite entre groupes dans les tests.
- Latence p95 inférieure à 30 s pour une demande simple sur le matériel retenu.
