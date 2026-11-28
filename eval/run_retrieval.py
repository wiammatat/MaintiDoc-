"""Évaluation de la recherche : Recall@5, nDCG@5, MRR@10 et fuites entre groupes (CPU suffit).

Appelle directement retrieval.search (sans VLM) pour chaque scénario qui a des documents attendus ou interdits,
avec plusieurs configurations de reranking, pour remplir le tableau de l'étape 2.4 :

    python -m eval.run_retrieval                                  # sans reranker + reranker d'origine
    python -m eval.run_retrieval --reranker-adapte models/reranker_v1   # + reranker fine-tuné
    python -m eval.run_retrieval --sans-photo                     # multimodalité : mêmes scénarios, photo retirée
    python -m eval.run_retrieval --candidats 10 20 0              # reranker sur 10, 20 ou tous les candidats
    python -m eval.run_retrieval --mlflow                         # enregistre aussi les résultats dans MLflow

Prérequis : index construit (python -m pipeline.index ou scripts du README), Qdrant accessible.
Résultats : eval/results/recherche_<date>.json (détail par scénario) et tableau affiché.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

from common.access import get_user
from common.config import get_settings
from eval.metrics import mean, percentile, ranked_documents, retrieval_scores
from eval.scenarios import CATEGORIES, DATA, JEU_VERSION, load
from retrieval.search import search

RESULTS = Path(__file__).with_name("results")
METRICS = ("recall@5", "ndcg@5", "mrr@10")


def evaluate(scenarios: list[dict], reranker: str | None, with_photo: bool = True,
             candidats: int | None = None) -> list[dict]:
    rows = []
    for s in scenarios:
        a = s["attendu"]
        user = get_user(s["user"])
        photo = Image.open(DATA / "photos" / s["photo"]).convert("RGB") if with_photo and s.get("photo") else None
        t0 = time.perf_counter()
        res = search(user, s["question"], photo, top_passages=10, reranker=reranker, rerank_candidates=candidats)
        latency = time.perf_counter() - t0
        hits = res.passages + res.images
        ranked = ranked_documents([h.payload["document_id"] for h in res.passages])
        interdits = set(a.get("documents_interdits", []))
        # Fuite : document explicitement interdit, ou document dont les groupes n'incluent pas l'utilisateur.
        leaks = sorted({h.payload["document_id"] for h in hits
                        if h.payload["document_id"] in interdits
                        or not set(h.payload["groupes_autorises"]) & set(user.groupes)})
        row = {"id": s["id"], "categorie": s["categorie"], "source": s["source"], "user": s["user"],
               "photo": bool(photo), "classement": ranked[:10], "attendus": a.get("documents", []),
               "fuites": leaks, "latence_s": round(latency, 3)}
        if a.get("documents"):
            row |= {k: round(v, 4) for k, v in retrieval_scores(ranked, a["documents"]).items()}
        rows.append(row)
        print(f"  {s['id']:<8} {' '.join(f'{m}={row[m]:.2f}' for m in METRICS if m in row):<42}"
              f"{'FUITE ' + ','.join(leaks) if leaks else ''}")
    return rows


def aggregate(rows: list[dict]) -> dict:
    scored = [r for r in rows if "recall@5" in r]
    out = {"scenarios": len(rows), "notes_en_recherche": len(scored),
           "fuites": sum(1 for r in rows if r["fuites"]),
           "latence_mediane_s": percentile([r["latence_s"] for r in rows], 50),
           "latence_p95_s": percentile([r["latence_s"] for r in rows], 95)}
    out |= {m: mean([r[m] for r in scored]) for m in METRICS}
    out["par_categorie"] = {c: {m: mean([r[m] for r in scored if r["categorie"] == c]) for m in METRICS}
                            | {"n": sum(1 for r in scored if r["categorie"] == c)}
                            for c in CATEGORIES if any(r["categorie"] == c for r in scored)}
    if any(r["source"] == "humain" for r in scored):
        out["scenarios_humains"] = {m: mean([r[m] for r in scored if r["source"] == "humain"]) for m in METRICS}
    return out


def print_table(results: dict) -> None:
    print("\n| Configuration | Recall@5 | nDCG@5 | MRR@10 | Fuites | Latence médiane (s) |")
    print("|---|---|---|---|---|---|")
    for name, r in results.items():
        a = r["agregats"]
        print(f"| {name} | {a['recall@5']} | {a['ndcg@5']} | {a['mrr@10']} | {a['fuites']} | {a['latence_mediane_s']} |")
    for name, r in results.items():
        print(f"\n{name} par catégorie :")
        for c, v in r["agregats"]["par_categorie"].items():
            print(f"  {c:<10} n={v['n']:<3} Recall@5={v['recall@5']}  nDCG@5={v['ndcg@5']}  MRR@10={v['mrr@10']}")


def log_mlflow(results: dict, path: Path, label: str) -> None:
    import mlflow

    s = get_settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    mlflow.set_experiment("maintidoc-eval-recherche")
    for name, r in results.items():
        with mlflow.start_run(run_name=f"{name}{label}"):
            mlflow.log_params({"configuration": name, "reranker": r["reranker"] or "aucun", "photo": not label,
                               "candidats_reranker": r["candidats"] if r["reranker"] else "-",
                               "index_version": s.index_version, "scenarios": r["agregats"]["scenarios"]})
            a = r["agregats"]
            mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in a.items() if isinstance(v, (int, float))})
            for c, v in a["par_categorie"].items():
                mlflow.log_metrics({f"{c}_{m.replace('@', '_at_')}": v[m] for m in METRICS if v[m] is not None})
            mlflow.log_artifact(str(path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Évaluation de la recherche MaintiDoc")
    parser.add_argument("--reranker-adapte", help="chemin du reranker fine-tuné à comparer")
    parser.add_argument("--sans-photo", action="store_true", help="retirer les photos (comparaison texte seul)")
    parser.add_argument("--categorie", choices=list(CATEGORIES), help="n'évaluer qu'une catégorie")
    parser.add_argument("--candidats", nargs="+", type=int,
                        help="nombres de candidats relus par le reranker à comparer (0 = tous) ; "
                             "défaut : RERANK_CANDIDATES")
    parser.add_argument("--mlflow", action="store_true", help="enregistrer les résultats dans MLflow")
    args = parser.parse_args()

    scenarios = [s for s in load() if s["attendu"].get("documents") or s["attendu"].get("documents_interdits")]
    if args.categorie:
        scenarios = [s for s in scenarios if s["categorie"] == args.categorie]
    s = get_settings()
    rerankers = {"reranker_origine": "default"}
    if args.reranker_adapte:
        rerankers["reranker_adapte"] = args.reranker_adapte
    candidats = args.candidats or [s.rerank_candidates]
    configs = {"sans_reranker": (None, None)}
    for name, reranker in rerankers.items():
        for n in candidats:
            configs[f"{name}_{n or 'tous'}" if len(candidats) > 1 else name] = (reranker, n)

    print(f"Reranker sur {s.reranker_device or s.device} ({s.reranker_dtype or s.model_dtype}), "
          f"candidats relus : {', '.join(str(n or 'tous') for n in candidats)}")
    results = {}
    for name, (reranker, n) in configs.items():
        print(f"\n== {name} ({len(scenarios)} scénarios{', sans photo' if args.sans_photo else ''})")
        rows = evaluate(scenarios, reranker, with_photo=not args.sans_photo, candidats=n)
        results[name] = {"reranker": reranker, "candidats": n, "agregats": aggregate(rows), "details": rows}

    RESULTS.mkdir(exist_ok=True)
    label = "_sans_photo" if args.sans_photo else ""
    path = RESULTS / f"recherche{label}_{datetime.now():%Y%m%d_%H%M}.json"
    meta = {"date": datetime.now().isoformat(timespec="seconds"), "index_version": s.index_version,
            "sans_photo": args.sans_photo, "jeu": JEU_VERSION,
            "reranker_device": s.reranker_device or s.device, "reranker_dtype": s.reranker_dtype or s.model_dtype}
    path.write_text(json.dumps({"meta": meta, "resultats": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print_table(results)
    print(f"\nDétail : {path}")
    if args.mlflow:
        log_mlflow(results, path, label)
    return 1 if any(r["agregats"]["fuites"] for r in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
