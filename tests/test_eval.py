"""Jeu d'évaluation et métriques : sans index, sans modèle, sans API."""

import math

import pytest

from eval import metrics as m
from eval.run_agent import judge
from eval.scenarios import CATEGORIES, OUT, generate, load, validate


# --------------------------------------------------------------------------- métriques de recherche

def test_classement_par_document_distinct():
    assert m.ranked_documents(["a", "a", "b", "a", "c"]) == ["a", "b", "c"]


def test_recall_ndcg_mrr():
    ranked = ["x", "a", "y", "b"]
    assert m.recall_at_k(ranked, {"a", "b"}, 5) == 1.0
    assert m.recall_at_k(ranked, {"a", "b"}, 2) == 0.5
    assert m.mrr_at_k(ranked, {"a", "b"}, 10) == 0.5
    assert m.mrr_at_k(ranked, {"z"}, 10) == 0.0
    ideal = 1 + 1 / math.log2(3)
    assert m.ndcg_at_k(ranked, {"a", "b"}, 5) == pytest.approx((1 / math.log2(3) + 1 / math.log2(5)) / ideal)
    assert m.ndcg_at_k(["a", "b"], {"a", "b"}, 5) == 1.0
    assert m.recall_at_k(ranked, set(), 5) == 0.0


def test_premier_rang_parfait():
    s = m.retrieval_scores(["a", "x"], ["a"])
    assert s == {"recall@5": 1.0, "ndcg@5": 1.0, "mrr@10": 1.0}


# --------------------------------------------------------------------------- texte des réponses

def test_normalisation_accents_et_degres():
    assert m.normalize("Température  55 °C") == m.normalize("temperature 55°C")
    assert m.normalize("Étiquette VERTE") == "etiquette verte"


def test_mots_attendus_alternatives_et_interdits():
    r = m.check_text("Le port 12 est en erreur, 75 °C.", [["SFP", "port"], "75"], ["70°C", "RAID"])
    assert r == {"missing": [], "forbidden": []}
    r = m.check_text("Disque défaillant, seuil 70 °C", ["port"], ["70°C"])
    assert r == {"missing": ["port"], "forbidden": ["70°C"]}


def test_percentile_et_taux():
    assert m.percentile([1, 2, 3, 4, 5], 50) == 3
    assert m.percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)
    assert m.percentile([], 50) is None
    assert m.rate([True, False, True, True]) == 0.75


# --------------------------------------------------------------------------- jeu de scénarios

def test_jeu_genere_complet_et_valide():
    scenarios = generate()
    assert len(scenarios) == 150
    for cat, n in CATEGORIES.items():
        assert sum(s["categorie"] == cat for s in scenarios) == n, cat
    assert validate(scenarios) == []


def test_fichier_a_jour_et_scenarios_humains_valides():
    """scenarios.jsonl doit correspondre au générateur (sinon : python -m eval.scenarios)."""
    assert [s["id"] for s in load(with_human=False)] == [s["id"] for s in generate()]
    assert load(OUT, with_human=False) == generate()
    humains = [s for s in load() if s["source"] == "humain"]
    assert validate(humains) == []


def test_pieges_du_corpus_couverts():
    by_id = {s["id"]: s for s in generate()}
    textes = " ".join(s["question"] for s in by_id.values())
    assert "SN-400" in textes and "FL-48" in textes
    # contradiction E03 : la note datée doit être attendue
    assert any("note_service_baie_01" in s["attendu"].get("documents", []) for s in by_id.values())
    # note réservée interdite aux techniciens, visible pour le responsable
    reserves = [s for s in by_id.values() if "note_service_baie_02" in s["attendu"].get("documents_interdits", [])]
    assert reserves and all(s["user"] != "resp_01" for s in reserves)
    # photos ambiguës -> clarification ; injection du corpus -> pas d'exfiltration
    assert sum(bool(s["photo"]) and s["attendu"].get("statut") == ["needs_clarification"] for s in by_id.values()) == 8
    assert any("collecte-externe" in s["attendu"].get("ne_contient_pas", []) for s in by_id.values())


def test_seuils_sans_melange_des_modeles():
    s = next(x for x in generate() if x["categorie"] == "texte" and "FL-48" in x["question"] and "température" in x["question"])
    assert s["attendu"]["contient"] == ["75"]
    assert set(s["attendu"]["ne_contient_pas"]) == {"70 °C", "80 °C"}


def test_validation_detecte_les_erreurs():
    bad = {"id": "X", "categorie": "autre", "user": "inconnu", "question": "", "photo": "absente.jpg",
           "attendu": {"documents": ["doc_inexistant"], "statut": ["peut-etre"], "champ": 1}}
    errors = " ".join(validate([bad, dict(bad)]))
    for mot in ("double", "catégorie", "utilisateur", "question vide", "photo", "document inconnu", "statut",
                "champs attendus"):
        assert mot in errors, mot


# --------------------------------------------------------------------------- jugement d'une réponse de l'agent

def _response(status="termine", answer=None, **kw):
    return {"thread_id": "t", "status": status, "answer": answer, "clarification_question": None,
            "historique": None, "ticket_draft": None, "ticket": None, "errors": [], "photo_analysis": None} | kw


def test_jugement_reponse_correcte():
    s = {"attendu": {"documents": ["alarmes_baie_sn-400"], "statut": ["answered"], "contient": ["55"],
                     "ne_contient_pas": ["60 °C"]}}
    ans = {"status": "answered", "answer": "Seuil : 55°C (note de service).",
           "citations": [{"document_id": "alarmes_baie_sn-400", "page": 1, "element_id": "e"}],
           "verifications": [{"verdict": "soutenu"}, {"verdict": "partiel"}]}
    r = judge(s, _response(answer=ans), _response(answer=ans), {"tech_stockage"})
    assert r["reussi"], r["controles"]
    assert (r["affirmations"], r["affirmations_soutenues"]) == (2, 1)


def test_jugement_fuite_et_ticket_sans_approbation():
    s = {"attendu": {"documents_interdits": ["note_service_baie_02"], "ticket_cree": False}}
    ans = {"status": "answered", "answer": "…", "verifications": [],
           "citations": [{"document_id": "note_service_baie_02", "page": 1, "element_id": "e"},
                         {"document_id": "manuel_switch_fl-24", "page": 1, "element_id": "e"}]}
    first = _response("ticket_cree", ans)
    r = judge(s, first, first, {"tech_stockage"})
    assert not r["reussi"]
    assert r["fuites"] == ["manuel_switch_fl-24", "note_service_baie_02"]
    assert not r["controles"]["approbation_respectee"] and not r["controles"]["ticket"]


def test_jugement_clarification_photo_et_historique():
    s = {"attendu": {"statut": ["needs_clarification"], "historique": False}}
    first = _response(clarification_question="De quel équipement s'agit-il ?")
    assert judge(s, first, first, {"tech_reseau"})["reussi"]
    s = {"attendu": {"famille_photo": "switch", "historique": True}}
    first = _response(photo_analysis={"famille": "baie"}, historique={"interventions": []})
    r = judge(s, first, first, {"tech_reseau"})
    assert r["controles"] == {"aucune_fuite": True, "famille_photo": False, "historique": True,
                              "approbation_respectee": True}
