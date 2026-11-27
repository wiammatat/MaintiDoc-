"""Vérification des citations (étape 1.6) : chaque affirmation est-elle soutenue par les sources qu'elle cite ?

Le contrôle d'existence (la citation pointe vers une source récupérée) est fait dans answer.py.
Ici, un second appel au modèle juge le soutien réel, affirmation par affirmation.
"""

import re

from pydantic import ValidationError

from generation.guards import models_in_question
from generation.prompting import parse, render
from generation.schema import ClaimCheck, EvidenceOutput
from generation.vlm import VLM
from retrieval.fusion import Hit

LABEL_RE = re.compile(r"\[(S\d+)\]")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_claims(text: str, known: set[str], cited: list[str] | None = None) -> list[tuple[str, list[str]]]:
    """Découpe la réponse en phrases et garde celles qui citent au moins une source connue.

    Si le texte ne contient aucune étiquette mais que le modèle a rempli la liste "citations"
    (cas observé avec Qwen2.5-VL), la réponse entière est vérifiée face à ces sources.
    """
    claims = []
    for sentence in SENTENCE_RE.split(text.strip()):
        labels = [l for l in dict.fromkeys(LABEL_RE.findall(sentence)) if l in known]
        if labels:
            claims.append((sentence.strip(), labels))
    if not claims:
        fallback = [l for l in dict.fromkeys(cited or []) if l in known]
        if fallback and text.strip():
            claims.append((text.strip(), fallback))
    return claims


def claims_block(claims: list[tuple[str, list[str]]], labels: dict[str, Hit]) -> str:
    """Affirmations numérotées et texte des sources qu'elles citent (variable {claims} du prompt)."""
    blocks = []
    for i, (claim, cited) in enumerate(claims, 1):
        # Le titre de la source permet au vérificateur de repérer une information attribuée au mauvais équipement.
        sources = "\n".join(f"  [{l}] ({labels[l].payload['titre']}, page {labels[l].payload['page']}) "
                            f"{labels[l].payload.get('text') or '(image sans texte)'}" for l in cited)
        blocks.append(f"Affirmation {i} : {claim}\nSources citées :\n{sources}")
    return "\n\n".join(blocks)


def parse_checks(raw: str, n_claims: int) -> EvidenceOutput:
    out = parse(EvidenceOutput, raw)   # validation par PydanticOutputParser (LangChain)
    if sorted(c.id for c in out.checks) != list(range(1, n_claims + 1)):
        raise ValueError("la vérification ne couvre pas exactement toutes les affirmations")
    return out


NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
CITATION_RE = re.compile(r"\[[^\]]*\bS\d+\b[^\]]*\]")   # [S1], [S1, S2], [S1 : non vérifié]


def values_supported(claim: str, sources: list[Hit]) -> bool:
    """Contrôle déterministe : toutes les valeurs chiffrées de l'affirmation figurent dans les sources citées,
    et chaque modèle d'équipement cité (SN-400...) apparaît dans ces sources (titre ou texte).

    Sert à corriger le vérificateur (même modèle 7B que le générateur) quand il rejette à tort une affirmation
    exacte. Il ne valide jamais une valeur attribuée à un autre modèle que celui de la source.
    """
    claim = CITATION_RE.sub(" ", claim)
    values = {v.replace(",", ".") for v in NUMBER_RE.findall(claim)}
    if not values:
        return False
    corpus = " ".join(f"{h.payload.get('titre', '')} {h.payload.get('text') or ''}" for h in sources)
    if any(model.lower() not in corpus.lower() for model in models_in_question(claim)):
        return False
    return values <= {v.replace(",", ".") for v in NUMBER_RE.findall(corpus)}


def check_evidence(vlm: VLM, prompt_name: str, answer_text: str, labels: dict[str, Hit],
                   max_calls: int, cited: list[str] | None = None) -> tuple[list[ClaimCheck], set[str], list[str], int]:
    """Renvoie (vérifications, étiquettes à retirer, incertitudes, nombre d'appels au modèle).

    Une étiquette est retirée si toutes les affirmations qui la citent sont jugées non soutenues.
    Si la vérification échoue, rien n'est retiré mais une incertitude est signalée.
    """
    claims = split_claims(answer_text, set(labels), cited)
    if not claims or max_calls <= 0:
        return [], set(), [], 0

    system, user_msg = render(prompt_name, claims=claims_block(claims, labels))
    calls = 0
    for _ in range(min(2, max_calls)):
        calls += 1
        try:
            out = parse_checks(vlm.generate(system, user_msg, []), len(claims))
            break
        except (ValueError, ValidationError):
            continue
    else:
        return [], set(), ["La vérification des citations n'a pas pu être effectuée."], calls

    by_id = {c.id: c for c in out.checks}
    checks = [ClaimCheck(claim=claim, sources=cited, verdict=by_id[i].verdict, raison=by_id[i].raison)
              for i, (claim, cited) in enumerate(claims, 1)]
    for c in checks:
        if c.verdict == "non_soutenu" and values_supported(c.claim, [labels[l] for l in c.sources]):
            c.verdict = "partiel"
            c.raison = (c.raison + " | " if c.raison else "") + ("verdict corrigé : toutes les valeurs chiffrées "
                                                                 "et les modèles cités figurent dans les sources")

    supported = {l for c in checks if c.verdict != "non_soutenu" for l in c.sources}
    unsupported = {l for c in checks if c.verdict == "non_soutenu" for l in c.sources} - supported
    uncertainties = [f"Affirmation non soutenue par les sources citées : « {c.claim} »" + (f" ({c.raison})" if c.raison else "")
                     for c in checks if c.verdict == "non_soutenu"]
    uncertainties += [f"Affirmation partiellement soutenue : « {c.claim} »" + (f" ({c.raison})" if c.raison else "")
                      for c in checks if c.verdict == "partiel"]
    return checks, unsupported, uncertainties, calls
