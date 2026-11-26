"""Étape 5 : génération de la réponse sourcée.

recherche -> paquet de preuves -> prompt -> VLM -> JSON validé -> citations vérifiées.
"""

import json
import re
from collections.abc import Callable
from functools import lru_cache

import yaml
from PIL import Image
from pydantic import ValidationError

from common.access import User
from common.config import get_settings
from generation.guards import models_without_access
from generation.schema import Answer, Citation, ModelOutput, output_schema_hint
from generation.verify import check_evidence
from generation.vlm import VLM, default_vlm, fit_image
from retrieval.evidence import build_evidence
from retrieval.fusion import Hit
from retrieval.search import SearchResult, search

MAX_ATTEMPTS = 2   # une nouvelle tentative si la sortie n'est pas un JSON valide
LABEL_RE = re.compile(r"\[(S\d+)\]")
INJECTION_CLAIM = re.compile(r"consigne.{0,60}(ignor|suspect|retir)|injection", re.IGNORECASE)


@lru_cache
def load_prompt(name: str) -> dict:
    return yaml.safe_load((get_settings().prompts_dir / f"{name}.yaml").read_text(encoding="utf-8"))


def parse_output(raw: str) -> ModelOutput:
    """Extrait le premier objet JSON de la sortie du modèle et le valide."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("aucun objet JSON dans la sortie du modèle")
    return ModelOutput.model_validate(json.loads(match.group(0)))


def resolve_citations(output: ModelOutput, labels: dict[str, Hit]) -> tuple[list[Citation], list[str]]:
    """Ne garde que les citations vers des sources réellement récupérées."""
    cited = list(dict.fromkeys(output.citations + LABEL_RE.findall(output.answer)))
    valid = [Citation(**labels[c].citation) for c in cited if c in labels]
    invalid = [c for c in cited if c not in labels]
    return valid, invalid


def answer(user: User, question: str, photo: Image.Image | None = None, *, vlm: VLM | None = None,
           search_fn: Callable[..., SearchResult] = search, verify: bool = True) -> Answer:
    """verify : second appel au modèle pour vérifier que les sources soutiennent les affirmations (étape 1.6)."""
    s = get_settings()
    vlm = vlm or default_vlm()
    prompt = load_prompt(s.answer_prompt)
    evidence_prompt = load_prompt(s.evidence_prompt)
    versions = {"modele": vlm.name, "prompt": prompt["version"], "prompt_verification": evidence_prompt["version"],
                "reranker": s.reranker_model, "index": s.index_version}

    result = search_fn(user, question, photo)
    if not result.passages and not result.images:
        return Answer(status="abstained", answer="Aucune source accessible ne traite cette question.",
                      citations=[], versions=versions)
    missing = models_without_access(question, user)
    if missing:
        # Garde-fou : jamais d'information transposée depuis un autre équipement (voyants de sens différent).
        return Answer(status="abstained", citations=[], versions=versions,
                      answer=f"Aucune source accessible ne concerne {', '.join(missing)}. Les informations d'un autre "
                             "équipement ne s'appliquent pas : les voyants n'ont pas le même sens selon l'équipement.",
                      uncertainties=[f"Équipement cité sans source accessible : {', '.join(missing)}."])

    n_photo = 1 if photo is not None else 0
    evidence = build_evidence(result, max(s.vlm_max_images - n_photo, 0), first_image_number=n_photo + 1)
    labels, lines, image_hits = evidence.labels, evidence.lines, evidence.images
    # Réduction des images ici : la photo de l'utilisateur garde une meilleure résolution que les images récupérées.
    images = ([fit_image(photo, s.vlm_photo_max_pixels)] if photo is not None else []) + [
        fit_image(Image.open(s.data_dir / h.payload["image_path"]), s.vlm_max_pixels) for h in image_hits]
    system = prompt["system"].replace("{schema}", output_schema_hint())
    user_msg = prompt["user"].format(
        question=question, sources="\n\n".join(lines),
        photo="jointe (IMAGE n°1)" if photo is not None else "aucune")

    errors, calls = [], 0
    for _ in range(min(MAX_ATTEMPTS, s.max_model_calls)):
        calls += 1
        raw = vlm.generate(system, user_msg, images)
        try:
            output = parse_output(raw)
            break
        except (ValueError, ValidationError) as e:
            errors.append(str(e)[:200])
    else:
        return Answer(status="abstained", answer="La réponse du modèle n'a pas pu être validée.",
                      citations=[], uncertainties=errors, model_calls=calls, versions=versions)

    # Les signalements d'injection viennent du code (retrieval/sanitize.py), pas du modèle : ceux du modèle sont
    # écartés, car il en inventait sur des sources saines et ratait la vraie (essai Kaggle, prompt v5).
    uncertainties = [f"La source [{label}] ({titre}) contenait une consigne suspecte : elle a été retirée avant "
                     "l'envoi au modèle." for label, titre in evidence.neutralized.items()]
    uncertainties += [u for u in output.uncertainties if not INJECTION_CLAIM.search(u)]
    if output.clarification_question and output.status != "needs_clarification":
        # Statut et question incohérents : la question est conservée comme point ouvert.
        uncertainties.append(f"Question ouverte : {output.clarification_question}")
        output.clarification_question = None
    checks, unsupported = [], set()
    if verify:
        checks, unsupported, notes, n = check_evidence(vlm, evidence_prompt, output.answer, labels,
                                                       max_calls=s.max_model_calls - calls, cited=output.citations)
        calls += n
        uncertainties += notes
        output.citations = [c for c in output.citations if c not in unsupported]
        for label in unsupported:   # la citation reste visible dans le texte, mais n'est plus présentée comme preuve
            output.answer = output.answer.replace(f"[{label}]", f"[{label} : non vérifié]")

    citations, invalid = resolve_citations(output, labels)
    if output.status == "answered" and not citations:
        # Une réponse sans aucune source vérifiable n'est pas présentée comme établie.
        uncertainties.append("Aucune citation valide : la réponse n'est pas étayée par les sources.")
        output.status = "abstained"
    return Answer(status=output.status, answer=output.answer, citations=citations, uncertainties=uncertainties,
                  clarification_question=output.clarification_question,
                  proposed_action=output.proposed_action, invalid_citations=invalid,
                  unsupported_citations=sorted(unsupported), verifications=checks, model_calls=calls,
                  versions=versions)
