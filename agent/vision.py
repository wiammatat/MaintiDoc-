"""Analyse de la photo (prompt visual_analysis_v1) : le VLM décrit, le code vérifie et décide.

Motif : sur Kaggle, une photo de SN-200 a été lue comme une SN-400, et une photo au nom masqué a reçu une réponse
au lieu d'une demande de clarification. Les baies et les switchs ont la même face avant mais des voyants de sens
différent : identifier la FAMILLE de l'équipement est indispensable avant de répondre.

Décision (sujet, section 8 : clarification si la photo est illisible ou l'équipement non identifiable) :
- famille identifiée par la photo ou par la question -> recherche filtrée sur cette famille ;
- famille inconnue -> clarification, sans recherche ni réponse ;
- photo et question contradictoires -> clarification.
"""

import re
from dataclasses import asdict, dataclass, field

from PIL import Image
from pydantic import BaseModel, Field

from common.config import get_settings
from generation.guards import model_groups, models_in_question
from generation.prompting import parse, render
from generation.vlm import VLM, VLMUnavailable, fit_image

BRANDS = {"stornova": "baie", "fabrilink": "switch"}   # marque -> famille (corpus SAN fictif)
CODE_PREFIX = {"baie": "E", "switch": "F"}
CODE_RE = re.compile(r"^[EF]\d{2}$")

CLARIFY_UNREADABLE = ("Je ne parviens pas à identifier l'équipement sur la photo (marque et modèle illisibles). "
                      "Pouvez-vous indiquer le modèle (par exemple SN-400 ou FL-48) ou photographier la plaque signalétique ?")


class LitLed(BaseModel):
    voyant: str = Field(max_length=10)
    couleur: str = Field(max_length=20)


class VisualDescription(BaseModel):
    """Ce que le VLM doit produire."""
    marque: str | None = None
    modele: str | None = None
    voyants_allumes: list[LitLed] = Field(default_factory=list, max_length=4)
    code_affiche: str | None = None
    texte_lisible: bool = False
    observations: str = Field(default="", max_length=300)


@dataclass
class PhotoAnalysis:
    famille: str | None = None        # famille retenue (photo ou question)
    modele: str | None = None         # modèle retenu, seulement s'il est certain
    voyants: list[dict] = field(default_factory=list)
    code: str | None = None
    clarification: str | None = None  # si non vide : demander une précision, ne pas répondre
    notes: list[str] = field(default_factory=list)

    def context(self) -> str:
        """Résumé transmis au modèle de réponse (variable {photo} du prompt)."""
        parts = [f"équipement de la famille « {self.famille} »" if self.famille else "équipement non identifié"]
        parts.append(f"modèle {self.modele}" if self.modele else "modèle non lisible : ne donner aucune valeur propre à un modèle")
        if self.voyants:
            parts.append("voyant(s) allumé(s) : " + ", ".join(f"{v['voyant']} {v['couleur']}" for v in self.voyants))
        if self.code:
            parts.append(f"code affiché : {self.code}")
        return "analyse préalable de la photo : " + " ; ".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)


def _family_of(model: str) -> str | None:
    import json
    meta = json.loads((get_settings().corpus_dir / "metadata.json").read_text(encoding="utf-8"))
    return next((m["famille"] for m in meta if m.get("modele") == model), None)


def normalize_model(text: str | None) -> str | None:
    """« sn400 », « SN 400 » -> « SN-400 » s'il est au catalogue ; sinon None."""
    if not text:
        return None
    compact = re.sub(r"[\s-]", "", text).upper()
    return next((m for m in model_groups() if m.replace("-", "") == compact), None)


def interpret(desc: VisualDescription | None, question: str) -> PhotoAnalysis:
    """Vérification déterministe de la description et décision."""
    result = PhotoAnalysis()
    photo_family = photo_model = None
    if desc is not None:
        photo_model = normalize_model(desc.modele) if desc.texte_lisible else None
        brand_family = BRANDS.get((desc.marque or "").strip().lower())
        model_family = _family_of(photo_model) if photo_model else None
        conflict = bool(photo_model and brand_family and model_family != brand_family)
        if conflict:   # lecture contradictoire : ni la marque ni le modèle ne sont retenus
            result.notes.append(f"Marque et modèle incohérents sur la photo ({desc.marque} / {desc.modele}) : ignorés.")
            photo_model = brand_family = model_family = None
        photo_family = model_family or brand_family
        if desc.modele and not photo_model and not conflict:
            result.notes.append(f"Modèle lu sur la photo non reconnu ou incertain : « {desc.modele} ».")
        result.voyants = [v.model_dump() for v in desc.voyants_allumes if re.fullmatch(r"L[1-4]", v.voyant.upper())]
        code = (desc.code_affiche or "").strip().upper()
        if CODE_RE.match(code) and (photo_family is None or code[0] == CODE_PREFIX[photo_family]):
            result.code = code
        elif code:
            result.notes.append(f"Code affiché « {desc.code_affiche} » incohérent avec l'équipement : ignoré.")
    else:
        result.notes.append("Analyse de la photo impossible (sortie du modèle invalide).")

    question_models = models_in_question(question)
    question_families = {_family_of(m) for m in question_models}
    if photo_family and question_families and photo_family not in question_families:
        result.clarification = (f"La photo semble montrer un équipement de la famille « {photo_family} », alors que la "
                                f"question cite {', '.join(question_models)}. De quel équipement s'agit-il ?")
        return result
    result.famille = photo_family or (next(iter(question_families)) if len(question_families) == 1 else None)
    result.modele = photo_model or (question_models[0] if len(question_models) == 1 else None)
    if result.famille is None:
        result.clarification = CLARIFY_UNREADABLE
    return result


def analyze_photo(vlm: VLM, question: str, photo: Image.Image) -> tuple[PhotoAnalysis, int]:
    """Renvoie (analyse vérifiée, nombre d'appels au modèle)."""
    system, user = render("visual_analysis_v1", question=question)
    try:
        desc = parse(VisualDescription, vlm.generate(system, user, [fit_image(photo, get_settings().vlm_photo_max_pixels)]))
    except (ValueError, VLMUnavailable):
        desc = None
    return interpret(desc, question), 1
