"""Sorties structurées (section 7 du sujet), validées par Pydantic."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Status = Literal["answered", "needs_clarification", "abstained"]


class ModelOutput(BaseModel):
    """Ce que le VLM doit produire : il cite des étiquettes de sources (S1, S2...)."""
    status: Status
    answer: str
    citations: list[str] = Field(default_factory=list, description="étiquettes des sources utilisées, ex. ['S1']")
    uncertainties: list[str] = Field(default_factory=list)
    clarification_question: str | None = None
    proposed_action: dict | None = None

    @field_validator("citations", "uncertainties", mode="before")
    @classmethod
    def _texte_seul_en_liste(cls, value):
        # Observé sur Kaggle : Qwen écrit parfois une chaîne au lieu d'une liste. On la garde plutôt
        # que de rejeter toute la réponse.
        if value is None:
            return []
        return [value] if isinstance(value, str) else value


class Citation(BaseModel):
    document_id: str
    page: int
    element_id: str


Verdict = Literal["soutenu", "partiel", "non_soutenu"]


class EvidenceVerdict(BaseModel):
    """Ce que le VLM produit lors de la vérification, pour chaque affirmation numérotée."""
    id: int
    verdict: Verdict
    raison: str = ""


class EvidenceOutput(BaseModel):
    checks: list[EvidenceVerdict]


class ClaimCheck(BaseModel):
    """Résultat de la vérification d'une affirmation de la réponse."""
    claim: str
    sources: list[str]            # étiquettes citées par l'affirmation
    verdict: Verdict
    raison: str = ""


class Answer(BaseModel):
    """Réponse finale renvoyée à l'utilisateur, avec citations résolues et traçabilité."""
    status: Status
    answer: str
    citations: list[Citation]
    uncertainties: list[str] = Field(default_factory=list)
    clarification_question: str | None = None
    proposed_action: dict | None = None
    invalid_citations: list[str] = Field(default_factory=list, description="étiquettes citées mais inconnues")
    unsupported_citations: list[str] = Field(default_factory=list,
                                             description="étiquettes retirées : la source ne soutient pas l'affirmation")
    verifications: list[ClaimCheck] = Field(default_factory=list)
    model_calls: int = 0
    versions: dict[str, str] = Field(default_factory=dict, description="modèle, prompts, reranker, index")


def output_schema_hint() -> str:
    """Description compacte du JSON attendu, insérée dans le prompt."""
    return ('{"status": "answered" | "needs_clarification" | "abstained", '
            '"answer": "texte avec citations [S1]", "citations": ["S1"], "uncertainties": ["..."], '
            '"clarification_question": "..." ou null, "proposed_action": null}')
