"""Formats des réponses de l'API."""

from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class Me(BaseModel):
    user_id: str
    nom: str
    groupes: list[str]


class Evidence(BaseModel):
    """Une preuve citable renvoyée par la recherche."""
    uid: str
    document_id: str
    titre: str
    page: int
    element_id: str
    type: str
    section: str | None = None
    text: str | None = None
    rerank_score: float | None = None
    image_url: str | None = None       # pour les figures et les pages, servie après contrôle d'accès


class SearchResponse(BaseModel):
    question: str
    passages: list[Evidence]
    images: list[Evidence]
    index_version: str


class AgentResponse(BaseModel):
    """Résultat de l'agent : réponse sourcée, historique MCP et, si demandé, brouillon de ticket à approuver."""
    thread_id: str
    status: str                        # termine | en_attente_approbation | ticket_cree | ticket_refuse | echec_creation
    answer: dict | None = None
    clarification_question: str | None = None
    historique: dict | None = None
    tickets_ouverts: dict | None = None
    ticket_draft: dict | None = None
    ticket: dict | None = None
    errors: list[str] = []
    mcp_calls: int = 0
    model_calls: int = 0


class ApprovalRequest(BaseModel):
    """Décision humaine. Le contenu envoyé est celui qui sera créé : s'il diffère du brouillon, c'est lui qui est approuvé."""
    approved: bool
    title: str | None = None
    description: str | None = None
