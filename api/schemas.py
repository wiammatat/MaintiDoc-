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
