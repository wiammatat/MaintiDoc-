"""Format d'un élément extrait d'un document : passage, tableau, figure ou page."""

from typing import Literal

from pydantic import BaseModel

ElementType = Literal["passage", "table", "figure", "page"]


class Element(BaseModel):
    # Identification (sert aux citations : document_id + page + element_id)
    document_id: str
    version: str
    page: int
    element_id: str
    type: ElementType
    source: Literal["pdf", "ocr"] = "pdf"

    # Contenu
    section: str | None = None
    text: str | None = None
    image_path: str | None = None   # chemin local relatif à data/
    s3_key: str | None = None       # clé dans le stockage S3
    bbox: list[float] | None = None

    # Métadonnées héritées du document (filtrage et contrôle d'accès)
    titre: str
    famille: str
    modele: str | None = None
    doc_type: str
    groupes_autorises: list[str]

    @property
    def uid(self) -> str:
        return f"{self.document_id}:{self.version}:{self.element_id}"
