"""Profil léger, sans Docker (Lightning AI Studio) : fichiers servis depuis le disque, métadonnées lues dans
le corpus (metadata.json) et dans les éléments préparés (elements.jsonl), au lieu de MinIO et PostgreSQL.

Les informations renvoyées (groupes autorisés compris) sont les mêmes que dans le profil Docker :
le contrôle d'accès de l'API est identique.
"""

import json
from functools import lru_cache
from pathlib import Path

from common.config import get_settings


@lru_cache
def _documents() -> dict[str, dict]:
    meta = json.loads((get_settings().corpus_dir / "metadata.json").read_text(encoding="utf-8"))
    return {m["document_id"]: m for m in meta}


@lru_cache
def _elements() -> dict[str, dict]:
    path = get_settings().data_dir / "processed" / "elements.jsonl"
    if not path.exists():
        return {}
    elements = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        elements[f"{e['document_id']}:{e['version']}:{e['element_id']}"] = e
    return elements


def get_document(document_id: str) -> dict | None:
    """Chemin du PDF source et groupes autorisés, ou None s'il n'existe pas."""
    m = _documents().get(document_id)
    if m is None:
        return None
    return {"path": get_settings().data_dir / m["fichier"], "groupes_autorises": m["groupes_autorises"],
            "titre": m["titre"]}


def get_element(uid: str) -> dict | None:
    """Chemin de l'image d'un élément et groupes autorisés, ou None s'il n'existe pas."""
    e = _elements().get(uid)
    if e is None:
        return None
    path: Path | None = get_settings().data_dir / e["image_path"] if e.get("image_path") else None
    return {"path": path, "groupes_autorises": e["groupes_autorises"], "type": e["type"]}
