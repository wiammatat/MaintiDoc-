"""Paquet de preuves : passages, tableaux, schémas originaux et pages transmis au VLM.

Chaque preuve reçoit une étiquette (S1, S2...). Le VLM ne cite que ces étiquettes ; elles sont
ensuite retraduites en citations exactes (document, page, élément).
"""

from dataclasses import dataclass, field

from common.config import get_settings
from retrieval.fusion import Hit
from retrieval.sanitize import neutralize
from retrieval.search import SearchResult


@dataclass
class EvidencePack:
    labels: dict[str, Hit]   # étiquette -> preuve (texte neutralisé si une consigne suspecte y a été détectée)
    lines: list[str]         # texte des preuves, pour le prompt
    images: list[Hit]        # preuves visuelles à joindre au VLM, dans l'ordre
    neutralized: dict[str, str] = field(default_factory=dict)   # étiquette -> titre, pour les sources nettoyées


def _sanitized(hit: Hit) -> tuple[Hit, bool]:
    """Copie de la preuve dont les consignes suspectes sont retirées (le modèle et le vérificateur voient
    la version nettoyée ; la citation, elle, reste celle de l'élément d'origine)."""
    text, changed = neutralize(hit.payload.get("text") or "")
    if not changed:
        return hit, False
    return Hit(uid=hit.uid, payload=hit.payload | {"text": text}, rrf_score=hit.rrf_score,
               rerank_score=hit.rerank_score, ranks=hit.ranks), True


def build_evidence(result: SearchResult, max_images: int, first_image_number: int = 1) -> EvidencePack:
    """max_images : images récupérées à joindre ; elles sont numérotées à partir de first_image_number
    (après la photo de l'utilisateur)."""
    labels, lines, images, neutralized = {}, [], [], {}
    for hit in result.passages[:get_settings().vlm_max_passages]:
        label = f"S{len(labels) + 1}"
        hit, changed = _sanitized(hit)
        if changed:
            neutralized[label] = hit.payload["titre"]
        labels[label] = hit
        p = hit.payload
        where = f"{p['titre']} (version {p['version']}), page {p['page']}" + (f", {p['section']}" if p.get("section") else "")
        lines.append(f"[{label}] {where}\n{p['text']}")
    # Les figures d'abord : une page entière apporte surtout du texte, déjà présent dans les preuves,
    # et coûte beaucoup plus de tokens visuels.
    for hit in sorted(result.images, key=lambda h: h.payload["type"] != "figure"):
        if len(images) >= max_images:
            break
        if hit.uid in {h.uid for h in result.passages}:
            continue
        label = f"S{len(labels) + 1}"
        labels[label] = hit
        images.append(hit)
        p = hit.payload
        lines.append(f"[{label}] IMAGE n°{first_image_number + len(images) - 1} : {p['titre']}, page {p['page']}"
                     + (f" - {p['text']}" if p.get("text") else ""))
    return EvidencePack(labels=labels, lines=lines, images=images, neutralized=neutralized)
