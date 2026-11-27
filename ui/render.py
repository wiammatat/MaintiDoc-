"""Mise en forme des réponses de l'API en Markdown, sans dépendance à l'interface (testable seule).

Tout le texte venant des documents, du modèle ou des outils est échappé : il ne peut ni ajouter de lien ou
d'image, ni modifier la mise en page (injection indirecte).
"""

import re

STATUS = {
    "answered": "✅ Réponse sourcée",
    "needs_clarification": "❓ Précision nécessaire",
    "abstained": "⛔ Pas de réponse (sources insuffisantes)",
}
AGENT_STATUS = {
    "termine": "Terminé", "en_attente_approbation": "Ticket en attente de votre approbation",
    "ticket_cree": "Ticket créé", "ticket_refuse": "Ticket refusé : rien n'a été créé",
    "echec_creation": "Échec de la création du ticket",
}
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$])")


def esc(text) -> str:
    """Échappe le Markdown : un texte non fiable reste du texte."""
    return _MD_SPECIAL.sub(r"\\\1", str(text or "")).replace("\n", "  \n")


def answer_md(ans: dict) -> list[str]:
    out = [f"### {STATUS.get(ans['status'], esc(ans['status']))}", esc(ans["answer"])]
    if ans.get("clarification_question"):
        out.append(f"> ❓ {esc(ans['clarification_question'])}")
    out += [f"> ⚠️ {esc(u)}" for u in ans.get("uncertainties", [])]
    if ans.get("citations"):
        out.append("**Sources citées**")
        out += [f"- {esc(c['document_id'])}, page {int(c['page'])} · {esc(c['element_id'])}" for c in ans["citations"]]
    removed = ans.get("unsupported_citations", []) + ans.get("invalid_citations", [])
    if removed:
        out.append(f"*Citations retirées par la vérification : {esc(', '.join(removed))}*")
    if ans.get("verifications"):
        out.append("**Vérification des affirmations**")
        out += [f"- **{esc(v['verdict'])}** : {esc(v['claim'])} ({esc(', '.join(v['sources']))})"
                for v in ans["verifications"]]
    return out


def photo_md(a: dict) -> list[str]:
    out = ["**Analyse de la photo** : "
           f"famille **{esc(a.get('famille') or 'non identifiée')}** · modèle **{esc(a.get('modele') or 'non lisible')}**"
           f" · code **{esc(a.get('code') or 'aucun')}**"]
    if a.get("voyants"):
        out.append("Voyants allumés : " + esc(", ".join(f"{v['voyant']} {v['couleur']}" for v in a["voyants"])))
    out += [f"*{esc(n)}*" for n in a.get("notes", [])]
    return out


def agent_md(r: dict) -> str:
    out = [f"## {AGENT_STATUS.get(r['status'], esc(r['status']))}"]
    if r.get("photo_analysis"):
        out += photo_md(r["photo_analysis"])
    if r.get("clarification_question"):
        out.append(f"> ❓ **{esc(r['clarification_question'])}**")
    if r.get("answer"):
        out += answer_md(r["answer"])
    t = r.get("ticket")
    if r["status"] == "ticket_cree" and t:
        out.append(f"**{'Ticket créé' if t.get('created') else 'Ticket déjà existant (aucun doublon créé)'} : "
                   f"{esc(t['ticket_id'])}**")
    return "\n\n".join(out)


def trace_md(r: dict) -> str:
    out = [f"Demande `{esc(r['thread_id'])}` · appels au modèle : {r['model_calls']} · appels MCP : {r['mcp_calls']}"]
    out += [f"- Recherche {a['round']} : « {esc(a['query'])} » → {esc(a['status'])}, {a['citations']} citation(s)"
            for a in r.get("search_attempts", [])]
    out += [f"- *{esc(e)}*" for e in r.get("errors", [])]
    versions = (r.get("answer") or {}).get("versions", {})
    if versions:
        out.append(esc(" · ".join(f"{k} : {v}" for k, v in versions.items())))
    return "\n\n".join(out)


def search_md(res: dict) -> str:
    out = [f"*Index {esc(res['index_version'])} · seuls les documents autorisés pour votre profil sont recherchés.*"]
    for p in res["passages"]:
        score = f" · score {p['rerank_score']:.2f}" if p.get("rerank_score") is not None else ""
        out.append(f"**{esc(p['titre'])}**, page {int(p['page'])} · {esc(p['type'])}{score}"
                   + (f"  \n*{esc(p['section'])}*" if p.get("section") else "")
                   + f"  \n{esc((p.get('text') or '')[:800])}")
    if not res["passages"]:
        out.append("Aucun passage accessible ne correspond.")
    return "\n\n---\n\n".join(out)
