"""Mise en forme HTML des réponses de l'API, sans dépendance à l'interface (testable seule).

Sécurité : TOUT texte venant des documents, du modèle ou des outils passe par h() (échappement HTML) avant
d'être inséré. Un document ou une réponse piégés ne peuvent donc ajouter ni balise, ni lien, ni script, ni style
(injection indirecte). Seules les balises écrites ici, avec des classes fixes, sont produites.
"""

from html import escape

STATUS = {
    "answered": ("ok", "Réponse sourcée"),
    "needs_clarification": ("warn", "Précision nécessaire"),
    "abstained": ("muted", "Pas de réponse : sources insuffisantes"),
}
AGENT_STATUS = {
    "termine": ("ok", "Terminé"),
    "en_attente_approbation": ("warn", "Ticket en attente de votre approbation"),
    "ticket_cree": ("ok", "Ticket créé"),
    "ticket_refuse": ("muted", "Ticket refusé : rien n'a été créé"),
    "echec_creation": ("ko", "Échec de la création du ticket"),
}
VERDICT = {"soutenu": "ok", "partiel": "warn", "non_soutenu": "ko"}
LED_COLORS = {"vert": "#22c55e", "orange": "#f97316", "rouge": "#ef4444", "bleu": "#3b82f6", "jaune": "#eab308",
              "blanc": "#e5e7eb", "ambre": "#f59e0b"}


def h(text) -> str:
    """Échappement HTML d'un texte non fiable (guillemets compris)."""
    return escape(str(text if text is not None else ""), quote=True)


def badge(kind: str, label: str) -> str:
    return f'<span class="md-badge md-{kind}">{h(label)}</span>'


def card(body: str, title: str = "", kind: str = "") -> str:
    head = f'<div class="md-card-title">{title}</div>' if title else ""
    return f'<div class="md-card {f"md-card-{kind}" if kind else ""}">{head}{body}</div>'


def notice(kind: str, text: str) -> str:
    return f'<div class="md-notice md-{kind}">{h(text)}</div>'


def _para(text: str) -> str:
    return "".join(f"<p>{h(line)}</p>" for line in str(text or "").split("\n") if line.strip())


# --------------------------------------------------------------------------- assistant

def answer_html(ans: dict) -> str:
    kind, label = STATUS.get(ans["status"], ("muted", ans["status"]))
    parts = [badge(kind, label), f'<div class="md-answer">{_para(ans["answer"])}</div>']
    if ans.get("clarification_question"):
        parts.append(notice("warn", ans["clarification_question"]))
    parts += [notice("muted", u) for u in ans.get("uncertainties", [])]
    out = [card("".join(parts), "Réponse")]

    if ans.get("citations"):
        items = "".join(
            f'<li><span class="md-doc">{h(c["document_id"])}</span>'
            f'<span class="md-chip">page {int(c["page"])}</span><span class="md-sub">{h(c["element_id"])}</span></li>'
            for c in ans["citations"])
        removed = ans.get("unsupported_citations", []) + ans.get("invalid_citations", [])
        extra = (f'<div class="md-sub">Citations retirées par la vérification : {h(", ".join(removed))}</div>'
                 if removed else "")
        out.append(card(f'<ul class="md-sources">{items}</ul>{extra}', "Sources citées"))

    if ans.get("verifications"):
        rows = "".join(f'<li>{badge(VERDICT.get(v["verdict"], "muted"), v["verdict"])} {h(v["claim"])} '
                       f'<span class="md-sub">({h(", ".join(v["sources"]))})</span></li>'
                       for v in ans["verifications"])
        out.append(card(f'<ul class="md-checks">{rows}</ul>', "Vérification des affirmations"))
    return "".join(out)


def photo_html(a: dict) -> str:
    def field(label, value, empty):
        return (f'<div class="md-field"><span class="md-sub">{label}</span>'
                f'<strong>{h(value) if value else f"<em>{empty}</em>"}</strong></div>')
    leds = "".join(
        f'<span class="md-led"><i style="background:{LED_COLORS.get(str(v["couleur"]).lower(), "#9ca3af")}"></i>'
        f'{h(v["voyant"])} {h(v["couleur"])}</span>' for v in a.get("voyants", []))
    body = (f'<div class="md-fields">{field("Famille", a.get("famille"), "non identifiée")}'
            f'{field("Modèle", a.get("modele"), "non lisible")}{field("Code affiché", a.get("code"), "aucun")}</div>'
            + (f'<div class="md-leds">{leds}</div>' if leds else "")
            + "".join(f'<div class="md-sub">{h(n)}</div>' for n in a.get("notes", [])))
    return card(body, "Analyse de la photo")


def agent_html(r: dict) -> str:
    kind, label = AGENT_STATUS.get(r["status"], ("muted", r["status"]))
    out = [f'<div class="md-status">{badge(kind, label)}</div>']
    if r.get("clarification_question"):
        out.append(notice("warn", r["clarification_question"]))
    if r.get("photo_analysis"):
        out.append(photo_html(r["photo_analysis"]))
    if r.get("answer"):
        out.append(answer_html(r["answer"]))
    t = r.get("ticket")
    if r["status"] == "ticket_cree" and t:
        msg = "Ticket créé" if t.get("created") else "Ticket déjà existant : aucun doublon créé"
        out.append(card(f'<div class="md-ticket-id">{h(t["ticket_id"])}</div><div class="md-sub">{h(msg)}</div>',
                        "Ticket de maintenance", "ok"))
    return f'<div class="md-result">{"".join(out)}</div>'


def trace_html(r: dict) -> str:
    stats = (f'<div class="md-stats"><span class="md-chip">demande {h(r["thread_id"])}</span>'
             f'<span class="md-chip">{int(r["model_calls"])} appel(s) au modèle</span>'
             f'<span class="md-chip">{int(r["mcp_calls"])} appel(s) MCP</span></div>')
    rounds = "".join(f'<li>Recherche {int(a["round"])} : « {h(a["query"])} » → {h(a["status"])}, '
                     f'{int(a["citations"])} citation(s)</li>' for a in r.get("search_attempts", []))
    errors = "".join(f"<li>{h(e)}</li>" for e in r.get("errors", []))
    versions = (r.get("answer") or {}).get("versions", {})
    vers = "".join(f'<span class="md-chip">{h(k)} : {h(v)}</span>' for k, v in versions.items())
    return (stats + (f"<ul>{rounds}</ul>" if rounds else "") + (f'<ul class="md-errors">{errors}</ul>' if errors else "")
            + (f'<div class="md-stats">{vers}</div>' if vers else ""))


# --------------------------------------------------------------------------- recherche

def search_html(res: dict) -> str:
    head = (f'<div class="md-sub">Index {h(res["index_version"])} · seuls les documents autorisés pour votre profil '
            f'sont recherchés · {len(res["passages"])} passage(s)</div>')
    if not res["passages"]:
        return head + notice("muted", "Aucun passage accessible ne correspond à cette recherche.")
    cards = []
    for p in res["passages"]:
        score = f'<span class="md-chip">score {p["rerank_score"]:.2f}</span>' if p.get("rerank_score") is not None else ""
        title = (f'{h(p["titre"])} <span class="md-chip">page {int(p["page"])}</span>'
                 f'<span class="md-chip">{h(p["type"])}</span>{score}')
        section = f'<div class="md-sub">{h(p["section"])}</div>' if p.get("section") else ""
        text = (p.get("text") or "")[:800]
        cards.append(card(f'{section}<div class="md-excerpt">{_para(text)}</div>', title))
    return head + "".join(cards)


def user_html(user: dict) -> str:
    groups = "".join(f'<span class="md-chip">{h(g)}</span>' for g in user["groupes"])
    return (f'<div class="md-user"><div class="md-avatar">{h(user["nom"][:1].upper())}</div>'
            f'<div><strong>{h(user["nom"])}</strong> <span class="md-sub">{h(user["user_id"])}</span>'
            f'<div>{groups}</div></div></div>')


CSS = """
.md-header {display:flex; align-items:center; justify-content:space-between; gap:16px; padding:18px 22px;
  border-radius:14px; background:linear-gradient(120deg,#1e3a8a,#0e7490); color:#fff; margin-bottom:6px;}
.md-header h1 {margin:0; font-size:1.55rem; color:#fff;}
.md-header p {margin:2px 0 0; opacity:.85; font-size:.92rem;}
.md-pill {padding:5px 12px; border-radius:999px; background:rgba(255,255,255,.16); font-size:.82rem; white-space:nowrap;}
.md-login {max-width:440px; margin:28px auto !important;}
.md-card {border:1px solid var(--border-color-primary); border-radius:12px; padding:14px 16px; margin:10px 0;
  background:var(--block-background-fill);}
.md-card-ok {border-left:4px solid #16a34a;}
.md-card-title {font-weight:600; margin-bottom:8px; display:flex; align-items:center; gap:6px; flex-wrap:wrap;}
.md-badge {display:inline-block; padding:3px 10px; border-radius:999px; font-size:.8rem; font-weight:600;}
.md-ok {background:rgba(22,163,74,.14); color:#15803d;}
.md-warn {background:rgba(234,88,12,.14); color:#c2410c;}
.md-ko {background:rgba(220,38,38,.14); color:#b91c1c;}
.md-muted {background:rgba(100,116,139,.14); color:var(--body-text-color-subdued);}
.md-notice {padding:10px 14px; border-radius:10px; margin:8px 0; font-weight:500;}
.md-answer p {margin:8px 0; line-height:1.55;}
.md-chip {display:inline-block; padding:1px 8px; margin:2px 4px 2px 0; border-radius:6px; font-size:.78rem;
  background:var(--background-fill-secondary); border:1px solid var(--border-color-primary);}
.md-sub {color:var(--body-text-color-subdued); font-size:.85rem;}
.md-sources, .md-checks {list-style:none; padding:0; margin:0;}
.md-sources li, .md-checks li {padding:6px 0; border-bottom:1px dashed var(--border-color-primary);}
.md-sources li:last-child, .md-checks li:last-child {border-bottom:none;}
.md-doc {font-weight:600; margin-right:8px;}
.md-fields {display:flex; gap:22px; flex-wrap:wrap;}
.md-field {display:flex; flex-direction:column;}
.md-leds {margin-top:10px;}
.md-led {display:inline-flex; align-items:center; gap:6px; margin-right:12px; font-size:.88rem;}
.md-led i {width:12px; height:12px; border-radius:50%; box-shadow:0 0 6px currentColor; display:inline-block;}
.md-ticket-id {font-size:1.4rem; font-weight:700;}
.md-status {margin-bottom:4px;}
.md-stats {margin:4px 0;}
.md-errors li {color:#b91c1c;}
.md-excerpt p {margin:4px 0; font-size:.92rem;}
.md-user {display:flex; align-items:center; gap:12px;}
.md-avatar {width:38px; height:38px; border-radius:50%; background:#1e3a8a; color:#fff; display:flex;
  align-items:center; justify-content:center; font-weight:700;}
.md-ticket-box {border:2px solid #f97316 !important; border-radius:12px !important; padding:6px !important;}
.md-empty {text-align:center; padding:40px 10px; color:var(--body-text-color-subdued);}
"""
