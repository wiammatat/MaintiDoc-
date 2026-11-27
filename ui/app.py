"""Application Gradio de MaintiDoc (étape 7 de la feuille de route) : interface du technicien.

    python -m ui.app                 (l'API doit tourner : uvicorn api.main:app) -> http://127.0.0.1:7860

- Assistant : question et photo facultative -> agent (analyse de la photo, historique MCP, réponse sourcée,
  vérification). Si un ticket est demandé, le brouillon s'affiche : le technicien le relit, le modifie, puis
  l'approuve ou le refuse. Rien n'est créé sans ce clic.
- Recherche documentaire : preuves brutes (passages, tableaux, figures), utilisable même sans VLM.

L'interface n'a aucun droit propre : chaque appel passe par l'API avec le jeton de l'utilisateur connecté, conservé
dans l'état de session Gradio (côté serveur). Les fonctions de rappel (on_login, on_ask, ...) ne dépendent pas de
l'affichage : les tests les appellent directement.
"""

import io
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")   # aucune donnée d'usage envoyée à l'extérieur

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from common.config import get_settings  # noqa: E402
from ui.client import ApiClient, ApiError, Photo  # noqa: E402
from ui.render import CSS, agent_html, h, notice, search_html, trace_html, user_html  # noqa: E402

MAX_PDFS = 3
EMPTY_RESULT = '<div class="md-empty">Posez une question à gauche : la réponse sourcée s\'affichera ici.</div>'
EXAMPLES = [
    "Que signifie le voyant L2 orange sur une baie SN-400 ?",
    "Historique des interventions sur BAIE-SN400-002 ?",
    "Voyant orange sur BAIE-SN400-002, prépare un ticket.",
    "Quel est le seuil de température de l'alarme E03 ?",
]
HISTORY_COLUMNS = {"date": "Date", "type": "Type", "code_alarme": "Code", "description": "Description",
                   "technicien": "Technicien", "duree_h": "Durée (h)"}
TICKET_COLUMNS = {"ticket_id": "Ticket", "titre": "Titre", "cree_par": "Créé par", "cree_le": "Créé le"}

THEME = gr.themes.Soft(
    primary_hue="blue", secondary_hue="cyan", neutral_hue="slate", radius_size="lg",
    font=["Segoe UI", "system-ui", "-apple-system", "sans-serif"],   # polices locales : aucun appel externe
    font_mono=["Cascadia Code", "Consolas", "monospace"],
)


def make_client(token: str | None = None) -> ApiClient:
    """Remplacée dans les tests par un client branché sur l'API en mémoire."""
    s = get_settings()
    return ApiClient(s.ui_api_url, token=token, timeout=s.ui_timeout)


def new_session() -> dict:
    return {"token": None, "user": None, "thread_id": None}


def _photo(image: Image.Image | None) -> Photo | None:
    if image is None:
        return None
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return Photo("photo.jpg", buf.getvalue(), "image/jpeg")


def _error(e: ApiError, session: dict) -> str:
    if e.status == 401:
        session.update(new_session())
        return notice("ko", "Session expirée : reconnectez-vous.")
    if e.status == 503:
        return notice("warn", f"{e} L'onglet « Recherche documentaire » reste utilisable.")
    return notice("ko", str(e))


def _table(rows: list[dict], columns: dict) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return df[[c for c in columns if c in df.columns]].rename(columns=columns) if not df.empty else df


def _pdfs(client: ApiClient, r: dict) -> list[str]:
    """PDF des documents cités, récupérés par l'API (accès revérifié), dans un dossier temporaire de la session."""
    ids = list(dict.fromkeys(c["document_id"] for c in ((r.get("answer") or {}).get("citations") or [])))[:MAX_PDFS]
    folder = Path(tempfile.mkdtemp(prefix="maintidoc_"))
    paths = []
    for doc_id in ids:
        if not re.fullmatch(r"[\w.-]+", doc_id):
            continue
        try:
            (folder / f"{doc_id}.pdf").write_bytes(client.document_pdf(doc_id))
            paths.append(str(folder / f"{doc_id}.pdf"))
        except ApiError:
            pass
    return paths


# --------------------------------------------------------------------------- rappels

def on_health() -> str:
    try:
        hl = make_client().health()
        vlm = "désactivé" if hl.get("vlm_backend") == "disabled" else hl.get("vlm_backend")
        return f"● API connectée · index {h(hl.get('index_version'))} · modèle : {h(vlm)}"
    except ApiError:
        return "○ API injoignable : lancer « uvicorn api.main:app »"


def header_html(status: str = "") -> str:
    return (f'<div class="md-header"><div><h1>MaintiDoc</h1><p>Assistant de maintenance SAN · réponses sourcées, '
            f'actions validées par un humain</p></div><span class="md-pill">{status}</span></div>')


def on_load() -> str:
    return header_html(on_health())


def on_login(user_id: str, password: str, session: dict):
    """-> (session, message, écran de connexion, espace de travail, carte utilisateur)"""
    client = make_client()
    try:
        user = client.login((user_id or "").strip(), password or "")
    except ApiError as e:
        msg = notice("ko", "Identifiant ou mot de passe incorrect." if e.status == 401 else str(e))
        return session, msg, gr.update(visible=True), gr.update(visible=False), ""
    session = new_session() | {"token": client.token, "user": user}
    return session, "", gr.update(visible=False), gr.update(visible=True), user_html(user)


def on_logout(session: dict):
    return (new_session(), notice("muted", "Vous êtes déconnecté."), gr.update(visible=True),
            gr.update(visible=False), "")


def _agent_outputs(client: ApiClient, r: dict, session: dict) -> tuple:
    """-> (session, résultat, traçabilité, historique, tickets ouverts, zone ticket, titre, description, PDF)"""
    session = session | {"thread_id": r["thread_id"]}
    pending = r["status"] == "en_attente_approbation" and r.get("ticket_draft")
    draft = r.get("ticket_draft") or {}
    hist = r.get("historique") or {}
    tickets = (r.get("tickets_ouverts") or {}).get("tickets", [])
    return (session, agent_html(r), trace_html(r), _table(hist.get("interventions", []), HISTORY_COLUMNS),
            _table(tickets, TICKET_COLUMNS), gr.update(visible=bool(pending)), draft.get("title", ""),
            draft.get("description", ""), _pdfs(client, r))


def _failed(message: str, session: dict) -> tuple:
    return session, message, "", pd.DataFrame(), pd.DataFrame(), gr.update(visible=False), "", "", []


def on_ask(question: str, image: Image.Image | None, session: dict) -> tuple:
    if not session.get("token"):
        return _failed(notice("warn", "Connectez-vous d'abord."), session)
    if not (question or "").strip():
        return _failed(notice("warn", "Écrivez une question."), session)
    client = make_client(session["token"])
    try:
        return _agent_outputs(client, client.agent(question.strip(), _photo(image)), session)
    except ApiError as e:
        return _failed(_error(e, session), session)


def on_decide(approved: bool, title: str, description: str, session: dict) -> tuple:
    if not session.get("token") or not session.get("thread_id"):
        return _failed(notice("warn", "Aucune demande en cours."), session)
    client = make_client(session["token"])
    try:
        # Le contenu envoyé est celui affiché (éventuellement corrigé) : c'est lui qui est approuvé puis créé.
        r = client.approve(session["thread_id"], approved, title=title, description=description)
        return _agent_outputs(client, r, session)
    except ApiError as e:
        return _failed(_error(e, session), session)


def on_search(question: str, image: Image.Image | None, session: dict) -> tuple:
    """-> (passages, galerie d'images)"""
    if not session.get("token"):
        return notice("warn", "Connectez-vous d'abord."), []
    if not (question or "").strip():
        return notice("warn", "Écrivez une question."), []
    client = make_client(session["token"])
    try:
        res = client.search(question.strip(), _photo(image))
    except ApiError as e:
        return _error(e, session), []
    gallery = []
    for im in res["images"]:
        if im.get("image_url"):
            try:
                gallery.append((Image.open(io.BytesIO(client.element_image(im["image_url"]))),
                                f"{im['titre']}, page {im['page']}"))
            except (ApiError, OSError):
                pass
    return search_html(res), gallery


# --------------------------------------------------------------------------- interface

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="MaintiDoc", theme=THEME, css=CSS) as demo:
        session = gr.State(new_session())
        header = gr.HTML(header_html())

        # ------------------------------------------------ écran de connexion
        with gr.Column(visible=True, elem_classes="md-login") as login_view:
            gr.Markdown("### Connexion\nVos droits d'accès aux documents et aux équipements dépendent de votre profil.")
            user_id = gr.Textbox(label="Identifiant", placeholder="ex. tech_s01")
            password = gr.Textbox(label="Mot de passe", type="password")
            login_btn = gr.Button("Se connecter", variant="primary", size="lg")
        login_msg = gr.HTML()

        # ------------------------------------------------ espace de travail
        with gr.Column(visible=False) as app_view:
            with gr.Row(equal_height=True):
                user_card = gr.HTML()
                logout_btn = gr.Button("Se déconnecter", size="sm", variant="secondary", scale=0, min_width=150)

            with gr.Tab("Assistant"):
                with gr.Row():
                    with gr.Column(scale=2, min_width=320):
                        question = gr.Textbox(label="Votre question", lines=4, max_length=1000,
                                              placeholder="Décrivez le symptôme, l'équipement (ex. BAIE-SN400-002) "
                                                          "et ce que vous voulez (explication, historique, ticket).")
                        photo = gr.Image(label="Photo de l'équipement (facultative)", type="pil", sources=["upload"],
                                         height=230)
                        with gr.Row():
                            clear_btn = gr.Button("Effacer", variant="secondary")
                            ask_btn = gr.Button("Envoyer", variant="primary", scale=2)
                        gr.Examples(EXAMPLES, inputs=question, label="Exemples")

                    with gr.Column(scale=3, min_width=360):
                        result = gr.HTML(EMPTY_RESULT)
                        with gr.Group(visible=False, elem_classes="md-ticket-box") as ticket_box:
                            gr.Markdown("#### Brouillon de ticket à valider\nRelisez et corrigez : c'est le contenu "
                                        "exact affiché ici qui sera créé, et seulement après votre approbation.")
                            title = gr.Textbox(label="Titre", max_length=120)
                            description = gr.Textbox(label="Description", lines=6, max_length=4000)
                            with gr.Row():
                                refuse_btn = gr.Button("Refuser", variant="stop")
                                approve_btn = gr.Button("Approuver et créer", variant="primary", scale=2)
                        pdfs = gr.File(label="Documents cités (PDF)", file_count="multiple", interactive=False)
                        with gr.Accordion("Historique de l'équipement (MCP)", open=False):
                            history = gr.Dataframe(label="Dernières interventions", interactive=False, wrap=True)
                            open_tickets = gr.Dataframe(label="Tickets ouverts", interactive=False, wrap=True)
                        with gr.Accordion("Traçabilité", open=False):
                            trace = gr.HTML()

            with gr.Tab("Recherche documentaire"):
                gr.Markdown("Recherche directe dans les manuels et procédures autorisés pour votre profil : "
                            "passages, tableaux et figures, avec leur score. Fonctionne sans modèle de génération.")
                with gr.Row():
                    with gr.Column(scale=3):
                        s_question = gr.Textbox(label="Rechercher", max_length=1000,
                                                placeholder="ex. voyant orange reconstruction RAID")
                        s_btn = gr.Button("Rechercher", variant="primary")
                    s_photo = gr.Image(label="Photo (facultative)", type="pil", sources=["upload"], height=150,
                                       scale=1)
                s_result = gr.HTML()
                s_gallery = gr.Gallery(label="Figures et pages", columns=3, height="auto", show_label=True)

        # ------------------------------------------------ événements
        agent_out = [session, result, trace, history, open_tickets, ticket_box, title, description, pdfs]
        login_out = [session, login_msg, login_view, app_view, user_card]
        demo.load(on_load, outputs=header)
        login_btn.click(on_login, [user_id, password, session], login_out)
        password.submit(on_login, [user_id, password, session], login_out)
        logout_btn.click(on_logout, [session], login_out).then(
            lambda: (EMPTY_RESULT, "", None, "", None), outputs=[result, question, photo, trace, pdfs])
        ask_btn.click(on_ask, [question, photo, session], agent_out)
        question.submit(on_ask, [question, photo, session], agent_out)
        clear_btn.click(lambda: ("", None, EMPTY_RESULT, gr.update(visible=False), None),
                        outputs=[question, photo, result, ticket_box, pdfs])
        approve_btn.click(lambda t, d, s: on_decide(True, t, d, s), [title, description, session], agent_out)
        refuse_btn.click(lambda t, d, s: on_decide(False, t, d, s), [title, description, session], agent_out)
        s_btn.click(on_search, [s_question, s_photo, session], [s_result, s_gallery])
        s_question.submit(on_search, [s_question, s_photo, session], [s_result, s_gallery])
    return demo


if __name__ == "__main__":
    # 127.0.0.1 : pas d'exposition sur le réseau ; share=False : pas de lien public Gradio.
    build_ui().queue().launch(server_name="127.0.0.1", server_port=7860, share=False)
