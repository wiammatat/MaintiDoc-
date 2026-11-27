"""Agent MaintiDoc (LangGraph) : workflow de la section 6 du sujet.

    vérifier l'accès -> analyser la demande -> [analyser la photo] -> [historique via MCP] -> rechercher et répondre (RAG + vérification)
    -> [boucle contrôlée : si les preuves sont insuffisantes, reformuler et rechercher à nouveau]
    -> [préparer un ticket] -> attendre l'approbation humaine -> créer le ticket via MCP

- L'état est mémorisé par thread (checkpointer) : le graphe s'interrompt avant toute action et reprend
  quand un humain approuve ou refuse le contenu EXACT du ticket.
- Budgets : au plus MAX_MCP_CALLS appels MCP et MAX_MODEL_CALLS appels au modèle par demande ; au-delà du temps
  maximal, les étapes facultatives (relance de recherche, ticket) sont abandonnées.
- Boucle contrôlée : au plus AGENT_MAX_SEARCH_ROUNDS recherches, et seulement si le budget permet un tour complet
  (reformulation + réponse + vérification). Pas de relance quand le garde-fou d'équipement a bloqué la demande.
- Analyse de la photo (prompt visual_analysis_v1) : le VLM décrit, le code vérifie (agent/vision.py). Famille
  identifiée -> recherche filtrée sur cette famille ; équipement non identifiable ou contradictoire -> clarification,
  sans recherche ni réponse.
- Le jeton de connexion de l'utilisateur est passé dans la configuration de chaque appel, jamais dans l'état
  mémorisé : le serveur MCP vérifie lui-même les droits à chaque appel.
"""

import base64
import io
import time
from collections.abc import Callable
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from PIL import Image

from agent.intent import analyze
from agent.mcp_client import MCPClient, MCPError
from agent.rewrite import rewrite_query
from agent.vision import analyze_photo
from agent.ticket import draft_ticket
from common.access import get_user
from common.config import get_settings
from common.tokens import create_approval, ticket_hash
from generation.answer import answer
from generation.vlm import VLM
from retrieval.search import SearchResult, search

CHECKPOINTER = InMemorySaver()   # prototype : les approbations en attente sont perdues au redémarrage de l'API


class AgentState(TypedDict, total=False):
    user_id: str
    question: str
    photo_b64: str | None
    started: float
    intent: dict
    clarification_question: str | None
    photo_analysis: dict | None
    search_filters: dict | None
    photo_context: str | None
    history: dict | None
    open_tickets: dict | None
    answer: dict | None
    search_query: str | None
    search_rounds: int
    search_attempts: list[dict]
    ticket_draft: dict | None
    draft_origin: str | None
    decision: dict | None
    ticket: dict | None
    status: str
    mcp_calls: int
    model_calls: int
    errors: list[str]


def _token(config: RunnableConfig) -> str:
    return config["configurable"]["auth_token"]


def _photo(state: AgentState) -> Image.Image | None:
    return Image.open(io.BytesIO(base64.b64decode(state["photo_b64"]))) if state.get("photo_b64") else None


def build_agent(vlm: VLM | None, mcp: MCPClient, search_fn: Callable[..., SearchResult] = search,
                checkpointer=CHECKPOINTER):
    s = get_settings()

    def mcp_call(state: AgentState, config: RunnableConfig, tool: str, args: dict) -> tuple[dict | None, dict]:
        """Appel MCP dans le budget. Renvoie (résultat ou None, mise à jour de l'état)."""
        calls = state.get("mcp_calls", 0)
        if calls >= s.max_mcp_calls:
            return None, {"errors": state.get("errors", []) + [f"Budget MCP atteint ({s.max_mcp_calls} appels) : {tool} non appelé"]}
        try:
            result = mcp.call(tool, args | {"auth_token": _token(config)})
            return result, {"mcp_calls": calls + 1}
        except MCPError as e:
            return None, {"mcp_calls": calls + 1, "errors": state.get("errors", []) + [f"{tool} : {e}"]}

    # ----------------------------------------------------------------- nœuds

    def verifier_acces(state: AgentState) -> dict:
        user = get_user(state["user_id"])           # lève PermissionError si inconnu
        if not user.groupes:
            raise PermissionError("Utilisateur sans groupe")
        return {"started": time.time(), "mcp_calls": 0, "model_calls": 0, "errors": [], "status": "en_cours"}

    def analyser(state: AgentState) -> dict:
        intent = analyze(state["question"])
        update: dict[str, Any] = {"intent": intent.to_dict(), "clarification_question": None}
        if (intent.wants_history or intent.wants_ticket) and not intent.equipment_id:
            update["clarification_question"] = ("Quel est l'identifiant de l'équipement concerné "
                                                "(par exemple BAIE-SN400-002 ou SW-FL48-002) ?")
        return update

    def analyser_image(state: AgentState) -> dict:
        used = state.get("model_calls", 0)
        if vlm is None or used >= s.max_model_calls:
            return {"errors": state.get("errors", []) + ["Photo non analysée (modèle indisponible ou budget atteint)"]}
        analysis, calls = analyze_photo(vlm, state["question"], _photo(state))
        update: dict[str, Any] = {"photo_analysis": analysis.to_dict(), "model_calls": used + calls}
        if analysis.clarification:
            update["clarification_question"] = analysis.clarification
        else:
            update["search_filters"] = {"famille": analysis.famille}
            update["photo_context"] = analysis.context()
        return update

    def consulter_historique(state: AgentState, config: RunnableConfig) -> dict:
        eq = state["intent"]["equipment_id"]
        history, update = mcp_call(state, config, "get_equipment_history", {"equipment_id": eq})
        state = state | update
        tickets, update2 = mcp_call(state, config, "get_open_tickets", {"equipment_id": eq}) if history else (None, {})
        return update | update2 | {"history": history, "open_tickets": tickets}

    def repondre(state: AgentState) -> dict:
        photo = _photo(state)
        query = state.get("search_query") or state["question"]
        used = state.get("model_calls", 0)
        filters = state.get("search_filters")

        def run_search(user, _question, image=None):
            # La recherche utilise la requête courante (éventuellement reformulée), filtrée sur la famille de la photo.
            return search_fn(user, query, image, filters=filters) if filters else search_fn(user, query, image)
        res = answer(get_user(state["user_id"]), state["question"], photo, vlm=vlm, search_fn=run_search,
                     max_calls=s.max_model_calls - used, photo_context=state.get("photo_context"))
        attempts = state.get("search_attempts", []) + [{"round": state.get("search_rounds", 0) + 1, "query": query,
                                                        "status": res.status, "citations": len(res.citations)}]
        return {"answer": res.model_dump(), "model_calls": used + res.model_calls,
                "search_rounds": state.get("search_rounds", 0) + 1, "search_attempts": attempts}

    def reformuler(state: AgentState) -> dict:
        previous = state.get("search_query") or state["question"]
        query = rewrite_query(vlm, state["question"], previous, outcome=state["answer"]["answer"])
        update: dict[str, Any] = {"model_calls": state.get("model_calls", 0) + 1}
        if query is None:   # pas de reformulation exploitable : on n'insiste pas
            update["errors"] = state.get("errors", []) + ["Reformulation impossible : pas de nouvelle recherche"]
        else:
            update["search_query"] = query
        return update

    def preparer_ticket(state: AgentState) -> dict:
        if time.time() - state["started"] > s.max_processing_seconds:
            return {"errors": state["errors"] + ["Temps maximal dépassé : ticket non préparé"], "status": "termine"}
        draft, calls, origin = draft_ticket(vlm, state["intent"]["equipment_id"], state["question"], state["answer"],
                                            state.get("history"), state.get("open_tickets"),
                                            max_calls=s.max_model_calls - state.get("model_calls", 0))
        return {"ticket_draft": draft.model_dump(), "draft_origin": origin,
                "model_calls": state.get("model_calls", 0) + calls, "status": "en_attente_approbation"}

    def attendre_approbation(state: AgentState) -> dict:
        # Pause : l'API renvoie le brouillon ; le graphe reprend avec la décision humaine.
        decision = interrupt({"ticket_draft": state["ticket_draft"]})
        return {"decision": decision}

    def creer_ticket(state: AgentState, config: RunnableConfig) -> dict:
        decision = state["decision"]
        if not decision.get("approved"):
            return {"status": "ticket_refuse"}
        eq = state["ticket_draft"]["equipment_id"]
        title, description = decision["title"], decision["description"]   # contenu exact approuvé (éventuellement modifié)
        approval = create_approval(state["user_id"], ticket_hash(eq, title, description))
        ticket, update = mcp_call(state, config, "create_ticket", {"equipment_id": eq, "title": title,
                                                                  "description": description, "approval_token": approval})
        return update | {"ticket": ticket, "status": "ticket_cree" if ticket else "echec_creation"}

    def conclure(state: AgentState) -> dict:
        if state.get("status") in ("ticket_cree", "ticket_refuse", "echec_creation"):
            return {}
        return {"status": "termine"}

    # ----------------------------------------------------------------- routes

    def apres_analyse(state: AgentState) -> str:
        if state.get("photo_b64"):
            return "analyser_image"
        return apres_photo(state)

    def apres_photo(state: AgentState) -> str:
        if (state.get("photo_analysis") or {}).get("clarification"):
            return "conclure"   # équipement non identifiable : on demande une précision au lieu de répondre
        i = state["intent"]
        return "consulter_historique" if i["equipment_id"] and (i["wants_history"] or i["wants_ticket"]) else "repondre"

    def doit_relancer(state: AgentState) -> bool:
        """Boucle contrôlée : preuves insuffisantes, limite de tours, budget et temps."""
        a = state["answer"]
        insuffisant = a["status"] == "abstained" and not a.get("citations")
        bloque = any(u.startswith("Équipement cité sans source accessible") for u in a.get("uncertainties", []))
        budget_ok = state.get("model_calls", 0) + 3 <= s.max_model_calls   # reformulation + réponse + vérification
        temps_ok = time.time() - state["started"] <= s.max_processing_seconds
        return (insuffisant and not bloque and vlm is not None and budget_ok and temps_ok
                and state.get("search_rounds", 1) < s.agent_max_search_rounds)

    def apres_reponse(state: AgentState) -> str:
        if doit_relancer(state):
            return "reformuler"
        i = state["intent"]
        # Pas de ticket si l'accès MCP à l'équipement a échoué (droits) : rien à proposer.
        return "preparer_ticket" if i["wants_ticket"] and i["equipment_id"] and state.get("history") else "conclure"

    def apres_reformulation(state: AgentState) -> str:
        if state.get("search_query") and state["search_query"] != (state["search_attempts"][-1]["query"]):
            return "repondre"
        i = state["intent"]
        return "preparer_ticket" if i["wants_ticket"] and i["equipment_id"] and state.get("history") else "conclure"

    def apres_preparation(state: AgentState) -> str:
        return "attendre_approbation" if state.get("ticket_draft") else "conclure"

    g = StateGraph(AgentState)
    for name, fn in [("verifier_acces", verifier_acces), ("analyser", analyser), ("analyser_image", analyser_image),
                     ("consulter_historique", consulter_historique), ("repondre", repondre), ("reformuler", reformuler),
                     ("preparer_ticket", preparer_ticket), ("attendre_approbation", attendre_approbation),
                     ("creer_ticket", creer_ticket), ("conclure", conclure)]:
        g.add_node(name, fn)
    g.add_edge(START, "verifier_acces")
    g.add_edge("verifier_acces", "analyser")
    g.add_conditional_edges("analyser", apres_analyse, ["analyser_image", "consulter_historique", "repondre"])
    g.add_conditional_edges("analyser_image", apres_photo, ["consulter_historique", "repondre", "conclure"])
    g.add_edge("consulter_historique", "repondre")
    g.add_conditional_edges("repondre", apres_reponse, ["reformuler", "preparer_ticket", "conclure"])
    g.add_conditional_edges("reformuler", apres_reformulation, ["repondre", "preparer_ticket", "conclure"])
    g.add_conditional_edges("preparer_ticket", apres_preparation, ["attendre_approbation", "conclure"])
    g.add_edge("attendre_approbation", "creer_ticket")
    g.add_edge("creer_ticket", END)
    g.add_edge("conclure", END)
    return g.compile(checkpointer=checkpointer)
