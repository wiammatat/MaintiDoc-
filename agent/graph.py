"""Agent MaintiDoc (LangGraph) : workflow de la section 6 du sujet.

    vérifier l'accès -> analyser la demande -> [historique via MCP] -> rechercher et répondre (RAG + vérification)
    -> [préparer un ticket] -> attendre l'approbation humaine -> créer le ticket via MCP

- L'état est mémorisé par thread (checkpointer) : le graphe s'interrompt avant toute action et reprend
  quand un humain approuve ou refuse le contenu EXACT du ticket.
- Budgets : au plus MAX_MCP_CALLS appels MCP et MAX_MODEL_CALLS appels au modèle par demande ; au-delà du temps
  maximal, les étapes facultatives (ticket) sont abandonnées.
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
    history: dict | None
    open_tickets: dict | None
    answer: dict | None
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

    def consulter_historique(state: AgentState, config: RunnableConfig) -> dict:
        eq = state["intent"]["equipment_id"]
        history, update = mcp_call(state, config, "get_equipment_history", {"equipment_id": eq})
        state = state | update
        tickets, update2 = mcp_call(state, config, "get_open_tickets", {"equipment_id": eq}) if history else (None, {})
        return update | update2 | {"history": history, "open_tickets": tickets}

    def repondre(state: AgentState) -> dict:
        photo = Image.open(io.BytesIO(base64.b64decode(state["photo_b64"]))) if state.get("photo_b64") else None
        res = answer(get_user(state["user_id"]), state["question"], photo, vlm=vlm, search_fn=search_fn)
        return {"answer": res.model_dump(), "model_calls": state.get("model_calls", 0) + res.model_calls}

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
        i = state["intent"]
        return "consulter_historique" if i["equipment_id"] and (i["wants_history"] or i["wants_ticket"]) else "repondre"

    def apres_reponse(state: AgentState) -> str:
        i = state["intent"]
        # Pas de ticket si l'accès MCP à l'équipement a échoué (droits) : rien à proposer.
        return "preparer_ticket" if i["wants_ticket"] and i["equipment_id"] and state.get("history") else "conclure"

    def apres_preparation(state: AgentState) -> str:
        return "attendre_approbation" if state.get("ticket_draft") else "conclure"

    g = StateGraph(AgentState)
    for name, fn in [("verifier_acces", verifier_acces), ("analyser", analyser),
                     ("consulter_historique", consulter_historique), ("repondre", repondre),
                     ("preparer_ticket", preparer_ticket), ("attendre_approbation", attendre_approbation),
                     ("creer_ticket", creer_ticket), ("conclure", conclure)]:
        g.add_node(name, fn)
    g.add_edge(START, "verifier_acces")
    g.add_edge("verifier_acces", "analyser")
    g.add_conditional_edges("analyser", apres_analyse, ["consulter_historique", "repondre"])
    g.add_edge("consulter_historique", "repondre")
    g.add_conditional_edges("repondre", apres_reponse, ["preparer_ticket", "conclure"])
    g.add_conditional_edges("preparer_ticket", apres_preparation, ["attendre_approbation", "conclure"])
    g.add_edge("attendre_approbation", "creer_ticket")
    g.add_edge("creer_ticket", END)
    g.add_edge("conclure", END)
    return g.compile(checkpointer=checkpointer)
