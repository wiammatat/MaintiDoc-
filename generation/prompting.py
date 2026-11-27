"""Prompting et validation des sorties avec LangChain (section 6 du sujet : « LangChain fournit les composants
de récupération, de prompting et de validation des sorties »).

- Les prompts restent versionnés en YAML (prompts/*.yaml) ; ils sont chargés en ChatPromptTemplate LangChain.
- Les sorties du modèle sont validées par un PydanticOutputParser LangChain.
"""

import re
from functools import lru_cache

import yaml
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from common.config import get_settings

PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@lru_cache
def load_prompt(name: str) -> dict:
    """Prompt versionné brut (version, description, system, user)."""
    return yaml.safe_load((get_settings().prompts_dir / f"{name}.yaml").read_text(encoding="utf-8"))


def _template(text: str) -> str:
    """Les exemples JSON du prompt contiennent des accolades : on les protège, sauf les variables {nom}."""
    escaped = text.replace("{", "{{").replace("}", "}}")
    return re.sub(r"\{\{(\w+)\}\}", r"{\1}", escaped)


@lru_cache
def chat_prompt(name: str) -> ChatPromptTemplate:
    p = load_prompt(name)
    return ChatPromptTemplate.from_messages([("system", _template(p["system"])), ("human", _template(p["user"]))])


def render(name: str, **variables) -> tuple[str, str]:
    """Renvoie (message système, message utilisateur) pour le VLM."""
    system, human = chat_prompt(name).format_messages(**variables)
    return system.content, human.content


def parse(model: type[BaseModel], raw: str) -> BaseModel:
    """Valide la sortie du modèle. Qwen entoure souvent le JSON de texte : on extrait le bloc avant le parseur.

    Lève ValueError si la sortie n'est pas un JSON conforme au modèle Pydantic.
    """
    match = JSON_RE.search(raw)
    if not match:
        raise ValueError("aucun objet JSON dans la sortie du modèle")
    try:
        return PydanticOutputParser(pydantic_object=model).parse(match.group(0))
    except OutputParserException as e:
        raise ValueError(str(e)[:300]) from None
