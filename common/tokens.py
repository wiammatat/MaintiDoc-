"""Jetons signés partagés par l'API et le serveur MCP (même secret, même machine).

- Jeton de connexion (JWT HS256, JWT_SECRET) : ne porte que l'identifiant ; les groupes sont relus côté serveur.
- Jeton d'approbation (HMAC-SHA256, APPROVAL_SECRET) : lie une approbation humaine à un utilisateur et au contenu
  EXACT d'un ticket. Toute modification du contenu invalide le jeton (sujet, section 6).
"""

import hashlib
import hmac
import json
import time
from datetime import UTC, datetime, timedelta

import jwt

from common.access import User, load_users
from common.config import get_settings

ALGORITHM = "HS256"
APPROVAL_TTL_SECONDS = 15 * 60


class InvalidToken(Exception):
    pass


def _secret(name: str) -> str:
    secret = getattr(get_settings(), name).get_secret_value()
    if len(secret) < 32:
        raise RuntimeError(f"{name.upper()} absent ou trop court dans .env (32 caractères minimum)")
    return secret


# --------------------------------------------------------------------------- connexion

def create_token(user: User) -> tuple[str, int]:
    """Renvoie (jeton, durée de validité en secondes)."""
    ttl = get_settings().jwt_ttl_minutes * 60
    now = datetime.now(UTC)
    claims = {"sub": user.user_id, "iat": now, "exp": now + timedelta(seconds=ttl)}
    return jwt.encode(claims, _secret("jwt_secret"), algorithm=ALGORITHM), ttl


def user_from_token(token: str) -> User:
    try:
        claims = jwt.decode(token, _secret("jwt_secret"), algorithms=[ALGORITHM], options={"require": ["sub", "exp"]})
    except jwt.PyJWTError:
        raise InvalidToken("jeton de connexion invalide ou expiré") from None
    user = load_users().get(claims["sub"])
    if user is None:
        raise InvalidToken("utilisateur inconnu")
    return user


# --------------------------------------------------------------------------- approbation d'un ticket

def ticket_hash(equipment_id: str, title: str, description: str) -> str:
    """Empreinte du contenu exact du ticket."""
    canonical = json.dumps({"equipment_id": equipment_id, "title": title, "description": description},
                           ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _approval_signature(user_id: str, content_hash: str, expires: int) -> str:
    message = f"{user_id}|{content_hash}|{expires}".encode()
    return hmac.new(_secret("approval_secret").encode(), message, hashlib.sha256).hexdigest()


def create_approval(user_id: str, content_hash: str, now: float | None = None) -> str:
    """Jeton remis APRÈS l'approbation humaine du contenu exact : « expiration.signature »."""
    expires = int((now or time.time()) + APPROVAL_TTL_SECONDS)
    return f"{expires}.{_approval_signature(user_id, content_hash, expires)}"


def verify_approval(token: str, user_id: str, content_hash: str, now: float | None = None) -> None:
    """Lève InvalidToken si le jeton ne correspond pas à cet utilisateur et à ce contenu, ou s'il a expiré."""
    try:
        expires_str, signature = token.split(".", 1)
        expires = int(expires_str)
    except ValueError:
        raise InvalidToken("jeton d'approbation mal formé") from None
    if (now or time.time()) > expires:
        raise InvalidToken("approbation expirée : une nouvelle validation est nécessaire")
    if not hmac.compare_digest(signature, _approval_signature(user_id, content_hash, expires)):
        raise InvalidToken("approbation invalide : le contenu ou l'utilisateur ne correspond pas à ce qui a été approuvé")


def idempotency_key(user_id: str, content_hash: str) -> str:
    """Même utilisateur + même contenu approuvé = même clé : un second envoi ne crée pas de doublon."""
    return hashlib.sha256(f"{user_id}|{content_hash}".encode()).hexdigest()
