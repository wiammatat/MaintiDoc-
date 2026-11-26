"""Authentification de l'API : jetons JWT signés (HS256) et utilisateur courant.

Le jeton ne porte que l'identifiant : les groupes sont relus côté serveur à chaque requête,
pour qu'un changement de droits s'applique immédiatement et qu'un jeton ne puisse pas les modifier.
"""

from datetime import UTC, datetime, timedelta

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from common.access import User, load_users
from common.config import get_settings

ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

UNAUTHORIZED = HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentification requise ou jeton invalide",
                             headers={"WWW-Authenticate": "Bearer"})


def _secret() -> str:
    secret = get_settings().jwt_secret.get_secret_value()
    if len(secret) < 32:
        raise RuntimeError("JWT_SECRET absent ou trop court dans .env (32 caractères minimum)")
    return secret


def create_token(user: User) -> tuple[str, int]:
    """Renvoie (jeton, durée de validité en secondes)."""
    ttl = get_settings().jwt_ttl_minutes * 60
    now = datetime.now(UTC)
    claims = {"sub": user.user_id, "iat": now, "exp": now + timedelta(seconds=ttl)}
    return jwt.encode(claims, _secret(), algorithm=ALGORITHM), ttl


def current_user(token: str = Depends(oauth2_scheme)) -> User:
    try:
        claims = jwt.decode(token, _secret(), algorithms=[ALGORITHM], options={"require": ["sub", "exp"]})
    except jwt.PyJWTError:
        raise UNAUTHORIZED from None
    user = load_users().get(claims["sub"])
    if user is None:          # compte supprimé depuis l'émission du jeton
        raise UNAUTHORIZED
    return user
