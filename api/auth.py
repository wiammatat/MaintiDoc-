"""Authentification de l'API : jetons JWT signés et utilisateur courant.

Le jeton ne porte que l'identifiant : les groupes sont relus côté serveur à chaque requête,
pour qu'un changement de droits s'applique immédiatement et qu'un jeton ne puisse pas les modifier.
La création et la vérification des jetons sont partagées avec le serveur MCP (common/tokens.py).
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from common.access import User
from common.tokens import InvalidToken, create_token, user_from_token

__all__ = ["create_token", "current_user", "oauth2_scheme"]

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

UNAUTHORIZED = HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentification requise ou jeton invalide",
                             headers={"WWW-Authenticate": "Bearer"})


def current_user(token: str = Depends(oauth2_scheme)) -> User:
    try:
        return user_from_token(token)
    except InvalidToken:
        raise UNAUTHORIZED from None
