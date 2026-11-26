"""Hachage des mots de passe (PBKDF2-SHA256, bibliothèque standard) et vérification à temps constant."""

import hashlib
import hmac
import os

ITERATIONS = 200_000


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Renvoie « pbkdf2_sha256$itérations$sel$empreinte » (sel et empreinte en hexadécimal)."""
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return f"pbkdf2_sha256${ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iterations, salt, digest = stored.split("$")
    except ValueError:
        return False
    if algo != "pbkdf2_sha256":
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iterations))
    return hmac.compare_digest(candidate.hex(), digest)
