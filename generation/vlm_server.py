"""Serveur du VLM, lancé sur une machine avec GPU et appelé à distance par RemoteVLM (VLM_BACKEND=remote).

Cas d'usage : Qwen tourne sur le PC équipé de la carte NVIDIA, l'API tourne sur un autre PC du même réseau.
Il n'expose que la génération : la recherche, le contrôle d'accès et la vérification restent côté API.

Lancement sur le PC avec GPU :   python -m generation.vlm_server --host 0.0.0.0 --port 8001
(le jeton à recopier dans le .env de l'autre PC est affiché au démarrage)
Protections :
- jeton secret obligatoire (comparaison à temps constant), sinon 401 ;
- nombre et taille des images bornés, textes bornés ;
- une seule génération à la fois (le GPU ne supporte pas deux générations en parallèle).
"""

import base64
import binascii
import hmac
import io
import threading
import time

from fastapi import FastAPI, Header, HTTPException, status
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from generation.vlm import VLM

MAX_IMAGES = 4
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_TEXT_CHARS = 60_000


class GenerateRequest(BaseModel):
    system: str = Field(max_length=MAX_TEXT_CHARS)
    user: str = Field(max_length=MAX_TEXT_CHARS)
    images: list[str] = Field(default_factory=list, max_length=MAX_IMAGES)   # images encodées en base64


class GenerateResponse(BaseModel):
    text: str
    input_tokens: int | None = None


def _decode(image_b64: str) -> Image.Image:
    try:
        data = base64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Image base64 invalide") from None
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Image trop volumineuse")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Fichier image illisible") from None


def create_app(vlm: VLM, token: str) -> FastAPI:
    if len(token) < 32:
        raise ValueError("Le jeton du serveur VLM doit faire au moins 32 caractères")
    app = FastAPI(title="MaintiDoc - serveur VLM")
    lock = threading.Lock()

    def check(authorization: str | None) -> None:
        expected = f"Bearer {token}"
        if not authorization or not hmac.compare_digest(authorization.encode(), expected.encode()):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Jeton invalide")

    @app.get("/health")
    def health(authorization: str | None = Header(default=None)) -> dict:
        check(authorization)
        return {"status": "ok", "model": vlm.name}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest, authorization: str | None = Header(default=None)) -> GenerateResponse:
        check(authorization)
        images = [_decode(i) for i in req.images]
        with lock:   # une génération à la fois sur le GPU
            text = vlm.generate(req.system, req.user, images)
            tokens = getattr(vlm, "token_log", None)
            return GenerateResponse(text=text, input_tokens=tokens[-1] if tokens else None)

    return app


def serve_in_background(app: FastAPI, port: int = 8001) -> threading.Thread:
    """Lance uvicorn dans un fil d'exécution (utilisé par les tests)."""
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(50):
        if server.started:
            return thread
        time.sleep(0.2)
    raise RuntimeError("Le serveur VLM n'a pas démarré")


def main() -> None:
    import argparse
    import secrets

    import uvicorn

    from generation.vlm import QwenVL

    parser = argparse.ArgumentParser(description="Serveur VLM MaintiDoc (Qwen sur ce GPU)")
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 pour accepter les autres PC du réseau local")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    token = secrets.token_urlsafe(32)   # nouveau jeton à chaque lancement
    app = create_app(QwenVL(), token)
    print(f"Serveur VLM prêt. Dans le .env du PC qui fait tourner l'API :\n"
          f"VLM_BACKEND=remote\nVLM_REMOTE_URL=http://<adresse-IP-de-ce-PC>:{args.port}\nVLM_REMOTE_TOKEN={token}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
