"""Serveur du VLM, lancé sur une machine avec GPU (Kaggle) et appelé à distance par RemoteVLM.

Il n'expose que la génération : la recherche, le contrôle d'accès et la vérification restent sur le PC.
Protections :
- jeton secret obligatoire (comparaison à temps constant), sinon 401 ;
- nombre et taille des images bornés, textes bornés ;
- une seule génération à la fois (le GPU ne supporte pas deux générations en parallèle).
"""

import base64
import binascii
import hmac
import io
import re
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, status
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from generation.vlm import VLM

MAX_IMAGES = 4
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_TEXT_CHARS = 60_000
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"


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
    """Lance uvicorn dans un fil d'exécution (le notebook reste utilisable)."""
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(50):
        if server.started:
            return thread
        time.sleep(0.2)
    raise RuntimeError("Le serveur VLM n'a pas démarré")


def start_tunnel(port: int = 8001, workdir: Path = Path("/kaggle/working"), timeout: int = 90) -> tuple[str, subprocess.Popen]:
    """Ouvre un tunnel Cloudflare rapide (sans compte) vers le port local. Renvoie (adresse publique, processus)."""
    binary = shutil.which("cloudflared") or str(workdir / "cloudflared")
    if not Path(binary).exists():
        urllib.request.urlretrieve(CLOUDFLARED_URL, binary)
        Path(binary).chmod(0o755)
    proc = subprocess.Popen([binary, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if match:
            # Continuer à lire la sortie de cloudflared : sinon le tampon se remplit et le tunnel se bloque.
            threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True).start()
            return match.group(0), proc
    proc.terminate()
    raise RuntimeError("Adresse du tunnel introuvable : vérifier que l'accès Internet est activé dans Kaggle")
