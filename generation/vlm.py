"""Modèles vision-langage interchangeables.

- QwenVL : Qwen2.5-VL sur GPU NVIDIA, en 4 bits ou en fp16.
- ScriptedVLM : renvoie des réponses prédéfinies, pour les tests sur CPU.
"""

from functools import lru_cache
from typing import Protocol

from PIL import Image

from common.config import get_settings


def fit_image(img: Image.Image, max_pixels: int) -> Image.Image:
    """Réduit l'image (proportions conservées) pour que largeur x hauteur <= max_pixels.

    Le nombre de tokens visuels, et donc la mémoire de l'attention, en dépend directement.
    Fait ici plutôt que par le processeur, dont le réglage de taille n'était pas appliqué sur Kaggle.
    """
    img = img.convert("RGB")
    w, h = img.size
    if w * h <= max_pixels:
        return img
    scale = (max_pixels / (w * h)) ** 0.5
    return img.resize((max(28, int(w * scale)), max(28, int(h * scale))), Image.Resampling.LANCZOS)


class VLM(Protocol):
    name: str

    def generate(self, system: str, user: str, images: list[Image.Image]) -> str: ...


class ScriptedVLM:
    """Faux VLM : renvoie successivement les réponses fournies (la dernière est répétée)."""

    def __init__(self, *replies: str, name: str = "scripted"):
        self.replies, self.name, self.calls = list(replies), name, []

    def generate(self, system: str, user: str, images: list[Image.Image]) -> str:
        self.calls.append({"system": system, "user": user, "n_images": len(images),
                           "image_pixels": [im.width * im.height for im in images]})
        return self.replies[min(len(self.calls), len(self.replies)) - 1]


class VLMUnavailable(RuntimeError):
    """Le VLM ne peut pas répondre (serveur distant arrêté, jeton refusé...) : l'API renvoie 503."""


class RemoteVLM:
    """Client du serveur VLM distant (generation/vlm_server.py), par exemple Qwen sur le GPU d'un autre PC du réseau."""

    def __init__(self, url: str, token: str, timeout: float = 180, client=None, name: str | None = None):
        import httpx
        self.url, self.token, self.timeout = url.rstrip("/"), token, timeout
        self.client = client or httpx.Client()
        self.name = name or f"{get_settings().vlm_model} (distant)"
        self.token_log: list[int] = []

    def generate(self, system: str, user: str, images: list[Image.Image]) -> str:
        import base64
        import io

        import httpx

        encoded = []
        for im in images:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, "JPEG", quality=90)
            encoded.append(base64.b64encode(buf.getvalue()).decode())
        try:
            r = self.client.post(f"{self.url}/generate", json={"system": system, "user": user, "images": encoded},
                                 headers={"Authorization": f"Bearer {self.token}"}, timeout=self.timeout)
        except httpx.HTTPError as e:
            raise VLMUnavailable("Serveur VLM injoignable : serveur arrêté ou adresse incorrecte (VLM_REMOTE_URL) "
                                 f"({type(e).__name__})") from None
        if r.status_code == 401:
            raise VLMUnavailable("Jeton refusé par le serveur VLM : vérifier VLM_REMOTE_TOKEN dans .env")
        if r.status_code != 200:
            raise VLMUnavailable(f"Erreur du serveur VLM (HTTP {r.status_code})")
        body = r.json()
        if body.get("input_tokens"):
            self.token_log.append(body["input_tokens"])
        return body["text"]


class QwenVL:
    def __init__(self, model_name: str | None = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        s = get_settings()
        self.name = model_name or s.vlm_model
        self.token_log: list[int] = []   # longueur du prompt (tokens) de chaque appel
        kwargs = {"device_map": "auto", "dtype": torch.float16}
        if s.vlm_quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        elif torch.cuda.device_count() >= 2:
            # Le GPU 0 porte l'encodeur d'images et les activations : on y limite les poids
            # pour garder de la marge (sinon « CUDA out of memory » avec plusieurs images).
            kwargs["max_memory"] = {0: f"{s.vlm_gpu0_max_gb}GiB"} | {
                i: "13GiB" for i in range(1, torch.cuda.device_count())}
        self.model = AutoModelForImageTextToText.from_pretrained(self.name, **kwargs).eval()
        # size = bornes en nombre de pixels (remplace min_pixels / max_pixels, obsolètes).
        self.processor = AutoProcessor.from_pretrained(
            self.name, size={"shortest_edge": 56 * 56, "longest_edge": s.vlm_photo_max_pixels})

    def generate(self, system: str, user: str, images: list[Image.Image]) -> str:
        import torch

        s = get_settings()
        # Les images arrivent déjà réduites (answer.py) ; ce plafond est une sécurité.
        content = ([{"type": "image", "image": fit_image(im, s.vlm_photo_max_pixels)} for im in images]
                   + [{"type": "text", "text": user}])
        messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                    return_dict=True, return_tensors="pt").to(self.model.device)
        # La mémoire de l'attention croît avec le carré de cette longueur : utile pour diagnostiquer.
        self.token_log.append(int(inputs["input_ids"].shape[1]))
        try:
            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=s.vlm_max_new_tokens, do_sample=False)
            return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        finally:
            del inputs
            torch.cuda.empty_cache()   # libère les activations entre deux questions


@lru_cache
def default_vlm() -> VLM:
    return QwenVL()
