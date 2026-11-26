"""Vérifie le VLM distant : serveur (generation/vlm_server.py) et client (RemoteVLM), reliés en mémoire."""

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from generation.vlm import RemoteVLM, ScriptedVLM, VLMUnavailable
from generation.vlm_server import MAX_IMAGES, create_app

TOKEN = "t" * 40


@pytest.fixture
def setup():
    model = ScriptedVLM("réponse du modèle", name="qwen-test")
    http = TestClient(create_app(model, TOKEN))
    return model, http, RemoteVLM("http://testserver", TOKEN, client=http)


def b64_png():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 200, 10)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_jeton_court_refuse():
    with pytest.raises(ValueError):
        create_app(ScriptedVLM("x"), "court")


def test_aller_retour_avec_images(setup):
    model, _, remote = setup
    text = remote.generate("système", "question", [Image.new("RGB", (64, 48)), Image.new("RGB", (20, 20))])
    assert text == "réponse du modèle"
    call = model.calls[0]
    assert call["system"] == "système" and call["user"] == "question" and call["n_images"] == 2
    assert call["image_pixels"] == [64 * 48, 20 * 20]


@pytest.mark.parametrize("header", [None, "Bearer mauvais", f"Bearer {TOKEN}x", TOKEN])
def test_serveur_refuse_sans_le_bon_jeton(setup, header):
    model, http, _ = setup
    headers = {"Authorization": header} if header else {}
    assert http.post("/generate", json={"system": "s", "user": "u"}, headers=headers).status_code == 401
    assert http.get("/health", headers=headers).status_code == 401
    assert model.calls == []


def test_client_signale_un_jeton_refuse(setup):
    _, http, _ = setup
    with pytest.raises(VLMUnavailable, match="Jeton refusé"):
        RemoteVLM("http://testserver", "mauvais" * 6, client=http).generate("s", "u", [])


def test_trop_d_images_ou_image_invalide(setup):
    _, http, _ = setup
    auth = {"Authorization": f"Bearer {TOKEN}"}
    trop = http.post("/generate", json={"system": "s", "user": "u", "images": [b64_png()] * (MAX_IMAGES + 1)}, headers=auth)
    assert trop.status_code == 422
    invalide = http.post("/generate", json={"system": "s", "user": "u", "images": ["pas-du-base64!"]}, headers=auth)
    assert invalide.status_code == 422


def test_serveur_injoignable():
    remote = RemoteVLM("http://127.0.0.1:9", TOKEN, timeout=2)   # port fermé
    with pytest.raises(VLMUnavailable, match="injoignable"):
        remote.generate("s", "u", [])


def test_api_ask_renvoie_503_si_kaggle_injoignable(monkeypatch):
    from api.main import app, get_search_fn, get_vlm
    from tests.test_api import FAKE, auth

    app.dependency_overrides[get_search_fn] = lambda: (lambda user, question, photo=None: FAKE)
    app.dependency_overrides[get_vlm] = lambda: RemoteVLM("http://127.0.0.1:9", TOKEN, timeout=2)
    try:
        r = TestClient(app).post("/ask", headers=auth("tech_s01"), data={"question": "voyant orange ?"})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 503 and "injoignable" in r.json()["detail"]


def test_vrai_serveur_http_en_arriere_plan():
    import socket

    from generation.vlm_server import serve_in_background

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    serve_in_background(create_app(ScriptedVLM("via HTTP"), TOKEN), port=port)
    assert RemoteVLM(f"http://127.0.0.1:{port}", TOKEN, timeout=10).generate("s", "u", [Image.new("RGB", (8, 8))]) == "via HTTP"
