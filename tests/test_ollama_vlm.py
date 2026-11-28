import base64
import io
import json
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from generation.vlm import OllamaVLM, VLMUnavailable


def test_ollama_chat_sends_prompt_and_images():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={
            "message": {"content": "réponse Ollama"},
            "prompt_eval_count": 42,
        })

    client = httpx.Client(transport=httpx.MockTransport(respond))
    vlm = OllamaVLM("http://localhost:11434/", "qwen3-vl:8b", client=client)
    image = Image.new("RGB", (8, 8), (10, 20, 30))

    assert vlm.generate("instructions", "question", [image]) == "réponse Ollama"
    body = json.loads(requests[0].content)
    decoded_image = Image.open(io.BytesIO(base64.b64decode(body["messages"][1]["images"][0])))

    assert requests[0].url == "http://localhost:11434/api/chat"
    assert body["model"] == "qwen3-vl:8b"
    assert body["messages"][0] == {"role": "system", "content": "instructions"}
    assert body["messages"][1]["role"] == "user"
    assert body["messages"][1]["content"] == "question"
    assert decoded_image.size == (8, 8)
    assert body["stream"] is False
    assert body["options"]["num_ctx"] == 8192
    assert vlm.token_log == [42]


def test_ollama_missing_model_has_actionable_error():
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(404, json={"error": "model not found"}),
    ))
    vlm = OllamaVLM("http://localhost:11434", "qwen3-vl:8b", client=client)

    with pytest.raises(VLMUnavailable, match="ollama pull qwen3-vl:8b"):
        vlm.generate("s", "u", [])


def test_ollama_http_error_includes_server_detail():
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(400, json={"error": "context length exceeded"}),
    ))
    vlm = OllamaVLM("http://localhost:11434", "qwen3-vl:8b", client=client)

    with pytest.raises(VLMUnavailable, match="HTTP 400.*context length exceeded"):
        vlm.generate("s", "u", [])


def test_api_selects_ollama_backend(monkeypatch):
    from api.main import get_vlm

    expected = OllamaVLM("http://localhost:11434", "qwen3-vl:8b")
    monkeypatch.setattr("api.main.get_settings", lambda: SimpleNamespace(vlm_backend="ollama"))
    monkeypatch.setattr("api.main._ollama_vlm", lambda: expected)

    assert get_vlm() is expected
