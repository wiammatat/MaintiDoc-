"""Client HTTP de l'API MaintiDoc, utilisé par l'application Gradio.

L'interface ne décide de rien côté sécurité : authentification, droits sur les documents et les équipements,
garde-fous et approbation sont vérifiés par l'API et par le serveur MCP. Le jeton de connexion reste dans la
session Gradio (côté serveur) et n'apparaît jamais dans une URL.
"""

from dataclasses import dataclass

import httpx


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass
class Photo:
    name: str
    data: bytes
    mime: str = "image/jpeg"


def _message(r: httpx.Response) -> str:
    try:
        detail = r.json().get("detail")
    except ValueError:
        return r.text[:300] or f"Erreur HTTP {r.status_code}"
    if isinstance(detail, list):   # erreur de validation FastAPI
        return "; ".join(str(d.get("msg", d)) for d in detail)
    return str(detail or f"Erreur HTTP {r.status_code}")


class ApiClient:
    def __init__(self, base_url: str = "http://localhost:8000", token: str | None = None, timeout: float = 300,
                 http: httpx.Client | None = None):
        # http : client injecté pour les tests (TestClient de FastAPI)
        self.http = http or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self.token = token

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _request(self, method: str, path: str, **kw) -> httpx.Response:
        try:
            r = self.http.request(method, path, headers=self._headers(), **kw)
        except httpx.TimeoutException:
            raise ApiError(504, "L'API n'a pas répondu à temps.") from None
        except httpx.TransportError:
            raise ApiError(0, "API injoignable : lancer « uvicorn api.main:app ».") from None
        if r.status_code >= 400:
            raise ApiError(r.status_code, _message(r))
        return r

    @staticmethod
    def _files(photo: Photo | None) -> dict | None:
        return {"photo": (photo.name, photo.data, photo.mime)} if photo else None

    # ------------------------------------------------------------------ routes

    def health(self) -> dict:
        return self._request("GET", "/health").json()

    def login(self, user_id: str, password: str) -> dict:
        body = self._request("POST", "/auth/token", data={"username": user_id, "password": password}).json()
        self.token = body["access_token"]
        return self.me()

    def me(self) -> dict:
        return self._request("GET", "/me").json()

    def search(self, question: str, photo: Photo | None = None) -> dict:
        return self._request("POST", "/search", data={"question": question}, files=self._files(photo)).json()

    def agent(self, question: str, photo: Photo | None = None) -> dict:
        return self._request("POST", "/agent", data={"question": question}, files=self._files(photo)).json()

    def approve(self, thread_id: str, approved: bool, title: str | None = None, description: str | None = None) -> dict:
        body = {"approved": approved, "title": title, "description": description}
        return self._request("POST", f"/agent/{thread_id}/approval", json=body).json()

    def document_pdf(self, document_id: str) -> bytes:
        return self._request("GET", f"/documents/{document_id}/pdf").content

    def element_image(self, url: str) -> bytes:
        """url : chemin renvoyé par l'API (image_url d'une preuve), jamais construit à partir d'une saisie."""
        if not url.startswith("/elements/"):
            raise ApiError(400, "Adresse d'image inattendue")
        return self._request("GET", url).content
