"""Configuration centrale, lue depuis les variables d'environnement ou le fichier .env."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # Chemins
    data_dir: Path = ROOT / "data"
    prompts_dir: Path = ROOT / "prompts"

    # Modèles
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    text_embedding_model: str = "BAAI/bge-m3"
    image_embedding_model: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # OCR
    tesseract_cmd: str = "tesseract"
    tessdata_prefix: Path = ROOT / "models" / "tessdata"
    ocr_lang: str = "fra"

    # Services
    qdrant_url: str = "http://localhost:6333"
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_db: str = "maintidoc"
    postgres_user: str = "maintidoc"
    postgres_password: SecretStr = SecretStr("")
    s3_endpoint: str = "http://localhost:8333"
    s3_access_key: str = "maintidoc"
    s3_secret_key: SecretStr = SecretStr("")
    s3_bucket: str = "maintidoc"
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Sécurité (SecretStr : jamais affiché dans les journaux)
    approval_secret: SecretStr = SecretStr("")

    # Budgets par demande
    max_mcp_calls: int = 3
    max_model_calls: int = 6
    max_processing_seconds: int = 60

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / "corpus"

    @property
    def tessdata_dir(self) -> Path:
        """Chemin absolu du dossier tessdata (relatif à la racine du projet si besoin)."""
        p = self.tessdata_prefix
        return p if p.is_absolute() else ROOT / p

    @property
    def postgres_dsn(self) -> str:
        return (f"postgresql://{self.postgres_user}:{self.postgres_password.get_secret_value()}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}")


@lru_cache
def get_settings() -> Settings:
    return Settings()
