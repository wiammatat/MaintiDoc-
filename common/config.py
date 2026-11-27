"""Configuration centrale, lue depuis les variables d'environnement ou le fichier .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

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
    image_embedding_model: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1"  # texte -> espace CLIP
    image_encoder_model: str = "sentence-transformers/clip-ViT-B-32"                     # image -> espace CLIP
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Précision des modèles d'embeddings et du reranker. float32 par défaut : sur un CPU sans
    # instructions bfloat16 natives (ex. Intel i5 de 12e génération), bfloat16 est émulé et
    # des dizaines de fois plus lent (voir docs/choix_techniques.md). bfloat16 reste utile sur GPU.
    model_dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    device: str = "cpu"   # modèles de recherche : "cuda" si la carte a assez de mémoire, sinon "cpu"

    # VLM (exécuté sur un GPU NVIDIA)
    vlm_quantization: Literal["none", "4bit"] = "4bit"   # 7B en fp16 dépasse 16 Go : 4 bits sur un seul T4
    vlm_max_new_tokens: int = 512
    vlm_max_pixels: int = 384 * 28 * 28   # images récupérées : ~300 000 pixels, ~384 tokens visuels au plus
    vlm_photo_max_pixels: int = 512 * 28 * 28   # photo de l'utilisateur, plus importante : ~400 000 pixels
    vlm_max_passages: int = 4             # passages transmis au VLM (borne la longueur du prompt)
    vlm_gpu0_max_gb: float = 7.0          # avec 2 GPU : poids du VLM limités sur le GPU 0, qui accueille aussi la recherche (~2,8 Go)
    vlm_max_images: int = 2               # photo de l'utilisateur + images récupérées (mémoire GPU sur T4)
    answer_prompt: str = "answer_v6"
    evidence_prompt: str = "evidence_check_v2"

    # Index vectoriel (le numéro de version est porté par le nom des collections)
    index_version: str = "v1"
    qdrant_path: Path | None = None   # si défini : Qdrant embarqué sur disque, sans serveur (profil léger)

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
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "maintidoc"
    s3_secret_key: SecretStr = SecretStr("")
    s3_bucket: str = "maintidoc"
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Sécurité (SecretStr : jamais affiché dans les journaux)
    approval_secret: SecretStr = SecretStr("")
    jwt_secret: SecretStr = SecretStr("")    # signature des jetons de connexion de l'API
    jwt_ttl_minutes: int = 60

    # Budgets par demande
    max_mcp_calls: int = 3
    max_model_calls: int = 6
    max_processing_seconds: int = 60
    agent_max_search_rounds: int = 2   # boucle contrôlée : recherches au plus par demande (1re + relances)

    # Stockage des fichiers et des métadonnées : docker (MinIO + PostgreSQL) ou local (disque, sans Docker)
    storage_backend: Literal["docker", "local"] = "docker"

    # API
    vlm_backend: Literal["disabled", "qwen", "remote"] = "disabled"   # qwen : GPU de ce PC ; remote : serveur VLM d'un autre PC
    vlm_remote_url: str = ""                  # adresse affichée par python -m generation.vlm_server
    vlm_remote_token: SecretStr = SecretStr("")   # jeton affiché par python -m generation.vlm_server
    vlm_remote_timeout: int = 180             # secondes par génération
    max_upload_mb: int = 5                                    # taille maximale d'une photo envoyée

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / "corpus"

    @property
    def text_collection(self) -> str:
        return f"maintidoc_text_{self.index_version}"

    @property
    def image_collection(self) -> str:
        return f"maintidoc_images_{self.index_version}"

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
