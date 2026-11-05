"""Vérifie que les services Docker répondent avec les identifiants du fichier .env.

Usage : python scripts/check_services.py
"""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import get_settings  # noqa: E402


def check_http(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as r:
        return f"HTTP {r.status}"


def check_postgres() -> str:
    import psycopg
    with psycopg.connect(get_settings().postgres_dsn, connect_timeout=5) as con:
        return con.execute("SELECT version()").fetchone()[0].split(",")[0]


def check_s3() -> str:
    import boto3
    from botocore.config import Config
    s = get_settings()
    s3 = boto3.client("s3", endpoint_url=s.s3_endpoint, aws_access_key_id=s.s3_access_key,
                      aws_secret_access_key=s.s3_secret_key.get_secret_value(), region_name="us-east-1",
                      config=Config(retries={"max_attempts": 1}, connect_timeout=5))
    buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]
    if s.s3_bucket not in buckets:
        s3.create_bucket(Bucket=s.s3_bucket)
        return f"bucket '{s.s3_bucket}' créé"
    return f"bucket '{s.s3_bucket}' présent"


def main() -> int:
    s = get_settings()
    checks = {
        "Qdrant": lambda: check_http(f"{s.qdrant_url}/readyz"),
        "PostgreSQL": check_postgres,
        "Stockage S3": check_s3,
        "MLflow": lambda: check_http(f"{s.mlflow_tracking_uri}/health"),
    }
    ok = True
    for name, fn in checks.items():
        try:
            print(f"[OK]     {name:<12} {fn()}")
        except Exception as e:  # noqa: BLE001 - on veut afficher toute erreur de connexion
            ok = False
            print(f"[ERREUR] {name:<12} {type(e).__name__}: {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
