"""Stockage des fichiers (S3 / SeaweedFS) et des métadonnées (PostgreSQL)."""

from functools import lru_cache
from pathlib import Path

import boto3
import psycopg
from botocore.config import Config
from psycopg.types.json import Jsonb

from common.config import get_settings
from ingestion.schema import Element

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id       TEXT PRIMARY KEY,
    version           TEXT NOT NULL,
    titre             TEXT NOT NULL,
    famille           TEXT NOT NULL,
    modele            TEXT,
    doc_type          TEXT NOT NULL,
    groupes_autorises TEXT[] NOT NULL,
    s3_key            TEXT,
    ingested_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS elements (
    uid               TEXT PRIMARY KEY,
    document_id       TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    version           TEXT NOT NULL,
    page              INTEGER NOT NULL,
    element_id        TEXT NOT NULL,
    type              TEXT NOT NULL,
    source            TEXT NOT NULL,
    section           TEXT,
    text              TEXT,
    s3_key            TEXT,
    bbox              JSONB,
    groupes_autorises TEXT[] NOT NULL
);
CREATE INDEX IF NOT EXISTS elements_document_idx ON elements (document_id);
"""


@lru_cache
def s3_client():
    s = get_settings()
    return boto3.client("s3", endpoint_url=s.s3_endpoint, aws_access_key_id=s.s3_access_key,
                        aws_secret_access_key=s.s3_secret_key.get_secret_value(), region_name="us-east-1",
                        config=Config(retries={"max_attempts": 3}, connect_timeout=5))


def ensure_bucket() -> None:
    s3, bucket = s3_client(), get_settings().s3_bucket
    if bucket not in [b["Name"] for b in s3.list_buckets()["Buckets"]]:
        s3.create_bucket(Bucket=bucket)


def upload_files(pdf_path: Path, meta: dict, elements: list[Element]) -> str:
    """Envoie le PDF source et les images des éléments. Renseigne s3_key sur chaque élément."""
    s = get_settings()
    s3 = s3_client()
    prefix = f"{meta['document_id']}/{meta['version']}"
    pdf_key = f"{prefix}/source.pdf"
    s3.upload_file(str(pdf_path), s.s3_bucket, pdf_key)
    for el in elements:
        if el.image_path:
            local = s.data_dir / el.image_path
            el.s3_key = f"{prefix}/{local.name}"
            s3.upload_file(str(local), s.s3_bucket, el.s3_key)
    return pdf_key


def connect() -> psycopg.Connection:
    # autocommit : chaque bloc con.transaction() est alors une vraie transaction, validée à sa sortie.
    con = psycopg.connect(get_settings().postgres_dsn, connect_timeout=5, autocommit=True)
    con.execute(SCHEMA)
    return con


def save_document(con: psycopg.Connection, meta: dict, elements: list[Element], pdf_key: str | None) -> None:
    """Remplace le document et tous ses éléments (ré-ingestion idempotente)."""
    with con.transaction():
        con.execute("DELETE FROM documents WHERE document_id = %s", (meta["document_id"],))
        con.execute(
            "INSERT INTO documents (document_id, version, titre, famille, modele, doc_type, groupes_autorises, s3_key)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (meta["document_id"], meta["version"], meta["titre"], meta["famille"], meta.get("modele"),
             meta["type"], meta["groupes_autorises"], pdf_key))
        with con.cursor() as cur:
            cur.executemany(
                "INSERT INTO elements (uid, document_id, version, page, element_id, type, source, section, text,"
                " s3_key, bbox, groupes_autorises) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [(e.uid, e.document_id, e.version, e.page, e.element_id, e.type, e.source, e.section, e.text,
                  e.s3_key, Jsonb(e.bbox) if e.bbox else None, e.groupes_autorises) for e in elements])
