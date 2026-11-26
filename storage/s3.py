"""Stockage des fichiers (PDF sources, figures, pages) dans MinIO via l'API S3."""

from functools import lru_cache
from pathlib import Path

import boto3
from botocore.config import Config

from common.config import get_settings
from preparation.schema import Element


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


def clear_bucket() -> int:
    """Supprime tous les fichiers du bucket (changement de corpus). Renvoie le nombre de fichiers supprimés."""
    s3, bucket, n = s3_client(), get_settings().s3_bucket, 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if keys:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": keys})
            n += len(keys)
    return n


def read_file(key: str) -> bytes:
    """Contenu d'un fichier du bucket."""
    return s3_client().get_object(Bucket=get_settings().s3_bucket, Key=key)["Body"].read()
