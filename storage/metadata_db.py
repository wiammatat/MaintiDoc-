"""Stockage des métadonnées des documents et des éléments dans PostgreSQL."""

import psycopg
from psycopg.types.json import Jsonb

from common.config import get_settings
from preparation.schema import Element

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


def reset(con: psycopg.Connection) -> None:
    """Vide les tables (changement de corpus) ; les éléments sont supprimés en cascade."""
    con.execute("TRUNCATE documents CASCADE")


def get_document(con: psycopg.Connection, document_id: str) -> dict | None:
    """Clé du PDF source et groupes autorisés d'un document, ou None s'il n'existe pas."""
    row = con.execute("SELECT s3_key, groupes_autorises, titre FROM documents WHERE document_id = %s",
                      (document_id,)).fetchone()
    return dict(zip(("s3_key", "groupes_autorises", "titre"), row)) if row else None


def get_element(con: psycopg.Connection, uid: str) -> dict | None:
    """Clé de l'image et groupes autorisés d'un élément, ou None s'il n'existe pas."""
    row = con.execute("SELECT s3_key, groupes_autorises, type FROM elements WHERE uid = %s", (uid,)).fetchone()
    return dict(zip(("s3_key", "groupes_autorises", "type"), row)) if row else None
