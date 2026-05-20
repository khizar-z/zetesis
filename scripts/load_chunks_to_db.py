#!/usr/bin/env python3
"""Load chunk metadata and embeddings into Postgres with pgvector."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from text_repair import fix_mojibake

EXPECTED_DIMENSION = 768
DEFAULT_BATCH_SIZE = 250
DEFAULT_PROGRESS_INTERVAL = 500
SCHEMA_PATH = Path("db/init/01-schema.sql")
INDEX_NAME = "chunks_embedding_hnsw_idx"
CREATE_INDEX_SQL = f"""
CREATE INDEX IF NOT EXISTS {INDEX_NAME}
ON chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
""".strip()
DROP_INDEX_SQL = f"DROP INDEX IF EXISTS {INDEX_NAME};"
UPSERT_SQL = """
INSERT INTO chunks (
    id,
    entry_slug,
    entry_title,
    section_number,
    section_title,
    url,
    chunk_text,
    chunk_index,
    embedding
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE SET
    entry_slug = EXCLUDED.entry_slug,
    entry_title = EXCLUDED.entry_title,
    section_number = EXCLUDED.section_number,
    section_title = EXCLUDED.section_title,
    url = EXCLUDED.url,
    chunk_text = EXCLUDED.chunk_text,
    chunk_index = EXCLUDED.chunk_index,
    embedding = EXCLUDED.embedding;
""".strip()
REQUIRED_CHUNK_FIELDS = {
    "chunk_id",
    "entry_slug",
    "entry_title",
    "section_number",
    "section_title",
    "url",
    "chunk_text",
    "chunk_index",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load Zetesis chunks and embeddings into a pgvector-enabled Postgres database.",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("sep_chunks.json"),
        help="Path to sep_chunks.json.",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("sep_embeddings.npy"),
        help="Path to sep_embeddings.npy.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of rows to upsert per database batch.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=DEFAULT_PROGRESS_INTERVAL,
        help="Print progress every N inserted rows.",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate the chunks table before loading.",
    )
    parser.add_argument(
        "--drop-index-first",
        action="store_true",
        help="Drop the HNSW index before loading, then recreate it afterwards.",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip creating the HNSW index after loading rows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files and configuration without connecting to the database.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of rows to load, useful for smoke tests.",
    )
    return parser.parse_args()


def validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def resolve_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url

    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB")

    if user and password and database:
        quoted_password = quote_plus(password)
        return f"postgresql://{user}:{quoted_password}@{host}:{port}/{database}"

    raise ValueError(
        "DATABASE_URL is not set. Add it to your .env file or provide the POSTGRES_* variables."
    )


def load_chunks(path: Path, limit: int | None = None) -> list[dict[str, object]]:
    chunks = json.loads(path.read_text(encoding="utf-8"))
    if limit is not None:
        return chunks[:limit]
    return chunks


def load_embeddings(path: Path, limit: int | None = None) -> np.ndarray:
    embeddings = np.load(path, mmap_mode="r")
    if limit is not None:
        return embeddings[:limit]
    return embeddings


def validate_inputs(chunks: list[dict[str, object]], embeddings: np.ndarray) -> None:
    if not chunks:
        raise ValueError("No chunk records found.")

    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {embeddings.shape}.")

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"Chunk count ({len(chunks)}) does not match embedding rows ({embeddings.shape[0]})."
        )

    if embeddings.shape[1] != EXPECTED_DIMENSION:
        raise ValueError(
            f"Expected embedding dimension {EXPECTED_DIMENSION}, got {embeddings.shape[1]}."
        )

    for index, chunk in enumerate(chunks):
        missing = REQUIRED_CHUNK_FIELDS - chunk.keys()
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Chunk at index {index} is missing required fields: {missing_list}")


def load_schema_sql() -> str:
    return SCHEMA_PATH.read_text(encoding="utf-8")


def execute_sql_script(cur: psycopg.Cursor[object], sql_text: str) -> None:
    statements = [statement.strip() for statement in sql_text.split(";") if statement.strip()]
    for statement in statements:
        cur.execute(statement)


def chunk_rows(
    chunks: list[dict[str, object]],
    embeddings: np.ndarray,
    start: int,
    end: int,
) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for index in range(start, end):
        chunk = chunks[index]
        embedding = embeddings[index]
        rows.append(
            (
                uuid.UUID(str(chunk["chunk_id"])),
                fix_mojibake(str(chunk["entry_slug"])),
                fix_mojibake(str(chunk["entry_title"])),
                fix_mojibake(str(chunk["section_number"])),
                fix_mojibake(str(chunk["section_title"])),
                str(chunk["url"]),
                fix_mojibake(str(chunk["chunk_text"])),
                int(chunk["chunk_index"]),
                np.asarray(embedding, dtype=np.float32),
            )
        )
    return rows


def main() -> None:
    load_dotenv()
    args = parse_args()
    validate_positive("--batch-size", args.batch_size)
    validate_positive("--progress-interval", args.progress_interval)

    chunks = load_chunks(args.chunks, args.limit)
    embeddings = load_embeddings(args.embeddings, args.limit)
    validate_inputs(chunks, embeddings)

    total_rows = len(chunks)
    print(f"Validated {total_rows} rows from {args.chunks} and {args.embeddings}")

    if args.dry_run:
        print("Dry run enabled; skipping database connection.")
        return

    database_url = resolve_database_url()
    schema_sql = load_schema_sql()

    with psycopg.connect(database_url) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        register_vector(conn)

        with conn.cursor() as cur:
            execute_sql_script(cur, schema_sql)

            if args.truncate:
                cur.execute("TRUNCATE TABLE chunks;")
                print("Truncated chunks table.")

            if args.drop_index_first:
                cur.execute(DROP_INDEX_SQL)
                print(f"Dropped index {INDEX_NAME} before loading.")

            conn.commit()

            next_progress_mark = args.progress_interval
            for start in range(0, total_rows, args.batch_size):
                end = min(start + args.batch_size, total_rows)
                rows = chunk_rows(chunks, embeddings, start, end)
                cur.executemany(UPSERT_SQL, rows)
                conn.commit()

                while end >= next_progress_mark:
                    print(f"Loaded {next_progress_mark}/{total_rows} rows")
                    next_progress_mark += args.progress_interval

            print(f"Finished loading {total_rows} rows into chunks.")

            if not args.skip_index:
                print(f"Creating HNSW index {INDEX_NAME}...")
                cur.execute(CREATE_INDEX_SQL)
                conn.commit()
                print(f"Created HNSW index {INDEX_NAME}.")


if __name__ == "__main__":
    main()
