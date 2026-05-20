#!/usr/bin/env python3
"""Build entry graph tables from SEP sections, chunks, and embeddings."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections import OrderedDict
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import psycopg
from build_sep_entries import DEFAULT_MAX_INTRO_WORDS, aggregate_entries
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from text_repair import fix_mojibake

EXPECTED_DIMENSION = 768
DEFAULT_BATCH_SIZE = 500
DEFAULT_PROGRESS_INTERVAL = 250
DEFAULT_SEMANTIC_THRESHOLD = 0.85
DEFAULT_MAX_SEMANTIC_NEIGHBORS = 5
SCHEMA_PATH = Path("db/init/01-schema.sql")

ENTRY_UPSERT_SQL = """
INSERT INTO entries (
    id,
    entry_slug,
    entry_title,
    url,
    section_count,
    section_outline,
    intro_text,
    search_text,
    embedding,
    graph_embedding
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (entry_slug) DO UPDATE SET
    entry_title = EXCLUDED.entry_title,
    url = EXCLUDED.url,
    section_count = EXCLUDED.section_count,
    section_outline = EXCLUDED.section_outline,
    intro_text = EXCLUDED.intro_text,
    search_text = EXCLUDED.search_text,
    graph_embedding = EXCLUDED.graph_embedding;
""".strip()

EXPLICIT_EDGE_UPSERT_SQL = """
INSERT INTO explicit_edges (source_slug, target_slug, weight)
VALUES (%s, %s, %s)
ON CONFLICT (source_slug, target_slug) DO UPDATE SET
    weight = EXCLUDED.weight;
""".strip()

SEMANTIC_EDGE_UPSERT_SQL = """
INSERT INTO semantic_edges (source_slug, target_slug, similarity)
VALUES (%s, %s, %s)
ON CONFLICT (source_slug, target_slug) DO UPDATE SET
    similarity = EXCLUDED.similarity;
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Zetesis concept graph by computing entry-level graph "
            "embeddings and populating explicit and semantic edge tables."
        ),
    )
    parser.add_argument(
        "--sections",
        type=Path,
        default=Path("sep_sections.json"),
        help="Path to the section-level SEP JSON file.",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("sep_chunks.json"),
        help="Path to the chunk-level SEP JSON file.",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("sep_embeddings.npy"),
        help="Path to the chunk embedding matrix.",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=DEFAULT_SEMANTIC_THRESHOLD,
        help="Keep semantic edges whose cosine similarity is strictly above this value.",
    )
    parser.add_argument(
        "--max-semantic-neighbors",
        type=int,
        default=DEFAULT_MAX_SEMANTIC_NEIGHBORS,
        help=(
            "Cap semantic edges to each entry's top N neighbors by similarity. "
            "The final graph keeps only mutual top-N pairs, which is stricter than "
            "a one-sided cap."
        ),
    )
    parser.add_argument(
        "--max-intro-words",
        type=int,
        default=DEFAULT_MAX_INTRO_WORDS,
        help="Maximum number of overview words when synthesizing entry metadata.",
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
        help="Print semantic-edge progress every N source entries.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and compute graph counts without writing to Postgres.",
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


def load_json_records(path: Path) -> list[dict[str, object]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected {path} to contain a JSON list.")
    if not records:
        raise ValueError(f"No records found in {path}.")
    return records


def load_embeddings(path: Path) -> np.ndarray:
    embeddings = np.load(path, mmap_mode="r")
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {embeddings.shape}.")
    if embeddings.shape[1] != EXPECTED_DIMENSION:
        raise ValueError(
            f"Expected embedding dimension {EXPECTED_DIMENSION}, got {embeddings.shape[1]}."
        )
    return embeddings


def validate_sections_have_graph_links(sections: list[dict[str, object]]) -> None:
    first = sections[0]
    if "outbound_links" in first or "outbound_link_counts" in first:
        return

    raise ValueError(
        "sep_sections.json does not contain outbound graph link metadata yet. "
        "Regenerate it with scripts/scrape_sep.py before running build_graph.py."
    )


def aggregate_chunk_embeddings(
    chunks: list[dict[str, object]],
    embeddings: np.ndarray,
) -> dict[str, np.ndarray]:
    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"Chunk count ({len(chunks)}) does not match embedding rows ({embeddings.shape[0]})."
        )

    totals: OrderedDict[str, np.ndarray] = OrderedDict()
    counts: dict[str, int] = {}

    for index, chunk in enumerate(chunks):
        slug = fix_mojibake(str(chunk["entry_slug"]))
        vector = np.asarray(embeddings[index], dtype=np.float64)

        if slug not in totals:
            totals[slug] = np.zeros(EXPECTED_DIMENSION, dtype=np.float64)
            counts[slug] = 0

        totals[slug] += vector
        counts[slug] += 1

    return {
        slug: (total / counts[slug]).astype(np.float32)
        for slug, total in totals.items()
    }


def build_entry_metadata(
    sections: list[dict[str, object]],
    *,
    max_intro_words: int,
) -> OrderedDict[str, dict[str, object]]:
    aggregated_entries = aggregate_entries(sections, max_intro_words=max_intro_words)
    return OrderedDict(
        (fix_mojibake(str(entry["entry_slug"])), entry) for entry in aggregated_entries
    )


def build_explicit_edges(
    sections: list[dict[str, object]],
    *,
    valid_slugs: set[str],
) -> tuple[list[tuple[str, str, int]], int]:
    outbound_by_entry: OrderedDict[str, dict[str, int]] = OrderedDict()

    for section in sections:
        source_slug = fix_mojibake(str(section["entry_slug"]))
        if source_slug in outbound_by_entry:
            continue

        if "outbound_link_counts" in section:
            raw_counts = dict(section["outbound_link_counts"])  # type: ignore[arg-type]
            counts = {
                fix_mojibake(str(target_slug)): int(weight)
                for target_slug, weight in raw_counts.items()
                if fix_mojibake(str(target_slug)) != source_slug
            }
        else:
            raw_links = list(section.get("outbound_links", []))
            counts = {
                fix_mojibake(str(target_slug)): 1
                for target_slug in raw_links
                if fix_mojibake(str(target_slug)) != source_slug
            }

        outbound_by_entry[source_slug] = counts

    explicit_edges: list[tuple[str, str, int]] = []
    skipped_missing_targets = 0

    for source_slug, target_counts in outbound_by_entry.items():
        if source_slug not in valid_slugs:
            continue

        for target_slug, weight in target_counts.items():
            if target_slug not in valid_slugs:
                skipped_missing_targets += 1
                continue
            explicit_edges.append((source_slug, target_slug, weight))

    return explicit_edges, skipped_missing_targets


def normalize_embeddings(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def build_semantic_edges(
    entry_embeddings: dict[str, np.ndarray],
    *,
    similarity_threshold: float,
    progress_interval: int,
    max_semantic_neighbors: int,
) -> list[tuple[str, str, float]]:
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("--semantic-threshold must be between -1.0 and 1.0.")
    validate_positive("--max-semantic-neighbors", max_semantic_neighbors)

    slugs = sorted(entry_embeddings)
    matrix = np.stack([entry_embeddings[slug] for slug in slugs]).astype(np.float32)
    normalized = normalize_embeddings(matrix)
    similarity_matrix = normalized @ normalized.T

    top_neighbors_by_slug: dict[str, dict[str, float]] = {}
    total_sources = len(slugs)

    for source_index, source_slug in enumerate(slugs):
        if source_index > 0 and source_index % progress_interval == 0:
            print(
                f"Computed semantic similarities for {source_index}/{total_sources} source entries"
            )

        row = similarity_matrix[source_index]
        candidate_indices = np.flatnonzero(row > similarity_threshold)
        candidate_indices = candidate_indices[candidate_indices != source_index]

        if candidate_indices.size == 0:
            top_neighbors_by_slug[source_slug] = {}
            continue

        ranked_indices = candidate_indices[np.argsort(row[candidate_indices])[::-1]]
        selected_indices = ranked_indices[:max_semantic_neighbors]
        top_neighbors_by_slug[source_slug] = {
            slugs[target_index]: float(row[target_index]) for target_index in selected_indices
        }

    semantic_edges: list[tuple[str, str, float]] = []
    for source_slug in slugs:
        source_neighbors = top_neighbors_by_slug[source_slug]
        for target_slug, similarity in source_neighbors.items():
            if source_slug >= target_slug:
                continue

            if source_slug not in top_neighbors_by_slug.get(target_slug, {}):
                continue

            semantic_edges.append((source_slug, target_slug, similarity))

    return semantic_edges


def execute_sql_script(cur: psycopg.Cursor[object], sql_text: str) -> None:
    statements = [statement.strip() for statement in sql_text.split(";") if statement.strip()]
    for statement in statements:
        cur.execute(statement)


def entry_rows(
    entry_metadata: OrderedDict[str, dict[str, object]],
    entry_embeddings: dict[str, np.ndarray],
) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []

    for entry_slug, metadata in entry_metadata.items():
        embedding = entry_embeddings[entry_slug]
        rows.append(
            (
                uuid.UUID(str(metadata["entry_id"])),
                entry_slug,
                fix_mojibake(str(metadata["entry_title"])),
                str(metadata["url"]),
                int(metadata["section_count"]),
                fix_mojibake(str(metadata["section_outline"])),
                fix_mojibake(str(metadata["intro_text"])),
                fix_mojibake(str(metadata["search_text"])),
                np.asarray(embedding, dtype=np.float32),
                np.asarray(embedding, dtype=np.float32),
            )
        )

    return rows


def batched[T](items: list[T], batch_size: int) -> list[list[T]]:
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def main() -> None:
    load_dotenv()
    args = parse_args()
    validate_positive("--batch-size", args.batch_size)
    validate_positive("--progress-interval", args.progress_interval)
    validate_positive("--max-intro-words", args.max_intro_words)
    validate_positive("--max-semantic-neighbors", args.max_semantic_neighbors)

    sections = load_json_records(args.sections)
    validate_sections_have_graph_links(sections)

    chunks = load_json_records(args.chunks)
    embeddings = load_embeddings(args.embeddings)

    entry_metadata = build_entry_metadata(sections, max_intro_words=args.max_intro_words)
    entry_embeddings = aggregate_chunk_embeddings(chunks, embeddings)

    section_slugs = set(entry_metadata)
    chunk_slugs = set(entry_embeddings)
    missing_chunk_embeddings = sorted(section_slugs - chunk_slugs)
    missing_section_metadata = sorted(chunk_slugs - section_slugs)

    if missing_chunk_embeddings:
        sample = ", ".join(missing_chunk_embeddings[:10])
        print(
            f"Warning: skipping {len(missing_chunk_embeddings)} scraped entries with no "
            f"chunk embeddings yet. Sample: {sample}"
        )

    if missing_section_metadata:
        sample = ", ".join(missing_section_metadata[:10])
        print(
            f"Warning: skipping {len(missing_section_metadata)} chunk-derived entries with no "
            f"matching section metadata. Sample: {sample}"
        )

    common_slugs = section_slugs & chunk_slugs
    if not common_slugs:
        raise ValueError("No overlapping entries between section metadata and chunk embeddings.")

    entry_metadata = OrderedDict(
        (slug, metadata) for slug, metadata in entry_metadata.items() if slug in common_slugs
    )
    entry_embeddings = {
        slug: embedding for slug, embedding in entry_embeddings.items() if slug in common_slugs
    }

    explicit_edges, skipped_missing_targets = build_explicit_edges(
        sections,
        valid_slugs=common_slugs,
    )
    semantic_edges = build_semantic_edges(
        entry_embeddings,
        similarity_threshold=args.semantic_threshold,
        progress_interval=args.progress_interval,
        max_semantic_neighbors=args.max_semantic_neighbors,
    )

    print(f"Loaded {len(sections)} sections from {args.sections}")
    print(f"Loaded {len(chunks)} chunks from {args.chunks}")
    print(f"Computed graph embeddings for {len(entry_embeddings)} entries")
    print(f"Built {len(explicit_edges)} explicit edges")
    if skipped_missing_targets:
        print(f"Skipped {skipped_missing_targets} explicit edges with missing target entries")
    print(
        f"Built {len(semantic_edges)} semantic edges above similarity "
        f"{args.semantic_threshold:.3f} with mutual top-{args.max_semantic_neighbors} pruning"
    )

    if args.dry_run:
        print("Dry run enabled; skipping database writes.")
        return

    database_url = resolve_database_url()
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    entry_payload = entry_rows(entry_metadata, entry_embeddings)

    with psycopg.connect(database_url) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        register_vector(conn)

        with conn.cursor() as cur:
            execute_sql_script(cur, schema_sql)
            conn.commit()

            for batch in batched(entry_payload, args.batch_size):
                cur.executemany(ENTRY_UPSERT_SQL, batch)
            conn.commit()
            print(f"Upserted {len(entry_payload)} entries.")

            cur.execute("TRUNCATE TABLE explicit_edges, semantic_edges;")
            conn.commit()
            print("Truncated explicit_edges and semantic_edges.")

            for batch in batched(explicit_edges, args.batch_size):
                cur.executemany(EXPLICIT_EDGE_UPSERT_SQL, batch)
            conn.commit()
            print(f"Upserted {len(explicit_edges)} explicit edges.")

            for batch in batched(semantic_edges, args.batch_size):
                cur.executemany(SEMANTIC_EDGE_UPSERT_SQL, batch)
            conn.commit()
            print(f"Upserted {len(semantic_edges)} semantic edges.")


if __name__ == "__main__":
    main()
