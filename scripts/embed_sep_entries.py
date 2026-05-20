#!/usr/bin/env python3
"""Generate sentence-transformer embeddings for SEP entry search documents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_BATCH_SIZE = 64
DEFAULT_PROGRESS_INTERVAL = 100
EXPECTED_DIMENSION = 768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed SEP entry documents with all-mpnet-base-v2 and save to .npy.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("sep_entries.json"),
        help="Path to the entry JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sep_entry_embeddings.npy"),
        help="Path to save the NumPy embedding matrix.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Sentence-transformers model name.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override, for example 'cpu' or 'mps'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for embedding generation.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=DEFAULT_PROGRESS_INTERVAL,
        help="Print progress every N entries.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of entries, useful for smoke tests.",
    )
    return parser.parse_args()


def load_entries(path: Path, limit: int | None = None) -> list[dict[str, object]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    if limit is not None:
        return entries[:limit]
    return entries


def validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def main() -> None:
    args = parse_args()
    validate_positive("--batch-size", args.batch_size)
    validate_positive("--progress-interval", args.progress_interval)

    entries = load_entries(args.input, args.limit)
    total_entries = len(entries)
    if total_entries == 0:
        raise ValueError(f"No entries found in {args.input}")

    print(f"Loading {total_entries} entries from {args.input}")
    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model, device=args.device)
    print(f"Using device: {model.device}")

    if hasattr(model, "get_embedding_dimension"):
        embedding_dimension = model.get_embedding_dimension()
    else:
        embedding_dimension = model.get_sentence_embedding_dimension()
    if embedding_dimension != EXPECTED_DIMENSION:
        raise ValueError(
            f"Expected embedding dimension {EXPECTED_DIMENSION}, "
            f"but model returned {embedding_dimension}."
        )

    texts = [str(entry["search_text"]) for entry in entries]
    embeddings = np.empty((total_entries, embedding_dimension), dtype=np.float32)

    next_progress_mark = args.progress_interval
    for start in range(0, total_entries, args.batch_size):
        end = min(start + args.batch_size, total_entries)
        batch_embeddings = model.encode(
            texts[start:end],
            batch_size=args.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        embeddings[start:end] = batch_embeddings.astype(np.float32, copy=False)

        while end >= next_progress_mark:
            print(f"Embedded {next_progress_mark}/{total_entries} entries")
            next_progress_mark += args.progress_interval

    np.save(args.output, embeddings)
    print(f"Saved embeddings with shape {embeddings.shape} to {args.output}")


if __name__ == "__main__":
    main()
