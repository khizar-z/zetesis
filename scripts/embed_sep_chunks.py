#!/usr/bin/env python3
"""Generate sentence-transformer embeddings for SEP chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_BATCH_SIZE = 64
DEFAULT_PROGRESS_INTERVAL = 500
DEFAULT_ENCODE_BLOCK_SIZE = 1024
EXPECTED_DIMENSION = 768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed SEP chunks with all-mpnet-base-v2 and save to .npy.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("sep_chunks.json"),
        help="Path to the chunk JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sep_embeddings.npy"),
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
        help="Print progress every N chunks.",
    )
    parser.add_argument(
        "--encode-block-size",
        type=int,
        default=DEFAULT_ENCODE_BLOCK_SIZE,
        help=(
            "How many texts to hand to SentenceTransformer.encode at once. "
            "The model still uses --batch-size internally."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of chunks, useful for smoke tests.",
    )
    return parser.parse_args()


def load_chunks(path: Path, limit: int | None = None) -> list[dict[str, object]]:
    chunks = json.loads(path.read_text(encoding="utf-8"))
    if limit is not None:
        return chunks[:limit]
    return chunks


def validate_batch_size(batch_size: int) -> None:
    if batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")


def validate_progress_interval(progress_interval: int) -> None:
    if progress_interval <= 0:
        raise ValueError("--progress-interval must be a positive integer.")


def validate_encode_block_size(encode_block_size: int, batch_size: int) -> None:
    if encode_block_size <= 0:
        raise ValueError("--encode-block-size must be a positive integer.")
    if encode_block_size < batch_size:
        raise ValueError("--encode-block-size must be greater than or equal to --batch-size.")


def main() -> None:
    args = parse_args()
    validate_batch_size(args.batch_size)
    validate_progress_interval(args.progress_interval)
    validate_encode_block_size(args.encode_block_size, args.batch_size)

    chunks = load_chunks(args.input, args.limit)
    total_chunks = len(chunks)
    if total_chunks == 0:
        raise ValueError(f"No chunks found in {args.input}")

    print(f"Loading {total_chunks} chunks from {args.input}")
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

    texts = [str(chunk["chunk_text"]) for chunk in chunks]
    embeddings = np.empty((total_chunks, embedding_dimension), dtype=np.float32)

    next_progress_mark = args.progress_interval
    for start in range(0, total_chunks, args.encode_block_size):
        end = min(start + args.encode_block_size, total_chunks)
        batch_embeddings = model.encode(
            texts[start:end],
            batch_size=args.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        embeddings[start:end] = batch_embeddings.astype(np.float32, copy=False)

        while end >= next_progress_mark:
            print(f"Embedded {next_progress_mark}/{total_chunks} chunks")
            next_progress_mark += args.progress_interval

    np.save(args.output, embeddings)
    print(f"Saved embeddings with shape {embeddings.shape} to {args.output}")


if __name__ == "__main__":
    main()
