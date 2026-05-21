#!/usr/bin/env python3
"""Semantic search pipeline for Zetesis."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from sentence_transformers import CrossEncoder, SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CANDIDATE_LIMIT = 40
DEFAULT_ENTRY_CANDIDATE_LIMIT = 40
DEFAULT_ENTRY_RESULT_LIMIT = 8
DEFAULT_ENTRY_CHUNK_LIMIT = 8
DEFAULT_RESULT_LIMIT = 7
DEFAULT_RERANK_BATCH_SIZE = 32
EXPECTED_DIMENSION = 768
RESULT_TO_CHUNK_CANDIDATE_RATIO = 8
RESULT_TO_ENTRY_CANDIDATE_RATIO = 6
SEARCH_CACHE_TTL_SECONDS = 600
SEARCH_CACHE_MAX_ENTRIES = 32
SEARCH_CACHE_PREFETCH_PAGES = 3

ENTRY_RERANK_OVERVIEW_WORDS = 220
ENTRY_RERANK_SECTION_LIMIT = 18

CHUNK_COSINE_WEIGHT = 0.75
CHUNK_ENTRY_SCORE_WEIGHT = 0.8
CHUNK_TITLE_BONUS_WEIGHT = 0.35
CHUNK_SECTION_BONUS_WEIGHT = 0.2

ENTRY_COSINE_WEIGHT = 1.0
ENTRY_TITLE_BONUS_WEIGHT = 1.0

LEXICAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "between",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "view",
    "what",
    "with",
}

CHUNK_RETRIEVE_SQL = """
SELECT
    id::text AS id,
    entry_slug,
    entry_title,
    section_title,
    url,
    chunk_text,
    1 - (embedding <=> %s) AS cosine_score
FROM chunks
ORDER BY embedding <=> %s
LIMIT %s;
""".strip()

ENTRY_RETRIEVE_SQL = """
SELECT
    id::text AS id,
    entry_slug,
    entry_title,
    url,
    section_outline,
    intro_text,
    search_text,
    1 - (embedding <=> %s) AS cosine_score
FROM entries
ORDER BY embedding <=> %s
LIMIT %s;
""".strip()

ENTRY_CHUNK_SQL = """
WITH ranked_chunks AS (
    SELECT
        id::text AS id,
        entry_slug,
        entry_title,
        section_title,
        url,
        chunk_text,
        1 - (embedding <=> %s) AS cosine_score,
        ROW_NUMBER() OVER (
            PARTITION BY entry_slug
            ORDER BY embedding <=> %s
        ) AS entry_rank
    FROM chunks
    WHERE entry_slug = ANY(%s)
)
SELECT
    id,
    entry_slug,
    entry_title,
    section_title,
    url,
    chunk_text,
    cosine_score
FROM ranked_chunks
WHERE entry_rank <= %s
ORDER BY cosine_score DESC
LIMIT %s;
""".strip()


@dataclass(frozen=True)
class SearchConfig:
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL
    reranker_model_name: str = DEFAULT_RERANKER_MODEL
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT
    entry_candidate_limit: int = DEFAULT_ENTRY_CANDIDATE_LIMIT
    entry_result_limit: int = DEFAULT_ENTRY_RESULT_LIMIT
    entry_chunk_limit: int = DEFAULT_ENTRY_CHUNK_LIMIT
    result_limit: int = DEFAULT_RESULT_LIMIT
    rerank_batch_size: int = DEFAULT_RERANK_BATCH_SIZE


@dataclass
class RetrievedChunk:
    id: str
    entry_slug: str
    entry_title: str
    section_title: str
    url: str
    chunk_text: str
    cosine_score: float


@dataclass
class RetrievedEntry:
    id: str
    entry_slug: str
    entry_title: str
    url: str
    section_outline: str
    intro_text: str
    search_text: str
    cosine_score: float


@dataclass
class ScoredEntry:
    candidate: RetrievedEntry
    reranker_score: float
    final_score: float


@dataclass
class SearchResult:
    entry_slug: str
    chunk_text: str
    entry_title: str
    section_title: str
    url: str
    cosine_score: float
    reranker_score: float
    final_score: float
    parent_entry_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_slug": self.entry_slug,
            "chunk_text": self.chunk_text,
            "entry_title": self.entry_title,
            "section_title": self.section_title,
            "url": self.url,
            "cosine_score": self.cosine_score,
            "reranker_score": self.reranker_score,
        }


@dataclass
class CachedSearchResults:
    results: list[dict[str, Any]]
    created_at: float
    requested_window: int
    is_exhaustive: bool


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


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def normalize_lexical_text(text: str) -> str:
    normalized = text.lower().replace("’", "'")
    normalized = re.sub(r"'s\b", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return normalized.strip()


def lexical_tokens(text: str, *, drop_stopwords: bool = False) -> list[str]:
    tokens = normalize_lexical_text(text).split()
    if not drop_stopwords:
        return tokens
    return [token for token in tokens if token not in LEXICAL_STOPWORDS]


def strip_leading_articles(tokens: list[str]) -> list[str]:
    stripped = list(tokens)
    while stripped and stripped[0] in {"a", "an", "the"}:
        stripped = stripped[1:]
    return stripped


def title_bonus(query: str, title: str) -> float:
    query_tokens = lexical_tokens(query, drop_stopwords=True)
    title_tokens = strip_leading_articles(lexical_tokens(title))
    if not query_tokens or not title_tokens:
        return 0.0

    query_text = " ".join(query_tokens)
    title_text = " ".join(title_tokens)
    query_token_set = set(query_tokens)
    title_token_set = set(title_tokens)
    shared_tokens = query_token_set & title_token_set

    bonus = 0.0
    if query_text == title_text:
        bonus += 2.5
    elif title_text and title_text in query_text:
        bonus += 1.5 if len(title_tokens) >= 2 else 1.0

    coverage = len(shared_tokens) / len(title_token_set)
    bonus += 0.8 * coverage

    if len(title_token_set) >= 2 and title_token_set.issubset(query_token_set):
        bonus += 0.6

    return bonus


def section_bonus(query: str, section_title: str) -> float:
    query_tokens = lexical_tokens(query, drop_stopwords=True)
    section_tokens = strip_leading_articles(lexical_tokens(section_title, drop_stopwords=True))
    if not query_tokens or not section_tokens:
        return 0.0

    query_text = " ".join(query_tokens)
    section_text = " ".join(section_tokens)
    query_token_set = set(query_tokens)
    section_token_set = set(section_tokens)
    shared_tokens = query_token_set & section_token_set

    bonus = 0.4 * (len(shared_tokens) / len(section_token_set))
    if len(section_tokens) >= 2 and section_text and section_text in query_text:
        bonus += 0.5
    return bonus


def format_chunk_for_rerank(candidate: RetrievedChunk) -> str:
    return (
        f"Entry: {candidate.entry_title}\n"
        f"Section: {candidate.section_title}\n"
        f"Passage: {candidate.chunk_text}"
    )


def format_entry_for_rerank(candidate: RetrievedEntry) -> str:
    section_lines = candidate.section_outline.splitlines()[:ENTRY_RERANK_SECTION_LIMIT]
    section_outline = "\n".join(section_lines).strip()
    intro_excerpt = truncate_words(candidate.intro_text, ENTRY_RERANK_OVERVIEW_WORDS)

    parts = [f"Entry: {candidate.entry_title}"]
    if section_outline:
        parts.append(f"Section headings:\n{section_outline}")
    if intro_excerpt:
        parts.append(f"Overview:\n{intro_excerpt}")
    return "\n\n".join(parts).strip()


def merge_chunk_candidates(*candidate_lists: list[RetrievedChunk]) -> list[RetrievedChunk]:
    merged: dict[str, RetrievedChunk] = {}
    for candidate_list in candidate_lists:
        for candidate in candidate_list:
            existing = merged.get(candidate.id)
            if existing is None or candidate.cosine_score > existing.cosine_score:
                merged[candidate.id] = candidate
    return list(merged.values())


class SearchService:
    """Embed a query, retrieve chunk and entry candidates, and rerank them."""

    def __init__(
        self,
        *,
        database_url: str,
        embedding_model: SentenceTransformer,
        reranker_model: CrossEncoder,
        config: SearchConfig | None = None,
    ) -> None:
        self.database_url = database_url
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.config = config or SearchConfig()
        self._result_cache: dict[tuple[str, int | None], CachedSearchResults] = {}

        validate_positive("candidate_limit", self.config.candidate_limit)
        validate_positive("entry_candidate_limit", self.config.entry_candidate_limit)
        validate_positive("entry_result_limit", self.config.entry_result_limit)
        validate_positive("entry_chunk_limit", self.config.entry_chunk_limit)
        validate_positive("result_limit", self.config.result_limit)
        validate_positive("rerank_batch_size", self.config.rerank_batch_size)

        embedding_dimension = self._embedding_dimension()
        if embedding_dimension != EXPECTED_DIMENSION:
            raise ValueError(
                f"Expected embedding dimension {EXPECTED_DIMENSION}, got {embedding_dimension}."
            )

    def _cache_key(self, query: str, candidate_limit: int | None) -> tuple[str, int | None]:
        return (query, candidate_limit)

    def _purge_expired_cache_entries(self) -> None:
        now = time.time()
        expired_keys = [
            key
            for key, cached in self._result_cache.items()
            if now - cached.created_at > SEARCH_CACHE_TTL_SECONDS
        ]
        for key in expired_keys:
            self._result_cache.pop(key, None)

    def _get_cached_results(
        self,
        query: str,
        candidate_limit: int | None,
    ) -> CachedSearchResults | None:
        self._purge_expired_cache_entries()
        return self._result_cache.get(self._cache_key(query, candidate_limit))

    def _store_cached_results(
        self,
        query: str,
        candidate_limit: int | None,
        *,
        requested_window: int,
        results: list[dict[str, Any]],
    ) -> CachedSearchResults:
        self._purge_expired_cache_entries()

        if len(self._result_cache) >= SEARCH_CACHE_MAX_ENTRIES:
            oldest_key = min(
                self._result_cache,
                key=lambda key: self._result_cache[key].created_at,
            )
            self._result_cache.pop(oldest_key, None)

        cached = CachedSearchResults(
            results=results,
            created_at=time.time(),
            requested_window=requested_window,
            is_exhaustive=len(results) < requested_window,
        )
        self._result_cache[self._cache_key(query, candidate_limit)] = cached
        return cached

    def _expanded_result_window(self, required_count: int, page_size: int) -> int:
        block_size = max(page_size, page_size * SEARCH_CACHE_PREFETCH_PAGES)
        blocks = max(1, (required_count + block_size - 1) // block_size)
        return blocks * block_size

    def _embedding_dimension(self) -> int:
        if hasattr(self.embedding_model, "get_embedding_dimension"):
            return int(self.embedding_model.get_embedding_dimension())
        return int(self.embedding_model.get_sentence_embedding_dimension())

    def embed_query(self, query: str) -> np.ndarray:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Query must not be empty.")

        embedding = self.embedding_model.encode(
            cleaned_query,
            batch_size=1,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return np.asarray(embedding, dtype=np.float32)

    def _connect(self) -> psycopg.Connection[dict[str, Any]]:
        conn = psycopg.connect(self.database_url, row_factory=dict_row)
        register_vector(conn)
        return conn

    def _retrieve_chunk_candidates(
        self,
        conn: psycopg.Connection[dict[str, Any]],
        query_embedding: np.ndarray,
        *,
        candidate_limit: int,
    ) -> list[RetrievedChunk]:
        vector = np.asarray(query_embedding, dtype=np.float32)
        with conn.cursor() as cur:
            cur.execute(CHUNK_RETRIEVE_SQL, (vector, vector, candidate_limit))
            rows = cur.fetchall()

        return [
            RetrievedChunk(
                id=str(row["id"]),
                entry_slug=str(row["entry_slug"]),
                entry_title=str(row["entry_title"]),
                section_title=str(row["section_title"]),
                url=str(row["url"]),
                chunk_text=str(row["chunk_text"]),
                cosine_score=float(row["cosine_score"]),
            )
            for row in rows
        ]

    def retrieve_candidates(
        self,
        query_embedding: np.ndarray,
        *,
        candidate_limit: int | None = None,
    ) -> list[RetrievedChunk]:
        limit = candidate_limit or self.config.candidate_limit
        validate_positive("candidate_limit", limit)

        with self._connect() as conn:
            return self._retrieve_chunk_candidates(
                conn,
                query_embedding,
                candidate_limit=limit,
            )

    def _retrieve_entry_candidates(
        self,
        conn: psycopg.Connection[dict[str, Any]],
        query_embedding: np.ndarray,
        *,
        candidate_limit: int,
    ) -> list[RetrievedEntry]:
        vector = np.asarray(query_embedding, dtype=np.float32)
        try:
            with conn.cursor() as cur:
                cur.execute(ENTRY_RETRIEVE_SQL, (vector, vector, candidate_limit))
                rows = cur.fetchall()
        except psycopg.Error as exc:
            if exc.sqlstate == "42P01":
                return []
            raise

        return [
            RetrievedEntry(
                id=str(row["id"]),
                entry_slug=str(row["entry_slug"]),
                entry_title=str(row["entry_title"]),
                url=str(row["url"]),
                section_outline=str(row["section_outline"]),
                intro_text=str(row["intro_text"]),
                search_text=str(row["search_text"]),
                cosine_score=float(row["cosine_score"]),
            )
            for row in rows
        ]

    def retrieve_entry_candidates(
        self,
        query_embedding: np.ndarray,
        *,
        candidate_limit: int | None = None,
    ) -> list[RetrievedEntry]:
        limit = candidate_limit or self.config.entry_candidate_limit
        validate_positive("entry_candidate_limit", limit)

        with self._connect() as conn:
            return self._retrieve_entry_candidates(
                conn,
                query_embedding,
                candidate_limit=limit,
            )

    def _retrieve_chunks_for_entries(
        self,
        conn: psycopg.Connection[dict[str, Any]],
        query_embedding: np.ndarray,
        entry_slugs: list[str],
        *,
        per_entry_limit: int,
    ) -> list[RetrievedChunk]:
        if not entry_slugs:
            return []

        overall_limit = len(entry_slugs) * per_entry_limit
        vector = np.asarray(query_embedding, dtype=np.float32)

        with conn.cursor() as cur:
            cur.execute(
                ENTRY_CHUNK_SQL,
                (
                    vector,
                    vector,
                    entry_slugs,
                    per_entry_limit,
                    overall_limit,
                ),
            )
            rows = cur.fetchall()

        return [
            RetrievedChunk(
                id=str(row["id"]),
                entry_slug=str(row["entry_slug"]),
                entry_title=str(row["entry_title"]),
                section_title=str(row["section_title"]),
                url=str(row["url"]),
                chunk_text=str(row["chunk_text"]),
                cosine_score=float(row["cosine_score"]),
            )
            for row in rows
        ]

    def retrieve_chunks_for_entries(
        self,
        query_embedding: np.ndarray,
        entry_slugs: list[str],
        *,
        per_entry_limit: int | None = None,
    ) -> list[RetrievedChunk]:
        limit = per_entry_limit or self.config.entry_chunk_limit
        validate_positive("entry_chunk_limit", limit)

        with self._connect() as conn:
            return self._retrieve_chunks_for_entries(
                conn,
                query_embedding,
                entry_slugs,
                per_entry_limit=limit,
            )

    def rerank_entries(
        self,
        query: str,
        candidates: list[RetrievedEntry],
    ) -> list[ScoredEntry]:
        if not candidates:
            return []

        pairs = [(query, format_entry_for_rerank(candidate)) for candidate in candidates]
        scores = self.reranker_model.predict(
            pairs,
            batch_size=self.config.rerank_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        flat_scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        ranked_entries: list[ScoredEntry] = []
        for candidate, score in zip(candidates, flat_scores, strict=True):
            lexical_bonus = title_bonus(query, candidate.entry_title)
            final_score = (
                float(score)
                + (ENTRY_COSINE_WEIGHT * candidate.cosine_score)
                + (ENTRY_TITLE_BONUS_WEIGHT * lexical_bonus)
            )
            ranked_entries.append(
                ScoredEntry(
                    candidate=candidate,
                    reranker_score=float(score),
                    final_score=final_score,
                )
            )

        ranked_entries.sort(key=lambda item: item.final_score, reverse=True)
        return ranked_entries

    def rerank_candidates(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        *,
        parent_entry_scores: dict[str, float] | None = None,
    ) -> list[SearchResult]:
        if not candidates:
            return []

        entry_scores = parent_entry_scores or {}
        pairs = [(query, format_chunk_for_rerank(candidate)) for candidate in candidates]
        scores = self.reranker_model.predict(
            pairs,
            batch_size=self.config.rerank_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        flat_scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        results: list[SearchResult] = []
        for candidate, score in zip(candidates, flat_scores, strict=True):
            parent_entry_score = max(0.0, entry_scores.get(candidate.entry_slug, 0.0))
            title_match_bonus = title_bonus(query, candidate.entry_title)
            section_match_bonus = section_bonus(query, candidate.section_title)
            final_score = (
                float(score)
                + (CHUNK_COSINE_WEIGHT * candidate.cosine_score)
                + (CHUNK_ENTRY_SCORE_WEIGHT * parent_entry_score)
                + (CHUNK_TITLE_BONUS_WEIGHT * title_match_bonus)
                + (CHUNK_SECTION_BONUS_WEIGHT * section_match_bonus)
            )
            results.append(
                SearchResult(
                    entry_slug=candidate.entry_slug,
                    chunk_text=candidate.chunk_text,
                    entry_title=candidate.entry_title,
                    section_title=candidate.section_title,
                    url=candidate.url,
                    cosine_score=candidate.cosine_score,
                    reranker_score=float(score),
                    final_score=final_score,
                    parent_entry_score=parent_entry_score,
                )
            )

        results.sort(key=lambda item: item.final_score, reverse=True)
        return results

    def _compute_result_window(
        self,
        query: str,
        *,
        candidate_limit: int | None = None,
        result_window: int,
    ) -> list[dict[str, Any]]:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Query must not be empty.")

        base_chunk_limit = candidate_limit or self.config.candidate_limit
        chunk_limit = max(
            base_chunk_limit,
            result_window * RESULT_TO_CHUNK_CANDIDATE_RATIO,
        )
        entry_candidate_limit = max(
            self.config.entry_candidate_limit,
            chunk_limit,
            result_window * RESULT_TO_ENTRY_CANDIDATE_RATIO,
        )
        entry_expansion_limit = max(
            self.config.entry_result_limit,
            result_window,
        )

        validate_positive("candidate_limit", chunk_limit)
        validate_positive("result_window", result_window)

        query_embedding = self.embed_query(cleaned_query)

        with self._connect() as conn:
            global_chunk_candidates = self._retrieve_chunk_candidates(
                conn,
                query_embedding,
                candidate_limit=chunk_limit,
            )
            entry_candidates = self._retrieve_entry_candidates(
                conn,
                query_embedding,
                candidate_limit=entry_candidate_limit,
            )
            ranked_entries = self.rerank_entries(cleaned_query, entry_candidates)

            top_entry_slugs = [
                item.candidate.entry_slug
                for item in ranked_entries[:entry_expansion_limit]
            ]
            entry_score_map = {
                item.candidate.entry_slug: item.final_score for item in ranked_entries
            }
            entry_expansion_candidates = self._retrieve_chunks_for_entries(
                conn,
                query_embedding,
                top_entry_slugs,
                per_entry_limit=self.config.entry_chunk_limit,
            )

        all_candidates = merge_chunk_candidates(
            global_chunk_candidates,
            entry_expansion_candidates,
        )
        reranked = self.rerank_candidates(
            cleaned_query,
            all_candidates,
            parent_entry_scores=entry_score_map,
        )
        return [result.to_dict() for result in reranked[:result_window]]

    def search(
        self,
        query: str,
        *,
        candidate_limit: int | None = None,
        result_limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Query must not be empty.")
        if offset < 0:
            raise ValueError("offset must be zero or greater.")

        page_size = result_limit or self.config.result_limit
        validate_positive("result_limit", page_size)

        required_end = offset + page_size
        cached = self._get_cached_results(cleaned_query, candidate_limit)
        if cached is not None:
            if cached.is_exhaustive or len(cached.results) >= required_end:
                return cached.results[offset:required_end]

        requested_window = self._expanded_result_window(required_end, page_size)
        computed_results = self._compute_result_window(
            cleaned_query,
            candidate_limit=candidate_limit,
            result_window=requested_window,
        )
        cached = self._store_cached_results(
            cleaned_query,
            candidate_limit,
            requested_window=requested_window,
            results=computed_results,
        )
        return cached.results[offset:required_end]


def create_search_service(
    *,
    database_url: str | None = None,
    embedding_model: SentenceTransformer | None = None,
    reranker_model: CrossEncoder | None = None,
    device: str | None = None,
    config: SearchConfig | None = None,
) -> SearchService:
    load_dotenv()
    search_config = config or SearchConfig()
    resolved_database_url = database_url or resolve_database_url()
    embedding_model = embedding_model or SentenceTransformer(
        search_config.embedding_model_name,
        device=device,
    )
    reranker_model = reranker_model or CrossEncoder(
        search_config.reranker_model_name,
        device=device,
    )
    return SearchService(
        database_url=resolved_database_url,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        config=search_config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Zetesis semantic search from the command line.",
    )
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=DEFAULT_CANDIDATE_LIMIT,
        help="How many global chunk candidates to retrieve before reranking.",
    )
    parser.add_argument(
        "--entry-candidate-limit",
        type=int,
        default=DEFAULT_ENTRY_CANDIDATE_LIMIT,
        help="How many entry candidates to retrieve before entry reranking.",
    )
    parser.add_argument(
        "--entry-result-limit",
        type=int,
        default=DEFAULT_ENTRY_RESULT_LIMIT,
        help="How many entry candidates to expand back into chunk candidates.",
    )
    parser.add_argument(
        "--entry-chunk-limit",
        type=int,
        default=DEFAULT_ENTRY_CHUNK_LIMIT,
        help="How many top vector chunks to pull from each selected entry.",
    )
    parser.add_argument(
        "--result-limit",
        type=int,
        default=DEFAULT_RESULT_LIMIT,
        help="How many reranked results to return.",
    )
    parser.add_argument(
        "--rerank-batch-size",
        type=int,
        default=DEFAULT_RERANK_BATCH_SIZE,
        help="Batch size for cross-encoder reranking.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional model device override, for example 'cpu' or 'mps'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SearchConfig(
        candidate_limit=args.candidate_limit,
        entry_candidate_limit=args.entry_candidate_limit,
        entry_result_limit=args.entry_result_limit,
        entry_chunk_limit=args.entry_chunk_limit,
        result_limit=args.result_limit,
        rerank_batch_size=args.rerank_batch_size,
    )
    service = create_search_service(device=args.device, config=config)
    results = service.search(
        args.query,
        candidate_limit=args.candidate_limit,
        result_limit=args.result_limit,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
