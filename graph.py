#!/usr/bin/env python3
"""Graph retrieval service for Zetesis concept graph endpoints."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

DEFAULT_MIN_EXPLICIT_DEGREE = 3

ENTRY_EXISTS_SQL = """
SELECT entry_slug, entry_title, subdiscipline, intro_text
FROM entries
WHERE entry_slug = %s;
""".strip()

FULL_GRAPH_NODE_SLUGS_SQL = """
WITH explicit_neighbors AS (
    SELECT source_slug AS slug, target_slug AS neighbor_slug
    FROM explicit_edges
    UNION
    SELECT target_slug AS slug, source_slug AS neighbor_slug
    FROM explicit_edges
)
SELECT slug
FROM explicit_neighbors
GROUP BY slug
HAVING COUNT(DISTINCT neighbor_slug) >= %s
ORDER BY slug;
""".strip()

NODES_BY_SLUG_SQL = """
SELECT entry_slug, entry_title, subdiscipline, intro_text
FROM entries
WHERE entry_slug = ANY(%s)
ORDER BY entry_title, entry_slug;
""".strip()

EXPLICIT_EDGES_FOR_NODES_SQL = """
SELECT source_slug, target_slug, weight
FROM explicit_edges
WHERE source_slug = ANY(%s) AND target_slug = ANY(%s)
ORDER BY source_slug, target_slug;
""".strip()

SEMANTIC_EDGES_FOR_NODES_SQL = """
SELECT source_slug, target_slug, similarity
FROM semantic_edges
WHERE source_slug = ANY(%s) AND target_slug = ANY(%s)
ORDER BY source_slug, target_slug;
""".strip()

INCIDENT_EXPLICIT_EDGES_SQL = """
SELECT source_slug, target_slug
FROM explicit_edges
WHERE source_slug = ANY(%s) OR target_slug = ANY(%s);
""".strip()

INCIDENT_SEMANTIC_EDGES_SQL = """
SELECT source_slug, target_slug
FROM semantic_edges
WHERE source_slug = ANY(%s) OR target_slug = ANY(%s);
""".strip()

@dataclass(frozen=True)
class ExplicitEdge:
    source: str
    target: str
    weight: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class SemanticEdge:
    source: str
    target: str
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "similarity": self.similarity,
        }


@dataclass(frozen=True)
class GraphNode:
    slug: str
    title: str
    subdiscipline: str | None
    degree: int
    intro_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "title": self.title,
            "subdiscipline": self.subdiscipline,
            "degree": self.degree,
            "intro_text": self.intro_text,
        }


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


def compute_degrees(
    node_slugs: set[str],
    explicit_edges: list[ExplicitEdge],
    semantic_edges: list[SemanticEdge],
) -> dict[str, int]:
    neighbors: dict[str, set[str]] = {slug: set() for slug in node_slugs}

    for edge in explicit_edges:
        neighbors.setdefault(edge.source, set()).add(edge.target)
        neighbors.setdefault(edge.target, set()).add(edge.source)

    for edge in semantic_edges:
        neighbors.setdefault(edge.source, set()).add(edge.target)
        neighbors.setdefault(edge.target, set()).add(edge.source)

    return {slug: len(adjacent) for slug, adjacent in neighbors.items()}


class GraphService:
    """Load cached and dynamic concept graph views from Postgres."""

    def __init__(
        self,
        *,
        database_url: str,
        min_explicit_degree: int = DEFAULT_MIN_EXPLICIT_DEGREE,
    ) -> None:
        if min_explicit_degree <= 0:
            raise ValueError("min_explicit_degree must be positive.")

        self.database_url = database_url
        self.min_explicit_degree = min_explicit_degree
        self._full_graph_cache: dict[str, Any] | None = None

    def _connect(self) -> psycopg.Connection[dict[str, Any]]:
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def _fetch_entry(self, conn: psycopg.Connection[dict[str, Any]], slug: str) -> dict[str, Any] | None:
        with conn.cursor() as cur:
            cur.execute(ENTRY_EXISTS_SQL, (slug,))
            return cur.fetchone()

    def _fetch_nodes(
        self,
        conn: psycopg.Connection[dict[str, Any]],
        node_slugs: list[str],
    ) -> list[dict[str, Any]]:
        if not node_slugs:
            return []
        with conn.cursor() as cur:
            cur.execute(NODES_BY_SLUG_SQL, (node_slugs,))
            return cur.fetchall()

    def _fetch_explicit_edges(
        self,
        conn: psycopg.Connection[dict[str, Any]],
        node_slugs: list[str],
    ) -> list[ExplicitEdge]:
        if not node_slugs:
            return []

        with conn.cursor() as cur:
            cur.execute(EXPLICIT_EDGES_FOR_NODES_SQL, (node_slugs, node_slugs))
            rows = cur.fetchall()

        return [
            ExplicitEdge(
                source=str(row["source_slug"]),
                target=str(row["target_slug"]),
                weight=int(row["weight"]),
            )
            for row in rows
        ]

    def _fetch_semantic_edges(
        self,
        conn: psycopg.Connection[dict[str, Any]],
        node_slugs: list[str],
    ) -> list[SemanticEdge]:
        if not node_slugs:
            return []

        with conn.cursor() as cur:
            cur.execute(SEMANTIC_EDGES_FOR_NODES_SQL, (node_slugs, node_slugs))
            rows = cur.fetchall()

        return [
            SemanticEdge(
                source=str(row["source_slug"]),
                target=str(row["target_slug"]),
                similarity=float(row["similarity"]),
            )
            for row in rows
        ]

    def _build_graph_response(
        self,
        node_rows: list[dict[str, Any]],
        explicit_edges: list[ExplicitEdge],
        semantic_edges: list[SemanticEdge],
    ) -> dict[str, Any]:
        node_slugs = {str(row["entry_slug"]) for row in node_rows}
        degrees = compute_degrees(node_slugs, explicit_edges, semantic_edges)

        nodes = [
            GraphNode(
                slug=str(row["entry_slug"]),
                title=str(row["entry_title"]),
                subdiscipline=(
                    None
                    if row["subdiscipline"] is None
                    else str(row["subdiscipline"])
                ),
                degree=degrees.get(str(row["entry_slug"]), 0),
                intro_text=(
                    ""
                    if row["intro_text"] is None
                    else str(row["intro_text"]).strip()
                ),
            ).to_dict()
            for row in node_rows
        ]

        nodes.sort(key=lambda node: (-int(node["degree"]), str(node["title"]), str(node["slug"])))

        return {
            "nodes": nodes,
            "explicit_edges": [edge.to_dict() for edge in explicit_edges],
            "semantic_edges": [edge.to_dict() for edge in semantic_edges],
        }

    def load_full_graph(self) -> dict[str, Any]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(FULL_GRAPH_NODE_SLUGS_SQL, (self.min_explicit_degree,))
                node_slugs = [str(row["slug"]) for row in cur.fetchall()]

            node_rows = self._fetch_nodes(conn, node_slugs)
            explicit_edges = self._fetch_explicit_edges(conn, node_slugs)
            semantic_edges = self._fetch_semantic_edges(conn, node_slugs)

        return self._build_graph_response(node_rows, explicit_edges, semantic_edges)

    def preload_full_graph_cache(self) -> dict[str, Any]:
        self._full_graph_cache = self.load_full_graph()
        return self._full_graph_cache

    def get_full_graph(self) -> dict[str, Any]:
        if self._full_graph_cache is None:
            return self.preload_full_graph_cache()
        return self._full_graph_cache

    def _expand_neighborhood_slugs(
        self,
        conn: psycopg.Connection[dict[str, Any]],
        *,
        center_slug: str,
        hops: int,
    ) -> list[str]:
        visited: set[str] = {center_slug}
        frontier: set[str] = {center_slug}

        for _ in range(hops):
            if not frontier:
                break

            frontier_list = sorted(frontier)
            next_frontier: set[str] = set()

            with conn.cursor() as cur:
                cur.execute(INCIDENT_EXPLICIT_EDGES_SQL, (frontier_list, frontier_list))
                for row in cur.fetchall():
                    source = str(row["source_slug"])
                    target = str(row["target_slug"])
                    if source in frontier and target not in visited:
                        next_frontier.add(target)
                    if target in frontier and source not in visited:
                        next_frontier.add(source)

                cur.execute(INCIDENT_SEMANTIC_EDGES_SQL, (frontier_list, frontier_list))
                for row in cur.fetchall():
                    source = str(row["source_slug"])
                    target = str(row["target_slug"])
                    if source in frontier and target not in visited:
                        next_frontier.add(target)
                    if target in frontier and source not in visited:
                        next_frontier.add(source)

            visited.update(next_frontier)
            frontier = next_frontier

        return sorted(visited)

    def get_neighborhood(self, slug: str, *, hops: int = 1) -> dict[str, Any]:
        cleaned_slug = slug.strip()
        if not cleaned_slug:
            raise ValueError("Entry slug must not be empty.")
        if hops not in {1, 2}:
            raise ValueError("hops must be 1 or 2.")

        with self._connect() as conn:
            entry = self._fetch_entry(conn, cleaned_slug)
            if entry is None:
                raise LookupError(f"Unknown entry slug: {cleaned_slug}")

            node_slugs = self._expand_neighborhood_slugs(
                conn,
                center_slug=cleaned_slug,
                hops=hops,
            )
            node_rows = self._fetch_nodes(conn, node_slugs)
            explicit_edges = self._fetch_explicit_edges(conn, node_slugs)
            semantic_edges = self._fetch_semantic_edges(conn, node_slugs)

        return self._build_graph_response(node_rows, explicit_edges, semantic_edges)

def create_graph_service(
    *,
    database_url: str | None = None,
    min_explicit_degree: int = DEFAULT_MIN_EXPLICIT_DEGREE,
) -> GraphService:
    load_dotenv()
    resolved_database_url = database_url or resolve_database_url()
    return GraphService(
        database_url=resolved_database_url,
        min_explicit_degree=min_explicit_degree,
    )
