#!/usr/bin/env python3
"""FastAPI backend for Zetesis semantic search."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from graph import GraphService, create_graph_service
from search import SearchService, create_search_service

logger = logging.getLogger("zetesis.api")


def parse_cors_origins() -> list[str]:
    raw_origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
    origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    if not origins:
        return ["*"]

    if "*" in origins:
        return ["*"]

    expanded_origins: set[str] = set(origins)
    for origin in origins:
        if "localhost" in origin:
            expanded_origins.add(origin.replace("localhost", "127.0.0.1"))
        if "127.0.0.1" in origin:
            expanded_origins.add(origin.replace("127.0.0.1", "localhost"))

    return sorted(expanded_origins)


def get_search_service(request: Request) -> SearchService:
    service = getattr(request.app.state, "search_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Search service is not available.")
    return service


def get_graph_service(request: Request) -> GraphService:
    service = getattr(request.app.state, "graph_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Graph service is not available.")
    return service


def create_app(
    search_service: SearchService | Any | None = None,
    graph_service: GraphService | Any | None = None,
) -> FastAPI:
    load_dotenv()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.search_service = search_service or create_search_service()
        app.state.graph_service = graph_service or create_graph_service()
        yield

    app = FastAPI(
        title="Zetesis API",
        version="0.1.0",
        lifespan=lifespan,
    )

    if search_service is not None:
        app.state.search_service = search_service
    if graph_service is not None:
        app.state.graph_service = graph_service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=parse_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/search")
    async def search_endpoint(
        request: Request,
        q: str = Query(..., max_length=300),
    ) -> list[dict[str, Any]]:
        cleaned_query = q.strip()
        if not cleaned_query:
            raise HTTPException(status_code=400, detail="Query must not be empty.")

        service = get_search_service(request)
        try:
            return service.search(cleaned_query)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except psycopg.Error as exc:
            logger.exception("Database-backed search failed for query: %s", cleaned_query)
            raise HTTPException(status_code=503, detail="Database search is unavailable.") from exc

    @app.get("/graph/full")
    async def graph_full_endpoint(request: Request) -> dict[str, Any]:
        service = get_graph_service(request)
        try:
            return service.get_full_graph()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except psycopg.Error as exc:
            logger.exception("Full graph retrieval failed.")
            raise HTTPException(status_code=503, detail="Graph data is unavailable.") from exc

    @app.get("/graph/neighborhood")
    async def graph_neighborhood_endpoint(
        request: Request,
        slug: str = Query(..., min_length=1),
        hops: int = Query(1, ge=1, le=2),
    ) -> dict[str, Any]:
        cleaned_slug = slug.strip()
        if not cleaned_slug:
            raise HTTPException(status_code=400, detail="Entry slug must not be empty.")

        service = get_graph_service(request)
        try:
            return service.get_neighborhood(cleaned_slug, hops=hops)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except psycopg.Error as exc:
            logger.exception("Neighborhood graph retrieval failed for slug: %s", cleaned_slug)
            raise HTTPException(status_code=503, detail="Graph data is unavailable.") from exc

    @app.get("/graph/related")
    async def graph_related_endpoint(
        request: Request,
        slug: str = Query(..., min_length=1),
        limit: int = Query(5, ge=1, le=10),
    ) -> list[dict[str, Any]]:
        cleaned_slug = slug.strip()
        if not cleaned_slug:
            raise HTTPException(status_code=400, detail="Entry slug must not be empty.")

        service = get_graph_service(request)
        try:
            return service.get_related_concepts(cleaned_slug, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except psycopg.Error as exc:
            logger.exception("Related concept lookup failed for slug: %s", cleaned_slug)
            raise HTTPException(status_code=503, detail="Graph data is unavailable.") from exc

    return app


app = create_app()
