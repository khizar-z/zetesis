#!/usr/bin/env python3
"""Populate graph subdiscipline labels for SEP entries."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus, urlparse

import psycopg
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

CONFIG_PATH = Path("shared/subdisciplines.json")
DEFAULT_DELAY_SECONDS = 1.0
DEFAULT_PROGRESS_INTERVAL = 50
DEFAULT_REQUEST_TIMEOUT = 30
DEFAULT_RETRY_LIMIT = 3

ENTRY_SELECT_SQL = """
SELECT entry_slug, entry_title, url, subdiscipline
FROM entries
{where_clause}
ORDER BY entry_slug
{limit_clause};
""".strip()

UPDATE_SUBDISCIPLINE_SQL = """
UPDATE entries
SET subdiscipline = %s
WHERE entry_slug = %s;
""".strip()

SUBDISCIPLINE_PATTERNS = [
    re.compile(r"\bsubdiscipline\s*:\s*([A-Za-z ,&/-]+)", re.IGNORECASE),
    re.compile(r"\bdiscipline\s*:\s*([A-Za-z ,&/-]+)", re.IGNORECASE),
    re.compile(r"\bcategory\s*:\s*([A-Za-z ,&/-]+)", re.IGNORECASE),
    re.compile(r"\barea\s*:\s*([A-Za-z ,&/-]+)", re.IGNORECASE),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch SEP entry pages and populate the entries.subdiscipline "
            "column using page metadata and SEP-linked InPhO topic metadata."
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help="Delay in seconds between SEP page requests.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on processed entries, useful for smoke tests.",
    )
    parser.add_argument(
        "--slugs",
        nargs="+",
        default=None,
        help="Optional list of specific entry slugs to process.",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Process only entries whose subdiscipline is currently NULL.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=DEFAULT_PROGRESS_INTERVAL,
        help="Print progress every N processed entries.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="HTTP timeout in seconds for each request.",
    )
    parser.add_argument(
        "--retry-limit",
        type=int,
        default=DEFAULT_RETRY_LIMIT,
        help="How many times to retry a failed network request.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to the shared subdiscipline configuration JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be updated without writing to the database.",
    )
    return parser.parse_args()


def validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def validate_non_negative_float(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


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


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required_keys = {
        "default_color",
        "subdiscipline_colors",
        "aliases",
        "inpho_topic_model",
    }
    missing = required_keys - config.keys()
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Config file {path} is missing required keys: {missing_list}")
    return config


def build_select_query(*, only_missing: bool, slugs: list[str] | None, limit: int | None) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []

    if only_missing:
        clauses.append("subdiscipline IS NULL")

    if slugs:
        clauses.append("entry_slug = ANY(%s)")
        params.append(sorted(set(slugs)))

    where_clause = ""
    if clauses:
        where_clause = "WHERE " + " AND ".join(clauses)

    limit_clause = ""
    if limit is not None:
        limit_clause = "LIMIT %s"
        params.append(limit)

    query = ENTRY_SELECT_SQL.format(
        where_clause=where_clause,
        limit_clause=limit_clause,
    )
    return query, params


def fetch_entries(
    conn: psycopg.Connection[Any],
    *,
    only_missing: bool,
    slugs: list[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    query, params = build_select_query(
        only_missing=only_missing,
        slugs=slugs,
        limit=limit,
    )
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    return [
        {
            "entry_slug": row[0],
            "entry_title": row[1],
            "url": row[2],
            "subdiscipline": row[3],
        }
        for row in rows
    ]


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "ZetesisSubdisciplineTagger/1.0 (+https://plato.stanford.edu/)",
        }
    )
    return session


def request_text(
    session: requests.Session,
    url: str,
    *,
    timeout: int,
    retry_limit: int,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retry_limit + 1):
        try:
            response = session.get(url, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            return response.text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == retry_limit:
                break
            time.sleep(min(2.0 * attempt, 5.0))

    raise RuntimeError(f"Failed to fetch {url}") from last_error


def request_json(
    session: requests.Session,
    url: str,
    *,
    timeout: int,
    retry_limit: int,
) -> Any:
    text = request_text(
        session,
        url,
        timeout=timeout,
        retry_limit=retry_limit,
    )
    return json.loads(text)


def normalize_candidate_label(candidate: str, alias_map: dict[str, str]) -> str | None:
    normalized = re.sub(r"\s+", " ", candidate.strip().lower())
    normalized = normalized.strip(" .,:;")
    return alias_map.get(normalized)


def extract_explicit_subdiscipline(
    soup: BeautifulSoup,
    *,
    alias_map: dict[str, str],
) -> str | None:
    for meta in soup.find_all("meta"):
        content = meta.get("content")
        if not content:
            continue
        label = normalize_candidate_label(str(content), alias_map)
        if label:
            return label

    content_root = soup.select_one("#content") or soup.select_one("#article") or soup
    text = " ".join(content_root.get_text(" ", strip=True).split())
    for pattern in SUBDISCIPLINE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        label = normalize_candidate_label(match.group(1), alias_map)
        if label:
            return label

    return None


def extract_inpho_sep_slug(soup: BeautifulSoup) -> str | None:
    link = soup.select_one('#academic-tools a[href*="inphoproject.org/entity?sep="]')
    if link is None:
        link = soup.select_one('a[href*="inphoproject.org/entity?sep="]')
    if link is None:
        return None

    href = link.get("href")
    if not href:
        return None

    parsed = urlparse(href)
    sep_values = parse_qs(parsed.query).get("sep", [])
    if not sep_values:
        return None
    return sep_values[0].strip() or None


def infer_subdiscipline_from_inpho(
    *,
    sep_slug: str,
    session: requests.Session,
    timeout: int,
    retry_limit: int,
    topic_model_config: dict[str, Any],
    cache: dict[str, str | None],
) -> str | None:
    if sep_slug in cache:
        return cache[sep_slug]

    model_k = int(topic_model_config["k"])
    min_probability = float(topic_model_config["min_topic_probability"])
    topic_to_subdiscipline = {
        str(key): str(value)
        for key, value in dict(topic_model_config["topic_to_subdiscipline"]).items()
    }

    url = (
        f"https://www.inphoproject.org/topics/sep/{model_k}/docs_topics/"
        f"{sep_slug}.json?n=1"
    )

    try:
        payload = request_json(
            session,
            url,
            timeout=timeout,
            retry_limit=retry_limit,
        )
    except Exception:  # noqa: BLE001
        cache[sep_slug] = None
        return None

    if not isinstance(payload, list) or not payload:
        cache[sep_slug] = None
        return None

    topics = payload[0].get("topics", {})
    if not isinstance(topics, dict) or not topics:
        cache[sep_slug] = None
        return None

    top_topic_id, top_probability = max(
        ((str(topic_id), float(probability)) for topic_id, probability in topics.items()),
        key=lambda item: item[1],
    )

    if top_probability < min_probability:
        cache[sep_slug] = None
        return None

    subdiscipline = topic_to_subdiscipline.get(top_topic_id)
    cache[sep_slug] = subdiscipline
    return subdiscipline


def determine_subdiscipline(
    *,
    html: str,
    session: requests.Session,
    timeout: int,
    retry_limit: int,
    alias_map: dict[str, str],
    topic_model_config: dict[str, Any],
    inpho_cache: dict[str, str | None],
) -> tuple[str | None, str]:
    soup = BeautifulSoup(html, "html.parser")

    explicit = extract_explicit_subdiscipline(soup, alias_map=alias_map)
    if explicit is not None:
        return explicit, "page"

    sep_slug = extract_inpho_sep_slug(soup)
    if sep_slug is None:
        return None, "none"

    inferred = infer_subdiscipline_from_inpho(
        sep_slug=sep_slug,
        session=session,
        timeout=timeout,
        retry_limit=retry_limit,
        topic_model_config=topic_model_config,
        cache=inpho_cache,
    )
    if inferred is not None:
        return inferred, "inpho"
    return None, "none"


def main() -> None:
    load_dotenv()
    args = parse_args()
    validate_non_negative_float("--delay", args.delay)
    validate_positive("--progress-interval", args.progress_interval)
    validate_positive("--request-timeout", args.request_timeout)
    validate_positive("--retry-limit", args.retry_limit)
    if args.limit is not None:
        validate_positive("--limit", args.limit)

    config = load_config(args.config)
    alias_map = {
        str(key): str(value)
        for key, value in dict(config["aliases"]).items()
    }
    topic_model_config = dict(config["inpho_topic_model"])

    database_url = resolve_database_url()
    session = create_session()
    inpho_cache: dict[str, str | None] = {}

    with psycopg.connect(database_url) as conn:
        entries = fetch_entries(
            conn,
            only_missing=args.only_missing,
            slugs=args.slugs,
            limit=args.limit,
        )

        total_entries = len(entries)
        if total_entries == 0:
            print("No matching entries found.")
            return

        print(f"Loaded {total_entries} entries from the database.")

        page_matches = 0
        inpho_matches = 0
        null_matches = 0
        error_count = 0

        processed = 0
        for index, entry in enumerate(entries, start=1):
            entry_slug = str(entry["entry_slug"])
            entry_title = str(entry["entry_title"])
            url = str(entry["url"])

            try:
                html = request_text(
                    session,
                    url,
                    timeout=args.request_timeout,
                    retry_limit=args.retry_limit,
                )
                subdiscipline, source = determine_subdiscipline(
                    html=html,
                    session=session,
                    timeout=args.request_timeout,
                    retry_limit=args.retry_limit,
                    alias_map=alias_map,
                    topic_model_config=topic_model_config,
                    inpho_cache=inpho_cache,
                )
            except Exception as exc:  # noqa: BLE001
                subdiscipline = None
                source = "error"
                error_count += 1
                print(f"[{index}/{total_entries}] error {entry_slug}: {exc}")

            if source == "page":
                page_matches += 1
            elif source == "inpho":
                inpho_matches += 1
            elif source == "none":
                null_matches += 1

            if args.dry_run:
                print(
                    f"[{index}/{total_entries}] {entry_slug} ({entry_title}) -> "
                    f"{subdiscipline!r} via {source}"
                )
            else:
                with conn.cursor() as cur:
                    cur.execute(UPDATE_SUBDISCIPLINE_SQL, (subdiscipline, entry_slug))
                conn.commit()

            processed += 1
            if processed % args.progress_interval == 0 or processed == total_entries:
                print(
                    f"Processed {processed}/{total_entries} entries "
                    f"(page={page_matches}, inpho={inpho_matches}, null={null_matches}, "
                    f"errors={error_count})"
                )

            if index < total_entries and args.delay > 0:
                time.sleep(args.delay)

    print("Finished tagging subdisciplines.")


if __name__ == "__main__":
    main()
