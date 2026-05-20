#!/usr/bin/env python3
"""Scrape SEP entries into section-level JSON records."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from requests import Response, Session
from requests.adapters import HTTPAdapter
from text_repair import fix_mojibake
from urllib3.util.retry import Retry

CONTENTS_URL = "https://plato.stanford.edu/contents.html"
ENTRY_URL_PATTERN = re.compile(r"/entries/([^/]+)/?$")
SECTION_NUMBER_PATTERN = re.compile(r"^(?P<number>\d+(?:\.\d+)*)\.?\s+(?P<title>.+)$")
HEADING_LEVELS = {"h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
TEXT_BLOCK_TAGS = {
    "p",
    "blockquote",
    "ul",
    "ol",
    "dl",
    "pre",
    "table",
    "div",
}


@dataclass
class SectionRecord:
    entry_slug: str
    entry_title: str
    section_number: str
    section_title: str
    section_text: str
    url: str
    outbound_links: list[str]
    outbound_link_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "entry_slug": self.entry_slug,
            "entry_title": self.entry_title,
            "section_number": self.section_number,
            "section_title": self.section_title,
            "section_text": self.section_text,
            "url": self.url,
            "outbound_links": self.outbound_links,
            "outbound_link_counts": self.outbound_link_counts,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Stanford Encyclopedia of Philosophy entries into section-level JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sep_sections.json"),
        help="Where to save the extracted sections.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between HTTP requests.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of entries, useful for smoke tests.",
    )
    return parser.parse_args()


def build_session() -> Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Zetesis/0.1 (+https://github.com/placeholder/zetesis) "
                "SEP semantic search research scraper"
            )
        }
    )
    retries = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch(session: Session, url: str) -> Response:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "charset=" not in content_type.lower() and response.apparent_encoding:
        response.encoding = response.apparent_encoding
    return response


def normalize_entry_url(url: str) -> str | None:
    parsed = urlparse(url)
    match = ENTRY_URL_PATTERN.search(parsed.path.rstrip("/"))
    if not match:
        return None

    slug = match.group(1)
    return f"{parsed.scheme}://{parsed.netloc}/entries/{slug}/"


def extract_entry_urls(session: Session, delay: float, limit: int | None = None) -> list[str]:
    soup = BeautifulSoup(fetch(session, CONTENTS_URL).text, "html.parser")
    seen: set[str] = set()
    urls: list[str] = []

    for link in soup.select("a[href]"):
        full_url = urljoin(CONTENTS_URL, link["href"])
        normalized = normalize_entry_url(full_url)
        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        urls.append(normalized)
        if limit is not None and len(urls) >= limit:
            break

    if delay > 0:
        time.sleep(delay)

    return urls


def extract_entry_slug(entry_url: str) -> str:
    match = ENTRY_URL_PATTERN.search(urlparse(entry_url).path.rstrip("/"))
    if not match:
        raise ValueError(f"Could not derive entry slug from URL: {entry_url}")
    return match.group(1)


def clean_text(text: str) -> str:
    text = fix_mojibake(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def collect_block_text(block: Tag) -> str:
    if block.name in {"ul", "ol"}:
        items = [clean_text(item.get_text(" ", strip=True)) for item in block.find_all("li", recursive=False)]
        return "\n".join(item for item in items if item)

    if block.name == "dl":
        items: list[str] = []
        for child in block.find_all(["dt", "dd"], recursive=False):
            text = clean_text(child.get_text(" ", strip=True))
            if text:
                items.append(text)
        return "\n".join(items)

    if block.name == "table":
        rows: list[str] = []
        for row in block.find_all("tr"):
            cells = [clean_text(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"])]
            cells = [cell for cell in cells if cell]
            if cells:
                rows.append(" | ".join(cells))
        return "\n".join(rows)

    return clean_text(block.get_text(" ", strip=True))


def split_section_heading(heading_text: str) -> tuple[str, str]:
    text = clean_text(heading_text)
    match = SECTION_NUMBER_PATTERN.match(text)
    if not match:
        return "", text
    return match.group("number"), match.group("title")


def heading_anchor_id(heading: Tag) -> str | None:
    anchor = heading.find("a")
    if anchor:
        if anchor.get("id"):
            return anchor["id"]
        if anchor.get("name"):
            return anchor["name"]

    if heading.get("id"):
        return heading["id"]

    return None


def extract_entry_title(soup: BeautifulSoup) -> str:
    meta_title = soup.select_one('meta[property="citation_title"]')
    if meta_title and meta_title.get("content"):
        return clean_text(meta_title["content"])

    page_title = soup.find("title")
    if page_title:
        return clean_text(page_title.get_text())

    raise ValueError("Could not locate entry title.")


def extract_outbound_links(main_text: Tag, *, entry_slug: str, entry_url: str) -> tuple[list[str], dict[str, int]]:
    outbound_link_counts: dict[str, int] = {}

    for anchor in main_text.select("a[href]"):
        href = anchor.get("href")
        if not href:
            continue

        full_url = urljoin(entry_url, href)
        normalized = normalize_entry_url(full_url)
        if not normalized:
            continue

        target_slug = extract_entry_slug(normalized)
        if target_slug == entry_slug:
            continue

        outbound_link_counts[target_slug] = outbound_link_counts.get(target_slug, 0) + 1

    return list(outbound_link_counts.keys()), outbound_link_counts


def extract_sections(entry_url: str, html: str) -> list[SectionRecord]:
    soup = BeautifulSoup(html, "html.parser")
    main_text = soup.select_one("#main-text")
    if main_text is None:
        raise ValueError(f"Could not locate #main-text for {entry_url}")

    entry_slug = extract_entry_slug(entry_url)
    entry_title = extract_entry_title(soup)
    outbound_links, outbound_link_counts = extract_outbound_links(
        main_text,
        entry_slug=entry_slug,
        entry_url=entry_url,
    )
    sections: list[SectionRecord] = []

    current_heading: Tag | None = None
    current_blocks: list[str] = []

    def flush_section() -> None:
        nonlocal current_heading, current_blocks
        if current_heading is None:
            return

        section_text = "\n\n".join(block for block in current_blocks if block).strip()
        if not section_text:
            current_heading = None
            current_blocks = []
            return

        heading_text = current_heading.get_text(" ", strip=True)
        section_number, section_title = split_section_heading(heading_text)
        anchor_id = heading_anchor_id(current_heading)
        section_url = entry_url if not anchor_id else f"{entry_url}#{anchor_id}"

        sections.append(
            SectionRecord(
                entry_slug=entry_slug,
                entry_title=entry_title,
                section_number=section_number,
                section_title=section_title,
                section_text=section_text,
                url=section_url,
                outbound_links=outbound_links,
                outbound_link_counts=outbound_link_counts,
            )
        )

        current_heading = None
        current_blocks = []

    for child in main_text.children:
        if isinstance(child, NavigableString):
            continue

        if child.name in HEADING_LEVELS:
            flush_section()
            current_heading = child
            continue

        if current_heading is None:
            # Skip the preamble text before the first numbered section.
            continue

        if child.name not in TEXT_BLOCK_TAGS:
            text = clean_text(child.get_text(" ", strip=True))
            if text:
                current_blocks.append(text)
            continue

        text = collect_block_text(child)
        if text:
            current_blocks.append(text)

    flush_section()

    if sections:
        return sections

    # Fallback for atypical entries with no visible section headings.
    blocks: list[str] = []
    for child in main_text.children:
        if isinstance(child, NavigableString):
            continue
        text = collect_block_text(child) if child.name in TEXT_BLOCK_TAGS else clean_text(child.get_text(" ", strip=True))
        if text:
            blocks.append(text)

    section_text = "\n\n".join(blocks).strip()
    if not section_text:
        return []

    return [
        SectionRecord(
            entry_slug=entry_slug,
            entry_title=entry_title,
            section_number="",
            section_title="Main Text",
            section_text=section_text,
            url=entry_url,
            outbound_links=outbound_links,
            outbound_link_counts=outbound_link_counts,
        )
    ]


def iter_entry_sections(
    session: Session,
    entry_urls: Iterable[str],
    delay: float,
) -> Iterable[SectionRecord]:
    entry_urls = list(entry_urls)
    total = len(entry_urls)

    for index, entry_url in enumerate(entry_urls, start=1):
        entry_slug = extract_entry_slug(entry_url)
        response = fetch(session, entry_url)
        sections = extract_sections(entry_url=entry_url, html=response.text)

        for section in sections:
            yield section

        print(f"[{index}/{total}] scraped {entry_slug} -> {len(sections)} sections")

        if delay > 0 and index < total:
            time.sleep(delay)


def main() -> None:
    args = parse_args()
    session = build_session()

    print("Collecting SEP entry URLs...")
    entry_urls = extract_entry_urls(session=session, delay=args.delay, limit=args.limit)
    print(f"Found {len(entry_urls)} SEP entry URLs.")

    sections = [section.to_dict() for section in iter_entry_sections(session, entry_urls, args.delay)]

    args.output.write_text(
        json.dumps(sections, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(sections)} sections to {args.output}")


if __name__ == "__main__":
    main()
