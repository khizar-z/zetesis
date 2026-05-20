#!/usr/bin/env python3
"""Build entry-level search documents from SEP section records."""

from __future__ import annotations

import argparse
import json
import re
import uuid
from collections import OrderedDict
from pathlib import Path

from text_repair import fix_mojibake

ENTRY_NAMESPACE = uuid.UUID("2b1f4a29-0f2a-4707-8588-a507a4f9622f")
DEFAULT_MAX_INTRO_WORDS = 350


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate SEP sections into entry-level search documents.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("sep_sections.json"),
        help="Path to the section-level JSON created by scrape_sep.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sep_entries.json"),
        help="Where to save the entry-level output.",
    )
    parser.add_argument(
        "--max-intro-words",
        type=int,
        default=DEFAULT_MAX_INTRO_WORDS,
        help="Maximum number of words to keep in the entry overview excerpt.",
    )
    return parser.parse_args()


def normalize_whitespace(text: str) -> str:
    text = fix_mojibake(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def trim_to_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        raise ValueError("--max-intro-words must be a positive integer.")

    words = normalize_whitespace(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).strip()


def base_entry_url(url: str) -> str:
    return url.split("#", maxsplit=1)[0]


def build_section_outline(sections: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for section in sections:
        number = normalize_whitespace(section["section_number"])
        title = normalize_whitespace(section["section_title"])
        if number:
            lines.append(f"{number}. {title}")
        else:
            lines.append(title)
    return "\n".join(lines).strip()


def build_intro_text(sections: list[dict[str, str]], max_words: int) -> str:
    intro_parts: list[str] = []
    total_words = 0

    for section in sections:
        section_text = normalize_whitespace(section["section_text"])
        if not section_text:
            continue

        remaining_words = max_words - total_words
        if remaining_words <= 0:
            break

        excerpt = trim_to_words(section_text, remaining_words)
        excerpt_words = word_count(excerpt)
        if excerpt_words == 0:
            continue

        intro_parts.append(excerpt)
        total_words += excerpt_words

        if total_words >= max_words:
            break

    return "\n\n".join(intro_parts).strip()


def build_search_text(
    *,
    entry_title: str,
    section_outline: str,
    intro_text: str,
) -> str:
    parts = [f"Entry: {entry_title}"]

    if section_outline:
        parts.append(f"Section headings:\n{section_outline}")

    if intro_text:
        parts.append(f"Overview:\n{intro_text}")

    return "\n\n".join(parts).strip()


def aggregate_entries(
    sections: list[dict[str, str]],
    *,
    max_intro_words: int,
) -> list[dict[str, str | int]]:
    grouped: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for section in sections:
        slug = str(section["entry_slug"])
        grouped.setdefault(slug, []).append(section)

    entries: list[dict[str, str | int]] = []
    for entry_slug, entry_sections in grouped.items():
        first = entry_sections[0]
        entry_title = normalize_whitespace(str(first["entry_title"]))
        url = base_entry_url(str(first["url"]))
        section_outline = build_section_outline(entry_sections)
        intro_text = build_intro_text(entry_sections, max_intro_words)
        search_text = build_search_text(
            entry_title=entry_title,
            section_outline=section_outline,
            intro_text=intro_text,
        )

        entries.append(
            {
                "entry_id": str(uuid.uuid5(ENTRY_NAMESPACE, entry_slug)),
                "entry_slug": entry_slug,
                "entry_title": entry_title,
                "url": url,
                "section_count": len(entry_sections),
                "section_outline": section_outline,
                "intro_text": intro_text,
                "search_text": search_text,
            }
        )

    return entries


def main() -> None:
    args = parse_args()
    sections = json.loads(args.input.read_text(encoding="utf-8"))
    entries = aggregate_entries(sections, max_intro_words=args.max_intro_words)

    args.output.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Loaded {len(sections)} sections from {args.input}")
    print(f"Saved {len(entries)} entries to {args.output}")


if __name__ == "__main__":
    main()
