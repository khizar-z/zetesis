#!/usr/bin/env python3
"""Turn SEP sections into retrieval chunks."""

from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path

from text_repair import fix_mojibake

MIN_WORDS = 300
MAX_WORDS = 400
TARGET_WORDS = 350
OVERLAP_FALLBACK_WORDS = 40
MAX_OVERLAP_SENTENCE_WORDS = 80
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split SEP section records into retrieval chunks.",
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
        default=Path("sep_chunks.json"),
        help="Where to save the chunked output.",
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


def split_paragraphs(text: str) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    paragraphs = [normalize_whitespace(part) for part in re.split(r"\n\s*\n", text)]
    return [paragraph for paragraph in paragraphs if paragraph]


def split_sentences(text: str) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    parts = re.split(SENTENCE_SPLIT_PATTERN, text)
    if len(parts) == 1 and re.search(r"[.!?]", text):
        parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def trailing_words(text: str, max_words: int = OVERLAP_FALLBACK_WORDS) -> str:
    words = normalize_whitespace(text).split()
    if not words:
        return ""
    return " ".join(words[-max_words:])


def last_sentence(text: str) -> str:
    sentences = split_sentences(text)
    if sentences:
        candidate = sentences[-1]
        if word_count(candidate) <= MAX_OVERLAP_SENTENCE_WORDS:
            return candidate
        return trailing_words(candidate)
    return trailing_words(text)


def build_word_chunks(text: str) -> list[str]:
    words = normalize_whitespace(text).split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + TARGET_WORDS, len(words))
        chunks.append(" ".join(words[start:end]))
        start = end

    return chunks


def build_sentence_chunks(paragraph: str) -> list[str]:
    """Fallback when a single paragraph is too large to fit a chunk."""
    sentences = split_sentences(paragraph)
    if not sentences:
        return build_word_chunks(paragraph)

    if len(sentences) == 1 and word_count(sentences[0]) > MAX_WORDS:
        return build_word_chunks(paragraph)

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = word_count(sentence)
        if sentence_words > MAX_WORDS:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_words = 0
            chunks.extend(build_word_chunks(sentence))
            continue

        projected = current_words + sentence_words

        if current_sentences and projected > MAX_WORDS:
            if current_words >= MIN_WORDS:
                chunks.append(" ".join(current_sentences))
                current_sentences = [sentence]
                current_words = sentence_words
                continue

            undershoot = MIN_WORDS - current_words
            overshoot = projected - MAX_WORDS
            if undershoot <= overshoot:
                chunks.append(" ".join(current_sentences))
                current_sentences = [sentence]
                current_words = sentence_words
                continue

        current_sentences.append(sentence)
        current_words = projected

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return [normalize_whitespace(chunk) for chunk in chunks if normalize_whitespace(chunk)]


def build_base_chunks(section_text: str) -> list[str]:
    paragraphs = split_paragraphs(section_text)
    if not paragraphs:
        return []

    if word_count(section_text) <= MAX_WORDS:
        return [normalize_whitespace(section_text)]

    chunks: list[str] = []
    current_paragraphs: list[str] = []
    current_words = 0

    for paragraph in paragraphs:
        paragraph_words = word_count(paragraph)

        if paragraph_words > MAX_WORDS:
            if current_paragraphs:
                chunks.append("\n\n".join(current_paragraphs))
                current_paragraphs = []
                current_words = 0
            chunks.extend(build_sentence_chunks(paragraph))
            continue

        projected = current_words + paragraph_words

        if current_paragraphs and projected > MAX_WORDS:
            if current_words >= MIN_WORDS:
                chunks.append("\n\n".join(current_paragraphs))
                current_paragraphs = [paragraph]
                current_words = paragraph_words
                continue

            undershoot = MIN_WORDS - current_words
            overshoot = projected - MAX_WORDS
            if undershoot <= overshoot:
                chunks.append("\n\n".join(current_paragraphs))
                current_paragraphs = [paragraph]
                current_words = paragraph_words
                continue

        current_paragraphs.append(paragraph)
        current_words = projected

    if current_paragraphs:
        chunks.append("\n\n".join(current_paragraphs))

    return [normalize_whitespace(chunk) for chunk in chunks if normalize_whitespace(chunk)]


def add_sentence_overlap(chunks: list[str]) -> list[str]:
    if not chunks:
        return []

    overlapped = [chunks[0]]
    for index in range(1, len(chunks)):
        overlap = last_sentence(chunks[index - 1])
        current = chunks[index]
        if overlap and not current.startswith(overlap):
            current = f"{overlap}\n\n{current}"
        overlapped.append(normalize_whitespace(current))
    return overlapped


def chunk_section(section: dict[str, str]) -> list[dict[str, str | int]]:
    section_text = section["section_text"]
    base_chunks = build_base_chunks(section_text)
    overlapped_chunks = add_sentence_overlap(base_chunks)

    results: list[dict[str, str | int]] = []
    for chunk_index, chunk_text in enumerate(overlapped_chunks):
        results.append(
            {
                "chunk_id": str(uuid.uuid4()),
                "entry_slug": section["entry_slug"],
                "entry_title": section["entry_title"],
                "section_number": section["section_number"],
                "section_title": section["section_title"],
                "url": section["url"],
                "chunk_text": chunk_text,
                "chunk_index": chunk_index,
            }
        )

    return results


def main() -> None:
    args = parse_args()
    sections = json.loads(args.input.read_text(encoding="utf-8"))

    chunks: list[dict[str, str | int]] = []
    total_sections = len(sections)
    for index, section in enumerate(sections, start=1):
        chunks.extend(chunk_section(section))
        if index % 1000 == 0 or index == total_sections:
            print(f"[{index}/{total_sections}] processed sections")

    args.output.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Loaded {len(sections)} sections from {args.input}")
    print(f"Saved {len(chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()
