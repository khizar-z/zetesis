#!/usr/bin/env python3
"""Manual retrieval evaluation for Zetesis."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from search import SearchConfig, SearchService, create_search_service

DEFAULT_RESULT_LIMIT = 3
DEFAULT_SNIPPET_LENGTH = 220

EVAL_QUERIES = [
    "what is truth",
    "what is knowledge",
    "what makes an action free",
    "relationship between free will and moral responsibility",
    "Parfit's reductionist view of personal identity",
    "problem of other minds",
    "is consciousness reducible to physical processes",
    "compatibilism versus incompatibilism",
    "Kant's synthetic a priori",
    "Aristotle on virtue and habituation",
    "Hume on causation",
    "Plato on justice in the Republic",
    "Nietzsche on ressentiment",
    "grounding and metaphysical explanation",
    "epistemic injustice",
    "moral status of animals",
    "double effect and trolley cases",
    "reference and descriptions",
    "Lewis on modal realism",
    "social ontology of institutions",
    "structural realism in philosophy of science",
    "phenomenology of intentionality",
    "divine hiddenness",
    "Rawls's original position",
    "relationship between testimony and knowledge",
]


@dataclass
class EvalResult:
    query: str
    results: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Zetesis retrieval evaluation over 25 curated philosophical queries.",
    )
    parser.add_argument(
        "--result-limit",
        type=int,
        default=DEFAULT_RESULT_LIMIT,
        help="How many top results to print per query.",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=20,
        help="How many vector candidates to retrieve before reranking.",
    )
    parser.add_argument(
        "--snippet-length",
        type=int,
        default=DEFAULT_SNIPPET_LENGTH,
        help="Maximum snippet length to print for each result.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional model device override, for example 'cpu' or 'mps'.",
    )
    return parser.parse_args()


def validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def build_snippet(text: str, max_length: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_length:
        return cleaned

    truncated = cleaned[:max_length]
    last_space = truncated.rfind(" ")
    if last_space <= max_length * 0.6:
        return f"{truncated.rstrip()}..."
    return f"{truncated[:last_space].rstrip()}..."


def run_evaluation(
    service: SearchService,
    *,
    queries: list[str] | None = None,
    candidate_limit: int,
    result_limit: int,
) -> list[EvalResult]:
    eval_queries = queries or EVAL_QUERIES
    report: list[EvalResult] = []

    for index, query in enumerate(eval_queries, start=1):
        print(f"[{index}/{len(eval_queries)}] {query}")
        results = service.search(
            query,
            candidate_limit=candidate_limit,
            result_limit=result_limit,
        )
        report.append(EvalResult(query=query, results=results))

    return report


def format_report(report: list[EvalResult], *, snippet_length: int) -> str:
    lines: list[str] = []
    for index, item in enumerate(report, start=1):
        lines.append(f"{index}. Query: {item.query}")

        if not item.results:
            lines.append("   No results returned.")
            lines.append("")
            continue

        for result_index, result in enumerate(item.results, start=1):
            snippet = build_snippet(str(result["chunk_text"]), snippet_length)
            entry = str(result["entry_title"])
            section = str(result["section_title"])
            lines.append(f"   {result_index}. Entry: {entry}")
            lines.append(f"      Section: {section}")
            lines.append(f"      Snippet: {snippet}")

        lines.append("")

    return "\n".join(lines).rstrip()


def main() -> None:
    args = parse_args()
    validate_positive("--result-limit", args.result_limit)
    validate_positive("--candidate-limit", args.candidate_limit)
    validate_positive("--snippet-length", args.snippet_length)

    config = SearchConfig(
        candidate_limit=args.candidate_limit,
        result_limit=args.result_limit,
    )
    service = create_search_service(device=args.device, config=config)
    report = run_evaluation(
        service,
        candidate_limit=args.candidate_limit,
        result_limit=args.result_limit,
    )
    print()
    print(format_report(report, snippet_length=args.snippet_length))


if __name__ == "__main__":
    main()
