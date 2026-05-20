#!/usr/bin/env python3
"""Repair common mojibake in SEP JSON artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from text_repair import repair_json_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair mojibake in SEP JSON artifacts in place.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        default=[
            Path("sep_sections.json"),
            Path("sep_chunks.json"),
            Path("sep_entries.json"),
        ],
        help="JSON files to repair in place.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for path in args.files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        repaired = repair_json_value(payload)
        path.write_text(
            json.dumps(repaired, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Repaired {path}")


if __name__ == "__main__":
    main()
