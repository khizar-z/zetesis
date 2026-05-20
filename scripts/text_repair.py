"""Utilities for repairing common mojibake in SEP text artifacts."""

from __future__ import annotations

import re
from typing import Any

MOJIBAKE_PATTERN = re.compile(r"[ÃÂâ]|[\u0080-\u009f]")


def looks_like_mojibake(text: str) -> bool:
    return bool(MOJIBAKE_PATTERN.search(text))


def fix_mojibake(text: str) -> str:
    fixed = text
    for _ in range(2):
        if not looks_like_mojibake(fixed):
            break
        try:
            candidate = fixed.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        if candidate == fixed:
            break
        fixed = candidate
    return fixed


def repair_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return fix_mojibake(value)
    if isinstance(value, list):
        return [repair_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: repair_json_value(item) for key, item in value.items()}
    return value
