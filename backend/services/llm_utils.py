"""Shared helpers for working with LLM completion output."""

from __future__ import annotations


def strip_json_fences(text: str) -> str:
    """Remove optional markdown JSON code fences from an LLM response."""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```")
        cleaned = cleaned.strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[: -3].strip()
    return cleaned
