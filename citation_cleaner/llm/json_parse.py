"""Parse JSON from LLM responses, repairing common malformations first."""

from __future__ import annotations

import json
import re
from typing import Any

try:
    import json_repair
except ImportError:  # pragma: no cover - optional at import time in dry tests
    json_repair = None  # type: ignore[assignment]

_LIST_WRAPPER_KEYS = (
    "references",
    "refs",
    "bibliography",
    "entries",
    "items",
    "data",
    "results",
)
_ITEM_TEXT_KEYS = ("raw", "reference", "text", "ref", "citation")


def strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_llm_json(text: str) -> Any:
    """Parse JSON from an LLM response.

    Tries strict ``json.loads`` first; on failure runs ``json_repair.loads``
    to fix truncated strings, trailing commas, etc.
    """
    cleaned = strip_json_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as first_err:
        if json_repair is None:
            raise RuntimeError(
                "json-repair is not installed. pip install json-repair"
            ) from first_err
        try:
            repaired = json_repair.loads(cleaned)
            print("[json] repaired malformed LLM JSON response")
            return repaired
        except Exception:
            # Last resort: extract the outermost array/object and repair that.
            for pattern in (r"\[[\s\S]*", r"\{[\s\S]*"):
                match = re.search(pattern, cleaned)
                if not match:
                    continue
                try:
                    repaired = json_repair.loads(match.group(0))
                    print("[json] repaired malformed LLM JSON response (extracted fragment)")
                    return repaired
                except Exception:
                    continue
            raise first_err


def _string_list_items(items: list[Any]) -> list[str]:
    out: list[str] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
            if text:
                out.append(text)
            continue
        if isinstance(item, dict):
            for key in _ITEM_TEXT_KEYS:
                val = item.get(key)
                if isinstance(val, str) and val.strip():
                    out.append(val.strip())
                    break
    return out


def coerce_json_string_list(parsed: Any) -> list[str] | None:
    """Normalize common LLM shapes into a list of reference strings."""
    if isinstance(parsed, list):
        strings = _string_list_items(parsed)
        return strings or None
    if isinstance(parsed, dict):
        for key in _LIST_WRAPPER_KEYS:
            val = parsed.get(key)
            if isinstance(val, list):
                strings = _string_list_items(val)
                if strings:
                    return strings
        list_vals = [val for val in parsed.values() if isinstance(val, list)]
        if len(list_vals) == 1:
            return coerce_json_string_list(list_vals[0])
    return None
