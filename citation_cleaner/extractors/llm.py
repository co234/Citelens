"""LLM-backed Stage 2 structured reference extraction."""

from __future__ import annotations

import json

from citation_cleaner.llm.anthropic import system_block
from citation_cleaner.llm.json_parse import parse_llm_json
from citation_cleaner.schemas.reference import ExtractedReference, validate_extracted_batch

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

EXTRACTION_SYSTEM = """\
You are a bibliographic data parser. Given raw reference strings extracted from
academic papers, return structured JSON for each. Inputs are heterogeneous:
authors may appear surname-first or given-first, names may be full or initialed,
titles may use inconsistent casing.

Rules:
- Preserve multi-word surname particles such as van der Vaart and de la Cruz.
- Return null for fields you cannot determine confidently. Do not guess venues.
- Return title text as it appears in the source; post-processing handles casing.
- For years such as 1996a, set year=1996 and year_suffix="a".
- Preserve citing_paper_id from the input object when provided.

Output ONLY a JSON array, one object per input, in input order. Schema per item:
{
  "raw": "...",
  "citing_paper_id": "...",
  "authors": [{"surname": "...", "given": "..." | null}, ...],
  "year": int | null,
  "year_suffix": "a"|"b"|... | null,
  "title": str | null,
  "venue": str | null,
  "type": "article"|"book"|"inproceedings"|"chapter"|"preprint"|"unknown",
  "doi": str | null
}
"""


def _build_user_message(batch: list[dict]) -> str:
    lines = "\n".join(
        f"{i + 1}. citing_paper_id={item.get('citing_paper_id', '')!r}; raw={item['raw']!r}"
        for i, item in enumerate(batch)
    )
    return f"Parse the following {len(batch)} reference strings:\n\n{lines}\n\nReturn a JSON array of {len(batch)} objects."


def _flatten_parsed_items(parsed: list) -> list:
    """Expand nested lists the LLM occasionally returns as a single array element."""
    flat: list = []
    for item in parsed:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def _call_llm(batch: list[dict], client, model: str) -> list[dict]:
    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0,
        system=system_block(EXTRACTION_SYSTEM, model),
        messages=[{"role": "user", "content": _build_user_message(batch)}],
    )
    parsed = parse_llm_json(resp.content[0].text)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")
    parsed = _flatten_parsed_items(parsed)
    if len(parsed) != len(batch):
        raise ValueError(f"LLM returned {len(parsed)} items for batch of {len(batch)}")
    return parsed


def extract_batch(batch: list[dict], client, model: str = DEFAULT_MODEL) -> tuple[list[ExtractedReference], list[dict]]:
    try:
        parsed = _call_llm(batch, client, model)
    except (json.JSONDecodeError, ValueError) as exc:
        if len(batch) <= 1:
            return [], [{"raw": batch[0]["raw"] if batch else "", "_error": f"single-item parse failure: {exc}"}]
        mid = len(batch) // 2
        valid_a, quarantine_a = extract_batch(batch[:mid], client, model)
        valid_b, quarantine_b = extract_batch(batch[mid:], client, model)
        return valid_a + valid_b, quarantine_a + quarantine_b
    return validate_extracted_batch(parsed)
