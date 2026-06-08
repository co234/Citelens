"""Stage 5 judge and canonical finalization."""

from __future__ import annotations

import json

from citation_cleaner.llm.anthropic import system_block
from citation_cleaner.llm.json_parse import parse_llm_json
from citation_cleaner.schemas.reference import CanonicalReference, make_canonical_id

DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"

JUDGE_SYSTEM = """\
You are a bibliographic entity resolution judge. Decide whether a cluster of
references all point to the SAME published work, and if so, produce a single
canonical record.

Same work = same authors, same title content allowing minor formatting changes,
and same year. Different editions of a book = same work. A book and a similarly
titled journal article = different works.

Output ONLY a JSON object:
{
  "verdict": "same" | "different" | "uncertain",
  "canonical": {
    "authors": [{"surname": "...", "given": "..."|null}, ...],
    "year": int|null,
    "title": str,
    "venue": str|null,
    "type": "article"|"book"|"inproceedings"|"chapter"|"preprint"|"unknown",
    "doi": str|null,
    "volume": str|null,
    "pages": str|null
  } | null,
  "confidence": <float 0..1>,
  "reasoning": "<2-3 sentences>",
  "needs_external_lookup": <bool>
}
If verdict != "same", set canonical to null.
"""


def extract_json_object(text: str) -> dict:
    parsed = parse_llm_json(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
    return parsed


def judge_cluster(cluster: dict, client, model: str = DEFAULT_JUDGE_MODEL) -> dict:
    user = json.dumps({"members": cluster["members"]}, ensure_ascii=False, indent=2)
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0,
        system=system_block(JUDGE_SYSTEM, model),
        messages=[{"role": "user", "content": f"Cluster to evaluate:\n{user}\n\nSame work?"}],
    )
    return extract_json_object(resp.content[0].text)


def judge_cluster_dryrun(cluster: dict) -> dict:
    members = cluster["members"]
    if len(members) <= 1:
        return {
            "verdict": "different",
            "canonical": None,
            "confidence": 1.0,
            "reasoning": "singleton",
            "needs_external_lookup": False,
        }

    def title_chars(member: dict) -> set[str]:
        return set((member.get("title") or "").lower())

    base = title_chars(members[0])
    if not base:
        verdict = "uncertain"
    else:
        overlaps = [
            len(base & title_chars(member)) / max(len(base | title_chars(member)), 1)
            for member in members[1:]
        ]
        verdict = "same" if all(score > 0.6 for score in overlaps) else "different"

    canonical = None
    if verdict == "same":
        best = max(members, key=lambda member: len(member.get("title") or ""))
        canonical = {key: best.get(key) for key in ("authors", "year", "title", "venue", "type", "doi", "volume", "pages")}
    return {
        "verdict": verdict,
        "canonical": canonical,
        "confidence": 0.6,
        "reasoning": "dry-run heuristic",
        "needs_external_lookup": False,
    }


def _titlecase(text: str) -> str:
    try:
        from titlecase import titlecase
    except ImportError:  # pragma: no cover - only when dependency missing
        return text.strip()
    return titlecase(text.strip())


def finalize_canonical(canonical: dict, raw_variants: list[str], confidence: float = 1.0) -> dict:
    canonical = dict(canonical)
    if canonical.get("title"):
        canonical["title"] = _titlecase(canonical["title"])
    canonical["raw_variants"] = raw_variants
    canonical["confidence"] = confidence
    canonical["canonical_id"] = make_canonical_id(
        authors=canonical.get("authors"),
        year=canonical.get("year"),
        title=canonical.get("title"),
    )
    return CanonicalReference.model_validate(canonical).model_dump()
