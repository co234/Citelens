"""Deterministic Stage 2 extraction for offline smoke tests."""

from __future__ import annotations

import re

from citation_cleaner.schemas.reference import ExtractedReference

_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})([a-z])?\b")


def extract_batch_dryrun(batch: list[str], citing_paper_ids: list[str] | None = None) -> tuple[list[ExtractedReference], list[dict]]:
    out: list[ExtractedReference] = []
    citing_paper_ids = citing_paper_ids or [""] * len(batch)
    for raw, citing_paper_id in zip(batch, citing_paper_ids):
        year_match = _YEAR_RE.search(raw)
        year = int(year_match.group(1)) if year_match else None
        suffix = year_match.group(2) if year_match and year_match.group(2) else None
        if year_match:
            head = raw[: year_match.start()].rstrip(" .,(")
            tail = raw[year_match.end() :].lstrip(" .,)")
        else:
            head, tail = raw, ""

        first_chunk = re.split(r"(?:,|\band\b|&)", head, maxsplit=1)[0].strip()
        tokens = [t for t in re.split(r"\s+", first_chunk) if t and not re.fullmatch(r"[A-Z]\.?", t)]
        surname = tokens[-1].strip(".,") if tokens else first_chunk or "Unknown"
        title = tail.split(".")[0].strip(" .") or None
        item = {
            "raw": raw,
            "citing_paper_id": citing_paper_id,
            "authors": [{"surname": surname, "given": None}],
            "year": year,
            "year_suffix": suffix,
            "title": title,
            "venue": None,
            "type": "unknown",
            "doi": None,
        }
        try:
            out.append(ExtractedReference.model_validate(item))
        except Exception as exc:
            return out, [{"raw": raw, "_error": str(exc)}]
    return out, []
