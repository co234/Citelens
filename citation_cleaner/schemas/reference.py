"""Reference schemas and stable helpers.

This module is the single source of truth for reference-shaped records.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError


class Author(BaseModel):
    surname: str
    given: Optional[str] = None


WorkType = Literal["article", "book", "inproceedings", "chapter", "preprint", "unknown"]


class RawRefRow(BaseModel):
    citing_paper_id: str = ""
    raw: str


class CleanedRef(BaseModel):
    citing_paper_id: str = ""
    raw: str
    cleaned: str


class ExtractedReference(BaseModel):
    raw: str
    citing_paper_id: str = ""
    authors: list[Author] = Field(default_factory=list)
    year: Optional[int] = None
    year_suffix: Optional[str] = None
    title: Optional[str] = None
    venue: Optional[str] = None
    type: WorkType = "unknown"
    doi: Optional[str] = None


class CanonicalReference(BaseModel):
    canonical_id: str
    authors: list[Author]
    year: Optional[int] = None
    title: str
    venue: Optional[str] = None
    type: WorkType = "unknown"
    doi: Optional[str] = None
    volume: Optional[str] = None
    pages: Optional[str] = None
    raw_variants: list[str] = Field(default_factory=list)
    confidence: float = 1.0


_PARTICLES = ("van", "von", "der", "de", "del", "la", "le", "da")


def normalize_surname_for_blocking(surname: str) -> str:
    """Return a comparison key; never use this as display text."""
    s = unicodedata.normalize("NFKD", surname or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z]", "", s.lower())
    for particle in _PARTICLES:
        if s.startswith(particle) and len(s) > len(particle) + 2:
            s = s[len(particle) :]
    return s


def make_canonical_id(
    *,
    authors: list[dict] | list[Author] | None,
    year: Optional[int],
    title: Optional[str],
) -> str:
    if authors and isinstance(authors[0], Author):
        first = authors[0].surname
    elif authors:
        first = str(authors[0].get("surname", ""))
    else:
        first = ""
    key = "|".join([first.lower(), str(year or ""), (title or "").lower()[:60]])
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def validate_extracted_batch(parsed: list[dict]) -> tuple[list[ExtractedReference], list[dict]]:
    valid: list[ExtractedReference] = []
    quarantined: list[dict] = []
    for item in parsed:
        try:
            valid.append(ExtractedReference.model_validate(item))
        except ValidationError as exc:
            quarantined.append({**item, "_error": str(exc)})
    return valid, quarantined
