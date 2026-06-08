"""Document-level schemas produced by Stage 0."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .citance import Citance
from .reference import Author


class HeaderMeta(BaseModel):
    title: Optional[str] = None
    authors: list[Author] = Field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None


class QualityReport(BaseModel):
    n_refs_extracted: int
    pct_with_year: float
    pct_with_authorlike: float
    needs_llm_fallback: bool
    llm_fallback_invoked: bool = False
    notes: list[str] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    pdf_path: str
    citing_paper_id: str
    header: HeaderMeta
    references: list[str]
    citances: list[Citance]
    quality: QualityReport
