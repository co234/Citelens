"""Quality signals for Stage 0 reference extraction."""

from __future__ import annotations

from dataclasses import dataclass
import re

from citation_cleaner.schemas.document import QualityReport

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b")
_AUTHORLIKE_RE = re.compile(r"^\s*(?:\[\d+\]\s*|\d+\.\s*)?[A-Z][A-Za-z'`-]{2,}")


@dataclass(frozen=True)
class QualityThresholds:
    min_refs: int = 5
    min_pct_with_year: float = 0.50
    min_pct_authorlike: float = 0.60

    @classmethod
    def default(cls) -> "QualityThresholds":
        return cls()


def score_reference_quality(
    references: list[str],
    thresholds: QualityThresholds = QualityThresholds.default(),
    llm_fallback_invoked: bool = False,
) -> QualityReport:
    n_refs = len(references)
    if n_refs == 0:
        return QualityReport(
            n_refs_extracted=0,
            pct_with_year=0.0,
            pct_with_authorlike=0.0,
            needs_llm_fallback=True,
            llm_fallback_invoked=llm_fallback_invoked,
            notes=["no references extracted"],
        )
    with_year = sum(1 for ref in references if _YEAR_RE.search(ref))
    authorlike = sum(1 for ref in references if _AUTHORLIKE_RE.search(ref))
    pct_with_year = with_year / n_refs
    pct_authorlike = authorlike / n_refs
    notes: list[str] = []
    if n_refs < thresholds.min_refs:
        notes.append(f"n_refs_extracted below threshold: {n_refs} < {thresholds.min_refs}")
    if pct_with_year < thresholds.min_pct_with_year:
        notes.append(f"pct_with_year below threshold: {pct_with_year:.2f} < {thresholds.min_pct_with_year:.2f}")
    if pct_authorlike < thresholds.min_pct_authorlike:
        notes.append(f"pct_with_authorlike below threshold: {pct_authorlike:.2f} < {thresholds.min_pct_authorlike:.2f}")
    return QualityReport(
        n_refs_extracted=n_refs,
        pct_with_year=pct_with_year,
        pct_with_authorlike=pct_authorlike,
        needs_llm_fallback=bool(notes),
        llm_fallback_invoked=llm_fallback_invoked,
        notes=notes,
    )
