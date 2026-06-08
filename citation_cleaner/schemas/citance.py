"""In-text citation occurrence schemas."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

CitanceStyle = Literal[
    "numbered_bracket",
    "numbered_superscript",
    "paren_author_year",
    "narrative_author_year",
]

LinkMethod = Literal[
    "numbered_direct",
    "author_year_unique",
    "llm_disambiguated",
    "unlinked",
]


class CitationMarker(BaseModel):
    style: CitanceStyle
    raw_marker: str
    page: int
    char_span: tuple[int, int]


class Citance(BaseModel):
    citing_paper_id: str
    raw_ref: Optional[str] = None
    context: str
    marker: CitationMarker
    link_method: LinkMethod
    link_confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _raw_ref_matches_link_method(self):
        if self.link_method == "unlinked" and self.raw_ref is not None:
            raise ValueError("raw_ref must be None when link_method is unlinked")
        if self.link_method != "unlinked" and not self.raw_ref:
            raise ValueError("raw_ref is required when link_method is linked")
        return self
