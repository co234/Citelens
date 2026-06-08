"""Detect in-text citations and link them to bibliography entries."""

from __future__ import annotations

from dataclasses import dataclass
import re

from citation_cleaner.schemas.citance import Citance, CitationMarker
from citation_cleaner.schemas.reference import normalize_surname_for_blocking

BRACKET_RE = re.compile(r"\[(\d+(?:\s*(?:,|-|\u2013)\s*\d+)*)\]")
PAREN_AY_RE = re.compile(r"\(([^()]{0,160}?(?:19|20)\d{2}[a-z]?[^()]*)\)")
NARRATIVE_AY_RE = re.compile(
    r"\b([A-Z][A-Za-z'`-]+)(?:\s+et\s+al\.?|\s+(?:and|&)\s+[A-Z][A-Za-z'`-]+)?\s*\(((?:19|20)\d{2})([a-z])?\)"
)
AY_PAIR_RE = re.compile(
    r"([A-Z][A-Za-z'`-]+)(?:\s+et\s+al\.?|\s+(?:and|&)\s+[A-Z][A-Za-z'`-]+)?[, ]+\s*((?:19|20)\d{2})([a-z])?"
)
YEAR_RE = re.compile(r"\b((?:19|20)\d{2})([a-z])?\b")


@dataclass(frozen=True)
class PageText:
    page: int
    text: str


def _parse_number_list(raw: str) -> list[int]:
    out: list[int] = []
    for piece in re.split(r"\s*,\s*", raw):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece or "\u2013" in piece:
            bits = re.split(r"\s*[-\u2013]\s*", piece)
            if len(bits) == 2 and bits[0].isdigit() and bits[1].isdigit():
                start, end = int(bits[0]), int(bits[1])
                if start <= end and end - start <= 50:
                    out.extend(range(start, end + 1))
            continue
        if piece.isdigit():
            out.append(int(piece))
    return out


def _first_author_surname_from_ref(raw_ref: str) -> str | None:
    ref = re.sub(r"^\s*(?:\[\d+\]|\d+\.)\s*", "", raw_ref).strip()
    year_match = YEAR_RE.search(ref)
    head = ref[: year_match.start()] if year_match else ref[:120]
    head = re.sub(r"\bet\s+al\.?", "", head, flags=re.IGNORECASE)
    if "," in head:
        candidate = head.split(",", 1)[0]
    else:
        candidate = re.split(r"\s+(?:and|&)\s+", head, maxsplit=1)[0]
        tokens = [t.strip(" .") for t in candidate.split() if t.strip(" .")]
        tokens = [t for t in tokens if not re.fullmatch(r"[A-Z]\.?", t)]
        candidate = tokens[-1] if tokens else candidate
    candidate = candidate.strip(" .;:")
    return candidate or None


def build_author_year_index(references: list[str]) -> dict[tuple[str, int], list[str]]:
    index: dict[tuple[str, int], list[str]] = {}
    for ref in references:
        year_match = YEAR_RE.search(ref)
        surname = _first_author_surname_from_ref(ref)
        if not year_match or not surname:
            continue
        key = (normalize_surname_for_blocking(surname), int(year_match.group(1)))
        index.setdefault(key, []).append(ref)
    return index


def extract_context(text: str, start: int, end: int, max_chars: int = 300) -> str:
    spans = list(re.finditer(r"[^.!?]+[.!?]?", text, flags=re.MULTILINE))
    selected = None
    for i, sentence in enumerate(spans):
        if sentence.start() <= start <= sentence.end():
            first = max(0, i - 1)
            last = min(len(spans), i + 2)
            selected = " ".join(s.group(0).strip() for s in spans[first:last])
            break
    if selected is None:
        selected = text[max(0, start - max_chars // 2) : min(len(text), end + max_chars // 2)]
    selected = re.sub(r"\s+", " ", selected).strip()
    if len(selected) <= max_chars:
        return selected
    marker_mid = max(0, min(len(selected), max_chars // 2))
    return selected[:max_chars].strip() if marker_mid else selected[:max_chars].strip()


def link_citances(
    page_texts: list[PageText],
    references: list[str],
    citing_paper_id: str,
    llm_fallback=None,
) -> list[Citance]:
    citances: list[Citance] = []
    ay_index = build_author_year_index(references)
    seen: set[tuple[str, int, int, str | None]] = set()

    for page in page_texts:
        text = page.text
        for match in BRACKET_RE.finditer(text):
            numbers = _parse_number_list(match.group(1))
            marker = CitationMarker(
                style="numbered_bracket",
                raw_marker=match.group(0),
                page=page.page,
                char_span=(match.start(), match.end()),
            )
            context = extract_context(text, match.start(), match.end())
            for number in numbers:
                raw_ref = references[number - 1] if 1 <= number <= len(references) else None
                method = "numbered_direct" if raw_ref else "unlinked"
                confidence = 1.0 if raw_ref else 0.0
                key = (marker.raw_marker, marker.page, number, raw_ref)
                if key in seen:
                    continue
                seen.add(key)
                citances.append(
                    Citance(
                        citing_paper_id=citing_paper_id,
                        raw_ref=raw_ref,
                        context=context,
                        marker=marker,
                        link_method=method,
                        link_confidence=confidence,
                    )
                )

        for match in PAREN_AY_RE.finditer(text):
            inner = match.group(1)
            for pair in AY_PAIR_RE.finditer(inner):
                surname, year = pair.group(1), int(pair.group(2))
                raw_ref, method, confidence = _link_author_year(
                    surname=surname,
                    year=year,
                    raw_marker=match.group(0),
                    context=extract_context(text, match.start(), match.end()),
                    ay_index=ay_index,
                    llm_fallback=llm_fallback,
                )
                marker = CitationMarker(
                    style="paren_author_year",
                    raw_marker=match.group(0),
                    page=page.page,
                    char_span=(match.start(), match.end()),
                )
                key = (marker.raw_marker, marker.page, match.start(), raw_ref)
                if key in seen:
                    continue
                seen.add(key)
                citances.append(
                    Citance(
                        citing_paper_id=citing_paper_id,
                        raw_ref=raw_ref,
                        context=extract_context(text, match.start(), match.end()),
                        marker=marker,
                        link_method=method,
                        link_confidence=confidence,
                    )
                )

        for match in NARRATIVE_AY_RE.finditer(text):
            surname, year = match.group(1), int(match.group(2))
            context = extract_context(text, match.start(), match.end())
            raw_ref, method, confidence = _link_author_year(
                surname=surname,
                year=year,
                raw_marker=match.group(0),
                context=context,
                ay_index=ay_index,
                llm_fallback=llm_fallback,
            )
            marker = CitationMarker(
                style="narrative_author_year",
                raw_marker=match.group(0),
                page=page.page,
                char_span=(match.start(), match.end()),
            )
            key = (marker.raw_marker, marker.page, match.start(), raw_ref)
            if key in seen:
                continue
            seen.add(key)
            citances.append(
                Citance(
                    citing_paper_id=citing_paper_id,
                    raw_ref=raw_ref,
                    context=context,
                    marker=marker,
                    link_method=method,
                    link_confidence=confidence,
                )
            )
    return citances


def _link_author_year(
    *,
    surname: str,
    year: int,
    raw_marker: str,
    context: str,
    ay_index: dict[tuple[str, int], list[str]],
    llm_fallback=None,
) -> tuple[str | None, str, float]:
    candidates = ay_index.get((normalize_surname_for_blocking(surname), year), [])
    if len(candidates) == 1:
        return candidates[0], "author_year_unique", 0.9
    if len(candidates) > 1 and llm_fallback is not None:
        index, confidence = llm_fallback.disambiguate_author_year(raw_marker, context, candidates)
        if index is not None:
            return candidates[index], "llm_disambiguated", confidence
    return None, "unlinked", 0.0
