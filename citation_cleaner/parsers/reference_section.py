"""Reference-section location and splitting."""

from __future__ import annotations

from dataclasses import dataclass
import re

REFERENCE_HEADING_RE = re.compile(r"^\s*(references|bibliography|works cited|参考文献)\s*$", re.IGNORECASE | re.MULTILINE)
NEXT_SECTION_RE = re.compile(
    r"^\s*(appendix|appendices|acknowledg(e)?ments?|supplementary materials?|funding|author contributions)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class ReferenceSection:
    text: str
    start_page: int
    heading: str


def _normalize_lines(text: str) -> list[str]:
    lines = []
    for line in text.replace("\r", "\n").split("\n"):
        cleaned = re.sub(r"\s+", " ", line).strip()
        if cleaned:
            lines.append(cleaned)
        else:
            lines.append("")
    return lines


def locate_reference_section(page_texts: list[str]) -> ReferenceSection | None:
    """Find the last plausible references heading and return all text after it."""
    candidates: list[tuple[int, int, re.Match[str]]] = []
    for page_index, text in enumerate(page_texts):
        for match in REFERENCE_HEADING_RE.finditer(text):
            candidates.append((page_index, match.start(), match))
    if not candidates:
        return None
    page_index, start, match = candidates[-1]
    section_parts = [page_texts[page_index][match.end() :]]
    section_parts.extend(page_texts[page_index + 1 :])
    text = "\n".join(section_parts)
    next_match = NEXT_SECTION_RE.search(text)
    if next_match:
        text = text[: next_match.start()]
    return ReferenceSection(text=text.strip(), start_page=page_index + 1, heading=match.group(1))


def _merge_wrapped_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []
    for line in lines:
        if not line:
            if merged and merged[-1]:
                merged.append("")
            continue
        if merged and merged[-1] and not _looks_like_new_reference(line):
            separator = "" if merged[-1].endswith("-") else " "
            if merged[-1].endswith("-"):
                merged[-1] = merged[-1][:-1] + line
            else:
                merged[-1] = merged[-1] + separator + line
        else:
            merged.append(line)
    return merged


def _looks_like_new_reference(line: str) -> bool:
    if re.match(r"^\s*(?:\[\d+\]|\d+\.)\s+", line):
        return True
    if re.match(r"^\s*[A-Z][A-Za-z'`-]+,\s+(?:[A-Z]\.|[A-Z][a-z]+)", line):
        return True
    if re.match(r"^\s*[A-Z][A-Za-z'`-]+\s+(?:and|&|et al\.|[A-Z]\.)", line):
        return True
    return False


def split_references(section_text: str) -> list[str]:
    """Split a references section into raw reference strings."""
    lines = _normalize_lines(section_text)
    text = "\n".join(lines)

    numbered_patterns = [
        r"(?m)^\s*\[(\d+)\]\s+",
        r"(?m)^\s*(\d+)\.\s+",
    ]
    for pattern in numbered_patterns:
        matches = list(re.finditer(pattern, text))
        if len(matches) >= 2:
            refs: list[str] = []
            for i, match in enumerate(matches):
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                refs.append(re.sub(r"\s+", " ", text[match.start() : end]).strip())
            return [ref for ref in refs if _valid_ref_candidate(ref)]

    if "\n\n" in text:
        parts = [re.sub(r"\s+", " ", part).strip() for part in re.split(r"\n\s*\n", text)]
        parts = [part for part in parts if _valid_ref_candidate(part)]
        if len(parts) >= 2:
            return parts

    merged = _merge_wrapped_lines(lines)
    refs: list[str] = []
    current: list[str] = []
    for line in merged:
        if not line:
            continue
        if current and _looks_like_new_reference(line):
            refs.append(re.sub(r"\s+", " ", " ".join(current)).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        refs.append(re.sub(r"\s+", " ", " ".join(current)).strip())
    return [ref for ref in refs if _valid_ref_candidate(ref)]


def _valid_ref_candidate(ref: str) -> bool:
    if len(ref) < 20:
        return False
    if re.fullmatch(r"\d+", ref):
        return False
    return True
