"""Rule-based reference precleaning."""

from __future__ import annotations

import re


def preclean(raw: str) -> str | None:
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if re.fullmatch(r"\d+", s):
        return None
    if len(s) < 6:
        return None

    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*&\s*", " and ", s)
    s = re.sub(r"'s\b", "", s)
    s = re.sub(r",\s*pp?\.\s*\d+(\s*[-\u2013]\s*\d+)?\s*\)", ")", s)
    s = re.sub(r"\s+pp?\.\s*\d+(\s*[-\u2013]\s*\d+)?\s*\)", ")", s)
    s = re.sub(r",\s*pp?\.\s*\d+(\s*[-\u2013]\s*\d+)?\s*$", "", s)
    s = re.sub(r"\s+pp?\.\s*\d+(\s*[-\u2013]\s*\d+)?\s*$", "", s)
    s = re.sub(r"\band([A-Z])", r"and \1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None
