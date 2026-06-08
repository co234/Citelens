"""Stage 3 surname/year blocking."""

from __future__ import annotations

from collections import defaultdict

from citation_cleaner.schemas.reference import ExtractedReference, normalize_surname_for_blocking


def make_block_key(record: ExtractedReference | dict) -> tuple[str, int | None]:
    data = record.model_dump() if isinstance(record, ExtractedReference) else record
    authors = data.get("authors") or []
    if authors and hasattr(authors[0], "surname"):
        first = authors[0].surname
    elif authors:
        first = authors[0].get("surname", "")
    else:
        first = ""
    return (normalize_surname_for_blocking(first), data.get("year"))


def group_by_block(records: list[ExtractedReference]) -> dict[tuple[str, int | None], list[ExtractedReference]]:
    blocks: dict[tuple[str, int | None], list[ExtractedReference]] = defaultdict(list)
    for record in records:
        blocks[make_block_key(record)].append(record)
    return dict(blocks)
