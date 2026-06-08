"""External tools available to the Stage 6 disambiguation agent."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import requests

CROSSREF_URL = "https://api.crossref.org/works"
OPENALEX_URL = "https://api.openalex.org/works"

_POLITE_EMAIL = os.environ.get("CITATION_CLEANER_EMAIL", "")
_UA = f"citation-cleaner-v2/0.2 (mailto:{_POLITE_EMAIL})" if _POLITE_EMAIL else "citation-cleaner-v2/0.2"
_citing_context_fn = None


def search_crossref(title: str, author: Optional[str] = None, year: Optional[int] = None, rows: int = 5) -> list[dict]:
    params: dict = {"query.bibliographic": title, "rows": rows}
    if author:
        params["query.author"] = author
    if year:
        params["filter"] = f"from-pub-date:{year},until-pub-date:{year}"
    response = requests.get(CROSSREF_URL, params=params, headers={"User-Agent": _UA}, timeout=10)
    response.raise_for_status()
    items = response.json().get("message", {}).get("items", [])
    return [
        {
            "title": (item.get("title") or [""])[0],
            "authors": [
                {"family": author_item.get("family"), "given": author_item.get("given")}
                for author_item in item.get("author", [])
            ],
            "year": (item.get("issued", {}).get("date-parts") or [[None]])[0][0],
            "doi": item.get("DOI"),
            "container_title": (item.get("container-title") or [None])[0],
            "type": item.get("type"),
        }
        for item in items
    ]


def search_openalex(query: str, per_page: int = 5) -> list[dict]:
    params = {"search": query, "per_page": per_page}
    if _POLITE_EMAIL:
        params["mailto"] = _POLITE_EMAIL
    response = requests.get(OPENALEX_URL, params=params, timeout=10)
    response.raise_for_status()
    return [
        {
            "id": work.get("id"),
            "title": work.get("title"),
            "authors": [
                author["author"]["display_name"]
                for author in work.get("authorships", [])
                if author.get("author", {}).get("display_name")
            ],
            "year": work.get("publication_year"),
            "doi": work.get("doi"),
            "type": work.get("type"),
            "venue": ((work.get("primary_location") or {}).get("source") or {}).get("display_name"),
        }
        for work in response.json().get("results", [])
    ]


def register_citing_context_provider(fn) -> None:
    global _citing_context_fn
    _citing_context_fn = fn


def load_citances_jsonl(path: str | os.PathLike) -> int:
    index: dict[tuple[str, str], str] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw_ref = row.get("raw_ref") or row.get("raw")
            if raw_ref is None:
                continue
            index[(row["citing_paper_id"], raw_ref)] = row["context"]

    def provider(raw_ref: str, citing_paper_id: str) -> Optional[str]:
        return index.get((citing_paper_id, raw_ref))

    register_citing_context_provider(provider)
    return len(index)


def lookup_citing_context(raw_ref: str, citing_paper_id: str) -> dict:
    if _citing_context_fn is None:
        return {"context": None, "note": "No citing-context provider registered."}
    context = _citing_context_fn(raw_ref, citing_paper_id)
    if context is None:
        return {"context": None, "note": "No citance found for this raw_ref and citing_paper_id."}
    return {"context": context[:500] if isinstance(context, str) else context}


TOOL_DISPATCH = {
    "search_crossref": search_crossref,
    "search_openalex": search_openalex,
    "lookup_citing_context": lookup_citing_context,
}
