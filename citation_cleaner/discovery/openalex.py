"""OpenAlex client for finding source papers and downloading their PDFs.

OpenAlex is a free, no-key scholarly index. Polite pool requires an email,
set via CITATION_CLEANER_EMAIL. Without it we still work, just rate-limited.

v4 additions:
  - search_by_author(): resolve an author name to an OpenAlex author id and
    pull their top-N works by cited_by_count.
  - search_by_terminology() now accepts a `sort_by` argument so callers can
    explicitly request relevance / recency / citation-count ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
import time
from pathlib import Path
from typing import Literal, Optional

import requests


OPENALEX_BASE = "https://api.openalex.org"
USER_AGENT = "citation-cleaner-v4 (https://github.com/local/citation-cleaner)"

# How callers ask for an ordering. Maps to OpenAlex `sort` strings inside the
# search functions. "relevance" delegates to OpenAlex's default ranking
# (which already factors in citations + textual match).
SortBy = Literal["relevance", "recency", "citations"]


def _polite_params() -> dict:
    """OpenAlex 'polite pool' params: pass mailto to get higher rate limits."""
    email = os.environ.get("CITATION_CLEANER_EMAIL")
    return {"mailto": email} if email else {}


@dataclass
class DiscoveredPaper:
    """A paper found by OpenAlex search, before we download its PDF."""

    openalex_id: str          # e.g. "W2741809807" or full URL
    title: str
    year: Optional[int]
    doi: Optional[str]
    cited_by_count: int
    pdf_url: Optional[str]    # best-effort open access PDF URL
    relevance_score: float    # OpenAlex's relevance_score field
    authors: list[str] = field(default_factory=list)
    venue: Optional[str] = None

    def short_id(self) -> str:
        """Return a filesystem-safe short id for naming PDF files."""
        # OpenAlex IDs look like "W2741809807" — keep just the work id
        match = re.search(r"W\d+", self.openalex_id)
        return match.group(0) if match else "Wunknown"


_WORK_SELECT = (
    "id,title,publication_year,doi,cited_by_count,relevance_score,"
    "open_access,best_oa_location,authorships,primary_location"
)


def _work_to_discovered(work: dict, require_pdf: bool) -> Optional["DiscoveredPaper"]:
    """Shared row builder used by every search_by_* function."""
    pdf_url = _extract_pdf_url(work)
    if require_pdf and not pdf_url:
        return None
    authors = [
        (a.get("author") or {}).get("display_name", "")
        for a in (work.get("authorships") or [])
    ]
    venue = ((work.get("primary_location") or {}).get("source") or {}).get(
        "display_name"
    )
    return DiscoveredPaper(
        openalex_id=work.get("id", ""),
        title=(work.get("title") or "").strip(),
        year=work.get("publication_year"),
        doi=work.get("doi"),
        cited_by_count=int(work.get("cited_by_count", 0) or 0),
        pdf_url=pdf_url,
        relevance_score=float(work.get("relevance_score") or 0.0),
        authors=[a for a in authors if a],
        venue=venue,
    )


def _apply_local_sort(
    papers: list["DiscoveredPaper"], sort_by: SortBy
) -> list["DiscoveredPaper"]:
    """Defensive client-side sort. OpenAlex usually returns rows in the order
    we asked for, but PDF-availability filtering can leave gaps, so we re-sort
    locally to make the contract obvious to the caller."""
    if sort_by == "recency":
        return sorted(papers, key=lambda p: (p.year or 0), reverse=True)
    if sort_by == "citations":
        return sorted(papers, key=lambda p: p.cited_by_count, reverse=True)
    # "relevance" — OpenAlex's relevance_score is only populated when the
    # request actually used a `search=` clause. Fall back to citation count
    # for results that don't carry one (e.g. author-works listings).
    return sorted(
        papers,
        key=lambda p: (p.relevance_score, p.cited_by_count),
        reverse=True,
    )


def search_by_terminology(
    query: str,
    *,
    n: int = 3,
    require_pdf: bool = True,
    max_candidates: int = 25,
    min_year: Optional[int] = None,
    sort_by: SortBy = "relevance",
) -> list[DiscoveredPaper]:
    """Search OpenAlex for papers matching `query`.

    Strategy:
      1. Search with the requested ordering. OpenAlex returns up to
         `max_candidates`.
      2. Filter to ones with an accessible PDF if `require_pdf`.
      3. Re-sort locally per `sort_by` and take top `n`.

    We oversample (max_candidates) so that filtering by PDF availability
    still leaves us with `n` results.

    `sort_by`:
      - "relevance" — default. Lets OpenAlex pick the order (textual match ×
        citations).
      - "recency" — most-recent first (publication_date:desc).
      - "citations" — most-cited first (cited_by_count:desc).
    """
    params: dict = {
        "search": query,
        "per-page": min(max_candidates, 50),
        "select": _WORK_SELECT,
        **_polite_params(),
    }
    # Tell OpenAlex how to sort *before* it paginates, so the rows we get back
    # are already biased toward what we want. Without `sort`, OpenAlex uses
    # its default relevance ordering.
    if sort_by == "recency":
        params["sort"] = "publication_date:desc"
    elif sort_by == "citations":
        params["sort"] = "cited_by_count:desc"
    if min_year:
        params["filter"] = f"from_publication_year:{min_year}"

    response = requests.get(
        f"{OPENALEX_BASE}/works",
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    discovered: list[DiscoveredPaper] = []
    for work in payload.get("results", []):
        rec = _work_to_discovered(work, require_pdf=require_pdf)
        if rec is not None:
            discovered.append(rec)

    discovered = _apply_local_sort(discovered, sort_by)
    return discovered[:n]


# ----- Author mode -----

@dataclass
class ResolvedAuthor:
    """An OpenAlex author resolved from a free-text name."""

    author_id: str            # full URL form, e.g. "https://openalex.org/A1969205032"
    display_name: str
    works_count: int
    cited_by_count: int

    def short_id(self) -> str:
        match = re.search(r"A\d+", self.author_id)
        return match.group(0) if match else "Aunknown"


def resolve_author(name: str, *, timeout: int = 30) -> Optional[ResolvedAuthor]:
    """Resolve a free-text author name to an OpenAlex author record.

    Uses OpenAlex's `?search=` on /authors, which does fuzzy / partial matching
    on display_name and known aliases. When several authors share a name we
    pick the one with the highest `cited_by_count` — that's almost always the
    senior author the user meant. For genuine ambiguity, callers can pass an
    explicit OpenAlex author ID (Axxxxxxxxx) instead of a name; we detect
    that and skip the search.
    """
    name = (name or "").strip()
    if not name:
        return None

    # If the user passed a raw OpenAlex author ID, look it up directly.
    if re.fullmatch(r"A\d+", name) or re.fullmatch(
        r"https?://openalex\.org/A\d+", name
    ):
        author_id_short = re.search(r"A\d+", name).group(0)  # type: ignore[union-attr]
        resp = requests.get(
            f"{OPENALEX_BASE}/authors/{author_id_short}",
            params={
                "select": "id,display_name,works_count,cited_by_count",
                **_polite_params(),
            },
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return None
        row = resp.json()
        return ResolvedAuthor(
            author_id=row.get("id", ""),
            display_name=row.get("display_name", name),
            works_count=int(row.get("works_count", 0) or 0),
            cited_by_count=int(row.get("cited_by_count", 0) or 0),
        )

    resp = requests.get(
        f"{OPENALEX_BASE}/authors",
        params={
            "search": name,
            "per-page": 10,
            "select": "id,display_name,works_count,cited_by_count",
            **_polite_params(),
        },
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    resp.raise_for_status()
    candidates = resp.json().get("results", [])
    if not candidates:
        return None
    # Multiple authors can share a name. Pick the one with the most citations,
    # which is in practice the well-known author the user is asking about.
    candidates.sort(key=lambda r: int(r.get("cited_by_count", 0) or 0), reverse=True)
    top = candidates[0]
    return ResolvedAuthor(
        author_id=top.get("id", ""),
        display_name=top.get("display_name", name),
        works_count=int(top.get("works_count", 0) or 0),
        cited_by_count=int(top.get("cited_by_count", 0) or 0),
    )


def search_by_author(
    name: str,
    *,
    n: int = 3,
    require_pdf: bool = True,
    max_candidates: int = 50,
    min_year: Optional[int] = None,
) -> tuple[Optional[ResolvedAuthor], list[DiscoveredPaper]]:
    """Find an author's top-N most-cited works.

    Returns (resolved_author, papers). If the author couldn't be resolved,
    returns (None, []). The author-mode contract is always "most cited first"
    — that's the only ordering this entry point exposes, per the v4 spec.
    """
    author = resolve_author(name)
    if author is None:
        return None, []

    author_id_short = author.short_id()
    # OpenAlex `filter=author.id:` accepts either the short or full id.
    filter_parts = [f"author.id:{author_id_short}"]
    if min_year:
        filter_parts.append(f"from_publication_year:{min_year}")

    params = {
        "filter": ",".join(filter_parts),
        "sort": "cited_by_count:desc",
        "per-page": min(max_candidates, 50),
        "select": _WORK_SELECT,
        **_polite_params(),
    }
    resp = requests.get(
        f"{OPENALEX_BASE}/works",
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()

    discovered: list[DiscoveredPaper] = []
    for work in payload.get("results", []):
        rec = _work_to_discovered(work, require_pdf=require_pdf)
        if rec is not None:
            discovered.append(rec)

    discovered = _apply_local_sort(discovered, "citations")
    return author, discovered[:n]


def _extract_pdf_url(work: dict) -> Optional[str]:
    """Pull a best-effort open-access PDF URL from an OpenAlex work record."""
    best = work.get("best_oa_location") or {}
    if best.get("pdf_url"):
        return best["pdf_url"]
    # Some records expose pdf via open_access.oa_url
    oa = work.get("open_access") or {}
    if oa.get("oa_url") and oa["oa_url"].lower().endswith(".pdf"):
        return oa["oa_url"]
    return None


def fetch_pdf(
    paper: DiscoveredPaper,
    out_dir: Path,
    *,
    timeout: int = 60,
    retries: int = 2,
) -> Optional[Path]:
    """Download `paper`'s PDF to out_dir. Returns the path or None on failure.

    File is named `<openalex_short_id>.pdf` for traceability — that id is what
    we'll use as `citing_paper_id` downstream.
    """
    if not paper.pdf_url:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{paper.short_id()}.pdf"
    if out_path.exists() and out_path.stat().st_size > 1000:
        return out_path  # already downloaded

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(
                paper.pdf_url,
                headers={"User-Agent": USER_AGENT},
                timeout=timeout,
                stream=True,
                allow_redirects=True,
            )
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "").lower()
            if "pdf" not in content_type and not paper.pdf_url.lower().endswith(".pdf"):
                # Some publishers serve HTML behind a "PDF" URL — skip.
                return None
            with out_path.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=65536):
                    fh.write(chunk)
            if out_path.stat().st_size < 1000:
                out_path.unlink(missing_ok=True)
                return None
            return out_path
        except (requests.RequestException, OSError) as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    # All retries failed.
    if last_err:
        print(f"  [warn] failed to fetch {paper.short_id()}: {last_err}")
    return None
