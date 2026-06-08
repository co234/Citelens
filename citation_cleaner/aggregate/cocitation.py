"""Co-citation aggregation across source papers.

Inputs (from the existing pipeline's workdir):
  - resolved.jsonl       canonical references (one row per dedup'd ref)
  - raw_to_canonical.csv (raw_string -> canonical_id mapping)
  - raw_refs.csv         (citing_paper_id, raw)   -- only present for PDF mode
  - citances.jsonl       in-text citation occurrences (PDF mode, optional)

Outputs:
  - cocited_refs.csv     canonical references ranked by co-citation frequency
  - cocited_refs.md      human-readable report w/ in-context occurrences
  - cocited_refs.json    JSON form for programmatic use
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import requests

from citation_cleaner.preclean.rules import preclean


def _resolve_to_canonical(
    raw: str, raw_to_canon: dict[str, str]
) -> Optional[str]:
    """Map a raw reference string to its canonical_id.

    raw_refs.csv and citances.jsonl store the *original* reference string
    (straight from Stage 0's reference splitter / citance linker), but
    raw_to_canonical.csv is keyed by the *precleaned* string — Stage 2 extracts
    from the precleaned form, so that's what ends up in raw_variants. Those two
    differ whenever preclean rewrites the text (e.g. "&" -> "and", stripped
    "'s", dropped trailing page numbers).

    Try the original first (covers preclean no-ops and CSV-mode fixtures), then
    fall back to the precleaned form so the in-text occurrences actually line up
    with their canonical reference instead of being silently dropped."""
    canon = raw_to_canon.get(raw)
    if canon:
        return canon
    cleaned = preclean(raw)
    if cleaned and cleaned != raw:
        return raw_to_canon.get(cleaned)
    return None


@dataclass
class InContextOccurrence:
    """One in-text citation pulled from citances.jsonl, after we've mapped
    its raw_ref back to a canonical_id.

    page may be None for the CSV-mode codepath, which never has a real
    source PDF, but in PDF mode it should always be set by Stage 0."""

    citing_paper_id: str
    page: Optional[int]
    marker_raw: Optional[str]    # e.g. "[12]" or "Smith et al. 2020"
    context: str                 # the sentence/paragraph around the marker


@dataclass
class CocitedRecord:
    """One canonical reference with cross-paper aggregation info."""

    canonical_id: str
    title: Optional[str]
    authors: list[str]
    year: Optional[int]
    venue: Optional[str]
    doi: Optional[str]
    cocitation_count: int            # how many of the N source papers cited it
    citing_source_ids: list[str]     # which source papers cited it
    global_citation_count: Optional[int] = None  # OpenAlex cited_by_count, if enriched
    confidence: float = 1.0
    # v4: in-text occurrences across all source papers for this reference.
    # Empty list when citances.jsonl wasn't produced (CSV mode, or PDFs the
    # Stage 0 parser couldn't extract markers from).
    occurrences: list[InContextOccurrence] = field(default_factory=list)


def _load_resolved(workdir: Path) -> dict[str, dict]:
    """Load resolved.jsonl into {canonical_id: row} for fast lookup."""
    path = workdir / "resolved.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"resolved.jsonl not found in {workdir}")
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["canonical_id"]] = row
    return out


def _load_raw_to_canonical(workdir: Path) -> dict[str, str]:
    """Map raw reference string -> canonical_id."""
    path = workdir / "raw_to_canonical.csv"
    if not path.exists():
        raise FileNotFoundError(f"raw_to_canonical.csv not found in {workdir}")
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            mapping[row["raw"]] = row["canonical_id"]
    return mapping


def _load_raw_refs(workdir: Path) -> list[tuple[str, str]]:
    """Return list of (citing_paper_id, raw) tuples. Only present in PDF mode."""
    path = workdir / "raw_refs.csv"
    if not path.exists():
        return []
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append((row["citing_paper_id"], row["raw"]))
    return rows


def _load_citances_by_canonical(
    workdir: Path,
    raw_to_canon: dict[str, str],
) -> dict[str, list[InContextOccurrence]]:
    """Walk citances.jsonl and bucket each in-text occurrence under the
    canonical_id its raw_ref resolves to.

    Citances whose raw_ref isn't in `raw_to_canon` are dropped silently:
    those are the unresolved-reference cases the rest of the pipeline already
    surfaces in review_queue.jsonl, so we don't double-report them here.
    """
    path = workdir / "citances.jsonl"
    if not path.exists():
        return {}

    by_canon: dict[str, list[InContextOccurrence]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw = row.get("raw_ref")
            if not raw:
                # unlinked citance — nothing to attach it to
                continue
            canon_id = _resolve_to_canonical(raw, raw_to_canon)
            if not canon_id:
                continue
            marker = row.get("marker") or {}
            page = marker.get("page")
            try:
                page_int = int(page) if page is not None else None
            except (TypeError, ValueError):
                page_int = None
            occ = InContextOccurrence(
                citing_paper_id=row.get("citing_paper_id", ""),
                page=page_int,
                marker_raw=marker.get("raw_marker"),
                context=(row.get("context") or "").strip(),
            )
            by_canon.setdefault(canon_id, []).append(occ)
    return by_canon


def aggregate_cocitations(workdir: Path) -> list[CocitedRecord]:
    """Compute co-citation frequency for every canonical reference.

    For each canonical_id, count how many distinct source papers cited it.
    Ties are broken by global citation count (if available), then by title.
    """
    resolved = _load_resolved(workdir)
    raw_to_canon = _load_raw_to_canonical(workdir)
    raw_refs = _load_raw_refs(workdir)
    occurrences_by_canon = _load_citances_by_canonical(workdir, raw_to_canon)

    if not raw_refs:
        # We're in CSV mode — no citing_paper_id info. Just emit each resolved
        # row with cocitation_count=1.
        return [
            CocitedRecord(
                canonical_id=row["canonical_id"],
                title=row.get("title"),
                authors=_flatten_authors(row.get("authors")),
                year=row.get("year"),
                venue=row.get("venue"),
                doi=row.get("doi"),
                cocitation_count=1,
                citing_source_ids=[],
                confidence=float(row.get("confidence", 1.0)),
                occurrences=occurrences_by_canon.get(row["canonical_id"], []),
            )
            for row in resolved.values()
        ]

    # PDF mode: do the real cross-paper counting.
    # canonical_id -> set of citing_paper_ids that cited it
    cocited: dict[str, set[str]] = {}
    for citing_id, raw in raw_refs:
        canon_id = _resolve_to_canonical(raw, raw_to_canon)
        if not canon_id:
            continue  # raw didn't survive to canonical (review queue, etc.)
        cocited.setdefault(canon_id, set()).add(citing_id)

    records: list[CocitedRecord] = []
    for canon_id, citing_ids in cocited.items():
        row = resolved.get(canon_id)
        if not row:
            continue
        records.append(
            CocitedRecord(
                canonical_id=canon_id,
                title=row.get("title"),
                authors=_flatten_authors(row.get("authors")),
                year=row.get("year"),
                venue=row.get("venue"),
                doi=row.get("doi"),
                cocitation_count=len(citing_ids),
                citing_source_ids=sorted(citing_ids),
                confidence=float(row.get("confidence", 1.0)),
                occurrences=occurrences_by_canon.get(canon_id, []),
            )
        )

    records.sort(
        key=lambda r: (
            -r.cocitation_count,
            -(r.global_citation_count or 0),
            (r.title or "").lower(),
        )
    )
    return records


def _flatten_authors(authors_field) -> list[str]:
    """resolved.jsonl stores authors as list of {surname, given} dicts."""
    if not authors_field:
        return []
    if isinstance(authors_field, str):
        # CSV-roundtripped JSON
        try:
            authors_field = json.loads(authors_field)
        except json.JSONDecodeError:
            return [authors_field]
    out: list[str] = []
    for a in authors_field:
        if isinstance(a, dict):
            surname = a.get("surname", "")
            given = a.get("given") or ""
            full = f"{given} {surname}".strip() if given else surname
            if full:
                out.append(full)
        elif isinstance(a, str):
            out.append(a)
    return out


# ----- Optional enrichment: pull OpenAlex citation counts via DOI -----

def enrich_with_citation_counts(
    records: list[CocitedRecord],
    *,
    only_top_n: Optional[int] = None,
    sleep_between: float = 0.1,
    timeout: int = 15,
) -> None:
    """Mutate `records` in place, filling in global_citation_count via OpenAlex.

    Only queries records with a DOI. To keep this cheap, you can restrict
    enrichment to the top N by co-citation count via `only_top_n`.
    """
    import os
    polite = {}
    email = os.environ.get("CITATION_CLEANER_EMAIL")
    if email:
        polite["mailto"] = email

    targets = records[:only_top_n] if only_top_n else records
    for rec in targets:
        if not rec.doi:
            continue
        doi = rec.doi.replace("https://doi.org/", "").strip()
        if not doi:
            continue
        try:
            resp = requests.get(
                f"https://api.openalex.org/works/doi:{doi}",
                params={"select": "cited_by_count", **polite},
                headers={"User-Agent": "citation-cleaner-v3"},
                timeout=timeout,
            )
            if resp.status_code == 200:
                rec.global_citation_count = int(
                    resp.json().get("cited_by_count", 0) or 0
                )
        except requests.RequestException:
            pass
        time.sleep(sleep_between)

    # Re-sort now that we have global counts.
    records.sort(
        key=lambda r: (
            -r.cocitation_count,
            -(r.global_citation_count or 0),
            (r.title or "").lower(),
        )
    )


# ----- Outputs -----

def build_incontext_section(
    records: list[CocitedRecord],
    source_papers: Optional[list[dict]] = None,
    *,
    top_n: int = 10,
    max_context_chars: int = 400,
) -> list[str]:
    """Build the "In-context appearances" markdown block, line by line.

    For the top `top_n` references (by co-citation rank), this dumps every
    sentence-level appearance across the source corpus, grouped by source
    paper, with page + marker attribution. Shared by the .md report and the
    Gradio UI so both render identically.

    Returns an empty list when no reference carries any in-text occurrence —
    that happens in CSV mode, or for PDFs whose markers Stage 0 couldn't parse.
    The caller decides what (if anything) to show in that case.
    """
    refs_with_occ = [r for r in records if r.occurrences]
    if not refs_with_occ:
        return []

    if source_papers:
        n_sources = len(source_papers)
    else:
        n_sources = len(
            {occ.citing_paper_id for r in refs_with_occ for occ in r.occurrences}
        )

    # Headline finding: the reference cited the most *times in text* — a
    # stronger "narrative weight" signal than merely appearing in N
    # bibliographies, since it means authors keep returning to it.
    most_quoted = max(refs_with_occ, key=lambda r: len(r.occurrences))
    lines: list[str] = ["", "## In-context appearances", ""]
    lines.append(
        f"Across the {n_sources} source paper(s), "
        f"**{(most_quoted.title or '(untitled)')[:100]}** "
        f"({most_quoted.year or '?'}) is the most-referenced work in text, "
        f"with **{len(most_quoted.occurrences)} in-text mentions**."
    )
    lines.append("")

    for rank, rec in enumerate(records[:top_n], start=1):
        if not rec.occurrences:
            continue
        title = (rec.title or "(untitled)")[:120]
        authors_short = "; ".join(rec.authors[:2])
        if len(rec.authors) > 2:
            authors_short += " et al."
        lines.append(f"### {rank}. {title} ({rec.year or '?'})")
        if authors_short:
            lines.append(f"*{authors_short}*")
        lines.append("")
        lines.append(
            f"Cited in **{rec.cocitation_count}** source paper(s); "
            f"appears **{len(rec.occurrences)}** time(s) in text."
        )
        lines.append("")
        # Group occurrences by source paper for readability.
        by_paper: dict[str, list[InContextOccurrence]] = {}
        for occ in rec.occurrences:
            by_paper.setdefault(occ.citing_paper_id, []).append(occ)
        for paper_id in sorted(by_paper):
            lines.append(f"**In `{paper_id}`:**")
            lines.append("")
            for occ in by_paper[paper_id]:
                page_str = f"p. {occ.page}" if occ.page is not None else "p. ?"
                marker_str = (
                    f"`{occ.marker_raw}`"
                    if occ.marker_raw
                    else "_(marker not captured)_"
                )
                # Trim very long context to keep the report readable.
                ctx = occ.context.replace("\n", " ").strip()
                if len(ctx) > max_context_chars:
                    ctx = ctx[:max_context_chars].rsplit(" ", 1)[0] + "…"
                # Render the sentence as a blockquote so the actual in-text
                # paragraph is what stands out, with page/marker as attribution.
                lines.append(f"> {ctx}")
                lines.append(">")
                lines.append(f"> — {page_str} · marker {marker_str}")
                lines.append("")
    return lines


def write_cocitation_outputs(
    records: list[CocitedRecord],
    workdir: Path,
    *,
    query: str = "",
    source_papers: Optional[list[dict]] = None,
) -> dict[str, Path]:
    """Write CSV + JSON + Markdown. Returns dict of artifact name -> path."""
    workdir.mkdir(parents=True, exist_ok=True)

    csv_path = workdir / "cocited_refs.csv"
    json_path = workdir / "cocited_refs.json"
    md_path = workdir / "cocited_refs.md"

    # CSV
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rank",
                "cocitation_count",
                "in_text_hits",
                "global_citation_count",
                "title",
                "authors",
                "year",
                "venue",
                "doi",
                "canonical_id",
                "citing_source_ids",
                "confidence",
            ],
        )
        writer.writeheader()
        for rank, rec in enumerate(records, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "cocitation_count": rec.cocitation_count,
                    "in_text_hits": len(rec.occurrences),
                    "global_citation_count": rec.global_citation_count
                    if rec.global_citation_count is not None
                    else "",
                    "title": rec.title or "",
                    "authors": "; ".join(rec.authors),
                    "year": rec.year or "",
                    "venue": rec.venue or "",
                    "doi": rec.doi or "",
                    "canonical_id": rec.canonical_id,
                    "citing_source_ids": ";".join(rec.citing_source_ids),
                    "confidence": f"{rec.confidence:.2f}",
                }
            )

    # JSON
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "query": query,
                "source_papers": source_papers or [],
                "records": [asdict(rec) for rec in records],
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    # Markdown report
    lines: list[str] = []
    lines.append(f"# Co-cited References Report")
    lines.append("")
    if query:
        lines.append(f"**Query:** `{query}`")
    if source_papers:
        lines.append("")
        lines.append("**Source papers analyzed:**")
        for sp in source_papers:
            sid = sp.get("short_id") or sp.get("openalex_id", "?")
            lines.append(
                f"- `{sid}` — {sp.get('title', '(no title)')} "
                f"({sp.get('year', '?')}, cited {sp.get('cited_by_count', '?')}×)"
            )
    lines.append("")
    lines.append(f"**Total unique references found:** {len(records)}")
    lines.append("")
    lines.append("## Top references by co-citation frequency")
    lines.append("")
    lines.append(
        "| Rank | Co-cited | Global cites | In-text hits | Title | Authors | Year | DOI |"
    )
    lines.append("|---:|---:|---:|---:|---|---|---:|---|")
    for rank, rec in enumerate(records[:50], start=1):
        title = (rec.title or "(untitled)").replace("|", "\\|")
        authors = "; ".join(rec.authors[:3]).replace("|", "\\|")
        if len(rec.authors) > 3:
            authors += " et al."
        global_c = (
            str(rec.global_citation_count)
            if rec.global_citation_count is not None
            else "—"
        )
        doi = rec.doi or ""
        lines.append(
            f"| {rank} | {rec.cocitation_count} | {global_c} | "
            f"{len(rec.occurrences)} | "
            f"{title} | {authors} | {rec.year or ''} | {doi} |"
        )

    # ---- v4: In-context appendix ----
    # Built by build_incontext_section so the .md report and the web UI render
    # the same sentence-level appearances. Empty list = nothing to append.
    lines.extend(build_incontext_section(records, source_papers))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"csv": csv_path, "json": json_path, "md": md_path}
