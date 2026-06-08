"""Stage -1 + Stage 7 wrapper: discover papers, run citation cleaner, aggregate.

This is the new top-level orchestrator for v3 (extended in v4). Flow:

    Input: terminology string, an uploaded PDF, OR an author name
           │
           ▼
    Stage -1  Discover N source papers via OpenAlex, download PDFs
              - terminology / upload: relevance / recency / citations sorting
              - author: always top-N most-cited
           │
           ▼
    Stage 0..6  (existing pipeline) parse → clean → canonical references
           │
           ▼
    Stage 7   Aggregate co-citations across the N source papers
              + attach in-text citation occurrences (citances)
           │
           ▼
    Outputs:  cocited_refs.csv / .md / .json  (plus all v2 artifacts)
"""

from __future__ import annotations

import threading
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from citation_cleaner.aggregate.cocitation import (
    aggregate_cocitations,
    enrich_with_citation_counts,
    write_cocitation_outputs,
)
from citation_cleaner.discovery.openalex import (
    DiscoveredPaper,
    ResolvedAuthor,
    SortBy,
    fetch_pdf,
    search_by_author,
    search_by_terminology,
)
from citation_cleaner.discovery.topic_from_paper import topic_from_paper
from citation_cleaner.pipelines import pdf_to_canonical
from citation_cleaner.pipelines.pipeline_log import pipeline_logger
from citation_cleaner.pipelines.result_bundle import build_result_zip
from citation_cleaner.pipelines.stages import StageContext


def _write_dryrun_stub(workdir: Path, query: str, papers: list) -> None:
    """Write minimal stub artifacts in dry-run mode so Stage 7 has data.

    This synthesizes a believable resolved.jsonl + raw_to_canonical.csv +
    raw_refs.csv across the N stub papers, with two shared references that
    every stub paper "cites", to exercise the co-citation logic.
    """
    import csv
    import json

    # Two shared canonical refs co-cited by all stub papers.
    shared = [
        {
            "canonical_id": "stub0001abcd",
            "authors": [{"surname": "Vaswani", "given": "Ashish"}],
            "year": 2017,
            "title": "Attention Is All You Need",
            "venue": "NeurIPS",
            "type": "inproceedings",
            "doi": "10.48550/arxiv.1706.03762",
            "volume": None,
            "pages": None,
            "raw_variants": ["Vaswani et al. 2017"],
            "confidence": 0.95,
        },
        {
            "canonical_id": "stub0002efgh",
            "authors": [{"surname": "Devlin", "given": "Jacob"}],
            "year": 2019,
            "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "venue": "NAACL",
            "type": "inproceedings",
            "doi": "10.48550/arxiv.1810.04805",
            "volume": None,
            "pages": None,
            "raw_variants": ["Devlin et al. 2019"],
            "confidence": 0.95,
        },
    ]
    # One unique-per-paper ref each.
    unique = [
        {
            "canonical_id": f"stubpaper{i:04d}",
            "authors": [{"surname": f"Author{i}", "given": None}],
            "year": 2020 + i,
            "title": f"Stub-only reference cited by paper {i}",
            "venue": None,
            "type": "unknown",
            "doi": None,
            "volume": None,
            "pages": None,
            "raw_variants": [f"Author{i} et al. 202{i}"],
            "confidence": 1.0,
        }
        for i in range(len(papers))
    ]

    resolved = shared + unique
    (workdir / "resolved.jsonl").write_text(
        "\n".join(json.dumps(r) for r in resolved) + "\n",
        encoding="utf-8",
    )

    # Build raw_refs.csv: each stub paper cites both shared + its unique ref.
    raw_rows: list[tuple[str, str]] = []
    raw_to_canon_rows: list[tuple[str, str, float]] = []
    for i, paper in enumerate(papers):
        sid = paper.short_id()
        for shared_ref in shared:
            raw = shared_ref["raw_variants"][0]
            raw_rows.append((sid, raw))
            raw_to_canon_rows.append((raw, shared_ref["canonical_id"], 0.95))
        u = unique[i]
        raw = u["raw_variants"][0]
        raw_rows.append((sid, raw))
        raw_to_canon_rows.append((raw, u["canonical_id"], 1.0))

    with (workdir / "raw_refs.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["citing_paper_id", "raw"])
        writer.writeheader()
        for citing_id, raw in raw_rows:
            writer.writerow({"citing_paper_id": citing_id, "raw": raw})

    # Deduplicate raw_to_canonical rows.
    seen = set()
    with (workdir / "raw_to_canonical.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["raw", "canonical_id", "confidence"])
        writer.writeheader()
        for raw, canon, conf in raw_to_canon_rows:
            if raw in seen:
                continue
            seen.add(raw)
            writer.writerow({"raw": raw, "canonical_id": canon, "confidence": conf})

    # Synthetic in-text citances so the "In-context appearances" section
    # actually renders in offline/dry-run demos. Each stub paper mentions the
    # two shared refs a few times; Vaswani is cited most so it surfaces as the
    # "most-referenced" work. raw_ref must match the raw_to_canonical keys above.
    vaswani_raw = shared[0]["raw_variants"][0]
    devlin_raw = shared[1]["raw_variants"][0]
    stub_citances: list[dict] = []
    for i, paper in enumerate(papers):
        sid = paper.short_id()
        stub_citances.append(
            {
                "citing_paper_id": sid,
                "raw_ref": vaswani_raw,
                "context": "This work was directly inspired by [1], whose multi-head "
                "self-attention we adopt as the backbone of our encoder.",
                "marker": {
                    "style": "numbered_bracket",
                    "raw_marker": "[1]",
                    "page": 1 + i,
                    "char_span": [35, 38],
                },
                "link_method": "numbered_direct",
                "link_confidence": 1.0,
            }
        )
        stub_citances.append(
            {
                "citing_paper_id": sid,
                "raw_ref": vaswani_raw,
                "context": "We build on the Transformer of Vaswani et al. (2017) to model "
                "long-range dependencies without recurrence.",
                "marker": {
                    "style": "narrative_author_year",
                    "raw_marker": "Vaswani et al. (2017)",
                    "page": 3 + i,
                    "char_span": [31, 52],
                },
                "link_method": "author_year_unique",
                "link_confidence": 0.9,
            }
        )
        stub_citances.append(
            {
                "citing_paper_id": sid,
                "raw_ref": devlin_raw,
                "context": "Pretrained bidirectional encoders such as BERT (Devlin et al., "
                "2019) have since become a standard baseline for transfer learning.",
                "marker": {
                    "style": "paren_author_year",
                    "raw_marker": "(Devlin et al., 2019)",
                    "page": 2 + i,
                    "char_span": [47, 68],
                },
                "link_method": "author_year_unique",
                "link_confidence": 0.9,
            }
        )

    (workdir / "citances.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in stub_citances) + "\n",
        encoding="utf-8",
    )


def run_discovery(
    *,
    query: Optional[str] = None,
    uploaded_pdf: Optional[Path] = None,
    author: Optional[str] = None,
    workdir: Path,
    n_papers: int = 3,
    dry_run: bool = False,
    enrich_citations: bool = True,
    enrich_top_n: int = 30,
    config: Optional[dict] = None,
    sort_by: SortBy = "relevance",
) -> dict:
    """End-to-end discovery → aggregation pipeline.

    Exactly one of `query`, `uploaded_pdf`, or `author` must be provided.

    Modes:
      - query=...     : search OpenAlex by terminology. Honors `sort_by`
                        ("relevance" | "recency" | "citations").
      - uploaded_pdf= : derive a topic from the PDF, then run query-mode.
                        Honors `sort_by`.
      - author=...    : resolve the name to an OpenAlex author and pull
                        their top-N most-cited works. `sort_by` is ignored
                        (always citations-desc, per the v4 spec).

    Returns a dict with: query, source_papers, records, output_paths, plus
    `mode` ("terminology" | "upload" | "author") and `author` (resolved
    author dict, only in author mode).
    """
    inputs_set = sum(x is not None for x in (query, uploaded_pdf, author))
    if inputs_set != 1:
        raise ValueError(
            "Provide exactly one of `query`, `uploaded_pdf`, or `author`."
        )

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    config = config or {}
    log = pipeline_logger(config)

    mode: str
    resolved_author: Optional[ResolvedAuthor] = None
    effective_query: str

    # --- Resolve the input → an effective_query string (for reporting) and
    # the set of discovered papers. ---
    if author is not None:
        mode = "author"
        effective_query = f"author:{author}"
    elif uploaded_pdf is not None:
        mode = "upload"
        log.begin("topic", f"extracting topic from {uploaded_pdf.name}…")
        effective_query = topic_from_paper(
            uploaded_pdf, dry_run=dry_run, config=config
        )
        log.done("topic", f"derived topic: {effective_query!r}")
    else:
        mode = "terminology"
        effective_query = query  # type: ignore[assignment]

    # --- Stage -1: search OpenAlex, download PDFs. ---
    pdf_dir = workdir / "source_pdfs"
    pdf_paths: list[Path] = []
    successful: list[DiscoveredPaper] = []

    if dry_run and uploaded_pdf is None:
        # In dry-run + (terminology | author) mode there are no real PDFs to
        # feed downstream. We synthesize a stub workdir directly and return.
        log.begin("search", "dry-run mode: skipping OpenAlex; emitting stub output.")
        if mode == "author":
            # Stub a resolved author so the result shape is consistent.
            resolved_author = ResolvedAuthor(
                author_id="https://openalex.org/A0000000",
                display_name=author or "Unknown Author",
                works_count=n_papers,
                cited_by_count=0,
            )
        successful = [
            DiscoveredPaper(
                openalex_id=f"W000{i}",
                title=f"(dry-run stub) Paper {i} on {effective_query}",
                year=2024 - i,
                doi=None,
                cited_by_count=100 - i * 10,
                pdf_url=None,
                relevance_score=1.0 - i * 0.1,
            )
            for i in range(n_papers)
        ]
        # Emit minimal stub artifacts so Stage 7 has something to read.
        _write_dryrun_stub(workdir, effective_query, successful)
        log.begin("cocite", "aggregating co-citations across source papers…")
        records = aggregate_cocitations(workdir)
        log.done("cocite", f"{len(records)} unique references in aggregate")
        source_papers_meta = [
            {**asdict(p), "short_id": p.short_id()} for p in successful
        ]
        output_paths = write_cocitation_outputs(
            records, workdir, query=effective_query, source_papers=source_papers_meta
        )
        output_paths["zip"] = build_result_zip(workdir)
        log.done("report", "stub report written")
        return {
            "mode": mode,
            "query": effective_query,
            "author": asdict(resolved_author) if resolved_author else None,
            "sort_by": "citations" if mode == "author" else sort_by,
            "source_papers": source_papers_meta,
            "records": records,
            "output_paths": {k: str(v) for k, v in output_paths.items()},
        }

    # --- Real-mode dispatch ---
    if mode == "author":
        log.begin("author", f"resolving author {author!r}…")
        resolved_author, discovered = search_by_author(author or "", n=n_papers)
        if resolved_author is None:
            raise SystemExit(
                f"Could not resolve author {author!r} on OpenAlex. "
                "Try the full display name, or pass an OpenAlex author ID "
                "(e.g. A1234567890)."
            )
        log.done(
            "author",
            f"resolved to {resolved_author.display_name} "
            f"({resolved_author.short_id()}, {resolved_author.works_count} works, "
            f"{resolved_author.cited_by_count} citations)",
        )
        log.begin("fetch", f"fetching top {n_papers} works by citation count…")
        if not discovered:
            raise SystemExit(
                f"No open-access papers found for {resolved_author.display_name}. "
                "Their work may not be on OA, or try increasing --n."
            )
        log.done("fetch", f"found {len(discovered)} candidate paper(s)")
    else:
        log.begin(
            "search",
            f"searching OpenAlex for {effective_query!r} (sort_by={sort_by})…",
        )
        discovered = search_by_terminology(
            effective_query, n=n_papers, sort_by=sort_by
        )
        if not discovered:
            raise SystemExit(
                f"No open-access papers found for query {effective_query!r}. "
                "Try a different terminology, or relax require_pdf."
            )
        log.done("search", f"found {len(discovered)} candidate paper(s)")

    log.begin("download", f"downloading {len(discovered)} PDF(s)…")
    download_workers = max(1, min(int(config.get("workers") or 4), len(discovered)))
    total_dl = len(discovered)
    dl_done = [0]
    dl_lock = threading.Lock()

    def _download_one(paper: DiscoveredPaper) -> tuple[DiscoveredPaper, Path | None]:
        log.info("download", f"downloading {paper.short_id()}: {paper.title[:60]}…")
        path = fetch_pdf(paper, pdf_dir)
        with dl_lock:
            dl_done[0] += 1
            n = dl_done[0]
        if path:
            log.progress("download", n, total_dl, f"done {paper.short_id()}")
            return paper, path
        log.progress("download", n, total_dl, f"skip {paper.short_id()} — no PDF available")
        return paper, None

    if download_workers <= 1 or total_dl <= 1:
        for paper in discovered:
            paper, path = _download_one(paper)
            if path:
                pdf_paths.append(path)
                successful.append(paper)
    else:
        log.info("download", f"using {download_workers} parallel workers")
        with ThreadPoolExecutor(max_workers=download_workers) as pool:
            futures = [pool.submit(_download_one, paper) for paper in discovered]
            for fut in as_completed(futures):
                paper, path = fut.result()
                if path:
                    pdf_paths.append(path)
                    successful.append(paper)
    log.done("download", f"{len(pdf_paths)} PDF(s) ready")

    # If the user also uploaded a PDF, include it as one of the source papers.
    if uploaded_pdf is not None and uploaded_pdf.exists():
        local_copy = pdf_dir / f"USER_{uploaded_pdf.stem}.pdf"
        if not local_copy.exists():
            pdf_dir.mkdir(parents=True, exist_ok=True)
            local_copy.write_bytes(uploaded_pdf.read_bytes())
        pdf_paths.insert(0, local_copy)

    if not pdf_paths:
        raise SystemExit(
            "No PDFs could be downloaded. Check network / OpenAlex availability."
        )

    log.begin("s0", f"running citation cleaner on {len(pdf_paths)} PDF(s)…")

    # --- Stage 0..6: existing pipeline. ---
    ctx = StageContext(
        workdir=workdir,
        dry_run=dry_run,
        resume=True,
        config=config,
        logger=log,
    )
    resolved = pdf_to_canonical.run(pdf_paths, ctx)
    log.info("s6", f"resolved {len(resolved)} canonical references")

    # --- Stage 7: aggregate co-citations. ---
    log.begin("cocite", "aggregating co-citations across source papers…")
    records = aggregate_cocitations(workdir)
    log.done("cocite", f"{len(records)} unique references in aggregate")

    if enrich_citations and not dry_run:
        log.begin("enrich", f"fetching global citation counts (top {enrich_top_n})…")
        enrich_with_citation_counts(records, only_top_n=enrich_top_n)
        log.done("enrich", f"enriched top {enrich_top_n} references")

    # --- Write outputs. ---
    source_papers_meta = [
        {**asdict(p), "short_id": p.short_id()} for p in successful
    ]
    if uploaded_pdf is not None:
        source_papers_meta.insert(
            0,
            {
                "short_id": f"USER_{uploaded_pdf.stem}",
                "title": f"(user upload) {uploaded_pdf.name}",
                "year": None,
                "doi": None,
                "cited_by_count": None,
            },
        )

    output_paths = write_cocitation_outputs(
        records,
        workdir,
        query=effective_query,
        source_papers=source_papers_meta,
    )
    output_paths["zip"] = build_result_zip(workdir)
    log.done("report", "ranking table & download bundle ready")

    return {
        "mode": mode,
        "query": effective_query,
        "author": asdict(resolved_author) if resolved_author else None,
        "sort_by": "citations" if mode == "author" else sort_by,
        "source_papers": source_papers_meta,
        "records": records,
        "output_paths": {k: str(v) for k, v in output_paths.items()},
    }
