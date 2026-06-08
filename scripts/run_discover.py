"""CLI for the discover pipeline.

Examples:
  # By terminology, dry-run (no API keys required)
  python scripts/run_discover.py --query "diffusion models" --workdir ./run_dm --dry-run

  # By uploaded paper, with real APIs
  python scripts/run_discover.py --upload mypaper.pdf --workdir ./run_mp --n 5

  # By author — top N most-cited works
  python scripts/run_discover.py --author "Yann LeCun" --workdir ./run_lecun --n 5

  # Terminology mode, sorted by recency (newest first)
  python scripts/run_discover.py --query "transformers" --workdir ./run_t --sort recency

  # Terminology mode, sorted by total citations (most cited first)
  python scripts/run_discover.py --query "transformers" --workdir ./run_t --sort citations

  # Skip the optional global-citation-count enrichment
  python scripts/run_discover.py --query "transformers" --workdir ./run_t --no-enrich
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from citation_cleaner.pipelines.discover_pipeline import run_discovery


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover related papers and aggregate co-cited references.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", help="A terminology / topic to search for.")
    group.add_argument("--upload", help="Path to an uploaded PDF to derive topic from.")
    group.add_argument("--author", help="Author name (free text) or OpenAlex author ID. Returns top-N most-cited works.")

    parser.add_argument("--workdir", required=True, help="Output directory.")
    parser.add_argument("--n", type=int, default=3, help="Number of source papers to find (default 3).")
    parser.add_argument(
        "--sort",
        choices=["relevance", "recency", "citations"],
        default="relevance",
        help=(
            "Sort source papers (terminology / upload mode only). "
            "relevance=OpenAlex's mix of textual match + citations (default); "
            "recency=most-recent first; citations=most-cited first. "
            "Ignored in --author mode (always citations)."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="No network/LLM calls. Smoke test only.")
    parser.add_argument("--no-enrich", action="store_true", help="Skip global citation count enrichment.")
    parser.add_argument("--enrich-top-n", type=int, default=30, help="How many top records to enrich (default 30).")
    parser.add_argument("--config", help="Optional JSON config passed to StageContext.")
    args = parser.parse_args()

    config = {}
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))

    result = run_discovery(
        query=args.query,
        uploaded_pdf=Path(args.upload) if args.upload else None,
        author=args.author,
        workdir=Path(args.workdir),
        n_papers=args.n,
        dry_run=args.dry_run,
        enrich_citations=not args.no_enrich,
        enrich_top_n=args.enrich_top_n,
        config=config,
        sort_by=args.sort,
    )

    print()
    print("=" * 60)
    print(f"Done. Mode: {result['mode']}")
    if result["mode"] == "author" and result.get("author"):
        a = result["author"]
        print(f"Author:        {a['display_name']} ({a.get('author_id', '?')})")
        print(f"  works:       {a.get('works_count', '?')}")
        print(f"  citations:   {a.get('cited_by_count', '?')}")
    else:
        print(f"Query:         {result['query']!r}")
        print(f"Sort by:       {result['sort_by']}")
    print(f"Source papers: {len(result['source_papers'])}")
    print(f"Co-cited references: {len(result['records'])}")
    print()
    print("Outputs:")
    for name, path in result["output_paths"].items():
        print(f"  {name:5s}  {path}")

    # Print top-10 preview to terminal.
    print()
    print("Top 10 by co-citation count:")
    print(f"  {'rank':>4}  {'co':>3}  {'cites':>6}  {'hits':>4}  title")
    for rank, rec in enumerate(result["records"][:10], start=1):
        title = (rec.title or "(untitled)")[:65]
        global_c = (
            str(rec.global_citation_count)
            if rec.global_citation_count is not None
            else "—"
        )
        hits = len(rec.occurrences)
        print(f"  {rank:>4}  {rec.cocitation_count:>3}  {global_c:>6}  {hits:>4}  {title}")


if __name__ == "__main__":
    main()
