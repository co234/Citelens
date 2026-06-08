"""Run Stage 0 PDF parsing only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from citation_cleaner.pipelines.stages import Stage0_Parse, StageContext


def _resolve_pdfs(values: list[str]) -> list[Path]:
    pdfs: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            pdfs.extend(sorted(path.glob("*.pdf")))
        else:
            pdfs.append(path)
    if not pdfs:
        raise SystemExit("No PDFs found.")
    return pdfs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdfs", nargs="+", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Disable Stage 0 LLM fallback.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--stage0-model")
    args = parser.parse_args()

    config = {}
    if args.workers:
        config["workers"] = args.workers
    if args.stage0_model:
        config["stage0_model"] = args.stage0_model
    ctx = StageContext(workdir=Path(args.workdir), dry_run=args.dry_run, resume=not args.no_resume, config=config)
    docs = Stage0_Parse().run_or_resume(_resolve_pdfs(args.pdfs), ctx)
    print(json.dumps({"parsed_documents": len(docs), "workdir": str(ctx.workdir)}, indent=2))


if __name__ == "__main__":
    main()
