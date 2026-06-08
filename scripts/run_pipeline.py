"""Run Citation Cleaner v2 end to end."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from citation_cleaner.pipelines import pdf_to_canonical, refs_to_canonical
from citation_cleaner.pipelines.stages import StageContext


STAGE_NAMES = ["parse", "preclean", "extract", "block_and_embed", "judge_and_resolve"]


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required for YAML config files. Use JSON or install pyyaml.") from exc
    return yaml.safe_load(text) or {}


def _resolve_pdfs(values: list[str]) -> list[Path]:
    pdfs: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            pdfs.extend(sorted(path.glob("*.pdf")))
        else:
            pdfs.append(path)
    missing = [str(path) for path in pdfs if not path.exists()]
    if missing:
        raise SystemExit(f"Missing PDF path(s): {', '.join(missing)}")
    if not pdfs:
        raise SystemExit("No PDFs found.")
    return pdfs


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdfs", nargs="+", help="PDF files or directories of PDFs.")
    group.add_argument("--refs", help="CSV with raw reference strings.")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--resume-from", choices=STAGE_NAMES)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--config")
    parser.add_argument("--raw-column", default="raw")
    parser.add_argument("--citances", help="Optional external citances JSONL for refs mode.")
    parser.add_argument("--stage0-model")
    args = parser.parse_args()

    config = _load_config(args.config)
    if args.workers:
        config["workers"] = args.workers
    if args.stage0_model:
        config["stage0_model"] = args.stage0_model
    if args.resume_from:
        config["_force_from"] = args.resume_from

    ctx = StageContext(
        workdir=Path(args.workdir),
        dry_run=args.dry_run,
        resume=not args.no_resume,
        config=config,
    )

    if args.pdfs:
        pdf_paths = _resolve_pdfs(args.pdfs)
        resolved = pdf_to_canonical.run(pdf_paths, ctx)
    else:
        resolved = refs_to_canonical.run(
            Path(args.refs),
            ctx,
            raw_column=args.raw_column,
            citances_path=Path(args.citances) if args.citances else None,
        )

    print(f"Done. Resolved {len(resolved)} canonical references.")
    print(f"Outputs in {ctx.workdir}")
    print("  - resolved.jsonl")
    print("  - canonical_refs.csv")
    print("  - raw_to_canonical.csv")
    print("  - review_queue.jsonl")


if __name__ == "__main__":
    main()
