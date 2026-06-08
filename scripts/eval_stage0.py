"""Evaluate Stage 0 reference extraction and citance linking."""

from __future__ import annotations

import argparse
import csv
from difflib import SequenceMatcher
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from citation_cleaner.pipelines.stages import Stage0_Parse, StageContext


def _norm(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _load_refs(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row.get("raw") or row.get("reference") or next(iter(row.values())) for row in rows]


def _score_refs(pred: list[str], truth: list[str], threshold: float = 0.85) -> dict:
    matched_truth: set[int] = set()
    matched_pred = 0
    for pred_ref in pred:
        best_i = None
        best_score = 0.0
        for i, truth_ref in enumerate(truth):
            if i in matched_truth:
                continue
            score = _ratio(pred_ref, truth_ref)
            if score > best_score:
                best_i, best_score = i, score
        if best_i is not None and best_score >= threshold:
            matched_truth.add(best_i)
            matched_pred += 1
    recall = len(matched_truth) / len(truth) if truth else 0.0
    precision = matched_pred / len(pred) if pred else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"ref_recall": recall, "ref_precision": precision, "ref_f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--workdir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    base = manifest_path.parent
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="citation-cleaner-stage0-"))
    per_pdf = []
    for row in manifest:
        pdf_path = base / row["pdf_path"]
        truth_refs = _load_refs(base / (Path(row["pdf_path"]).stem + ".refs.csv"))
        pdf_workdir = workdir / Path(row["pdf_path"]).stem
        ctx = StageContext(workdir=pdf_workdir, dry_run=args.dry_run, resume=False, config={"workers": 1})
        parsed = Stage0_Parse().run_or_resume([pdf_path], ctx)[0]
        ref_metrics = _score_refs(parsed.references, truth_refs)
        per_pdf.append(
            {
                "pdf_id": Path(row["pdf_path"]).stem,
                **ref_metrics,
                "citance_recall": None,
                "citance_link_accuracy": None,
                "llm_fallback_invoked": parsed.quality.llm_fallback_invoked,
                "quality_score": (
                    parsed.quality.pct_with_year + parsed.quality.pct_with_authorlike
                )
                / 2,
            }
        )
    aggregate = {}
    for key in ("ref_recall", "ref_precision", "ref_f1"):
        aggregate[key] = sum(item[key] for item in per_pdf) / len(per_pdf) if per_pdf else 0.0
    aggregate["fallback_rate"] = (
        sum(1 for item in per_pdf if item["llm_fallback_invoked"]) / len(per_pdf) if per_pdf else 0.0
    )
    json.dump({"per_pdf": per_pdf, "aggregate": aggregate}, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
