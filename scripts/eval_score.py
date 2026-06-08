"""Pair-F1 scoring for Stage 1..6 outputs."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import sys


def pair_f1(eval_rows: list[dict], assignment: dict[str, str]) -> dict:
    tp = fp = fn = 0
    for a, b in itertools.combinations(eval_rows, 2):
        gt_same = a["canonical_id"] == b["canonical_id"]
        pred_same = assignment.get(a["raw"]) == assignment.get(b["raw"])
        if gt_same and pred_same:
            tp += 1
        elif not gt_same and pred_same:
            fp += 1
        elif gt_same and not pred_same:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def load_assignment(resolved_path: Path) -> dict[str, str]:
    assignment: dict[str, str] = {}
    with resolved_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for raw in row.get("raw_variants", []):
                assignment[raw] = row["canonical_id"]
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved", required=True)
    parser.add_argument("--eval-set", required=True)
    args = parser.parse_args()
    with Path(args.eval_set).open(encoding="utf-8", newline="") as handle:
        eval_rows = list(csv.DictReader(handle))
    metrics = pair_f1(eval_rows, load_assignment(Path(args.resolved)))
    json.dump(metrics, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
