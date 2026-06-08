from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from citation_cleaner.pipelines.refs_to_canonical import run
from citation_cleaner.pipelines.stages import StageContext


def test_refs_pipeline_dry_run(tmp_path):
    refs_path = tmp_path / "refs.csv"
    with refs_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["citing_paper_id", "raw"])
        writer.writeheader()
        writer.writerow({"citing_paper_id": "p1", "raw": "Smith, J. (2020). Deep learning for citations."})
        writer.writerow({"citing_paper_id": "p2", "raw": "Jones, K. (2021). Reference parsing."})

    ctx = StageContext(workdir=tmp_path / "run", dry_run=True, resume=False)
    resolved = run(refs_path, ctx)
    assert len(resolved) == 2
    assert (ctx.workdir / "resolved.jsonl").exists()
    assert (ctx.workdir / "canonical_refs.csv").exists()
    assert (ctx.workdir / "raw_to_canonical.csv").exists()
