"""Stage 1..6 raw-refs-to-canonical orchestrator."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from citation_cleaner.pipelines.stages import (
    Stage,
    Stage1_Preclean,
    Stage2_Extract,
    Stage34_BlockEmbed,
    Stage56_JudgeResolve,
    StageContext,
)
from citation_cleaner.schemas.citance import Citance
from citation_cleaner.schemas.reference import RawRefRow


def build(*, dry_run: bool = False) -> list[Stage]:
    return [Stage1_Preclean(), Stage2_Extract(), Stage34_BlockEmbed(), Stage56_JudgeResolve()]


def load_refs_csv(path: Path, raw_column: str = "raw") -> list[RawRefRow]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    refs: list[RawRefRow] = []
    for row in rows:
        refs.append(
            RawRefRow(
                citing_paper_id=row.get("citing_paper_id", ""),
                raw=row.get(raw_column) or row.get("raw") or "",
            )
        )
    return refs


def load_citances(path: Path | None) -> list[Citance]:
    if path is None:
        return []
    citances: list[Citance] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                if "marker" in row:
                    citances.append(Citance.model_validate(row))
    return citances


def run(refs_path: Path, ctx: StageContext, raw_column: str = "raw", citances_path: Path | None = None) -> list[dict]:
    refs = load_refs_csv(refs_path, raw_column=raw_column)
    ctx.config.setdefault("citances_path", str(citances_path) if citances_path else None)
    cleaned = Stage1_Preclean().run_or_resume(refs, ctx)
    extracted = Stage2_Extract().run_or_resume(cleaned, ctx)
    clusters = Stage34_BlockEmbed().run_or_resume(extracted, ctx)
    return Stage56_JudgeResolve().run_or_resume((clusters, load_citances(citances_path)), ctx)
