"""Stage 0..6 PDF-to-canonical orchestrator."""

from __future__ import annotations

from pathlib import Path

from citation_cleaner.pipelines.stages import (
    Stage,
    Stage0_Parse,
    Stage1_Preclean,
    Stage2_Extract,
    Stage34_BlockEmbed,
    Stage56_JudgeResolve,
    StageContext,
)
from citation_cleaner.schemas.citance import Citance


def build(*, dry_run: bool = False) -> list[Stage]:
    return [
        Stage0_Parse(),
        Stage1_Preclean(),
        Stage2_Extract(),
        Stage34_BlockEmbed(),
        Stage56_JudgeResolve(),
    ]


def run(pdf_paths: list[Path], ctx: StageContext) -> list[dict]:
    stage0 = Stage0_Parse()
    stage1 = Stage1_Preclean()
    stage2 = Stage2_Extract()
    stage34 = Stage34_BlockEmbed()
    stage56 = Stage56_JudgeResolve()

    docs = stage0.run_or_resume(pdf_paths, ctx)
    cleaned = stage1.run_or_resume(docs, ctx)
    extracted = stage2.run_or_resume(cleaned, ctx)
    clusters = stage34.run_or_resume(extracted, ctx)
    citances: list[Citance] = [citance for doc in docs for citance in doc.citances]
    return stage56.run_or_resume((clusters, citances), ctx)
