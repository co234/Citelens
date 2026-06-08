"""Structured, stage-keyed pipeline logging with optional progress (current/total)."""

from __future__ import annotations

from dataclasses import dataclass, field
import sys
import threading
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from citation_cleaner.pipelines.stages import StageContext

StageStatus = Literal["pending", "running", "done", "error"]


@dataclass
class StageProgress:
    current: int = 0
    total: int = 0
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"current": self.current, "total": self.total, "label": self.label}


@dataclass
class StageRecord:
    lines: list[str] = field(default_factory=list)
    progress: StageProgress | None = None
    status: StageStatus = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "lines": list(self.lines),
            "progress": self.progress.to_dict() if self.progress else None,
            "status": self.status,
        }


# Keys match scripts/web_app.py STEP_DEFS (topic, s0, cocite, …).
_STDOUT_PREFIX: dict[str, str] = {
    "topic": "discover",
    "author": "discover",
    "fetch": "discover",
    "search": "discover",
    "download": "discover",
    "s0": "stage 0",
    "s1": "stage 1",
    "s2": "stage 2",
    "s3": "stage 3",
    "s4": "stage 4",
    "s5": "stage 5",
    "s6": "stage 6",
    "cocite": "discover",
    "enrich": "discover",
    "report": "discover",
}


def _format_line(stage: str, message: str, *, current: int | None, total: int | None) -> str:
    prefix = _STDOUT_PREFIX.get(stage, stage)
    if current is not None and total is not None and total > 0:
        return f"[{prefix}] ({current}/{total}) {message}"
    return f"[{prefix}] {message}"


class PipelineLogger:
    """Thread-safe logger that buckets messages by pipeline stage key."""

    def __init__(self, *, mirror_stdout: bool = True) -> None:
        self._stages: dict[str, StageRecord] = {}
        self._lock = threading.Lock()
        self._mirror_stdout = mirror_stdout
        self._active_stage: str | None = None

    @property
    def active_stage(self) -> str | None:
        with self._lock:
            return self._active_stage

    def _record(self, stage: str) -> StageRecord:
        if stage not in self._stages:
            self._stages[stage] = StageRecord()
        return self._stages[stage]

    def _emit(
        self,
        stage: str,
        message: str,
        *,
        current: int | None = None,
        total: int | None = None,
        status: StageStatus | None = None,
        append_line: bool = True,
    ) -> None:
        line = _format_line(stage, message, current=current, total=total)
        with self._lock:
            rec = self._record(stage)
            self._active_stage = stage
            if status:
                rec.status = status
            elif rec.status == "pending":
                rec.status = "running"
            if current is not None and total is not None:
                rec.progress = StageProgress(current=current, total=total, label=message)
            if append_line:
                rec.lines.append(line)
        if self._mirror_stdout:
            print(line, flush=True)

    def begin(self, stage: str, message: str) -> None:
        self._emit(stage, message, status="running")

    def info(self, stage: str, message: str) -> None:
        self._emit(stage, message)

    def progress(
        self,
        stage: str,
        current: int,
        total: int,
        message: str,
        *,
        append_line: bool = True,
    ) -> None:
        self._emit(
            stage,
            message,
            current=current,
            total=total,
            append_line=append_line,
        )

    def done(self, stage: str, message: str) -> None:
        self._emit(stage, message, status="done")

    def error(self, stage: str, message: str) -> None:
        self._emit(stage, message, status="error")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            stages = {key: rec.to_dict() for key, rec in self._stages.items()}
            return {
                "_meta": {"active_stage": self._active_stage},
                "stages": stages,
            }

    def flat_text(self) -> str:
        """Concatenate all stage lines in insertion order (for debugging)."""
        with self._lock:
            parts: list[str] = []
            for rec in self._stages.values():
                parts.extend(rec.lines)
            return "\n".join(parts)


def pipeline_logger(ctx: StageContext | dict | None) -> PipelineLogger:
    """Return the logger on *ctx* / config, creating one if needed."""
    if isinstance(ctx, dict):
        cfg = ctx
        existing = cfg.get("_pipeline_logger")
        if isinstance(existing, PipelineLogger):
            return existing
        pl = PipelineLogger(mirror_stdout=True)
        cfg["_pipeline_logger"] = pl
        return pl

    from citation_cleaner.pipelines.stages import StageContext as _Ctx

    if isinstance(ctx, _Ctx):
        if ctx.logger is not None:
            return ctx.logger
        pl = pipeline_logger(ctx.config)
        ctx.logger = pl
        return pl

    return PipelineLogger(mirror_stdout=True)


class _LoggerWriter:
    """Legacy stdout redirect target — forwards lines into the active stage."""

    def __init__(self, logger: PipelineLogger, default_stage: str = "report") -> None:
        self._logger = logger
        self._default_stage = default_stage
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                stage = self._logger.active_stage or self._default_stage
                self._logger.info(stage, line.strip())
        return len(s)

    def flush(self) -> None:
        if self._buf.strip():
            stage = self._logger.active_stage or self._default_stage
            self._logger.info(stage, self._buf.strip())
            self._buf = ""
        sys.stdout.flush()
