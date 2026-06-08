"""Anthropic client and prompt-cache routing helpers."""

from __future__ import annotations

import os
from typing import Any


def supports_anthropic_cache(model: str) -> bool:
    return model.startswith("claude-")


def system_block(text: str, model: str):
    if supports_anthropic_cache(model):
        return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
    return text


def cache_last_tool(tools: list[dict], model: str) -> list[dict]:
    if not tools or not supports_anthropic_cache(model):
        return tools
    out = list(tools)
    out[-1] = {**out[-1], "cache_control": {"type": "ephemeral"}}
    return out


def default_stage0_model() -> str:
    return os.environ.get("CITATION_CLEANER_STAGE0_MODEL", "claude-haiku-4-5-20251001")


def make_anthropic_client() -> Any:
    try:
        from anthropic import Anthropic
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("anthropic SDK not installed. Use --dry-run or install requirements.") from exc
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set. Use --dry-run or export it.")
    return Anthropic()
