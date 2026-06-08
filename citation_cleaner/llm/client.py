"""LLM client factory.

Picks the chat backend based on `config` (falling back to environment), so the
whole pipeline can run against either Anthropic directly or any model exposed
through OpenRouter without changing the individual call sites.

Recognized config keys (all optional):
  - llm_provider:        "anthropic" (default) | "openrouter"
  - openrouter_api_key:  key used when provider == "openrouter"

Environment fallbacks:
  - CITATION_CLEANER_LLM_PROVIDER
  - OPENROUTER_API_KEY
"""

from __future__ import annotations

import os
from typing import Any, Optional

from citation_cleaner.llm.anthropic import make_anthropic_client
from citation_cleaner.llm.openrouter import make_openrouter_client


def resolve_provider(config: Optional[dict] = None) -> str:
    config = config or {}
    provider = (
        config.get("llm_provider")
        or os.environ.get("CITATION_CLEANER_LLM_PROVIDER")
        or "anthropic"
    )
    return str(provider).strip().lower()


def make_llm_client(config: Optional[dict] = None) -> Any:
    config = config or {}
    provider = resolve_provider(config)
    if provider == "openrouter":
        return make_openrouter_client(config.get("openrouter_api_key"))
    return make_anthropic_client()
