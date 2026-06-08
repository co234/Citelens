"""Embedding providers for title similarity."""

from __future__ import annotations

import hashlib
import os

import numpy as np

DEFAULT_EMBED_MODEL = "text-embedding-3-small"

# OpenRouter exposes an OpenAI-compatible /embeddings endpoint, so the same
# client works by swapping the base_url + key and using a slugged model id.
OPENROUTER_EMBED_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_EMBED_MODEL = "qwen/qwen3-embedding-4b"


def embed_titles_openai(
    titles: list[str],
    model: str = DEFAULT_EMBED_MODEL,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
) -> np.ndarray:
    """Embed titles via any OpenAI-compatible /embeddings endpoint.

    Defaults to OpenAI proper. Pass base_url + api_key to target a compatible
    provider such as OpenRouter, Together, Jina, etc.
    """
    use_default_openai = base_url is None and not api_key
    if use_default_openai and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set. Use --dry-run or export it.")
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("openai SDK not installed. Use --dry-run or install requirements.") from exc
    client_kwargs: dict = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    resp = client.embeddings.create(model=model, input=titles)
    return np.array([item.embedding for item in resp.data])


def embed_titles_dryrun(titles: list[str], dim: int = 64) -> np.ndarray:
    out = np.zeros((len(titles), dim), dtype=np.float64)
    for i, title in enumerate(titles):
        text = (title or "").lower()
        for k in range(max(len(text) - 2, 0)):
            trigram = text[k : k + 3]
            h = int(hashlib.md5(trigram.encode()).hexdigest()[:8], 16)
            out[i, h % dim] += 1.0
        norm = np.linalg.norm(out[i])
        if norm > 0:
            out[i] /= norm
    return out
