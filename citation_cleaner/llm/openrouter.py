"""OpenRouter client adapter.

OpenRouter (https://openrouter.ai) exposes an OpenAI-compatible Chat Completions
API that proxies many model families (Anthropic, OpenAI, Google, Meta, ...).

The rest of this codebase was written against the Anthropic Messages API
(`client.messages.create(...)` returning `.content` blocks + `.stop_reason`,
including tool-use). Rather than rewrite every call site, this module provides a
thin adapter whose surface matches the slice of the Anthropic SDK we actually
use, while translating requests/responses to/from the OpenAI chat format under
the hood. That keeps Stage 2 extraction, the Stage 5 judge, the Stage 6
tool-using agent, and the Stage 0 fallback all working unchanged.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# --- Anthropic-shaped response blocks ---------------------------------------
@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class _Response:
    content: list
    stop_reason: str
    usage: Optional[dict] = field(default=None)


# --- block helpers (work for both our objects and Anthropic-style dicts) -----
def _btype(block: Any) -> Optional[str]:
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


def _battr(block: Any, name: str) -> Any:
    if isinstance(block, dict):
        return block.get(name)
    return getattr(block, name, None)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _system_to_text(system: Any) -> Optional[str]:
    if system is None:
        return None
    if isinstance(system, str):
        return system
    parts = []
    for block in system:
        if isinstance(block, dict):
            parts.append(block.get("text", ""))
        else:
            parts.append(getattr(block, "text", ""))
    text = "\n".join(p for p in parts if p)
    return text or None


def _convert_tools(tools: Optional[list]) -> Optional[list]:
    if not tools:
        return None
    out = []
    for tool in tools:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get(
                        "input_schema", {"type": "object", "properties": {}}
                    ),
                },
            }
        )
    return out


def _convert_messages(messages: list) -> list:
    """Translate Anthropic-style messages into OpenAI chat messages.

    Handles the three shapes this codebase produces:
      - {"role": ..., "content": "<str>"}
      - {"role": "assistant", "content": [TextBlock|ToolUseBlock, ...]}  (a prior turn)
      - {"role": "user", "content": [{"type": "tool_result", ...}, ...]}
    """
    out: list = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for block in content:
                btype = _btype(block)
                if btype == "text":
                    text_parts.append(_battr(block, "text") or "")
                elif btype == "tool_use":
                    tool_calls.append(
                        {
                            "id": _battr(block, "id"),
                            "type": "function",
                            "function": {
                                "name": _battr(block, "name"),
                                "arguments": json.dumps(
                                    _battr(block, "input") or {}, ensure_ascii=False
                                ),
                            },
                        }
                    )
            assistant_msg: dict = {
                "role": "assistant",
                "content": ("".join(text_parts) or None),
            }
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            out.append(assistant_msg)
            continue

        # role == "user" with a list: tool_result blocks and/or text blocks.
        pending_text: list[str] = []
        for block in content:
            btype = _btype(block)
            if btype == "tool_result":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": _battr(block, "tool_use_id"),
                        "content": _stringify(_battr(block, "content")),
                    }
                )
            elif btype == "text":
                pending_text.append(_battr(block, "text") or "")
            else:
                pending_text.append(_stringify(block))
        if pending_text:
            out.append({"role": "user", "content": "\n".join(pending_text)})

    return out


def _convert_response(resp: Any) -> _Response:
    choice = resp.choices[0]
    message = choice.message
    blocks: list = []

    if getattr(message, "content", None):
        blocks.append(TextBlock(text=message.content))

    tool_calls = getattr(message, "tool_calls", None) or []
    for call in tool_calls:
        try:
            args = json.loads(call.function.arguments or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {}
        blocks.append(ToolUseBlock(id=call.id, name=call.function.name, input=args))

    if not blocks:
        blocks.append(TextBlock(text=""))

    stop_reason = "tool_use" if tool_calls else "end_turn"

    usage = None
    raw_usage = getattr(resp, "usage", None)
    if raw_usage is not None:
        usage = {
            "input_tokens": getattr(raw_usage, "prompt_tokens", None),
            "output_tokens": getattr(raw_usage, "completion_tokens", None),
        }

    return _Response(content=blocks, stop_reason=stop_reason, usage=usage)


class _Messages:
    """Mimics `client.messages` from the Anthropic SDK."""

    def __init__(self, client: Any, extra_headers: dict) -> None:
        self._client = client
        self._extra_headers = extra_headers

    def create(
        self,
        *,
        model: str,
        messages: list,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system: Any = None,
        tools: Optional[list] = None,
        **_ignored: Any,
    ) -> _Response:
        oai_messages: list = []
        sys_text = _system_to_text(system)
        if sys_text:
            oai_messages.append({"role": "system", "content": sys_text})
        oai_messages.extend(_convert_messages(messages))

        kwargs: dict = {
            "model": model,
            "messages": oai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        oai_tools = _convert_tools(tools)
        if oai_tools:
            kwargs["tools"] = oai_tools
        if self._extra_headers:
            kwargs["extra_headers"] = self._extra_headers

        resp = self._client.chat.completions.create(**kwargs)
        return _convert_response(resp)


class OpenRouterClient:
    """Drop-in replacement for the slice of `anthropic.Anthropic` we use."""

    def __init__(self, api_key: str, *, app_title: str = "citation-cleaner-v4") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError(
                "openai SDK not installed. Use --dry-run or install requirements."
            ) from exc
        client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        extra_headers = {"X-Title": app_title}
        self.messages = _Messages(client, extra_headers)


def make_openrouter_client(api_key: Optional[str] = None) -> OpenRouterClient:
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OpenRouter API key not set. Provide it in the UI / config, set "
            "OPENROUTER_API_KEY, or use --dry-run."
        )
    return OpenRouterClient(key)
