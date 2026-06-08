"""Stage 6 tool-using resolver for uncertain clusters."""

from __future__ import annotations

import json

from citation_cleaner.llm.anthropic import cache_last_tool, system_block
from citation_cleaner.resolvers.judge import extract_json_object
from citation_cleaner.resolvers.tools import TOOL_DISPATCH

DEFAULT_AGENT_MODEL = "claude-sonnet-4-6"
AGENT_TOOL_BUDGET = 5

AGENT_SYSTEM = """\
You are a bibliographic disambiguation agent. The first-pass judge could not
resolve this cluster. Use available tools to gather authoritative metadata,
then return a final verdict.

Budget: at most 5 tool calls. Quality over quantity.

When ready, respond with ONLY:
{
  "verdict": "same" | "different" | "uncertain",
  "canonical": {
    "authors": [{"surname": "...", "given": "..."|null}, ...],
    "year": int|null, "title": str, "venue": str|null,
    "type": "article"|"book"|"inproceedings"|"chapter"|"preprint"|"unknown",
    "doi": str|null, "volume": str|null, "pages": str|null
  } | null,
  "confidence": <float 0..1>,
  "reasoning": "<2-3 sentences>",
  "needs_external_lookup": false
}
"""

TOOLS_SCHEMA = [
    {
        "name": "search_crossref",
        "description": "Query Crossref for article and conference metadata.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "author": {"type": "string"},
                "year": {"type": "integer"},
            },
            "required": ["title"],
        },
    },
    {
        "name": "search_openalex",
        "description": "Query OpenAlex, broader coverage including books and preprints.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "lookup_citing_context",
        "description": "Return surrounding sentence(s) for this raw reference in one citing paper.",
        "input_schema": {
            "type": "object",
            "properties": {"raw_ref": {"type": "string"}, "citing_paper_id": {"type": "string"}},
            "required": ["raw_ref", "citing_paper_id"],
        },
    },
]


def run_agent(cluster: dict, client, model: str = DEFAULT_AGENT_MODEL, max_calls: int = AGENT_TOOL_BUDGET) -> dict:
    user_payload = json.dumps({"members": cluster["members"]}, ensure_ascii=False, indent=2)
    messages = [
        {
            "role": "user",
            "content": (
                "The judge could not resolve this cluster. Determine whether it is one work or multiple.\n\n"
                f"Cluster:\n{user_payload}"
            ),
        }
    ]
    calls_used = 0
    final_text: str | None = None

    for _ in range(max_calls + 2):
        resp = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0.3,
            system=system_block(AGENT_SYSTEM, model),
            tools=cache_last_tool(TOOLS_SCHEMA, model),
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "end_turn":
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    final_text = block.text
                    break
            break
        if resp.stop_reason != "tool_use":
            break

        tool_results = []
        for block in resp.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            if calls_used >= max_calls:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Tool budget exhausted. Return final JSON verdict now.",
                        "is_error": True,
                    }
                )
                continue
            fn = TOOL_DISPATCH.get(block.name)
            if fn is None:
                result = {"error": f"Unknown tool: {block.name}"}
            else:
                try:
                    result = fn(**block.input)
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"}
            calls_used += 1
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)[:4000]})
        messages.append({"role": "user", "content": tool_results})

    if final_text is None:
        return {
            "verdict": "uncertain",
            "canonical": None,
            "confidence": 0.2,
            "reasoning": "agent did not produce a final verdict",
            "needs_external_lookup": False,
            "tool_calls_used": calls_used,
        }
    try:
        verdict = extract_json_object(final_text)
    except json.JSONDecodeError:
        verdict = {
            "verdict": "uncertain",
            "canonical": None,
            "confidence": 0.2,
            "reasoning": f"agent output was not JSON: {final_text[:200]}",
            "needs_external_lookup": False,
        }
    verdict["tool_calls_used"] = calls_used
    return verdict


def run_agent_dryrun(cluster: dict) -> dict:
    return {
        "verdict": "uncertain",
        "canonical": None,
        "confidence": 0.2,
        "reasoning": "dry-run agent; external lookup skipped",
        "needs_external_lookup": False,
        "tool_calls_used": 0,
    }
