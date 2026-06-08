"""Optional LLM rescue paths used by Stage 0."""

from __future__ import annotations

import json

from citation_cleaner.llm.anthropic import default_stage0_model, make_anthropic_client, system_block
from citation_cleaner.llm.json_parse import coerce_json_string_list, parse_llm_json

FALLBACK_SYSTEM = """\
You extract bibliography references from the References section of an academic
paper. Return only a JSON array of raw reference strings, in original order.
Do not normalize, rewrite, or invent references.
Do not wrap the array in an object — output must start with [ and end with ].
"""

FALLBACK_RETRY_SUFFIX = (
    "\n\nIMPORTANT: Reply with ONLY a JSON array of strings, "
    'for example ["Author A et al. Title. Venue, 2020.", "...]. '
    "No markdown fences, no wrapper object."
)

DISAMBIGUATE_SYSTEM = """\
You link an in-text author-year citation marker to one candidate bibliography
entry. Return only JSON: {"index": int|null, "confidence": float}.
Use zero-based index into the candidate list. Return null if uncertain.
"""


class LLMFallback:
    def __init__(self, client=None, model: str | None = None) -> None:
        self.model = model or default_stage0_model()
        self.client = client

    def _client(self):
        if self.client is None:
            self.client = make_anthropic_client()
        return self.client

    def parse_reference_section(self, section_text: str, *, max_attempts: int = 2) -> list[str]:
        last_error: Exception | None = None
        retry_hint = ""
        for _attempt in range(max_attempts):
            try:
                resp = self._client().messages.create(
                    model=self.model,
                    max_tokens=4096,
                    temperature=0,
                    system=system_block(FALLBACK_SYSTEM, self.model),
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Extract the bibliography entries from this References section.\n\n"
                                f"{section_text[:30000]}{retry_hint}"
                            ),
                        }
                    ],
                )
                parsed = parse_llm_json(resp.content[0].text)
                refs = coerce_json_string_list(parsed)
                if refs:
                    return refs
                last_error = ValueError(
                    f"Stage 0 fallback did not return a JSON array (got {type(parsed).__name__})"
                )
            except Exception as exc:  # noqa: BLE001 - retry then fall back upstream
                last_error = exc
            retry_hint = FALLBACK_RETRY_SUFFIX
        assert last_error is not None
        raise last_error

    def disambiguate_author_year(self, marker: str, context: str, candidates: list[str]) -> tuple[int | None, float]:
        payload = {"marker": marker, "context": context, "candidates": candidates}
        resp = self._client().messages.create(
            model=self.model,
            max_tokens=256,
            temperature=0,
            system=system_block(DISAMBIGUATE_SYSTEM, self.model),
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        )
        parsed = parse_llm_json(resp.content[0].text)
        if not isinstance(parsed, dict):
            raise ValueError("Stage 0 disambiguation did not return a JSON object")
        index = parsed.get("index")
        confidence = float(parsed.get("confidence", 0.0))
        if index is None:
            return None, confidence
        if not isinstance(index, int) or index < 0 or index >= len(candidates):
            return None, 0.0
        return index, confidence
