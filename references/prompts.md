# Prompt Contracts

Prompt text lives in code next to the caller:

- `citation_cleaner/extractors/llm.py` for Stage 2 extraction.
- `citation_cleaner/resolvers/judge.py` for Stage 5 judging.
- `citation_cleaner/resolvers/agent.py` for Stage 6 tool use.
- `citation_cleaner/parsers/llm_fallback.py` for the Stage 0 fallback.

Rules:

- Output JSON only.
- Preserve raw strings for traceability.
- Return `null` instead of guessing.
- Keep the Stage 0 fallback limited to raw-reference extraction; do not
  normalize or canonicalize there.
- Use temperature 0 for extraction and judge, 0.3 for the tool agent.
