"""Extract a short topic/keyword string from an uploaded PDF.

Used when the user uploads their own paper instead of typing a terminology:
we read the paper's title + abstract + a snippet of intro, ask Claude to
produce a focused search phrase, then feed that into the OpenAlex search.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from citation_cleaner.llm.client import make_llm_client


TOPIC_PROMPT = """\
You are helping a researcher find related papers. Given the title, abstract,
and intro snippet below, output ONE search phrase (3-8 words) that captures
the paper's core technical topic. Output only the phrase, no quotes, no
explanation.

Title: {title}

Abstract / opening text:
{snippet}
"""


def _read_pdf_head(pdf_path: Path, max_chars: int = 4000) -> tuple[str, str]:
    """Return (title_guess, opening_snippet) from a PDF.

    Title guess: first non-empty line of page 1. Snippet: first `max_chars`
    chars of page 1+2 text. Cheap, no LLM. Good enough for topic extraction.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise SystemExit(
            "pymupdf is required for topic extraction. pip install pymupdf"
        ) from exc

    doc = fitz.open(pdf_path)
    try:
        pages_text: list[str] = []
        for i in range(min(2, doc.page_count)):
            pages_text.append(doc.load_page(i).get_text("text") or "")
    finally:
        doc.close()

    full = "\n".join(pages_text)
    lines = [ln.strip() for ln in full.splitlines() if ln.strip()]
    title_guess = lines[0] if lines else ""
    snippet = full[:max_chars]
    return title_guess, snippet


def topic_from_paper(
    pdf_path: Path,
    *,
    dry_run: bool = False,
    model: Optional[str] = None,
    config: Optional[dict] = None,
) -> str:
    """Return a short search phrase derived from the uploaded paper."""
    title_guess, snippet = _read_pdf_head(pdf_path)

    if dry_run:
        # Deterministic offline behavior: use the title verbatim.
        return title_guess[:80] or "machine learning"

    config = config or {}
    client = make_llm_client(config)
    prompt = TOPIC_PROMPT.format(title=title_guess, snippet=snippet)
    response = client.messages.create(
        model=model or config.get("topic_model") or "claude-haiku-4-5-20251001",
        max_tokens=60,
        messages=[{"role": "user", "content": prompt}],
    )
    # Concatenate text blocks defensively.
    parts = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    ]
    phrase = " ".join(parts).strip().strip('"').strip("'")
    return phrase or title_guess[:80] or "machine learning"
