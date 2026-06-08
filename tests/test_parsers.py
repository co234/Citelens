from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from citation_cleaner.parsers.quality import score_reference_quality
from citation_cleaner.parsers.reference_section import locate_reference_section, split_references


def test_locate_and_split_numbered_references():
    pages = [
        "Title\n\nBody cites [1].\n\nReferences\n[1] Smith, J. (2020). Deep learning for citations.\n[2] Jones, K. (2021). Reference parsing."
    ]
    section = locate_reference_section(pages)
    assert section is not None
    refs = split_references(section.text)
    assert len(refs) == 2
    assert refs[0].startswith("[1] Smith")


def test_quality_flags_low_extraction():
    quality = score_reference_quality(["Smith, J. (2020). Short set."])
    assert quality.needs_llm_fallback
    assert quality.n_refs_extracted == 1


def test_heuristic_parser_on_synthetic_pdf(tmp_path):
    try:
        import fitz
    except ImportError:
        return

    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "A Synthetic Citation Paper\n"
        "Smith (2020) introduced the baseline. Later work [2] extended it.\n"
        "References\n"
        "[1] Smith, J. (2020). Deep learning for citations.\n"
        "[2] Jones, K. (2021). Reference parsing.\n",
    )
    doc.save(pdf_path)
    doc.close()

    from citation_cleaner.parsers.heuristic import HeuristicParser

    parsed = HeuristicParser(llm_fallback=None).parse(pdf_path)
    assert len(parsed.references) == 2
    assert any(c.raw_ref == parsed.references[1] for c in parsed.citances)
