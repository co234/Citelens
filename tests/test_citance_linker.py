from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from citation_cleaner.parsers.citance_linker import PageText, build_author_year_index, link_citances


def test_numbered_bracket_links_directly():
    refs = [
        "Smith, J. (2020). Deep learning for citations.",
        "Jones, K. (2021). Reference parsing.",
    ]
    pages = [PageText(page=1, text="Prior work [1] and later work [2] are both relevant.")]
    citances = link_citances(pages, refs, "paper-1")
    assert len(citances) == 2
    assert citances[0].raw_ref == refs[0]
    assert citances[0].link_method == "numbered_direct"


def test_bracket_range_expands():
    refs = ["A (2020). One.", "B (2021). Two.", "C (2022). Three."]
    pages = [PageText(page=1, text="Several results [1-3] apply.")]
    citances = link_citances(pages, refs, "paper-1")
    assert [c.raw_ref for c in citances] == refs


def test_author_year_unique_link():
    refs = ["Smith, J. (2020). Deep learning for citations."]
    pages = [PageText(page=1, text="We follow Smith (2020) in our setup.")]
    citances = link_citances(pages, refs, "paper-1")
    assert len(citances) == 1
    assert citances[0].raw_ref == refs[0]
    assert citances[0].link_method == "author_year_unique"


def test_author_year_index_handles_given_first():
    refs = ["J. Smith. 2020. Deep learning for citations."]
    index = build_author_year_index(refs)
    assert ("smith", 2020) in index
