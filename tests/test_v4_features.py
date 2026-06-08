"""Tests for v4 additions: author search, sort-by, and in-context occurrences.

These tests are designed to run fully offline. Anything that would talk to
OpenAlex is monkeypatched, and the in-context test writes its own synthetic
workdir.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from citation_cleaner.aggregate.cocitation import (
    CocitedRecord,
    InContextOccurrence,
    aggregate_cocitations,
    write_cocitation_outputs,
)
from citation_cleaner.discovery import openalex as oa


# ---------- helpers ----------

def _write_workdir_with_citances(wd: Path) -> None:
    """Build a minimal but realistic workdir: one canonical ref cited from
    three source papers with multiple in-text occurrences each."""
    wd.mkdir(parents=True, exist_ok=True)

    resolved = {
        "canonical_id": "abc123",
        "authors": [{"surname": "Vaswani", "given": "Ashish"}],
        "year": 2017,
        "title": "Attention Is All You Need",
        "venue": "NeurIPS",
        "doi": "10.48550/arxiv.1706.03762",
        "confidence": 0.95,
    }
    (wd / "resolved.jsonl").write_text(json.dumps(resolved) + "\n", encoding="utf-8")

    (wd / "raw_to_canonical.csv").write_text(
        "raw,canonical_id,confidence\n"
        '"Vaswani et al. 2017",abc123,0.95\n',
        encoding="utf-8",
    )
    (wd / "raw_refs.csv").write_text(
        "citing_paper_id,raw\n"
        'PAPER_A,"Vaswani et al. 2017"\n'
        'PAPER_B,"Vaswani et al. 2017"\n'
        'PAPER_C,"Vaswani et al. 2017"\n',
        encoding="utf-8",
    )

    citances = [
        {
            "citing_paper_id": "PAPER_A",
            "raw_ref": "Vaswani et al. 2017",
            "context": "We follow Vaswani et al. (2017) for the transformer baseline.",
            "marker": {
                "style": "narrative_author_year",
                "raw_marker": "Vaswani et al. (2017)",
                "page": 3,
                "char_span": [10, 30],
            },
            "link_method": "author_year_unique",
            "link_confidence": 0.9,
        },
        {
            "citing_paper_id": "PAPER_A",
            "raw_ref": "Vaswani et al. 2017",
            "context": "The attention mechanism (Vaswani et al., 2017) inspired this work.",
            "marker": {
                "style": "paren_author_year",
                "raw_marker": "(Vaswani et al., 2017)",
                "page": 5,
                "char_span": [25, 49],
            },
            "link_method": "author_year_unique",
            "link_confidence": 0.9,
        },
        {
            "citing_paper_id": "PAPER_B",
            "raw_ref": "Vaswani et al. 2017",
            "context": "Self-attention was introduced by Vaswani et al. (2017).",
            "marker": {
                "style": "narrative_author_year",
                "raw_marker": "Vaswani et al. (2017)",
                "page": 2,
                "char_span": [33, 54],
            },
            "link_method": "author_year_unique",
            "link_confidence": 0.9,
        },
    ]
    (wd / "citances.jsonl").write_text(
        "\n".join(json.dumps(c) for c in citances) + "\n",
        encoding="utf-8",
    )


# ---------- aggregate / in-context tests ----------

def test_in_context_occurrences_are_attached(tmp_path):
    _write_workdir_with_citances(tmp_path)
    records = aggregate_cocitations(tmp_path)
    assert len(records) == 1
    rec = records[0]
    assert rec.canonical_id == "abc123"
    # All three citances attached.
    assert len(rec.occurrences) == 3
    # Spans both source papers.
    by_paper = {o.citing_paper_id for o in rec.occurrences}
    assert by_paper == {"PAPER_A", "PAPER_B"}
    # Pages survived.
    pages_a = sorted(o.page for o in rec.occurrences if o.citing_paper_id == "PAPER_A")
    assert pages_a == [3, 5]
    # Markers survived.
    assert any("Vaswani et al." in (o.marker_raw or "") for o in rec.occurrences)


def test_markdown_report_includes_in_context_section(tmp_path):
    _write_workdir_with_citances(tmp_path)
    records = aggregate_cocitations(tmp_path)
    paths = write_cocitation_outputs(
        records,
        tmp_path,
        query="test",
        source_papers=[
            {"short_id": "PAPER_A", "title": "A paper", "year": 2024, "cited_by_count": 50},
            {"short_id": "PAPER_B", "title": "B paper", "year": 2024, "cited_by_count": 30},
            {"short_id": "PAPER_C", "title": "C paper", "year": 2024, "cited_by_count": 10},
        ],
    )
    md = paths["md"].read_text(encoding="utf-8")

    # Section header is there.
    assert "## In-context appearances" in md
    # Most-quoted highlight names the ref and the count.
    assert "Attention Is All You Need" in md
    assert "3 in-text mentions" in md
    # Per-paper subsections.
    assert "**In `PAPER_A`:**" in md
    assert "**In `PAPER_B`:**" in md
    # Pages and markers rendered.
    assert "p. 3" in md and "p. 5" in md and "p. 2" in md
    assert "Vaswani et al. (2017)" in md
    # The context sentence itself.
    assert "We follow Vaswani et al. (2017)" in md


def test_in_text_hits_column_in_csv(tmp_path):
    _write_workdir_with_citances(tmp_path)
    records = aggregate_cocitations(tmp_path)
    paths = write_cocitation_outputs(records, tmp_path, query="test")
    csv_text = paths["csv"].read_text(encoding="utf-8")
    # Header has the new column.
    header = csv_text.splitlines()[0]
    assert "in_text_hits" in header
    # The single ref has 3 in-text hits.
    assert ",3," in csv_text  # ranked first, with in_text_hits=3


def test_aggregate_works_without_citances_file(tmp_path):
    """If citances.jsonl is absent (no PDF-stage data), aggregation must
    still succeed and occurrences should be empty."""
    # No citances.jsonl this time.
    (tmp_path / "resolved.jsonl").write_text(
        json.dumps(
            {
                "canonical_id": "x1",
                "authors": [{"surname": "Smith"}],
                "year": 2020,
                "title": "Foo",
                "confidence": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "raw_to_canonical.csv").write_text(
        "raw,canonical_id,confidence\n"
        '"Smith 2020",x1,1.0\n',
        encoding="utf-8",
    )
    # No raw_refs.csv -> CSV-mode codepath.
    records = aggregate_cocitations(tmp_path)
    assert len(records) == 1
    assert records[0].occurrences == []


# ---------- sort_by tests (logic-only, no network) ----------

def test_apply_local_sort_recency():
    papers = [
        oa.DiscoveredPaper(
            openalex_id="W1", title="old", year=2010, doi=None,
            cited_by_count=999, pdf_url="", relevance_score=0.5,
        ),
        oa.DiscoveredPaper(
            openalex_id="W2", title="new", year=2024, doi=None,
            cited_by_count=5, pdf_url="", relevance_score=0.5,
        ),
        oa.DiscoveredPaper(
            openalex_id="W3", title="middle", year=2018, doi=None,
            cited_by_count=100, pdf_url="", relevance_score=0.5,
        ),
    ]
    sorted_papers = oa._apply_local_sort(papers, "recency")
    assert [p.year for p in sorted_papers] == [2024, 2018, 2010]


def test_apply_local_sort_citations():
    papers = [
        oa.DiscoveredPaper(
            openalex_id="W1", title="a", year=2020, doi=None,
            cited_by_count=10, pdf_url="", relevance_score=0.5,
        ),
        oa.DiscoveredPaper(
            openalex_id="W2", title="b", year=2020, doi=None,
            cited_by_count=1000, pdf_url="", relevance_score=0.1,
        ),
        oa.DiscoveredPaper(
            openalex_id="W3", title="c", year=2020, doi=None,
            cited_by_count=50, pdf_url="", relevance_score=0.9,
        ),
    ]
    sorted_papers = oa._apply_local_sort(papers, "citations")
    assert [p.cited_by_count for p in sorted_papers] == [1000, 50, 10]


def test_apply_local_sort_relevance_falls_back_to_citations():
    # When relevance_score is identical (e.g., author mode populates 0.0),
    # citation count breaks the tie.
    papers = [
        oa.DiscoveredPaper(
            openalex_id="W1", title="a", year=2020, doi=None,
            cited_by_count=10, pdf_url="", relevance_score=0.0,
        ),
        oa.DiscoveredPaper(
            openalex_id="W2", title="b", year=2020, doi=None,
            cited_by_count=500, pdf_url="", relevance_score=0.0,
        ),
    ]
    sorted_papers = oa._apply_local_sort(papers, "relevance")
    assert sorted_papers[0].cited_by_count == 500


# ---------- author resolution (mocked) ----------

class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_resolve_author_picks_most_cited_when_ambiguous(monkeypatch):
    """If OpenAlex returns multiple authors sharing a name, we pick the one
    with the highest cited_by_count."""
    fake_payload = {
        "results": [
            {
                "id": "https://openalex.org/A111",
                "display_name": "John Smith",
                "works_count": 5,
                "cited_by_count": 200,
            },
            {
                "id": "https://openalex.org/A222",
                "display_name": "John Smith",
                "works_count": 50,
                "cited_by_count": 50000,
            },
            {
                "id": "https://openalex.org/A333",
                "display_name": "John Smith",
                "works_count": 10,
                "cited_by_count": 1000,
            },
        ]
    }
    monkeypatch.setattr(
        oa.requests,
        "get",
        lambda *a, **kw: _FakeResponse(fake_payload),
    )
    author = oa.resolve_author("John Smith")
    assert author is not None
    assert author.short_id() == "A222"
    assert author.cited_by_count == 50000


def test_resolve_author_returns_none_when_no_results(monkeypatch):
    monkeypatch.setattr(
        oa.requests,
        "get",
        lambda *a, **kw: _FakeResponse({"results": []}),
    )
    assert oa.resolve_author("Nobody In Particular") is None


def test_resolve_author_direct_id_skips_search(monkeypatch):
    """Passing a raw 'A1234...' ID should fetch /authors/A1234 directly, not
    /authors?search=. We verify by checking the URL the mock sees."""
    seen_urls: list[str] = []

    def fake_get(url, *a, **kw):
        seen_urls.append(url)
        return _FakeResponse(
            {
                "id": "https://openalex.org/A999",
                "display_name": "Direct Id Author",
                "works_count": 1,
                "cited_by_count": 7,
            }
        )

    monkeypatch.setattr(oa.requests, "get", fake_get)
    author = oa.resolve_author("A999")
    assert author is not None
    assert author.display_name == "Direct Id Author"
    # Direct lookup should hit /authors/A999, not /authors?search=
    assert any("/authors/A999" in u for u in seen_urls)


# ---------- search_by_author end-to-end (mocked) ----------

def test_search_by_author_returns_top_n_by_citations(monkeypatch):
    """Mock both /authors (resolve) and /works (list) endpoints, then check
    that search_by_author returns the requested top-N papers, sorted by
    cited_by_count desc."""
    author_payload = {
        "results": [
            {
                "id": "https://openalex.org/A123",
                "display_name": "Jane Doe",
                "works_count": 100,
                "cited_by_count": 10000,
            }
        ]
    }
    works_payload = {
        "results": [
            {
                "id": f"https://openalex.org/W{i}",
                "title": f"Paper {i}",
                "publication_year": 2020 + i % 5,
                "doi": None,
                "cited_by_count": (10 - i) * 100,
                "relevance_score": None,
                "open_access": {},
                "best_oa_location": {"pdf_url": f"https://example.com/{i}.pdf"},
                "authorships": [
                    {"author": {"display_name": "Jane Doe"}}
                ],
                "primary_location": {"source": {"display_name": "Some Venue"}},
            }
            for i in range(10)
        ]
    }

    call_count = {"n": 0}

    def fake_get(url, *a, **kw):
        call_count["n"] += 1
        if "/authors" in url:
            return _FakeResponse(author_payload)
        if "/works" in url:
            return _FakeResponse(works_payload)
        return _FakeResponse({})

    monkeypatch.setattr(oa.requests, "get", fake_get)
    author, papers = oa.search_by_author("Jane Doe", n=3)
    assert author is not None
    assert author.short_id() == "A123"
    assert len(papers) == 3
    # cited_by_count descending: i=0 → 1000, i=1 → 900, i=2 → 800
    assert [p.cited_by_count for p in papers] == [1000, 900, 800]


def test_search_by_terminology_passes_sort_param(monkeypatch):
    """When sort_by='recency', the OpenAlex request must include
    sort=publication_date:desc."""
    captured_params: dict = {}

    def fake_get(url, *, params=None, **kw):
        captured_params.update(params or {})
        return _FakeResponse({"results": []})

    monkeypatch.setattr(oa.requests, "get", fake_get)
    oa.search_by_terminology("transformers", n=3, sort_by="recency")
    assert captured_params.get("sort") == "publication_date:desc"

    captured_params.clear()
    oa.search_by_terminology("transformers", n=3, sort_by="citations")
    assert captured_params.get("sort") == "cited_by_count:desc"

    captured_params.clear()
    oa.search_by_terminology("transformers", n=3, sort_by="relevance")
    # Relevance is OpenAlex's default — no explicit sort key.
    assert "sort" not in captured_params
