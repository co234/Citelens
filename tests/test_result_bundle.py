"""Tests for the result ZIP bundle."""

import zipfile
from pathlib import Path

from citation_cleaner.pipelines.result_bundle import build_result_zip


def test_build_result_zip_layout(tmp_path: Path):
    workdir = tmp_path / "run"
    workdir.mkdir()
    (workdir / "cocited_refs.csv").write_text("rank,title\n1,Test\n", encoding="utf-8")
    (workdir / "cocited_refs.md").write_text("# Report\n", encoding="utf-8")
    (workdir / "parsed_documents.jsonl").write_text('{"citing_paper_id":"W1"}\n', encoding="utf-8")
    pdf_dir = workdir / "source_pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "paper1.pdf").write_bytes(b"%PDF-1.4 stub")

    zip_path = build_result_zip(workdir)
    assert zip_path.is_file()

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    assert "README.txt" in names
    assert "reports/cocited_refs.csv" in names
    assert "reports/cocited_refs.md" in names
    assert "structured/parsed_documents.jsonl" in names
    assert "source_pdfs/paper1.pdf" in names
