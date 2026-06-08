"""Tests for structured pipeline logging."""

from citation_cleaner.pipelines.pipeline_log import PipelineLogger


def test_stage_dict_and_progress():
    pl = PipelineLogger(mirror_stdout=False)
    pl.begin("s0", "parsing 3 PDF(s)…")
    pl.progress("s0", 1, 3, "done paper_a.pdf")
    pl.progress("s0", 2, 3, "done paper_b.pdf")
    snap = pl.snapshot()
    assert snap["_meta"]["active_stage"] == "s0"
    s0 = snap["stages"]["s0"]
    assert s0["status"] == "running"
    assert s0["progress"]["current"] == 2
    assert s0["progress"]["total"] == 3
    assert any("(2/3)" in line for line in s0["lines"])


def test_mirror_stdout_format(capsys):
    pl = PipelineLogger(mirror_stdout=True)
    pl.progress("download", 2, 5, "done W123")
    out = capsys.readouterr().out
    assert "[discover] (2/5) done W123" in out


def test_flat_text_joins_stages():
    pl = PipelineLogger(mirror_stdout=False)
    pl.info("search", "found papers")
    pl.info("s2", "extracting")
    text = pl.flat_text()
    assert "found papers" in text
    assert "extracting" in text
