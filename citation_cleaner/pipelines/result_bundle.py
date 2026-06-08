"""Pack pipeline artifacts into a single downloadable ZIP bundle."""

from __future__ import annotations

import zipfile
from pathlib import Path

# Parsed / intermediate structured outputs (JSONL + related CSV).
_STRUCTURED_NAMES = (
    "parsed_documents.jsonl",
    "extracted.jsonl",
    "candidates.jsonl",
    "resolved.jsonl",
    "citances.jsonl",
    "quarantine.jsonl",
    "review_queue.jsonl",
    "precleaned.csv",
    "canonical_refs.csv",
    "raw_to_canonical.csv",
    "raw_refs.csv",
)

_REPORT_NAMES = (
    "cocited_refs.csv",
    "cocited_refs.md",
    "cocited_refs.json",
)


_README = """\
Citation Discoverer v4 — 结果包说明
==================================

本 ZIP 由 Citation Discoverer 流水线自动生成，包含共被引分析的最终报告、
下载的源论文 PDF，以及各阶段的结构化中间产物。

目录结构
--------

reports/
  最终共被引排名与报告（可直接阅读或导入表格软件）。

  cocited_refs.csv
    排名表：共被引次数、文内出现次数、全球被引、标题、作者、年份、DOI 等。

  cocited_refs.md
    完整 Markdown 报告，含排名表与「文内引用原文」附录。

  cocited_refs.json
    与 CSV 相同的聚合结果，JSON 格式，便于程序读取。

source_pdfs/
  从 OpenAlex 下载（或用户上传）的源论文 PDF 全文。
  文件名通常含 OpenAlex 短 ID（如 W1234567890.pdf）或 USER_<名称>.pdf。

structured/
  Stage 0–6 解析与规范化产出的结构化数据（JSON Lines 或 CSV）。
  若某步被跳过、缓存命中或 dry-run，对应文件可能不存在。

  parsed_documents.jsonl
    Stage 0：每篇 PDF 一行。含 citing_paper_id、正文片段、参考文献列表、
    citances（文内引用位置）等 ParsedDocument 字段。

  precleaned.csv
    Stage 1：预清洗后的参考文献字符串（citing_paper_id, raw, cleaned）。

  extracted.jsonl
    Stage 2：LLM 抽取的结构化字段（作者、年份、标题、venue、DOI 等）。

  candidates.jsonl
    Stage 3–4：按姓氏/年份分块后，标题 embedding 聚类得到的候选簇。

  resolved.jsonl
    Stage 5–6：经 LLM 判定与外部解析后的规范化引用（canonical_id、置信度等）。

  citances.jsonl
    文内引用：引用标记、上下文句子、页码、与 raw 参考文献的链接方式。

  canonical_refs.csv / raw_to_canonical.csv / raw_refs.csv
    规范化引用表、raw→canonical 映射、各源论文的 raw 参考文献列表。

  quarantine.jsonl
    Stage 2 抽取失败、暂隔离的条目（若存在）。

  review_queue.jsonl
    Stage 5–6 未能自动 resolve、需人工复核的簇（若存在）。

使用建议
--------
  · 快速浏览：打开 reports/cocited_refs.md
  · 表格分析：reports/cocited_refs.csv
  · 复现/调试：structured/ 下的 JSONL，每行一条 JSON 记录
  · 对照原文：source_pdfs/ 与 structured/citances.jsonl

生成工具：Citation Discoverer v4 (citation-cleaner)
"""


def build_result_zip(workdir: Path, *, zip_name: str = "citation_discoverer_bundle.zip") -> Path:
    """Create a ZIP with reports, source PDFs, and structured parse outputs.

    Layout inside the archive::

        README.txt
        reports/cocited_refs.{csv,md,json}
        source_pdfs/*.pdf
        structured/{parsed_documents,extracted,resolved,...}.jsonl
    """
    workdir = Path(workdir)
    zip_path = workdir / zip_name

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.txt", _README)

        for name in _REPORT_NAMES:
            path = workdir / name
            if path.is_file():
                zf.write(path, f"reports/{name}")

        for name in _STRUCTURED_NAMES:
            path = workdir / name
            if path.is_file():
                zf.write(path, f"structured/{name}")

        pdf_dir = workdir / "source_pdfs"
        if pdf_dir.is_dir():
            for pdf in sorted(pdf_dir.glob("*.pdf")):
                zf.write(pdf, f"source_pdfs/{pdf.name}")

    return zip_path
