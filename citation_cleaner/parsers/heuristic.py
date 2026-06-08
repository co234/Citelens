"""PyMuPDF-backed Stage 0 parser for modern publisher PDFs."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

from citation_cleaner.parsers.base import PDFParser
from citation_cleaner.parsers.citance_linker import PageText, link_citances
from citation_cleaner.parsers.quality import QualityThresholds, score_reference_quality
from citation_cleaner.parsers.reference_section import locate_reference_section, split_references
from citation_cleaner.schemas.document import HeaderMeta, ParsedDocument

DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
ARXIV_RE = re.compile(r"\barXiv:\s*([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


class HeuristicParser(PDFParser):
    def __init__(
        self,
        llm_fallback=None,
        quality_thresholds: QualityThresholds = QualityThresholds.default(),
    ) -> None:
        self.llm_fallback = llm_fallback
        self.quality_thresholds = quality_thresholds

    def parse(self, pdf_path: Path) -> ParsedDocument:
        pdf_path = Path(pdf_path)
        page_texts, first_page_spans = self._extract_pages(pdf_path)
        page_strings = [page.text for page in page_texts]
        section = locate_reference_section(page_strings)
        references = split_references(section.text) if section else []
        quality = score_reference_quality(references, self.quality_thresholds)

        if quality.needs_llm_fallback and self.llm_fallback is not None and section is not None:
            try:
                fallback_refs = self.llm_fallback.parse_reference_section(section.text)
            except Exception as exc:  # noqa: BLE001 - keep heuristic refs for this PDF
                quality.llm_fallback_invoked = True
                quality.notes.append(f"llm fallback failed ({exc}); kept heuristic output")
            else:
                fallback_quality = score_reference_quality(
                    fallback_refs,
                    self.quality_thresholds,
                    llm_fallback_invoked=True,
                )
                if fallback_quality.n_refs_extracted >= quality.n_refs_extracted:
                    references = fallback_refs
                    quality = fallback_quality
                else:
                    quality.llm_fallback_invoked = True
                    quality.notes.append(
                        "llm fallback returned fewer refs than heuristic; kept heuristic output"
                    )

        header = self._extract_header(pdf_path, page_strings[0] if page_strings else "", first_page_spans)
        citing_paper_id = header.doi or header.arxiv_id or hashlib.sha1(pdf_path.name.encode()).hexdigest()[:12]
        body_pages = page_texts
        if section is not None:
            body_pages = page_texts[: max(section.start_page - 1, 0)]
            if section.start_page - 1 < len(page_texts):
                body_pages.append(PageText(page=section.start_page, text=page_texts[section.start_page - 1].text.split(section.heading, 1)[0]))
        citances = link_citances(body_pages, references, citing_paper_id, llm_fallback=self.llm_fallback)
        return ParsedDocument(
            pdf_path=str(pdf_path),
            citing_paper_id=citing_paper_id,
            header=header,
            references=references,
            citances=citances,
            quality=quality,
        )

    def _extract_pages(self, pdf_path: Path) -> tuple[list[PageText], list[dict]]:
        try:
            import fitz
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError("PyMuPDF is required for Stage 0. Install pymupdf or run --refs.") from exc

        page_texts: list[PageText] = []
        first_page_spans: list[dict] = []
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                text = page.get_text("text")
                page_texts.append(PageText(page=i + 1, text=text))
                if i == 0:
                    first_page_spans = _collect_spans(page.get_text("dict"))
        return page_texts, first_page_spans

    def _extract_header(self, pdf_path: Path, first_page_text: str, spans: list[dict]) -> HeaderMeta:
        doi_match = DOI_RE.search(first_page_text)
        arxiv_match = ARXIV_RE.search(first_page_text)
        year_match = YEAR_RE.search(first_page_text[:2500])
        title = _guess_title(first_page_text, spans)
        return HeaderMeta(
            title=title,
            year=int(year_match.group(0)) if year_match else None,
            doi=doi_match.group(0).rstrip(".") if doi_match else None,
            arxiv_id=arxiv_match.group(1) if arxiv_match else None,
        )


def _collect_spans(page_dict: dict) -> list[dict]:
    spans: list[dict] = []
    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = re.sub(r"\s+", " ", span.get("text", "")).strip()
                if text:
                    spans.append({"text": text, "size": float(span.get("size", 0)), "bbox": span.get("bbox", [])})
    return spans


def _guess_title(first_page_text: str, spans: list[dict]) -> str | None:
    candidates = [
        span
        for span in spans
        if len(span["text"]) >= 8
        and not DOI_RE.search(span["text"])
        and not ARXIV_RE.search(span["text"])
        and not re.search(r"\b(abstract|keywords|introduction)\b", span["text"], re.IGNORECASE)
    ]
    if candidates:
        max_size = max(span["size"] for span in candidates)
        title_spans = [span["text"] for span in candidates if span["size"] >= max_size - 0.5]
        title = " ".join(title_spans[:3])
        title = re.sub(r"\s+", " ", title).strip()
        if 8 <= len(title) <= 250:
            return title

    for line in first_page_text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) >= 12 and not re.search(r"\b(abstract|doi|arxiv)\b", line, re.IGNORECASE):
            return line[:250]
    return None
