"""Parser interface for Stage 0 backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from citation_cleaner.schemas.document import ParsedDocument


class PDFParser(ABC):
    @abstractmethod
    def parse(self, pdf_path: Path) -> ParsedDocument:
        """Parse one PDF into references, citances, metadata, and quality."""
