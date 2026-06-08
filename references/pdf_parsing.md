# PDF Parsing

Stage 0 is an upstream parser for modern text-based PDF files.

## Supported

- PDF files with embedded selectable text.
- `References`, `Bibliography`, `Works Cited`, or common CJK reference-heading variants.
- Numbered references such as `[1]` and `1.`.
- Blank-line and simple hanging-indent author-year reference lists.
- Numbered in-text markers such as `[12]`, `[12-15]`, and `[12, 15]`.
- Parenthetical and narrative author-year markers.

## Not Supported

- Scanned PDFs requiring OCR.
- Full GROBID-style layout reconstruction.
- Runtime Claude vision parsing.
- Non-English reference styles beyond basic heading detection.

Unsupported PDFs should surface through low quality scores rather than silent
success.

## Quality Fallback

LLM fallback is considered when:

- fewer than 5 references are extracted,
- fewer than 50 percent contain a year,
- fewer than 60 percent look author-like.

Dry-run disables fallback. Real runs use `CITATION_CLEANER_STAGE0_MODEL`,
`--stage0-model`, or the default Haiku model.
