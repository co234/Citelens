# Citation Cleaner Test Cases and Results

Last verified: 2026-05-20

This directory contains the deterministic test suite for the skill. The tests
are designed to run without API keys. They cover the schema helpers, Stage 0 PDF
parsing path, citance linking, clustering, and dry-run pipeline orchestration.

## Test Cases

| File | Coverage |
|---|---|
| `test_smoke.py` | Canonical ID stability, surname normalization, Anthropic cache routing, preclean rules, and the Stage 4 union-find clustering regression. |
| `test_citance_linker.py` | Numbered citation linking, bracket-range expansion, author-year unique linking, and author-year indexing from raw references. |
| `test_parsers.py` | References-section location and splitting, quality scoring, and a synthetic text-PDF parse through `HeuristicParser`. |
| `test_pipelines.py` | Stage 1 through Stage 6 dry-run orchestration from a raw references CSV. |

## Verification Commands

Run from the `citation-cleaner-v2/` directory.

```powershell
python -B -m pytest -p no:cacheprovider tests -q
```

Result:

```text
13 passed in 0.70s
```

CSV dry-run smoke:

```powershell
python -B scripts\run_pipeline.py --refs demo\raw_refs_sample.csv --workdir tests\_tmp_csv_smoke --dry-run --no-resume
```

Result:

```text
Done. Resolved 4 canonical references.
Outputs in tests\_tmp_csv_smoke
  - resolved.jsonl
  - canonical_refs.csv
  - raw_to_canonical.csv
  - review_queue.jsonl
```

PDF Stage 0 smoke used a synthetic text PDF containing two reference entries
and two in-text citation markers.

```powershell
python -B scripts\parse_pdfs.py --pdfs tests\_tmp_pdf_input\sample.pdf --workdir tests\_tmp_pdf_parse --dry-run --no-resume
```

Result:

```json
{
  "parsed_documents": 1,
  "workdir": "tests\\_tmp_pdf_parse"
}
```

Observed Stage 0 output:

```csv
citing_paper_id,raw
c9eed62bd153,"[1] Smith, J. (2020). Deep learning for citations."
c9eed62bd153,"[2] Jones, K. (2021). Reference parsing."
```

Observed citance links:

```text
[2] -> [2] Jones, K. (2021). Reference parsing. via numbered_direct
Smith (2020) -> [1] Smith, J. (2020). Deep learning for citations. via author_year_unique
```

PDF end-to-end dry-run:

```powershell
python -B scripts\run_pipeline.py --pdfs tests\_tmp_pdf_input\sample.pdf --workdir tests\_tmp_pdf_pipeline --dry-run --no-resume
```

Result:

```text
Done. Resolved 2 canonical references.
Outputs in tests\_tmp_pdf_pipeline
  - resolved.jsonl
  - canonical_refs.csv
  - raw_to_canonical.csv
  - review_queue.jsonl
```

The temporary `_tmp_*` directories used for this verification are intentionally
not included in the packaged skill. The commands above recreate them.

## Notes

- `python -B` prevents Python from writing `__pycache__` files into the
  packaged skill.
- `-p no:cacheprovider` prevents pytest from writing `.pytest_cache`.
- All tests and smoke runs use `--dry-run`, so no Anthropic, OpenAI, Crossref,
  or OpenAlex calls are made.
- The PDF path tested here is the intended Stage 0 boundary: modern PDFs with
  embedded selectable text. Scanned PDFs still require an OCR or vision backend.
