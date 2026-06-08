---
name: citation-cleaner-v2
description: Use whenever the user wants to extract, normalize, deduplicate, or clean bibliographic references, including references read directly from modern text-based PDF files. Trigger this for cleaning citations, reference normalization, bibliography deduplication, inconsistent author names, extracting references from PDFs, parsing citation contexts from academic papers, building a clean dataset of works from a paper corpus, linking in-text citations back to reference entries, searching for an author's top-cited papers, or producing co-citation reports with in-text appearance contexts. Do NOT use for in-text citation intent classification.
---

# Citation Cleaner v2 / v3 / v4

Citation Cleaner turns a corpus of modern text-based PDF files (or a CSV of
raw reference strings) into canonical bibliographic records, a
raw-to-canonical mapping, and review artifacts.

v3 added the discovery pipeline: take a terminology / upload, find N related
papers on OpenAlex, run them all through the pipeline, and aggregate
co-cited references.

v4 (this version) adds **author input mode** (top-N most-cited papers for a
named author), **sort options** for terminology / upload modes
(`relevance | recency | citations`), and an **in-context appearances**
section in the markdown report — listing the actual page numbers, citation
markers, and context sentences where each top co-cited reference appears in
each source paper. See `README_v4.md` for the new entry points.

The first principle is traceable entity resolution. A source PDF contains page
text, reference strings, and in-text citation markers. The skill must keep those
objects visible while it normalizes and clusters them, because the main failure
is not messy formatting; it is silently merging different works or losing the
context that would have disambiguated them.

## Default Entry Points

Run from the `citation-cleaner-v2/` directory.

```bash
python scripts/run_pipeline.py --pdfs ./papers --workdir ./run_2026 --dry-run
python scripts/run_pipeline.py --refs demo/raw_refs_sample.csv --workdir ./run_refs --dry-run
python scripts/parse_pdfs.py --pdfs ./papers --workdir ./stage0_run --dry-run
```

Use `--dry-run` for offline smoke tests. Omit it only after
`scripts/check_api.py` passes with the required API keys.

## Pipeline

```text
PDF files
  -> Stage 0 parse PDF text, references, and citation contexts
  -> Stage 1 preclean raw references
  -> Stage 2 structured extraction
  -> Stage 3 surname/year blocking
  -> Stage 4 title embedding candidate clusters
  -> Stage 5 LLM judge
  -> Stage 6 tool-using resolver for uncertain clusters
  -> canonical references, raw mapping, review queue
```

For CSV input, the pipeline starts at Stage 1. Stages 1 through 6 use the same
implementations in both modes.

## Stage 0

Stage 0 uses `citation_cleaner.parsers.heuristic.HeuristicParser`.

It is designed for modern publisher PDFs with embedded text. It does not claim
OCR support for scanned PDFs. The parser:

1. extracts page text with PyMuPDF,
2. locates a References or Bibliography heading,
3. splits the section into raw reference strings,
4. scores extraction quality,
5. optionally invokes an LLM fallback when extraction quality is poor,
6. extracts title, DOI, arXiv id, and year from the first page,
7. scans body text for numbered and author-year citation markers,
8. links markers to references by index or `(first_author_surname, year)`,
9. writes `ParsedDocument` records.

The fallback is disabled in dry-run mode. In real mode, the Stage 0 fallback
model is `CITATION_CLEANER_STAGE0_MODEL` if set, otherwise
`claude-haiku-4-5-20251001`. `--stage0-model` overrides both.

## Outputs

Every full run writes these files in `--workdir`:

- `parsed_documents.jsonl` for PDF mode
- `raw_refs.csv` for PDF mode, flattened from parsed documents
- `citances.jsonl` for PDF mode and Stage 6 context lookup
- `precleaned.csv`
- `extracted.jsonl`
- `candidates.jsonl`
- `resolved.jsonl`
- `canonical_refs.csv`
- `raw_to_canonical.csv`
- `review_queue.jsonl`

Review queue entries are not failures to hide. They are the cases where the
pipeline could not produce a canonical reference with enough confidence.

## Real API Run

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export CITATION_CLEANER_EMAIL=you@example.com
python scripts/check_api.py
python scripts/run_pipeline.py --pdfs ./papers --workdir ./run_real
```

Stage 2 defaults to Haiku. Stages 5 and 6 default to Sonnet. Stage 4 uses
OpenAI embeddings. Crossref and OpenAlex are used only by the Stage 6 agent on
uncertain clusters.

## Resume

By default, stages load existing artifacts from `--workdir`.

```bash
python scripts/run_pipeline.py --pdfs ./papers --workdir ./run_real
python scripts/run_pipeline.py --pdfs ./papers --workdir ./run_real --resume-from extract
python scripts/run_pipeline.py --pdfs ./papers --workdir ./run_real --no-resume
```

`--resume-from` loads earlier artifacts and recomputes from the named stage
forward.

## Evaluation

Stage 1 through Stage 6:

```bash
python scripts/run_pipeline.py --refs eval/eval_set.csv --workdir run_eval --dry-run
python scripts/eval_score.py --resolved run_eval/resolved.jsonl --eval-set eval/eval_set.csv
```

Stage 0:

```bash
python scripts/eval_stage0.py --manifest eval/pdf_eval_set/manifest.csv --dry-run
```

The Stage 0 eval set should include representative PDF layouts and corrected
`.refs.csv` / `.citances.jsonl` files. Use `references/eval.md` for the target
metrics.

## Reference Files

- `references/schema.md` for Pydantic models and artifact shapes
- `references/pdf_parsing.md` for Stage 0 parser boundaries
- `references/prompts.md` for LLM prompt contracts
- `references/tools.md` for Stage 6 tool schemas
- `references/eval.md` for metrics and acceptance thresholds

## Coverage Boundary

This skill is self-contained. Raw reference tables use Stage 1 through Stage 6:
precleaning, structured extraction, blocking, embedding candidates, LLM judging,
and tool-assisted resolution. PDF input adds Stage 0, then feeds the extracted
references and citation contexts into the same Stage 1 through Stage 6 path.
