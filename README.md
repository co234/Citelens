# Citation Cleaner

> **From a topic, an author, or a paper title — to the references that actually matter, ranked by how often the field cites them, with every citing passage traced back to its source.**

<!-- Replace the placeholder links below with your actual URLs -->
[🌐 Product Page](#) · [📄 Technical Explainer](./citation_cleaner_v4_explainer.pdf) · [🚀 Web Demo](#)

---

## What it does

Citation Cleaner is an LLM-powered citation intelligence agent. Give it a research topic, an author name, or a paper — it retrieves the most relevant papers in the corpus, reads who they cite, tallies every reference across the set, and returns a frequency-ranked, evidence-backed map of the conversation.

The headline output is a co-citation report where each entry shows not just how many papers cite it, but **exactly where**: the in-context passage, the page number, and the citing paper.

```
Attention Is All You Need (2017) — cited 9× across the top-10 set.

  • in Zhang et al., 2023, p. 4:  "…building on the estimator proposed by [A]…"
  • in Okoye, 2022, p. 11:        "…we adopt the same blocking strategy as [A]…"
  • …and 7 more citing passages, each linked to its page and source.
```

---

## Three input modes

```bash
# 1. Terminology
python scripts/run_discover.py --query "diffusion models" --workdir ./run_dm
python scripts/run_discover.py --query "diffusion models" --workdir ./run_dm --sort recency
python scripts/run_discover.py --query "diffusion models" --workdir ./run_dm --sort citations

# 2. Upload a paper — topic is extracted automatically
python scripts/run_discover.py --upload mypaper.pdf --workdir ./run_mp --sort citations

# 3. Author — returns top-N most-cited papers for that author
python scripts/run_discover.py --author "Yann LeCun" --workdir ./run_lecun --n 5

# Use an OpenAlex author ID for disambiguation
python scripts/run_discover.py --author "A2208157607" --workdir ./run_lecun --n 5
```

`--sort` accepts `relevance` (default), `recency`, or `citations`. In author mode, sort is fixed to `citations`.

---

## Why this is an agent, not a script

Ranking citations sounds like counting. It isn't. The same paper can appear across a corpus as reordered author names, initials vs. full names, drifting title casing, typos, and edition variants. Count naively and you split one work into five. The mirror-image failure: two genuinely different books both titled *Generalized Additive Models* — merge them and you fabricate a highly-cited work that never existed.

These are judgment problems, not formatting problems. The architecture is deliberately frugal about where it spends LLM calls:

| Layer | What does the work |
|---|---|
| Format noise, field reordering | Deterministic regex / rules |
| Abbreviation, casing, typos | Title embeddings + clustering |
| Same / different judgment | LLM judge (only on uncertain clusters) |
| Unresolvable clusters | Agent loop → Crossref + OpenAlex lookups |

The LLM is the scalpel, not the hammer.

---

## Pipeline

Eight stages turn a query (or a folder of PDFs, or a raw reference CSV) into a ranked, traceable citation report.

```
Input (query / PDF folder / CSV)
  └─▶ Retrieval          rank corpus, keep top N
       └─▶ Stage 0       PDF parse — split refs, scan citances        [PDF mode only]
            └─▶ Stage 1  Pre-clean — regex, drop noise
                 └─▶ Stage 2  Extract — LLM → typed records
                      └─▶ Stage 3  Blocking — group by surname × year
                           └─▶ Stage 4  Similarity — title embeddings
                                └─▶ Stage 5  LLM judge — same / different / unsure
                                     └─▶ Stage 6  Agent loop — Crossref + OpenAlex  [uncertain only]
                                          └─▶ Frequency ranking → Report
```

**CSV mode** starts at Stage 1. **PDF mode** adds Stage 0, which also produces the in-text citing passages that power the headline report.

---

## Installation

```bash
git clone https://github.com/YOUR_ORG/citation-cleaner.git
cd citation-cleaner
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.9, plus the packages below.

```
pydantic>=2.5      anthropic>=0.40    pymupdf>=1.24
numpy>=1.24        openai>=1.50       gradio>=4.0
requests>=2.31     titlecase>=2.4     pandas>=2.0
```

API keys (set in `.env` or environment):

```bash
ANTHROPIC_API_KEY=...   # required for Stage 2 / 5 LLM calls
OPENAI_API_KEY=...      # optional — for embedding provider
```

Check connectivity before a real run:

```bash
python scripts/check_api.py
```

---

## Quick start

```bash
# Dry-run (no API calls) — smoke test the pipeline
python scripts/run_pipeline.py --pdfs ./papers --workdir ./run_2026 --dry-run
python scripts/run_pipeline.py --refs demo/raw_refs_sample.csv --workdir ./run_refs --dry-run

# Real run
python scripts/run_discover.py --query "conformal prediction" --workdir ./run_cp
```

**Web UI** (three tabs: Terminology / Upload paper / Author):

```bash
python scripts/web_app.py
```

Resume from a named stage after a partial run:

```bash
python scripts/run_pipeline.py --pdfs ./papers --workdir ./run_2026 --resume-from stage4
```

---

## Output

Each run writes to `--workdir`:

| File | Contents |
|---|---|
| `cocited_refs.md` | Ranked co-citation report with in-context appearances |
| `cocited_refs.csv` | Same data in tabular form (`in_text_hits` column included) |
| `raw_to_canonical.csv` | Every raw reference string mapped to its canonical record |
| `citances.jsonl` | Page-level in-text citation contexts (PDF mode) |
| `resolved.jsonl` | Full cluster resolution log |

The markdown report has two sections. The first is the co-citation ranking table (columns: `Rank`, `Co-cited`, `Global cites`, `In-text hits`, `Title`, `Authors`, `Year`, `DOI`). The second is the in-context appearances appendix — for each top-10 co-cited work, every occurrence in every source paper, with page number, citation marker, and context sentence.

---

## Python API

```python
from pathlib import Path
from citation_cleaner.pipelines.discover_pipeline import run_discovery

# Terminology, sorted by recency
result = run_discovery(
    query="diffusion models",
    workdir=Path("./run"),
    n_papers=5,
    sort_by="recency",
)

# Author mode
result = run_discovery(
    author="Yann LeCun",
    workdir=Path("./run_lecun"),
    n_papers=5,
)
print(result["mode"])    # "author"
print(result["author"])  # {'author_id': ..., 'display_name': ..., ...}

# In-context occurrences
for rec in result["records"][:3]:
    print(rec.title, "—", len(rec.occurrences), "in-text mentions")
    for occ in rec.occurrences:
        print(f"  in {occ.citing_paper_id} p.{occ.page}: {occ.context[:80]}")
```

---

## Repository layout

```
citation_cleaner/
├── schemas/        reference.py, citance.py, document.py
├── parsers/        heuristic.py, reference_section.py, citance_linker.py,
│                   llm_fallback.py, quality.py
├── preclean/       rules.py
├── extractors/     llm.py, dry_run.py, cache.py
├── blocking/       surname.py
├── embedding/      providers.py, clustering.py
├── resolvers/      judge.py, agent.py, tools.py
├── pipelines/      stages.py, pdf_to_canonical.py, refs_to_canonical.py,
│                   discover_pipeline.py
└── llm/            client.py, anthropic.py, openrouter.py, json_parse.py

scripts/            run_pipeline.py, run_discover.py, parse_pdfs.py,
                    web_app.py, check_api.py, eval_stage0.py, eval_score.py
demo/               raw_refs_sample.csv, citances_sample.jsonl, sample_pdfs/
references/         schema.md, pdf_parsing.md, prompts.md, tools.md, eval.md
tests/              test_smoke.py, test_parsers.py, test_citance_linker.py,
                    test_pipelines.py, test_v4_features.py
SKILL.md            Agent-facing contract (MCP skill descriptor)
```

---

## Evaluation

Stages 1–6 are scored with pair-level F1 and cluster purity against a hand-labeled reference set. Stage 0 earns its own PDF-level metrics: reference recall, reference precision, citance recall, citance-link accuracy, fallback rate, and whether the LLM fallback actually improved extraction quality.

```bash
python scripts/eval_score.py --workdir ./run_2026 --eval-set eval/eval_set.csv
python scripts/eval_stage0.py --pdfs eval/pdf_eval_set --manifest eval/pdf_eval_set/manifest.csv
```

The governing question: how much does PDF parsing degrade downstream canonicalization — and therefore citation ranking — compared with hand-extracted reference strings from the same papers?

---

## Tests

```bash
python -m pytest tests/ -q
# 25 passed
```

Smoke tests for v4 features:

```bash
python scripts/run_discover.py --author "Yann LeCun" --workdir /tmp/smoke_au --dry-run --no-enrich
python scripts/run_discover.py --query "transformers" --workdir /tmp/smoke_sort --dry-run --no-enrich --sort recency
```

---

## Changelog

| Version | What's new |
|---|---|
| **v4** | Author input mode · `--sort` option · In-context appearances appendix in report |
| **v3** | Discovery pipeline (OpenAlex retrieval) · Web UI · Co-citation aggregation |
| **v2** | Importable typed package · PDF ingestion (Stage 0) · Pydantic schema layer · Resume behavior · Full test suite |
| **v1** | Script-oriented prototype · Raw reference string input only |

---

## Further reading

- [Technical explainer (PDF)](./citation_cleaner_v4_explainer.pdf) — pipeline architecture, design rationale, and stage-by-stage breakdown
- [Schema reference](./references/schema.md)
- [PDF parsing notes](./references/pdf_parsing.md)
- [Prompt templates](./references/prompts.md)
- [Evaluation protocol](./references/eval.md)
