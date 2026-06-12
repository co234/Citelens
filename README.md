# Citelens - Citation Cleaners

> **From a topic, an author, or a paper title — to the references that actually matter, ranked by how often the field cites them, with every citing passage traced back to its source.**

<!-- Replace the placeholder links below with your actual URLs -->
[🌐 Playground](http://8.141.116.119:7867) · [📄 Technical Explainer](./citelens-citation_cleaner_v4_explainer.pdf)

<a href="https://youtu.be/dwhjaUQTbLw">
  <img src="./video.png" width="400">
</a>

---
## What it does

Citelens is an LLM-powered citation intelligence agent. Give it a research topic, an author name, or a paper — it retrieves the most relevant papers in the corpus, reads who they cite, tallies every reference across the set, and returns a frequency-ranked, evidence-backed map of the conversation.

The headline output is a co-citation report where each entry shows not just how many papers cite it, but **exactly where**: the in-context passage, the page number, and the citing paper.

Here's a real example from querying `diffusion models` — the paper *Diffusion Models Beat GANs on Image Synthesis* (2021, cited 2171×) was retrieved and processed, yielding 73 references, 135 citances, and 71 unique canonical works after deduplication:

```
Denoising Diffusion Probabilistic Models (2020) — most-referenced in text, 10 in-text mentions.

Analyzing and Improving the Image Quality of StyleGAN (2019)
  Tero Karras; Samuli Laine et al.
  Cited in 1 source paper(s); appears 6 time(s) in text.

  In 2105.05233v4:
  — p. 1 · marker [5, 28, 51]:  "…infinite high-quality synthetic images [5, 28, 51]
                                  and highly diverse human speech and music…"
  — p. 4 · marker [27, 28, 25]: "…some papers [27, 28, 25] compare against arbitrary
                                  subsets of the training set…"
  — p. 10 · marker [28]:        "StyleGAN2 [28]  3.84"
  …and 3 more citing passages.
```

---

## Three input modes

```bash
# 1. By topic
python scripts/run_discover.py --query "diffusion models" --workdir ./run_dm
python scripts/run_discover.py --query "diffusion models" --workdir ./run_dm --sort recency
python scripts/run_discover.py --query "diffusion models" --workdir ./run_dm --sort citations

# 2. Upload a paper — topic is extracted automatically
python scripts/run_discover.py --upload mypaper.pdf --workdir ./run_mp --sort citations

# 3. By author — returns top-N most-cited papers for that author
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

Nine stages turn a query (or a folder of PDFs, or a raw reference CSV) into a ranked, traceable citation report.

```
Input (query / PDF folder / CSV)
  └─▶ Retrieval      search OpenAlex, keep top N papers
       └─▶ Stage 0   PDF parse — extract text, split refs, scan citances   [PDF mode only]
            └─▶ Stage 1  Pre-clean — regex, drop noise
                 └─▶ Stage 2  Extract — LLM → typed records (batched)
                      └─▶ Stage 3  Block — group by surname × year
                           └─▶ Stage 4  Embed — title embeddings & clustering
                                └─▶ Stage 5  Judge — LLM same/different verdict
                                     └─▶ Stage 6  Resolve — agent + Crossref/OpenAlex  [uncertain only]
                                          └─▶ Stage 7  Co-citation — cross-paper aggregation
                                               └─▶ Enrich — global citation counts (top 30)
                                                    └─▶ Report — ranking table + in-context appendix
```

**CSV mode** starts at Stage 1. **PDF mode** adds Stage 0, which also produces the in-text citances that power the in-context appendix.

---

## Installation

```bash
git clone https://github.com/YOUR_ORG/citelens.git
cd citelens
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.9, plus the packages below.

```
pydantic>=2.5      pymupdf>=1.24      gradio>=4.0
numpy>=1.24        titlecase>=2.4     pandas>=2.0
requests>=2.31     json-repair>=0.30  python-dotenv>=1.0
```

LLM and embedding calls go through **OpenRouter**. Set your key in `.env` or the web UI:

```bash
OPENROUTER_API_KEY=...        # required — chat model + embedding model
CITATION_CLEANER_EMAIL=...    # optional — enables OpenAlex polite pool
```

Check connectivity before a real run:

```bash
python scripts/check_api.py
```

---

## Quick start

```bash
# Dry-run (no API calls) — smoke-test the full pipeline offline
python scripts/run_pipeline.py --pdfs ./papers --workdir ./run_2026 --dry-run
python scripts/run_pipeline.py --refs demo/raw_refs_sample.csv --workdir ./run_refs --dry-run

# Real run
python scripts/run_discover.py --query "conformal prediction" --workdir ./run_cp
```

**Web UI** — three tabs: By topic / Upload paper / By author:

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

The markdown report has two sections. The ranking table leads with `Co-cited` (how many source papers include it) and `In-text hits` (how many times it appears in body text across all source papers) — the two numbers often diverge, and both matter. The in-context appendix then lists every occurrence of each top-10 work: page number, citation marker, and surrounding sentence.

---

## Python API

```python
from pathlib import Path
from citation_cleaner.pipelines.discover_pipeline import run_discovery

# By topic, sorted by recency
result = run_discovery(
    query="diffusion models",
    workdir=Path("./run"),
    n_papers=5,
    sort_by="recency",
)

# By author
result = run_discovery(
    author="Yann LeCun",
    workdir=Path("./run_lecun"),
    n_papers=5,
)
print(result["mode"])    # "author"
print(result["author"])  # {'author_id': ..., 'display_name': ..., ...}

# Access in-context occurrences
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

## Further reading

- [Technical explainer (PDF)](./citation_cleaner_v4_explainer.pdf) — pipeline architecture, design rationale, and stage-by-stage breakdown
- [Schema reference](./references/schema.md)
- [PDF parsing notes](./references/pdf_parsing.md)
- [Prompt templates](./references/prompts.md)
- [Evaluation protocol](./references/eval.md)
