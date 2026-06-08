# Evaluation

## Stage 1 Through Stage 6

Use pair-level F1 and cluster purity against `eval/eval_set.csv`.

```bash
python scripts/run_pipeline.py --refs eval/eval_set.csv --workdir run_eval --dry-run
python scripts/eval_score.py --resolved run_eval/resolved.jsonl --eval-set eval/eval_set.csv
```

## Stage 0

Use `eval/pdf_eval_set/manifest.csv` plus one corrected `.refs.csv` file per
PDF.

```bash
python scripts/eval_stage0.py --manifest eval/pdf_eval_set/manifest.csv --dry-run
```

Acceptance targets:

| Metric | Usable | Production |
|---|---:|---:|
| ref_recall | 0.90 | 0.95 |
| ref_precision | 0.92 | 0.97 |
| citance_link_acc | 0.85 | 0.92 |
| fallback_rate | 0.20 | 0.10 |

Track changes in a dated eval log. Do not tune parser heuristics without
rerunning the representative set.
