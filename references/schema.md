# Schema Reference

All runtime schemas live under `citation_cleaner/schemas/`.

## Reference Records

- `Author`: display surname and optional given names.
- `RawRefRow`: `{citing_paper_id, raw}` for CSV input.
- `CleanedRef`: `{citing_paper_id, raw, cleaned}` after Stage 1.
- `ExtractedReference`: Stage 2 structured record.
- `CanonicalReference`: final one-work record with stable `canonical_id`.

Canonical ids are deterministic SHA1 prefixes over first author surname, year,
and title.

## Citances

`Citance` keeps the core citing-context fields plus extra analysis fields:

```json
{
  "citing_paper_id": "...",
  "raw_ref": "... or null",
  "context": "...",
  "marker": {
    "style": "numbered_bracket | numbered_superscript | paren_author_year | narrative_author_year",
    "raw_marker": "[12]",
    "page": 1,
    "char_span": [120, 124]
  },
  "link_method": "numbered_direct | author_year_unique | llm_disambiguated | unlinked",
  "link_confidence": 0.0
}
```

`raw_ref` is `null` only for unlinked citances.

## Parsed Documents

`ParsedDocument` is the Stage 0 output:

```json
{
  "pdf_path": "...",
  "citing_paper_id": "doi-or-arxiv-or-filehash",
  "header": {"title": "...", "authors": [], "year": 2020, "doi": "...", "arxiv_id": null},
  "references": ["raw ref string"],
  "citances": [],
  "quality": {
    "n_refs_extracted": 0,
    "pct_with_year": 0.0,
    "pct_with_authorlike": 0.0,
    "needs_llm_fallback": true,
    "llm_fallback_invoked": false,
    "notes": []
  }
}
```
