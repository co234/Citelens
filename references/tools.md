# Stage 6 Tools

Tool implementations live in `citation_cleaner/resolvers/tools.py`.

## search_crossref

Input:

```json
{"title": "string", "author": "surname or null", "year": 2020, "rows": 5}
```

Returns compact article or conference metadata from Crossref.

## search_openalex

Input:

```json
{"query": "free text", "per_page": 5}
```

Returns broader metadata including books and preprints.

## lookup_citing_context

Input:

```json
{"raw_ref": "raw reference", "citing_paper_id": "paper id"}
```

Returns surrounding sentence context from `citances.jsonl` or a registered
provider.
