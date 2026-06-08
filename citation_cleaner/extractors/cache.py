"""JSONL cache used by Stage 2 extraction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def cache_key(raw: str) -> str:
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        out[entry["key"]] = entry["result"]
    return out


def append_cache(path: Path, records: list[dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({"key": cache_key(record["raw"]), "result": record}, ensure_ascii=False) + "\n")
