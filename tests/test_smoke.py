from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from citation_cleaner.embedding.clustering import cluster_block
from citation_cleaner.llm.anthropic import cache_last_tool, supports_anthropic_cache, system_block
from citation_cleaner.preclean.rules import preclean
from citation_cleaner.schemas.reference import Author, make_canonical_id, normalize_surname_for_blocking


def test_make_canonical_id_deterministic():
    a = make_canonical_id(authors=[{"surname": "Smith"}], year=2020, title="Foo Bar")
    b = make_canonical_id(authors=[Author(surname="Smith")], year=2020, title="Foo Bar")
    assert a == b
    assert len(a) == 12


def test_normalize_surname():
    assert normalize_surname_for_blocking("van der Vaart") == "vaart"
    assert normalize_surname_for_blocking("de Boor") == "boor"
    assert normalize_surname_for_blocking("Muller") == "muller"


def test_cache_routing():
    assert isinstance(system_block("hi", "claude-haiku-4-5-20251001"), list)
    assert system_block("hi", "gpt-4o") == "hi"
    assert supports_anthropic_cache("claude-sonnet-4-6")
    assert not supports_anthropic_cache("gpt-4o")
    tools = [{"name": "a"}, {"name": "b"}]
    out = cache_last_tool(tools, "claude-sonnet-4-6")
    assert "cache_control" in out[-1]
    assert "cache_control" not in tools[-1]


def test_preclean_rules():
    assert preclean("23") is None
    assert preclean("Smith & Jones (1990).") == "Smith and Jones (1990)."
    assert preclean("Foo (1990), p. 154").endswith("(1990)")


def test_cluster_block_emits_each_record_once():
    recs = [{"raw": f"r{i}", "title": f"t{i}"} for i in range(4)]
    emb = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.85, 0.5267, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    clusters = cluster_block(recs, emb)
    seen = [member["raw"] for cluster in clusters for member in cluster["members"]]
    assert len(seen) == 4
    assert len(set(seen)) == 4
