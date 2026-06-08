"""Union-find clustering for Stage 4 candidate retrieval."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        a, b = self.find(i), self.find(j)
        if a != b:
            self.parent[a] = b


def cluster_block(
    records: list[dict],
    embeddings: np.ndarray,
    auto_merge: float = 0.95,
    candidate_high: float = 0.80,
    candidate_low: float = 0.65,
) -> list[dict]:
    n = len(records)
    if n == 0:
        return []

    auto_uf = UnionFind(n)
    candidate_edges: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            score = cosine(embeddings[i], embeddings[j])
            if score >= auto_merge:
                auto_uf.union(i, j)
            elif score >= candidate_low:
                candidate_edges.append((i, j, score))

    auto_groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        auto_groups[auto_uf.find(i)].append(i)

    roots = list(auto_groups.keys())
    root_idx = {root: k for k, root in enumerate(roots)}
    candidate_uf = UnionFind(len(roots))
    comp_max_sim: dict[int, float] = defaultdict(float)
    for i, j, _score in candidate_edges:
        ri = root_idx[auto_uf.find(i)]
        rj = root_idx[auto_uf.find(j)]
        candidate_uf.union(ri, rj)
    for i, j, score in candidate_edges:
        comp = candidate_uf.find(root_idx[auto_uf.find(i)])
        comp_max_sim[comp] = max(comp_max_sim[comp], score)

    components: dict[int, list[int]] = defaultdict(list)
    for root in roots:
        components[candidate_uf.find(root_idx[root])].append(root)

    clusters: list[dict] = []
    for comp, auto_roots in components.items():
        indexes = [i for root in auto_roots for i in auto_groups[root]]
        if len(auto_roots) == 1:
            members = [records[i] for i in indexes]
            if len(members) > 1:
                clusters.append({"tier": "auto", "max_sim": 1.0, "members": members})
            else:
                clusters.append({"tier": "singleton", "max_sim": 0.0, "members": members})
        else:
            max_score = comp_max_sim[comp]
            tier = "candidate" if max_score >= candidate_high else "borderline"
            clusters.append({"tier": tier, "max_sim": max_score, "members": [records[i] for i in indexes]})
    return clusters
