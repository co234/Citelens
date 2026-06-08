"""Typed stage abstraction and concrete v2 pipeline stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
import csv
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from citation_cleaner.pipelines.pipeline_log import PipelineLogger

from citation_cleaner.blocking.surname import group_by_block
from citation_cleaner.embedding.clustering import cluster_block
from citation_cleaner.embedding.providers import (
    DEFAULT_OPENROUTER_EMBED_MODEL,
    OPENROUTER_EMBED_BASE_URL,
    embed_titles_dryrun,
    embed_titles_openai,
)
from citation_cleaner.extractors.cache import append_cache, cache_key, load_cache
from citation_cleaner.extractors.dry_run import extract_batch_dryrun
from citation_cleaner.extractors.llm import DEFAULT_MODEL as DEFAULT_EXTRACT_MODEL
from citation_cleaner.extractors.llm import extract_batch
from citation_cleaner.llm.client import make_llm_client
from citation_cleaner.parsers.heuristic import HeuristicParser
from citation_cleaner.parsers.llm_fallback import LLMFallback
from citation_cleaner.pipelines.pipeline_log import pipeline_logger
from citation_cleaner.preclean.rules import preclean
from citation_cleaner.resolvers.agent import DEFAULT_AGENT_MODEL, run_agent, run_agent_dryrun
from citation_cleaner.resolvers.judge import DEFAULT_JUDGE_MODEL, finalize_canonical, judge_cluster, judge_cluster_dryrun
from citation_cleaner.resolvers.tools import load_citances_jsonl
from citation_cleaner.schemas.citance import Citance
from citation_cleaner.schemas.document import ParsedDocument
from citation_cleaner.schemas.reference import CleanedRef, ExtractedReference, RawRefRow

I = TypeVar("I")
O = TypeVar("O")


def _pool_workers(ctx: StageContext, *, cap: int = 8) -> int:
    """Thread-pool size for parallel work inside a stage."""
    if ctx.config.get("workers") is not None:
        return max(1, int(ctx.config["workers"]))
    return max(1, min(cap, os.cpu_count() or 1))


@dataclass
class StageContext:
    workdir: Path
    dry_run: bool = False
    resume: bool = True
    config: dict = field(default_factory=dict)
    logger: "PipelineLogger | None" = None

    def __post_init__(self) -> None:
        self.workdir = Path(self.workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)


class Stage(Generic[I, O], ABC):
    name: str
    artifact: str

    @abstractmethod
    def run(self, inp: I, ctx: StageContext) -> O:
        raise NotImplementedError

    @abstractmethod
    def load_artifact(self, ctx: StageContext) -> O | None:
        raise NotImplementedError

    @abstractmethod
    def save_artifact(self, out: O, ctx: StageContext) -> None:
        raise NotImplementedError

    def run_or_resume(self, inp: I, ctx: StageContext) -> O:
        force_from = ctx.config.get("_force_from")
        if force_from and not ctx.config.get("_force_started"):
            if self.name == force_from:
                ctx.config["_force_started"] = True
            elif ctx.resume:
                cached = self.load_artifact(ctx)
                if cached is not None:
                    return cached
        if ctx.resume and not ctx.config.get("_force_started"):
            cached = self.load_artifact(ctx)
            if cached is not None:
                return cached
        out = self.run(inp, ctx)
        self.save_artifact(out, ctx)
        return out


def _jsonl_path(ctx: StageContext, artifact: str) -> Path:
    return ctx.workdir / artifact


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class Stage0_Parse(Stage[list[Path], list[ParsedDocument]]):
    name = "parse"
    artifact = "parsed_documents.jsonl"

    def run(self, inp: list[Path], ctx: StageContext) -> list[ParsedDocument]:
        log = pipeline_logger(ctx)
        llm_fallback = None
        if not ctx.dry_run and not ctx.config.get("disable_stage0_fallback", False):
            llm_fallback = LLMFallback(
                client=make_llm_client(ctx.config),
                model=ctx.config.get("stage0_model"),
            )
        parser = ctx.config.get("parser") or HeuristicParser(llm_fallback=llm_fallback)
        workers = _pool_workers(ctx, cap=8)
        pdf_paths = [Path(path) for path in inp]
        total = len(pdf_paths)
        done_count = [0]
        done_lock = threading.Lock()

        def _parse_one(path: Path) -> ParsedDocument:
            log.info("s0", f"parsing {path.name}…")
            doc = parser.parse(path)
            with done_lock:
                done_count[0] += 1
                n = done_count[0]
            log.progress(
                "s0",
                n,
                total,
                f"done {path.name} — {len(doc.references)} reference(s), "
                f"{len(doc.citances)} citance(s)",
            )
            return doc

        log.begin("s0", f"parsing {total} PDF(s)…")
        if workers <= 1 or total <= 1:
            docs = [_parse_one(path) for path in pdf_paths]
        else:
            log.info("s0", f"using {workers} parallel workers")
            with ThreadPoolExecutor(max_workers=workers) as pool:
                docs = list(pool.map(_parse_one, pdf_paths))
        nrefs = sum(len(doc.references) for doc in docs)
        log.done("s0", f"{len(docs)} document(s), {nrefs} raw reference(s)")
        return docs

    def load_artifact(self, ctx: StageContext) -> list[ParsedDocument] | None:
        path = _jsonl_path(ctx, self.artifact)
        if not path.exists():
            return None
        return [ParsedDocument.model_validate(row) for row in _read_jsonl(path)]

    def save_artifact(self, out: list[ParsedDocument], ctx: StageContext) -> None:
        _write_jsonl(_jsonl_path(ctx, self.artifact), [doc.model_dump() for doc in out])
        self._write_flat_refs(out, ctx.workdir / "raw_refs.csv")
        self._write_citances(out, ctx.workdir / "citances.jsonl")

    @staticmethod
    def _write_flat_refs(docs: list[ParsedDocument], path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["citing_paper_id", "raw"])
            writer.writeheader()
            for doc in docs:
                for raw in doc.references:
                    writer.writerow({"citing_paper_id": doc.citing_paper_id, "raw": raw})

    @staticmethod
    def _write_citances(docs: list[ParsedDocument], path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for doc in docs:
                for citance in doc.citances:
                    handle.write(json.dumps(citance.model_dump(), ensure_ascii=False) + "\n")


class Stage1_Preclean(Stage[list[ParsedDocument] | list[RawRefRow], list[CleanedRef]]):
    name = "preclean"
    artifact = "precleaned.csv"

    def run(self, inp: list[ParsedDocument] | list[RawRefRow], ctx: StageContext) -> list[CleanedRef]:
        log = pipeline_logger(ctx)
        log.begin("s1", "pre-cleaning references…")
        rows: list[RawRefRow] = []
        if inp and isinstance(inp[0], ParsedDocument):
            for doc in inp:  # type: ignore[union-attr]
                for raw in doc.references:
                    rows.append(RawRefRow(citing_paper_id=doc.citing_paper_id, raw=raw))
        else:
            rows = list(inp)  # type: ignore[arg-type]

        cleaned_rows: list[CleanedRef] = []
        total = len(rows)
        for i, row in enumerate(rows, start=1):
            cleaned = preclean(row.raw)
            if cleaned:
                cleaned_rows.append(CleanedRef(citing_paper_id=row.citing_paper_id, raw=row.raw, cleaned=cleaned))
            if total and (i == total or i % max(1, total // 10) == 0):
                log.progress("s1", i, total, f"pre-cleaned {i} of {total} raw reference(s)", append_line=False)
        log.done("s1", f"{len(cleaned_rows)} cleaned row(s)")
        return cleaned_rows

    def load_artifact(self, ctx: StageContext) -> list[CleanedRef] | None:
        path = ctx.workdir / self.artifact
        if not path.exists():
            return None
        with path.open(encoding="utf-8", newline="") as handle:
            return [CleanedRef.model_validate(row) for row in csv.DictReader(handle)]

    def save_artifact(self, out: list[CleanedRef], ctx: StageContext) -> None:
        with (ctx.workdir / self.artifact).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["citing_paper_id", "raw", "cleaned"])
            writer.writeheader()
            for row in out:
                writer.writerow(row.model_dump())


class Stage2_Extract(Stage[list[CleanedRef], list[ExtractedReference]]):
    name = "extract"
    artifact = "extracted.jsonl"

    def run(self, inp: list[CleanedRef], ctx: StageContext) -> list[ExtractedReference]:
        log = pipeline_logger(ctx)
        log.begin("s2", "extracting structured fields (LLM)…")
        cache_path = ctx.workdir / ".extract_cache.jsonl"
        cache = load_cache(cache_path)
        batch_size = int(ctx.config.get("batch_size") or 30)
        results: list[ExtractedReference] = []
        quarantine: list[dict] = []
        pending: list[CleanedRef] = []

        if ctx.dry_run:
            client = None
        else:
            client = make_llm_client(ctx.config)
        model = ctx.config.get("extract_model") or DEFAULT_EXTRACT_MODEL

        for row in inp:
            key = cache_key(row.cleaned)
            if key in cache:
                record = ExtractedReference.model_validate(cache[key])
                record.citing_paper_id = row.citing_paper_id
                results.append(record)
                continue
            pending.append(row)

        if not pending:
            log.done("s2", f"{len(results)} extracted reference(s) (all cached)")
            return results

        batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
        total_batches = len(batches)
        batch_done = [0]
        batch_lock = threading.Lock()

        def run_batch(batch_idx: int, batch: list[CleanedRef]) -> tuple[list[ExtractedReference], list[dict]]:
            log.info("s2", f"extracting batch {batch_idx}/{total_batches} ({len(batch)} reference(s))…")
            if ctx.dry_run:
                valid, bad = extract_batch_dryrun(
                    [row.cleaned for row in batch],
                    [row.citing_paper_id for row in batch],
                )
            else:
                payload = [{"raw": row.cleaned, "citing_paper_id": row.citing_paper_id} for row in batch]
                valid, bad = extract_batch(payload, client, model)
            with batch_lock:
                batch_done[0] += 1
                n = batch_done[0]
            log.progress(
                "s2",
                n,
                total_batches,
                f"batch {batch_idx} done — {len(valid)} ok, {len(bad)} quarantined",
            )
            return valid, bad

        workers = _pool_workers(ctx, cap=4)
        if workers <= 1 or total_batches <= 1:
            for idx, batch in enumerate(batches, start=1):
                valid, bad = run_batch(idx, batch)
                results.extend(valid)
                quarantine.extend(bad)
                append_cache(cache_path, [record.model_dump() for record in valid])
        else:
            log.info("s2", f"running {total_batches} batch(es) with {workers} workers")
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(run_batch, idx, batch)
                    for idx, batch in enumerate(batches, start=1)
                ]
                for fut in as_completed(futures):
                    valid, bad = fut.result()
                    results.extend(valid)
                    quarantine.extend(bad)
                    append_cache(cache_path, [record.model_dump() for record in valid])

        if quarantine:
            _write_jsonl(ctx.workdir / "quarantine.jsonl", quarantine)
        log.done("s2", f"{len(results)} extracted reference(s)")
        return results

    def load_artifact(self, ctx: StageContext) -> list[ExtractedReference] | None:
        path = ctx.workdir / self.artifact
        if not path.exists():
            return None
        return [ExtractedReference.model_validate(row) for row in _read_jsonl(path)]

    def save_artifact(self, out: list[ExtractedReference], ctx: StageContext) -> None:
        _write_jsonl(ctx.workdir / self.artifact, [row.model_dump() for row in out])


class Stage34_BlockEmbed(Stage[list[ExtractedReference], list[dict]]):
    name = "block_and_embed"
    artifact = "candidates.jsonl"

    def run(self, inp: list[ExtractedReference], ctx: StageContext) -> list[dict]:
        log = pipeline_logger(ctx)
        log.begin("s3", "blocking by surname/year…")
        blocks = group_by_block(inp)
        log.done("s3", f"{len(blocks)} block(s)")

        log.begin("s4", "embedding & clustering titles…")
        clusters: list[dict] = []
        if ctx.dry_run or ctx.config.get("local_embeddings"):
            # Deterministic local embedder: offline / no embedding key available.
            embed = embed_titles_dryrun
        elif ctx.config.get("embed_provider") == "openrouter":
            # OpenRouter's OpenAI-compatible /embeddings endpoint: one key for
            # both the chat stages and the title embeddings.
            api_key = ctx.config.get("openrouter_api_key")
            model = ctx.config.get("embed_model") or DEFAULT_OPENROUTER_EMBED_MODEL

            def embed(titles: list[str]):
                return embed_titles_openai(
                    titles,
                    model=model,
                    base_url=OPENROUTER_EMBED_BASE_URL,
                    api_key=api_key,
                )
        else:
            model = ctx.config.get("embed_model")

            def embed(titles: list[str]):
                return embed_titles_openai(titles, model=model or "text-embedding-3-small")

        auto_merge = float(ctx.config.get("auto_merge", 0.95))
        candidate_high = float(ctx.config.get("candidate_high", 0.80))
        candidate_low = float(ctx.config.get("candidate_low", 0.65))

        block_items = list(blocks.values())
        total_blocks = len(block_items)
        block_done = [0]
        block_lock = threading.Lock()
        workers = _pool_workers(ctx, cap=4)

        def cluster_one_block(records: list) -> list[dict]:
            record_dicts = [record.model_dump() for record in records]
            titles = [record.title or "" for record in records]
            if not any(titles):
                out = [{"tier": "singleton", "max_sim": 0.0, "members": [record]} for record in record_dicts]
            else:
                embeddings = embed(titles)
                out = cluster_block(
                    record_dicts,
                    embeddings,
                    auto_merge=auto_merge,
                    candidate_high=candidate_high,
                    candidate_low=candidate_low,
                )
            with block_lock:
                block_done[0] += 1
                n = block_done[0]
            log.progress("s4", n, total_blocks, f"clustered block {n}/{total_blocks}")
            return out

        if workers <= 1 or total_blocks <= 1:
            for records in block_items:
                clusters.extend(cluster_one_block(records))
        else:
            log.info("s4", f"clustering {total_blocks} block(s) with {workers} workers")
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for block_clusters in pool.map(cluster_one_block, block_items):
                    clusters.extend(block_clusters)
        log.done("s4", f"{len(clusters)} cluster(s)")
        return clusters

    def load_artifact(self, ctx: StageContext) -> list[dict] | None:
        path = ctx.workdir / self.artifact
        return _read_jsonl(path) if path.exists() else None

    def save_artifact(self, out: list[dict], ctx: StageContext) -> None:
        _write_jsonl(ctx.workdir / self.artifact, out)


def _apply_cluster_verdict(
    cluster: dict,
    verdict: dict,
    resolved: list[dict],
    review: list[dict],
) -> None:
    members = cluster["members"]
    raw_variants = [member["raw"] for member in members]
    if verdict["verdict"] == "same" and verdict.get("canonical"):
        resolved.append(
            finalize_canonical(
                verdict["canonical"],
                raw_variants,
                confidence=float(verdict.get("confidence", 0.7)),
            )
        )
    elif verdict["verdict"] == "different":
        for member in members:
            canonical = _canonical_fields(member)
            if canonical.get("title"):
                resolved.append(
                    finalize_canonical(
                        canonical,
                        [member["raw"]],
                        confidence=float(verdict.get("confidence", 0.7)),
                    )
                )
            else:
                review.append({"cluster": {"members": [member]}, "reason": "split member has no title"})
    else:
        review.append({"cluster": cluster, "verdict": verdict})


def _resolve_llm_cluster(cluster: dict, judge, agent) -> tuple[list[dict], list[dict]]:
    """Judge (+ optional agent) one cluster; returns (resolved, review) lists."""
    resolved: list[dict] = []
    review: list[dict] = []
    verdict = judge(cluster)
    if verdict["verdict"] == "uncertain" or verdict.get("needs_external_lookup"):
        verdict = agent(cluster)
    _apply_cluster_verdict(cluster, verdict, resolved, review)
    return resolved, review


class Stage56_JudgeResolve(Stage[tuple[list[dict], list[Citance]], list[dict]]):
    name = "judge_and_resolve"
    artifact = "resolved.jsonl"

    def run(self, inp: tuple[list[dict], list[Citance]], ctx: StageContext) -> list[dict]:
        log = pipeline_logger(ctx)
        clusters, citances = inp
        if citances:
            self._register_citances(citances, ctx.workdir / "citances.jsonl")
        if ctx.config.get("citances_path"):
            load_citances_jsonl(ctx.config["citances_path"])

        if ctx.dry_run:
            judge = judge_cluster_dryrun
            agent = run_agent_dryrun
        else:
            client = make_llm_client(ctx.config)
            judge_model = ctx.config.get("judge_model") or DEFAULT_JUDGE_MODEL
            agent_model = ctx.config.get("agent_model") or DEFAULT_AGENT_MODEL

            def judge(cluster: dict) -> dict:
                return judge_cluster(cluster, client, judge_model)

            def agent(cluster: dict) -> dict:
                return run_agent(cluster, client, agent_model)

        resolved: list[dict] = []
        review: list[dict] = []
        llm_clusters: list[dict] = []

        for cluster in clusters:
            tier = cluster["tier"]
            members = cluster["members"]
            raw_variants = [member["raw"] for member in members]

            if tier == "auto":
                canonical = _canonical_fields(members[0])
                if canonical.get("title"):
                    resolved.append(finalize_canonical(canonical, raw_variants, confidence=0.95))
                else:
                    review.append({"cluster": cluster, "reason": "auto cluster has no title"})
                continue

            if tier == "singleton":
                member = members[0]
                canonical = _canonical_fields(member)
                if canonical.get("title"):
                    resolved.append(finalize_canonical(canonical, [member["raw"]], confidence=1.0))
                else:
                    review.append({"cluster": cluster, "reason": "singleton with no title"})
                continue

            llm_clusters.append(cluster)

        log.begin("s5", "LLM judge on clusters…")
        auto_count = len(resolved)
        if auto_count:
            log.info("s5", f"auto-resolved {auto_count} cluster(s) without LLM")

        workers = _pool_workers(ctx, cap=4)
        total_llm = len(llm_clusters)
        cluster_done = [0]
        cluster_lock = threading.Lock()

        def resolve_one(cluster: dict) -> tuple[list[dict], list[dict]]:
            out = _resolve_llm_cluster(cluster, judge, agent)
            with cluster_lock:
                cluster_done[0] += 1
                n = cluster_done[0]
            if total_llm:
                log.progress("s5", n, total_llm, f"resolved cluster {n}/{total_llm}")
            return out

        if llm_clusters:
            if workers <= 1 or total_llm <= 1:
                for cluster in llm_clusters:
                    part_resolved, part_review = resolve_one(cluster)
                    resolved.extend(part_resolved)
                    review.extend(part_review)
            else:
                log.info("s5", f"judging {total_llm} cluster(s) with {workers} workers")
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for part_resolved, part_review in pool.map(resolve_one, llm_clusters):
                        resolved.extend(part_resolved)
                        review.extend(part_review)
        log.done("s5", f"judged {total_llm} LLM cluster(s)")

        _write_jsonl(ctx.workdir / "review_queue.jsonl", review)
        self._write_csv_outputs(resolved, ctx)
        log.done("s6", f"{len(resolved)} canonical reference(s)")
        return resolved

    def load_artifact(self, ctx: StageContext) -> list[dict] | None:
        path = ctx.workdir / self.artifact
        return _read_jsonl(path) if path.exists() else None

    def save_artifact(self, out: list[dict], ctx: StageContext) -> None:
        _write_jsonl(ctx.workdir / self.artifact, out)

    @staticmethod
    def _register_citances(citances: list[Citance], path: Path) -> None:
        _write_jsonl(path, [citance.model_dump() for citance in citances])
        load_citances_jsonl(path)

    @staticmethod
    def _write_csv_outputs(resolved: list[dict], ctx: StageContext) -> None:
        canonical_path = ctx.workdir / "canonical_refs.csv"
        with canonical_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "canonical_id",
                "authors",
                "year",
                "title",
                "venue",
                "type",
                "doi",
                "volume",
                "pages",
                "raw_variants",
                "confidence",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in resolved:
                writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), list) else row.get(key) for key in fieldnames})

        mapping_path = ctx.workdir / "raw_to_canonical.csv"
        with mapping_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["raw", "canonical_id", "confidence"])
            writer.writeheader()
            for row in resolved:
                for raw in row.get("raw_variants", []):
                    writer.writerow({"raw": raw, "canonical_id": row["canonical_id"], "confidence": row["confidence"]})


def _canonical_fields(member: dict) -> dict:
    return {key: member.get(key) for key in ("authors", "year", "title", "venue", "type", "doi", "volume", "pages")}
