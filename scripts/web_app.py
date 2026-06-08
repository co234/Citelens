"""Gradio web UI for the discover pipeline.

Run: python scripts/web_app.py
Then open http://127.0.0.1:7860 in your browser.

Three input modes (one tab each): Terminology, Upload paper, Author.

This UI is built around a **live pipeline flow chart**: as a run progresses,
each stage (search → download → clean → co-cite → enrich → report) lights up in
order, and a **log panel** below it streams the detailed play-by-play and a
final conclusion block. The page is intentionally kept minimal.
"""

from __future__ import annotations

import html as html_module
import os
import tempfile
import threading
import time
import traceback
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


_load_dotenv()

try:
    import gradio as gr
except ImportError as exc:
    raise SystemExit("gradio is required. pip install gradio") from exc

import pandas as pd

from citation_cleaner.aggregate.cocitation import build_incontext_section
from citation_cleaner.pipelines.discover_pipeline import run_discovery
from citation_cleaner.pipelines.pipeline_log import PipelineLogger


# ---------------------------------------------------------------------------
# OpenRouter model configuration
# ---------------------------------------------------------------------------
# A curated short-list; the dropdown also accepts any custom OpenRouter model id.
OPENROUTER_MODELS = [
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-opus-4.8-fast",
    "openai/gpt-5.5",
    "moonshotai/kimi-k2.6",
    "z-ai/glm-5.1",
    "google/gemini-3.5-flash",
    "qwen/qwen-2.5-72b-instruct",
    "deepseek/deepseek-v4-flash",
]
DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-v4-flash"

# OpenRouter also exposes an OpenAI-compatible /embeddings endpoint, so the
# same key powers both the chat stages and the title embeddings.
OPENROUTER_EMBED_MODELS = [
    "qwen/qwen3-embedding-4b",
]
DEFAULT_OPENROUTER_EMBED_MODEL = "qwen/qwen3-embedding-4b"

# Browser localStorage key for persisting OpenRouter settings across refreshes.
_SETTINGS_STORE_KEY = "citation_discoverer_openrouter_v1"
_DEFAULT_STORED_SETTINGS = {
    "api_key": "",
    "model": DEFAULT_OPENROUTER_MODEL,
    "embed_model": DEFAULT_OPENROUTER_EMBED_MODEL,
}

# Special-user link: http://host:7860/?preset=vip
# The UI shows PRESET_API_KEY_PLACEHOLDER; the real key is resolved on the server from
# CITATION_CLEANER_PRESET_API_KEY in .env (see .env.example).
PRESET_QUERY_PARAM = "preset"
PRESET_QUERY_VALUE = "vip"
PRESET_API_KEY_PLACEHOLDER = "preset:vip"


def _preset_api_key() -> str:
    return os.environ.get("CITATION_CLEANER_PRESET_API_KEY", "").strip()


def _url_preset_value(request: "gr.Request | None") -> str:
    """Read ?preset=… from the page URL (Gradio injects gr.Request on load)."""
    if request is None:
        return ""
    qp = getattr(request, "query_params", None)
    if qp is None:
        return ""
    return str(dict(qp).get(PRESET_QUERY_PARAM, "") or "")


def _resolve_openrouter_api_key(api_key: str) -> str:
    """Map the VIP preset placeholder to the server-side key; pass through real keys."""
    token = (api_key or "").strip()
    if token == PRESET_API_KEY_PLACEHOLDER:
        return _preset_api_key()
    return token


def _pack_settings(api_key: str, model: str, embed_model: str) -> dict:
    return {
        "api_key": (api_key or "").strip(),
        "model": (model or "").strip() or DEFAULT_OPENROUTER_MODEL,
        "embed_model": (embed_model or "").strip() or DEFAULT_OPENROUTER_EMBED_MODEL,
    }


def _unpack_settings(stored) -> tuple[str, str, str]:
    if not stored or not isinstance(stored, dict):
        return "", DEFAULT_OPENROUTER_MODEL, DEFAULT_OPENROUTER_EMBED_MODEL
    return (
        stored.get("api_key") or "",
        stored.get("model") or DEFAULT_OPENROUTER_MODEL,
        stored.get("embed_model") or DEFAULT_OPENROUTER_EMBED_MODEL,
    )


def _wire_settings_persistence(
    app: "gr.Blocks",
    saved: "gr.BrowserState",
    api_key: "gr.Textbox",
    model: "gr.Dropdown",
    embed_model: "gr.Dropdown",
) -> None:
    """Restore OpenRouter settings on page load; save on every edit."""

    def save(api_key_val, model_val, embed_val):
        return _pack_settings(api_key_val, model_val, embed_val)

    def load(stored, request: gr.Request):
        """Restore saved settings; URL ?preset=vip fills the agreed placeholder token."""
        query_preset = _url_preset_value(request)
        if query_preset.strip() == PRESET_QUERY_VALUE:
            settings = _pack_settings(
                PRESET_API_KEY_PLACEHOLDER,
                DEFAULT_OPENROUTER_MODEL,
                DEFAULT_OPENROUTER_EMBED_MODEL,
            )
            api, model, embed = _unpack_settings(settings)
            return api, model, embed, settings
        api, model, embed = _unpack_settings(stored)
        return api, model, embed, gr.update()

    save_inputs = [api_key, model, embed_model]
    for event in (api_key.change, api_key.submit, model.change, embed_model.change):
        event(
            save,
            inputs=save_inputs,
            outputs=[saved],
            queue=False,
            show_progress="hidden",
        )

    app.load(
        load,
        inputs=[saved],
        outputs=[api_key, model, embed_model, saved],
        queue=False,
        show_progress="hidden",
    )


def _llm_config(api_key: str, model: str, embed_model: str, dry_run: bool) -> dict:
    """Build the StageContext config that routes the pipeline through
    OpenRouter (chat + embeddings). In dry-run mode no API is called."""
    if dry_run:
        return {}
    chosen = (model or "").strip() or DEFAULT_OPENROUTER_MODEL
    embed_chosen = (embed_model or "").strip() or DEFAULT_OPENROUTER_EMBED_MODEL
    return {
        "llm_provider": "openrouter",
        "openrouter_api_key": _resolve_openrouter_api_key(api_key),
        # Apply the single chosen chat model to every LLM stage.
        "extract_model": chosen,
        "judge_model": chosen,
        "agent_model": chosen,
        "stage0_model": chosen,
        "topic_model": chosen,
        # Route Stage 4 title embeddings through OpenRouter too.
        "embed_provider": "openrouter",
        "embed_model": embed_chosen,
        # Parallel PDF parse / LLM batches / cluster judge (default min(4, cpu)).
        "workers": 8,
    }


# ---------------------------------------------------------------------------
# Result shaping
# ---------------------------------------------------------------------------
def _records_to_df(records) -> "pd.DataFrame":
    rows = []
    for rank, rec in enumerate(records, start=1):
        rows.append(
            {
                "Rank": rank,
                "Co-cited": rec.cocitation_count,
                "In-text hits": len(rec.occurrences),
                "Global cites": rec.global_citation_count or "",
                "Title": rec.title or "",
                "Authors": "; ".join(rec.authors[:3])
                + (" et al." if len(rec.authors) > 3 else ""),
                "Year": rec.year or "",
                "Venue": rec.venue or "",
                "DOI": rec.doi or "",
            }
        )
    return pd.DataFrame(rows)


def _summary_md(result: dict) -> str:
    lines = []
    mode = result.get("mode", "?")
    if mode == "author" and result.get("author"):
        a = result["author"]
        lines.append(
            f"### Author: **{a['display_name']}** "
            f"(`{a.get('author_id', '?').rsplit('/', 1)[-1]}`)"
        )
        lines.append(
            f"Works on OpenAlex: {a.get('works_count', '?')} · "
            f"Total citations: {a.get('cited_by_count', '?')}"
        )
    else:
        lines.append(f"### Query: `{result['query']}`")
        lines.append(f"Sorted by: **{result.get('sort_by', 'relevance')}**")
    lines.append("")
    lines.append(f"**Source papers analyzed:** {len(result['source_papers'])}")
    for sp in result["source_papers"]:
        sid = sp.get("short_id", "?")
        title = (sp.get("title") or "(no title)")[:80]
        year = sp.get("year") or "?"
        cites = sp.get("cited_by_count")
        cites_str = f", cited {cites}×" if cites else ""
        lines.append(f"- `{sid}` — {title} ({year}{cites_str})")
    lines.append("")
    lines.append(f"**Unique references after deduplication:** {len(result['records'])}")
    return "\n".join(lines)


def _incontext_md(result: dict) -> str:
    """Render the In-context appearances block, reusing the exact builder the
    .md report uses so the two never drift apart."""
    records = result.get("records") or []
    source_papers = result.get("source_papers")
    section = build_incontext_section(records, source_papers)
    if not section:
        return (
            "## In-context appearances\n\n"
            "> _No in-text citation sentences were captured for this run._\n\n"
            "This is expected for CSV-only inputs, or for source PDFs whose citation "
            "markers the parser couldn't read (scanned or very old PDFs). Tick "
            "**Dry run** for an offline demo that always populates this section."
        )
    return "\n".join(section).lstrip("\n")


def _conclusion_text(result: dict) -> str:
    """A compact, human-readable conclusion appended to the log panel."""
    lines = ["", "==================== CONCLUSION ===================="]
    mode = result.get("mode")
    if mode == "author" and result.get("author"):
        a = result["author"]
        lines.append(
            f"Author: {a['display_name']} "
            f"({a.get('author_id', '').rsplit('/', 1)[-1]})"
        )
        lines.append(
            f"Works on OpenAlex: {a.get('works_count', '?')} · "
            f"Total citations: {a.get('cited_by_count', '?')}"
        )
    else:
        lines.append(f"Query: {result.get('query')}")
        lines.append(f"Sorted by: {result.get('sort_by', 'relevance')}")
    sps = result.get("source_papers", [])
    recs = result.get("records", [])
    lines.append(f"Source papers analyzed: {len(sps)}")
    lines.append(f"Unique references after deduplication: {len(recs)}")
    if recs:
        top = recs[0]
        lines.append(f"Top co-cited: \"{(top.title or '?')[:60]}\"")
        lines.append(
            f"  co-cited by {top.cocitation_count} source paper(s); "
            f"{len(top.occurrences)} in-text mention(s)"
        )
    op = result.get("output_paths", {})
    if op.get("zip"):
        lines.append(f"Download bundle: {op['zip']}")
    elif op.get("csv"):
        lines.append(f"CSV report: {op['csv']}")
    if op.get("md") and not op.get("zip"):
        lines.append(f"Markdown report: {op['md']}")
    lines.append("===================================================")
    return "\n".join(lines)


def _build_outputs(result: dict):
    """Shape a successful run into the report 4-tuple."""
    return (
        _summary_md(result),
        _records_to_df(result["records"]),
        _incontext_md(result),
        result["output_paths"]["zip"],
    )


# ---------------------------------------------------------------------------
# Pipeline flow chart (vertical: node left, log right, Stages 0–6 split)
# ---------------------------------------------------------------------------
STEP_DEFS: dict[str, tuple[str, str]] = {
    "topic": ("Extract topic", "Infer a query from the uploaded PDF"),
    "author": ("Resolve author", "Match the author on OpenAlex"),
    "fetch": ("Fetch top works", "Pull the author's top-N papers"),
    "search": ("Search OpenAlex", "Find related papers by topic"),
    "download": ("Download PDFs", "Fetch open-access full text"),
    "s0": ("Stage 0 · Parse PDFs", "Text, references & in-text citances"),
    "s1": ("Stage 1 · Pre-clean", "Normalize raw reference strings"),
    "s2": ("Stage 2 · Extract", "LLM structured field extraction"),
    "s3": ("Stage 3 · Block", "Surname/year blocking"),
    "s4": ("Stage 4 · Embed", "Title embedding & clustering"),
    "s5": ("Stage 5 · Judge", "LLM same/different verdict"),
    "s6": ("Stage 6 · Resolve", "Agent + external lookup"),
    "cocite": ("Co-citation", "Stage 7 cross-paper aggregation"),
    "enrich": ("Enrich cites", "Add global citation counts"),
    "report": ("Report", "Ranking table & in-context"),
}


def _steps_for(mode: str, dry_run: bool, enrich: bool) -> list[str]:
    """The ordered step keys that a given run will actually go through."""
    stub = dry_run and mode != "upload"
    if mode == "author":
        steps = ["author"] if stub else ["author", "fetch"]
    elif mode == "upload":
        steps = ["topic", "search"]
    else:
        steps = ["search"]
    if stub:
        return steps + ["cocite", "report"]
    steps += ["download", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "cocite"]
    if enrich and not dry_run:
        steps.append("enrich")
    steps.append("report")
    return steps


def _active_from_snapshot(snapshot: dict, steps: list[str]) -> str:
    """Pick the highlighted step from structured pipeline log metadata."""
    if not steps:
        return ""
    meta = snapshot.get("_meta") or {}
    active = meta.get("active_stage")
    if active in steps:
        return active
    stages = snapshot.get("stages") or {}
    for key in reversed(steps):
        rec = stages.get(key) or {}
        if rec.get("status") == "running":
            return key
    last = steps[0]
    for key in steps:
        rec = stages.get(key) or {}
        if rec.get("lines") or rec.get("status") in ("done", "error"):
            last = key
    return last


def _step_state(i: int, ai: int, state: str) -> str:
    if state == "idle":
        return "pending"
    if state == "done":
        return "done"
    if state == "error":
        if i < ai:
            return "done"
        if i == ai:
            return "error"
        return "pending"
    if i < ai:
        return "done"
    if i == ai:
        return "active"
    return "pending"


def _progress_html(progress: dict | None) -> str:
    if not progress:
        return ""
    total = int(progress.get("total") or 0)
    if total <= 0:
        return ""
    current = int(progress.get("current") or 0)
    pct = min(100, int(100 * current / total))
    label = html_module.escape(str(progress.get("label") or ""))
    return (
        f"<div class='cc-log-progress'>"
        f"<span class='cc-log-progress-text'>{current}/{total}</span>"
        f"<div class='cc-log-progress-bar' aria-hidden='true'>"
        f"<div class='cc-log-progress-fill' style='width:{pct}%'></div></div>"
        f"<span class='cc-log-progress-label'>{label}</span></div>"
    )


def _panel_html(steps: list[str], snapshot: dict, state: str) -> str:
    """Vertical pipeline: node on the left, that step's log on the right."""
    stages_data = snapshot.get("stages") or {}
    active_key = _active_from_snapshot(snapshot, steps)
    ai = steps.index(active_key) if active_key in steps else -1
    rows: list[str] = []
    for i, key in enumerate(steps):
        label, desc = STEP_DEFS[key]
        cls = _step_state(i, ai, state)
        glyph = "✓" if cls == "done" else ("!" if cls == "error" else str(i + 1))
        rec = stages_data.get(key) or {}
        lines = rec.get("lines") or []
        prog = _progress_html(rec.get("progress"))
        if lines:
            body = prog + "".join(
                f"<div class='cc-log-line'>{html_module.escape(ln)}</div>"
                for ln in lines
            )
        elif prog:
            body = prog
        elif cls == "active":
            body = "<div class='cc-log-line cc-log-muted'>Running…</div>"
        elif cls == "done":
            body = "<div class='cc-log-line cc-log-muted'>—</div>"
        else:
            body = "<div class='cc-log-line cc-log-muted'>Waiting…</div>"
        rows.append(
            f"<div class='cc-vrow {cls}'>"
            f"<div class='cc-vnode'>"
            f"<div class='cc-vdot'>{glyph}</div>"
            f"<div class='cc-vmeta'>"
            f"<div class='cc-vlabel'>{html_module.escape(label)}</div>"
            f"<div class='cc-vdesc'>{html_module.escape(desc)}</div>"
            f"</div></div>"
            f"<div class='cc-vlog'>{body}</div>"
            f"</div>"
        )
    return f"<div class='cc-vpanel'>{''.join(rows)}</div>"


def _idle_panel_html(mode: str) -> str:
    steps = _steps_for(mode, dry_run=False, enrich=True)
    return _panel_html(steps, {"_meta": {}, "stages": {}}, "idle")


# ---------------------------------------------------------------------------
# Streaming runner
# ---------------------------------------------------------------------------
def _err_tuple(msg: str):
    """5-tuple for a validation/early error (cleared outputs)."""
    err_panel = (
        f"<div class='cc-vpanel'><div class='cc-vrow error'>"
        f"<div class='cc-vnode'><div class='cc-vdot'>!</div>"
        f"<div class='cc-vmeta'><div class='cc-vlabel'>Error</div></div></div>"
        f"<div class='cc-vlog'><div class='cc-log-line'>{html_module.escape(msg)}</div></div>"
        f"</div></div>"
    )
    return err_panel, msg, None, "", None


def _stream(*, mode: str, kwargs: dict, dry_run: bool, enrich: bool):
    """Generator that runs run_discovery in a worker thread, streaming the
    vertical pipeline panel, then yielding the final report tuple.

    Yields 5-tuples: (panel_html, summary_md, dataframe, incontext_md, zip)
    """
    steps = _steps_for(mode, dry_run, enrich)
    pl = PipelineLogger(mirror_stdout=False)
    run_kwargs = dict(kwargs)
    cfg = dict(run_kwargs.get("config") or {})
    cfg["_pipeline_logger"] = pl
    run_kwargs["config"] = cfg
    holder: dict = {}

    def worker() -> None:
        try:
            holder["result"] = run_discovery(**run_kwargs)
        except BaseException as exc:  # noqa: BLE001 - surface SystemExit too
            holder["error"] = exc
            holder["traceback"] = traceback.format_exc()
            stage = pl.active_stage or (steps[-1] if steps else "report")
            pl.error(stage, str(exc))
            for line in traceback.format_exc().splitlines():
                pl.info(stage, line)

    pl.begin(steps[0] if steps else "search", "Starting…")
    yield _panel_html(steps, pl.snapshot(), "running"), "", None, "", None

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while thread.is_alive():
        yield (
            _panel_html(steps, pl.snapshot(), "running"),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
        time.sleep(0.3)
    thread.join()

    if "error" in holder:
        err = holder["error"]
        msg = f"❌ Run failed: {err}"
        yield _panel_html(steps, pl.snapshot(), "error"), msg, None, "", None
        return

    result = holder["result"]
    summary, df, incontext, zip_f = _build_outputs(result)
    pl.info("report", _conclusion_text(result))
    pl.done("report", "Run complete")
    yield (
        _panel_html(steps, pl.snapshot(), "done"),
        summary,
        df,
        incontext,
        zip_f,
    )


# --- Handlers ----------------------------------------------------------------
def _need_key(api_key: str, dry_run: bool) -> bool:
    """True if a real run is requested but no OpenRouter key was provided."""
    return not dry_run and not _resolve_openrouter_api_key(api_key)


def _run_terminology(
    query: str,
    n: int,
    sort_by: str,
    api_key: str,
    model: str,
    embed_model: str,
    dry_run: bool,
    enrich: bool,
):
    if not query or not query.strip():
        yield _err_tuple("Please enter a terminology / topic.")
        return
    if _need_key(api_key, dry_run):
        yield _err_tuple("Please enter your OpenRouter API key (or tick Dry run).")
        return
    workdir = Path(tempfile.mkdtemp(prefix="citation_web_"))
    yield from _stream(
        mode="terminology",
        kwargs=dict(
            query=query.strip(),
            workdir=workdir,
            n_papers=int(n),
            sort_by=sort_by,
            dry_run=bool(dry_run),
            enrich_citations=bool(enrich),
            config=_llm_config(api_key, model, embed_model, bool(dry_run)),
        ),
        dry_run=bool(dry_run),
        enrich=bool(enrich),
    )


def _run_upload(
    file_obj,
    n: int,
    sort_by: str,
    api_key: str,
    model: str,
    embed_model: str,
    dry_run: bool,
    enrich: bool,
):
    if file_obj is None:
        yield _err_tuple("Please upload a PDF file.")
        return
    if _need_key(api_key, dry_run):
        yield _err_tuple("Please enter your OpenRouter API key (or tick Dry run).")
        return
    workdir = Path(tempfile.mkdtemp(prefix="citation_web_"))
    yield from _stream(
        mode="upload",
        kwargs=dict(
            uploaded_pdf=Path(file_obj.name),
            workdir=workdir,
            n_papers=int(n),
            sort_by=sort_by,
            dry_run=bool(dry_run),
            enrich_citations=bool(enrich),
            config=_llm_config(api_key, model, embed_model, bool(dry_run)),
        ),
        dry_run=bool(dry_run),
        enrich=bool(enrich),
    )


def _run_author(
    name: str,
    n: int,
    api_key: str,
    model: str,
    embed_model: str,
    dry_run: bool,
    enrich: bool,
):
    if not name or not name.strip():
        yield _err_tuple("Please enter an author name (or OpenAlex author ID).")
        return
    if _need_key(api_key, dry_run):
        yield _err_tuple("Please enter your OpenRouter API key (or tick Dry run).")
        return
    workdir = Path(tempfile.mkdtemp(prefix="citation_web_"))
    yield from _stream(
        mode="author",
        kwargs=dict(
            author=name.strip().upper(),
            workdir=workdir,
            n_papers=int(n),
            dry_run=bool(dry_run),
            enrich_citations=bool(enrich),
            config=_llm_config(api_key, model, embed_model, bool(dry_run)),
        ),
        dry_run=bool(dry_run),
        enrich=bool(enrich),
    )


# ---------------------------------------------------------------------------
# Look & feel (minimal)
# ---------------------------------------------------------------------------
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(block_radius="12px")

CSS = """
.gradio-container {max-width: 1080px !important; margin: 0 auto !important;}
.cc-head {padding: 6px 2px 2px;}
.cc-head h1 {font-size: 1.5rem; font-weight: 800; margin: 0; color: #4338ca; letter-spacing:-.01em;}
.cc-head p {margin: 4px 0 0; color: #64748b; font-size: .92rem;}

/* ---- vertical pipeline panel (node left, log right) ---- */
.cc-run-panel-wrap {margin-top: 8px;}
.cc-vpanel {display: flex; flex-direction: column; gap: 0; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden; background: #fff;}
.cc-vrow {display: grid; grid-template-columns: 220px 1fr; gap: 0; border-bottom: 1px solid #f1f5f9; min-height: 52px;}
.cc-vrow:last-child {border-bottom: none;}
.cc-vrow.done {background: #fafbff;}
.cc-vrow.active {background: #f5f3ff;}
.cc-vrow.error {background: #fef2f2;}
.cc-vnode {display: flex; align-items: flex-start; gap: 10px; padding: 12px 14px; border-right: 1px solid #f1f5f9;}
.cc-vdot {
  flex-shrink: 0; width: 28px; height: 28px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: .78rem; background: #f1f5f9; color: #94a3b8; border: 2px solid #e2e8f0;
}
.cc-vrow.done .cc-vdot {background: #6366f1; color: #fff; border-color: #6366f1;}
.cc-vrow.active .cc-vdot {background: #fff; color: #4f46e5; border-color: #6366f1; box-shadow: 0 0 0 3px rgba(99,102,241,.15);}
.cc-vrow.error .cc-vdot {background: #ef4444; color: #fff; border-color: #ef4444;}
.cc-vmeta {min-width: 0;}
.cc-vlabel {font-size: .82rem; font-weight: 600; color: #334155; line-height: 1.25;}
.cc-vrow.active .cc-vlabel, .cc-vrow.done .cc-vlabel {color: #312e81;}
.cc-vrow.error .cc-vlabel {color: #b91c1c;}
.cc-vdesc {font-size: .68rem; color: #94a3b8; margin-top: 2px; line-height: 1.2;}
.cc-vlog {padding: 10px 14px; font-family: ui-monospace, "JetBrains Mono", monospace; font-size: .72rem; line-height: 1.45; color: #475569; overflow-x: auto; max-height: 140px; overflow-y: auto;}
.cc-log-line {white-space: pre-wrap; word-break: break-word;}
.cc-log-muted {color: #cbd5e1; font-style: italic;}
.cc-log-progress {
  display: flex; align-items: center; gap: 8px; margin-bottom: 6px;
  padding: 4px 8px; background: #eef2ff; border-radius: 6px; font-size: .78rem;
}
.cc-log-progress-text {font-weight: 700; color: #4338ca; font-variant-numeric: tabular-nums;}
.cc-log-progress-bar {
  flex: 1; height: 6px; background: #c7d2fe; border-radius: 999px; overflow: hidden;
}
.cc-log-progress-fill {height: 100%; background: #6366f1; border-radius: 999px; transition: width .25s;}
.cc-log-progress-label {color: #64748b; max-width: 55%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;}

/* ---- in-context panel ---- */
.cc-incontext {
  margin-top: 12px;
  padding: 16px 18px;
  background: #fafbff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
}
.cc-incontext h2 {
  color: #4338ca; font-weight: 700; font-size: 1.15rem;
  margin: 0 0 12px; padding-bottom: 10px;
  border-bottom: 1px solid #e2e8f0;
}
.cc-incontext h3 {
  color: #334155; margin-top: 20px; font-size: 1rem; font-weight: 600;
}
.cc-incontext h3:first-of-type { margin-top: 8px; }
.cc-incontext blockquote {
  border: none;
  background: #fff;
  box-shadow: inset 0 0 0 1px #e2e8f0;
  margin: 8px 0 4px;
  padding: 10px 14px;
  border-radius: 8px;
  color: #334155;
  font-style: normal;
  line-height: 1.55;
}
.cc-incontext blockquote p { margin: 0; }
.cc-footer {color: #94a3b8; font-size: .82rem; text-align: center; padding-top: 10px;}
.cc-footer code {background: #f1f5f9; color: #4338ca; padding: 1px 5px; border-radius: 5px;}

/* ---- ranked results table (horizontal scroll, readable headers) ---- */
.cc-results-table .table-wrap {
  overflow-x: auto !important;
  width: 100%;
  -webkit-overflow-scrolling: touch;
  border-radius: 10px;
}
.cc-results-table table {
  width: max-content !important;
  min-width: 100%;
  table-layout: auto !important;
}
.cc-results-table thead th {
  white-space: nowrap !important;
  padding: 10px 14px !important;
  font-size: .82rem;
  font-weight: 600;
  background: #f8fafc;
  vertical-align: bottom;
  line-height: 1.3;
}
.cc-results-table thead th:nth-child(1) { min-width: 52px; }
.cc-results-table thead th:nth-child(2) { min-width: 76px; }
.cc-results-table thead th:nth-child(3) { min-width: 96px; }
.cc-results-table thead th:nth-child(4) { min-width: 96px; }
.cc-results-table thead th:nth-child(5) { min-width: 280px; }
.cc-results-table thead th:nth-child(6) { min-width: 160px; }
.cc-results-table thead th:nth-child(7) { min-width: 64px; }
.cc-results-table thead th:nth-child(8) { min-width: 120px; }
.cc-results-table thead th:nth-child(9) { min-width: 180px; }
.cc-results-table tbody td {
  padding: 8px 14px !important;
  font-size: .84rem;
  vertical-align: top;
  white-space: nowrap;
}
.cc-results-table tbody td:nth-child(1),
.cc-results-table tbody td:nth-child(2),
.cc-results-table tbody td:nth-child(3),
.cc-results-table tbody td:nth-child(4),
.cc-results-table tbody td:nth-child(7) {
  text-align: center;
  font-variant-numeric: tabular-nums;
}
.cc-results-table tbody td:nth-child(5) {
  white-space: normal;
  min-width: 280px;
  max-width: 420px;
  line-height: 1.45;
}
.cc-results-table tbody td:nth-child(6),
.cc-results-table tbody td:nth-child(8) {
  white-space: normal;
  min-width: 160px;
  max-width: 240px;
}
"""

HEAD_HTML = """
<div class="cc-head">
  <h1>Citation Discoverer · v4</h1>
  <p>Discover related work by topic, paper, or author → clean references → rank by co-citation → show in-text context. The flow chart below tracks every step live.</p>
</div>
"""


def _llm_controls():
    """OpenRouter API key + chat/embedding model pickers.
    Returns (api_key_textbox, model_dropdown, embed_model_dropdown)."""
    api_key = gr.Textbox(
        label="OpenRouter API key",
        type="password",
        placeholder="sk-or-...",
        info=(
            "Saved in this browser (localStorage). "
        ),
    )
    with gr.Row():
        model = gr.Dropdown(
            choices=OPENROUTER_MODELS,
            value=DEFAULT_OPENROUTER_MODEL,
            allow_custom_value=True,
            label="Chat model (OpenRouter)",
        )
        embed_model = gr.Dropdown(
            choices=OPENROUTER_EMBED_MODELS,
            value=DEFAULT_OPENROUTER_EMBED_MODEL,
            allow_custom_value=True,
            label="Embedding model (OpenRouter)",
        )
    return api_key, model, embed_model


def _make_outputs(mode: str):
    """Create per-tab output components in click-handler order:
    panel, summary, table, in-context, zip bundle."""
    panel = gr.HTML(_idle_panel_html(mode), elem_classes="cc-run-panel-wrap")
    summary = gr.Markdown()
    with gr.Accordion("Ranked co-cited references", open=True):
        df = gr.Dataframe(
            interactive=False,
            wrap=False,
            elem_classes="cc-results-table",
        )
    incontext = gr.Markdown(elem_classes="cc-incontext")
    zip_f = gr.File(label="Download results (ZIP)")
    return panel, summary, df, incontext, zip_f


def build_app() -> "gr.Blocks":
    with gr.Blocks(title="Citation Discoverer v4") as app:
        gr.HTML(HEAD_HTML)

        saved_settings = gr.BrowserState(
            default_value=dict(_DEFAULT_STORED_SETTINGS),
            storage_key=_SETTINGS_STORE_KEY,
        )
        with gr.Accordion("OpenRouter settings", open=True):
            api_key, model, embed_model = _llm_controls()
        _wire_settings_persistence(app, saved_settings, api_key, model, embed_model)

        def _persist_settings(api_key_val, model_val, embed_val):
            return _pack_settings(api_key_val, model_val, embed_val)

        _save_on_run = dict(
            fn=_persist_settings,
            inputs=[api_key, model, embed_model],
            outputs=[saved_settings],
            queue=False,
            show_progress="hidden",
        )

        with gr.Tabs():
            # --- Terminology tab ---
            with gr.Tab("By topic"):
                query_input = gr.Textbox(
                    label="Terminology / topic",
                    placeholder="e.g. diffusion models for image generation",
                )
                with gr.Row():
                    n_term = gr.Slider(1, 10, value=3, step=1, label="# source papers")
                    sort_term = gr.Dropdown(
                        choices=["relevance", "recency", "citations"],
                        value="relevance",
                        label="Sort source papers by",
                    )
                with gr.Row():
                    dry_term = gr.Checkbox(label="Dry run (offline, no API calls)")
                    enrich_term = gr.Checkbox(value=True, label="Enrich with global citation counts")
                run_term_btn = gr.Button("Discover", variant="primary")

                outs_term = _make_outputs("terminology")
                run_term_btn.click(**_save_on_run)
                run_term_btn.click(
                    _run_terminology,
                    inputs=[
                        query_input,
                        n_term,
                        sort_term,
                        api_key,
                        model,
                        embed_model,
                        dry_term,
                        enrich_term,
                    ],
                    outputs=list(outs_term),
                )

            # --- Upload tab ---
            with gr.Tab("Upload paper"):
                upload_input = gr.File(label="Your paper (PDF)", file_types=[".pdf"])
                with gr.Row():
                    n_up = gr.Slider(1, 10, value=3, step=1, label="# related papers to find")
                    sort_up = gr.Dropdown(
                        choices=["relevance", "recency", "citations"],
                        value="relevance",
                        label="Sort related papers by",
                    )
                with gr.Row():
                    dry_up = gr.Checkbox(label="Dry run (offline, no API calls)")
                    enrich_up = gr.Checkbox(value=True, label="Enrich with global citation counts")
                run_up_btn = gr.Button("Discover", variant="primary")

                outs_up = _make_outputs("upload")
                run_up_btn.click(**_save_on_run)
                run_up_btn.click(
                    _run_upload,
                    inputs=[
                        upload_input,
                        n_up,
                        sort_up,
                        api_key,
                        model,
                        embed_model,
                        dry_up,
                        enrich_up,
                    ],
                    outputs=list(outs_up),
                )

            # --- Author tab ---
            with gr.Tab("By author"):
                gr.Markdown(
                    "Enter an author name (e.g. `Yann LeCun`) or an OpenAlex author ID "
                    "(e.g. `A2208157607`). For ambiguous names, the author with the most "
                    "total citations is chosen. **Sort is always citation-desc** — this "
                    "tab returns the author's top-N most-cited papers."
                )
                author_input = gr.Textbox(
                    label="Author name or OpenAlex author ID",
                    placeholder="Yann LeCun",
                )
                with gr.Row():
                    n_au = gr.Slider(1, 10, value=3, step=1, label="# top papers")
                    dry_au = gr.Checkbox(label="Dry run (offline, no API calls)")
                    enrich_au = gr.Checkbox(value=True, label="Enrich with global citation counts")
                run_au_btn = gr.Button("Discover", variant="primary")

                outs_au = _make_outputs("author")
                run_au_btn.click(**_save_on_run)
                run_au_btn.click(
                    _run_author,
                    inputs=[
                        author_input,
                        n_au,
                        api_key,
                        model,
                        embed_model,
                        dry_au,
                        enrich_au,
                    ],
                    outputs=list(outs_au),
                )

        gr.HTML(
            "<div class='cc-footer'>Both the LLM stages and the title embeddings run "
            "through <b>OpenRouter</b> — one <code>sk-or-...</code> key above (remembered "
            "in this browser). Set <code>CITATION_CLEANER_EMAIL</code> for OpenAlex's "
            "polite pool. <b>Dry run</b> bypasses all keys for an offline demo.</div>"
        )
    return app


if __name__ == "__main__":
    build_app().launch(
        server_name="127.0.0.1",
        server_port=7867,
        share=False,
        theme=THEME,
        css=CSS,
    )
