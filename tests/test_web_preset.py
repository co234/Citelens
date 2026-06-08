"""VIP preset placeholder → server-side API key resolution."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.web_app as wa  # noqa: E402


def test_resolve_preset_placeholder(monkeypatch):
    monkeypatch.setenv("CITATION_CLEANER_PRESET_API_KEY", "sk-or-v1-test-secret")
    assert wa._resolve_openrouter_api_key(wa.PRESET_API_KEY_PLACEHOLDER) == "sk-or-v1-test-secret"
    assert wa._resolve_openrouter_api_key("sk-or-v1-user-key") == "sk-or-v1-user-key"
    assert wa._need_key(wa.PRESET_API_KEY_PLACEHOLDER, dry_run=False) is False
    assert wa._need_key("", dry_run=False) is True


def test_llm_config_resolves_placeholder(monkeypatch):
    monkeypatch.setenv("CITATION_CLEANER_PRESET_API_KEY", "sk-or-v1-server-only")
    cfg = wa._llm_config(
        wa.PRESET_API_KEY_PLACEHOLDER,
        wa.DEFAULT_OPENROUTER_MODEL,
        wa.DEFAULT_OPENROUTER_EMBED_MODEL,
        False,
    )
    assert cfg["openrouter_api_key"] == "sk-or-v1-server-only"
