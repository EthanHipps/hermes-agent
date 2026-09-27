"""S6 import surfaces (R46; spec §9.7 "Agent import" and "OpenClaw migration script", §9.10 L1675/L1691).

In a Hermes home whose config requests authoritative memory, ``hermes import-agent`` and the
OpenClaw migration script never stat, read, create or write the native memory directory, never
contact the memory provider, and persist no migration state. That is proven with the native
sentinel plus a byte-and-mtime snapshot, never inferred from output. Every non-memory item
still imports, and an additive home is unchanged.
"""

import json
from pathlib import Path

import pytest
import yaml

from hermes_cli import agent_import
from hermes_cli.agent_import import AgentImporter
from tests.agent.memory_service.native_sentinel import native_memory_sentinel

# Ruling R44-3: the REQUESTED mode counts, even when the configuration is invalid.
VARIANTS = {"fail_closed": {}, "stateless": {"policy": "stateless"}, "invalid": {"principal": ""}}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """profile_env pattern (root AGENTS.md): Path.home() and HERMES_HOME inside tmp_path."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def provider_calls(monkeypatch):
    """Provider contact is the only 'upload' path; it is recorded and must never happen."""
    calls = []

    def factory(name):
        calls.append(name)
        raise AssertionError("an import surface contacted the memory provider")

    monkeypatch.setattr("plugins.memory.load_authoritative_backend_factory", factory)
    return calls


def _authoritative(home: Path, *, policy: str = "", principal: str = "ethan") -> Path:
    home.mkdir(parents=True, exist_ok=True)
    exe = home.parent / "provider.exe"
    exe.write_bytes(b"MZ")
    lines = ["memory:", "  provider: example", "  provider_mode: authoritative",
             f"  provider_executable: '{exe}'"]
    if principal:
        lines.append(f"  principal_id: {principal}")
    if policy:
        lines.append(f"  authoritative_failure_policy: {policy}")
    (home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return home


def _additive(home: Path) -> Path:
    (home / "config.yaml").write_text("model:\n  provider: openrouter\n", encoding="utf-8")
    return home


def _dormant(home: Path) -> Path:
    native = home / "memories"
    native.mkdir(exist_ok=True)
    (native / "MEMORY.md").write_text("dormant native memory\n", encoding="utf-8")
    (native / "USER.md").write_text("dormant native profile\n", encoding="utf-8")
    return native


def _snapshot(root: Path):
    """Bytes and mtime of every file under root; taken OUTSIDE the sentinel."""
    if not root.exists():
        return None
    return {p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file()}


def _claude_tree(root: Path) -> Path:
    root.mkdir()
    (root / "CLAUDE.md").write_text("# Rules\n\n- Always use type hints\n", encoding="utf-8")
    (root / "settings.json").write_text(
        json.dumps({"permissions": {"allow": ["Bash(npm run build)"]}}), encoding="utf-8")
    skill = root / "skills" / "deploy-helper"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: deploy-helper\n---\n\nDeploy.\n", encoding="utf-8")
    return root


def _codex_tree(root: Path) -> Path:
    root.mkdir()
    (root / "AGENTS.md").write_text("- Keep commits atomic\n", encoding="utf-8")
    (root / "config.toml").write_text('model = "gpt-5"\n', encoding="utf-8")
    (root / "memories").mkdir()
    (root / "memories" / "2026-01-01.md").write_text("- Project uses PostgreSQL\n", encoding="utf-8")
    return root


def _items(report, kind):
    return [i for i in report["items"] if i["kind"] == kind]


# --- hermes import-agent -------------------------------------------------------------------

@pytest.mark.parametrize("execute", [True, False], ids=["execute", "dry_run"])
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_authoritative_import_agent_never_touches_native_memory(env, tmp_path, provider_calls, execute, variant):
    home = _authoritative(env, **VARIANTS[variant])
    native = _dormant(home)
    before = _snapshot(native)
    tree = _claude_tree(tmp_path / ".claude")
    with native_memory_sentinel(native) as sentinel:
        report = AgentImporter("claude-code", tree, home, execute=execute).run()
    sentinel.assert_untouched()
    assert _snapshot(native) == before
    assert provider_calls == []
    assert not (home / "migrations").exists()
    [memory] = _items(report, "claude-md")
    assert (memory["status"], memory["reason"], memory["destination"]) == (
        "skipped", agent_import.PROVIDER_MANAGED_MEMORY_REASON, None)
    assert [i["status"] for i in _items(report, "command-allowlist")] == ["imported"]   # ruling R46-2
    if execute:
        config = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
        assert "npm run build" in config["command_allowlist"]
        assert config["memory"]["provider_mode"] == "authoritative"
        assert (home / "skills" / "claude-code-imports" / "deploy-helper" / "SKILL.md").exists()


def test_authoritative_codex_import_skips_both_memory_sources(env, tmp_path, provider_calls):
    home = _authoritative(env)
    native = _dormant(home)
    before = _snapshot(native)
    with native_memory_sentinel(native) as sentinel:
        report = AgentImporter("codex", _codex_tree(tmp_path / ".codex"), home, execute=True).run()
    sentinel.assert_untouched()
    assert _snapshot(native) == before
    assert provider_calls == []
    for kind in ("agents-md", "memories"):
        [item] = _items(report, kind)
        assert (item["status"], item["reason"]) == ("skipped", agent_import.PROVIDER_MANAGED_MEMORY_REASON)


def test_authoritative_import_never_creates_the_native_directory(env, tmp_path, provider_calls):
    home = _authoritative(env)
    native = home / "memories"
    assert not native.exists()
    with native_memory_sentinel(native) as sentinel:
        AgentImporter("claude-code", _claude_tree(tmp_path / ".claude"), home, execute=True).run()
    sentinel.assert_untouched()
    assert not native.exists()


def test_additive_import_agent_still_merges_into_native_memory(env, tmp_path):
    """Additive freeze (§9.10 L1674): the same source still lands in MEMORY.md."""
    home = _additive(env)
    report = AgentImporter("claude-code", _claude_tree(tmp_path / ".claude"), home, execute=True).run()
    [memory] = _items(report, "claude-md")
    assert memory["status"] == "imported"
    assert "Always use type hints" in (home / "memories" / "MEMORY.md").read_text(encoding="utf-8")
