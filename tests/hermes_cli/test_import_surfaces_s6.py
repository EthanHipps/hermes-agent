"""S6 import surfaces (R46; spec §9.7 "Agent import" and "OpenClaw migration script", §9.10 L1675/L1691).

In a Hermes home whose config requests authoritative memory, ``hermes import-agent`` and the
OpenClaw migration script never stat, read, create or write the native memory directory, never
contact the memory provider, and persist no migration state. That is proven with the native
sentinel plus a byte-and-mtime snapshot, never inferred from output. Every non-memory item
still imports, and an additive home is unchanged.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hermes_cli import agent_import
from hermes_cli.agent_import import AgentImporter
from hermes_cli.backup_memory import home_disposition
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


def test_import_agent_command_names_provider_managed_memory(env, tmp_path, provider_calls, monkeypatch):
    import hermes_cli.setup as setup_mod
    from hermes_cli.agent_import import import_agent_command

    # import_agent_command imports print_info from hermes_cli.setup at call time: patch where it reads.
    notices = []
    monkeypatch.setattr(setup_mod, "print_info", lambda *a, **k: notices.append(" ".join(map(str, a))))
    home = _authoritative(env)
    native = _dormant(home)
    before = _snapshot(native)
    tree = _claude_tree(tmp_path / ".claude")
    args = SimpleNamespace(agent="claude-code", source=str(tree), dry_run=False, overwrite=False, yes=True)
    with native_memory_sentinel(native) as sentinel:
        import_agent_command(args)
    sentinel.assert_untouched()
    assert _snapshot(native) == before
    assert provider_calls == []
    # The settings-block notice (ruling R46-7). The skipped report rows use plain print(), not print_info.
    assert any(agent_import.PROVIDER_MANAGED_MEMORY_REASON in line for line in notices)


# --- OpenClaw migration script: the target home's memory mode --------------------------------

SCRIPT_PATH = (Path(__file__).resolve().parents[2] / "optional-skills" / "migration"
               / "openclaw-migration" / "scripts" / "openclaw_to_hermes.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("openclaw_to_hermes_r46", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module   # @dataclass needs the module registered (Python 3.11+)
    spec.loader.exec_module(module)
    return module


PREDICATE_CASES = {
    "absent": None, "empty": "", "no_memory": "model: x\n", "memory_null": "memory:\n",
    "memory_list": "memory: [a]\n", "additive": "memory:\n  provider_mode: additive\n",
    "authoritative": "memory:\n  provider: example\n  provider_mode: authoritative\n",
    "padded": "memory:\n  provider_mode: '  authoritative '\n",
    "capitalised": "memory:\n  provider_mode: Authoritative\n",
    "non_string": "memory:\n  provider_mode: 1\n", "unparseable": "memory: [unclosed\n",
    "top_level_list": "- a\n",
}


@pytest.mark.parametrize("case", sorted(PREDICATE_CASES))
def test_standalone_predicate_agrees_with_hermes(tmp_path, case):
    """The script's stdlib twin and Hermes' per-home predicate decide every plain config alike."""
    mod = _load_script()
    home = tmp_path / case
    home.mkdir()
    if PREDICATE_CASES[case] is not None:
        (home / "config.yaml").write_text(PREDICATE_CASES[case], encoding="utf-8")
    assert mod._inline_requests_authoritative(home) is (home_disposition(home) is not None)


def test_hermes_predicate_is_used_whenever_hermes_is_importable(tmp_path, monkeypatch):
    """Only Hermes' pipeline expands env references, so this passes only on the canonical path."""
    mod = _load_script()
    monkeypatch.setenv("R46_MEMORY_MODE", "authoritative")
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("memory:\n  provider_mode: ${R46_MEMORY_MODE}\n", encoding="utf-8")
    assert mod.target_requests_authoritative_memory(home) is True
    assert mod._inline_requests_authoritative(home) is False


def test_standalone_fallback_when_hermes_is_not_importable(tmp_path, monkeypatch):
    mod = _load_script()
    monkeypatch.setitem(sys.modules, "hermes_cli.backup_memory", None)   # the import raises ImportError
    auth, plain = _authoritative(tmp_path / "a"), tmp_path / "p"
    plain.mkdir()
    (plain / "config.yaml").write_text("model: x\n", encoding="utf-8")
    assert mod.target_requests_authoritative_memory(auth) is True
    assert mod.target_requests_authoritative_memory(plain) is False


def test_without_pyyaml_any_provider_mode_counts_as_authoritative(tmp_path, monkeypatch):
    """Ruling R46-4: refusing is recoverable, writing into a dormant store is not."""
    mod = _load_script()
    monkeypatch.setattr(mod, "yaml", None)
    auth, plain = _authoritative(tmp_path / "a"), tmp_path / "p"
    plain.mkdir()
    (plain / "config.yaml").write_text("model: x\n", encoding="utf-8")
    assert mod._inline_requests_authoritative(auth) is True
    assert mod._inline_requests_authoritative(plain) is False   # no provider_mode key: additive, as at base
