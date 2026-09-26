"""S6 archive suite F: the session a restored home starts (§9.6 L1586, §9.8 L1639, §9.3 L1229).

A restore brings back Hermes configuration and the disposition record only. The next
session therefore binds through an ordinary ``new_session`` -- there is no reconnect
intent (ledger V-23) and no prior identity to validate. Legacy native files that came
back in a hand-built archive stay dormant, proven with the sentinel (§9.10 L1685).

Kept in its FRESH-SESSION form on purpose: R40 merges after this row and replaces the
binding resolver, and a fresh session id with no SessionDB row still resolves to "new"
(reconciliation correction R-4). Helpers are local (D-R44-f).
"""

import importlib
import os
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.agent_init import _init_memory
from agent.memory_service import wire as w
from agent.memory_service.errors import MemoryBlockedError
from agent.memory_service.service import MemoryDisposition
from tests.agent.memory_service.fake_backend import (
    FakeAuthoritativeBackend, FakeProviderStore, FakeRegistry,
)
from tests.agent.memory_service.native_sentinel import native_memory_sentinel


@pytest.fixture(autouse=True)
def _no_real_gateway_service(monkeypatch):
    import hermes_cli.gateway as gateway_mod
    monkeypatch.setattr(gateway_mod, "ensure_gateway_service", lambda **kw: False)
    monkeypatch.setattr(gateway_mod, "_is_service_running", lambda: False)


@pytest.fixture
def fake_backend(monkeypatch, tmp_path):
    """R36's fake at the real discovery seam, with the realpath'd cwd registered."""
    monkeypatch.chdir(tmp_path)
    registry = FakeRegistry(directories={os.path.realpath(os.getcwd()): ("repo-1", "proj-1", None)})
    backend = FakeAuthoritativeBackend(FakeProviderStore(registry=registry), provider="example")
    monkeypatch.setattr("plugins.memory.load_authoritative_backend_factory",
                        lambda name: (lambda cfg: backend))
    return backend


def _authoritative_yaml(exe: Path, provider: str, policy: str = "") -> str:
    extra = f"  authoritative_failure_policy: {policy}\n" if policy else ""
    return (f"memory:\n  provider: {provider}\n  provider_mode: authoritative\n"
            f"  provider_executable: '{exe}'\n  principal_id: ethan\n{extra}")


def _home(path: Path, *, provider=None, policy="") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if provider is None:
        (path / "config.yaml").write_text("model:\n  provider: openrouter\n", encoding="utf-8")
    else:
        exe = path.parent / f"{path.name}-provider.exe"
        exe.write_bytes(b"MZ")
        (path / "config.yaml").write_text(_authoritative_yaml(exe, provider, policy), encoding="utf-8")
    return path


def _dormant(home: Path) -> Path:
    native = home / "memories"
    native.mkdir(parents=True, exist_ok=True)
    (native / "MEMORY.md").write_text("dormant native memory\n", encoding="utf-8")
    (native / "USER.md").write_text("dormant native profile\n", encoding="utf-8")
    return native


def _restore(tmp_path, monkeypatch, *, provider="example", policy="") -> Path:
    """Back up an authoritative home, then restore it into a fresh target home."""
    backup = importlib.import_module("hermes_cli.backup")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    source = _home(tmp_path / "src", provider=provider, policy=policy)
    _dormant(source)
    monkeypatch.setenv("HERMES_HOME", str(source))
    backup.run_backup(Namespace(output=str(tmp_path / "a.zip")))
    target = tmp_path / "dst"
    target.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(target))
    backup.run_import(Namespace(zipfile=str(tmp_path / "a.zip"), force=True))
    return target


def _restore_legacy(tmp_path, monkeypatch, *, provider="example") -> Path:
    """Restore a hand-built archive that DOES carry native files (the real legacy case)."""
    import zipfile
    backup = importlib.import_module("hermes_cli.backup")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    exe = tmp_path / "p.exe"
    exe.write_bytes(b"MZ")
    archive = tmp_path / "legacy.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("config.yaml", _authoritative_yaml(exe, provider))
        zf.writestr(".env", "KEY=value\n")
        zf.writestr("memories/MEMORY.md", "legacy memory\n")
        zf.writestr("memories/USER.md", "legacy profile\n")
    target = tmp_path / "dst"
    target.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(target))
    backup.run_import(Namespace(zipfile=str(archive), force=True))
    return target


def _agent(session="post-restore-1"):
    return SimpleNamespace(enabled_toolsets=None, disabled_toolsets=None, session_id=session, tools=None)


def test_restored_home_starts_an_ordinary_new_session(tmp_path, monkeypatch, fake_backend):
    target = _restore(tmp_path, monkeypatch)
    from hermes_cli.config import load_config
    agent = _agent()
    with native_memory_sentinel(target / "memories") as sentinel:
        _init_memory(agent, load_config(), False, "cli")
    sentinel.assert_untouched()
    binds = [req for op, req in fake_backend.calls if op == "bind_session"]
    assert len(binds) == 1 and binds[0].bind_intent == "new_session" and binds[0].prior_identity is None
    assert fake_backend.count("validate_session") == 0
    assert agent._memory_store is None
    assert agent._memory_service.disposition is MemoryDisposition.AUTHORITATIVE


def test_restored_home_is_fail_closed_when_the_provider_is_unavailable(tmp_path, monkeypatch, fake_backend):
    target = _restore(tmp_path, monkeypatch)
    fake_backend.store.fail_transport("negotiate")
    from hermes_cli.config import load_config
    with native_memory_sentinel(target / "memories") as sentinel:
        with pytest.raises(MemoryBlockedError):
            _init_memory(_agent(), load_config(), False, "cli")
    sentinel.assert_untouched()


def test_restored_home_is_stateless_under_the_explicit_policy(tmp_path, monkeypatch, fake_backend):
    target = _restore(tmp_path, monkeypatch, policy="stateless")
    fake_backend.store.fail_transport("negotiate")
    from hermes_cli.config import load_config
    agent = _agent()
    with native_memory_sentinel(target / "memories") as sentinel:
        _init_memory(agent, load_config(), False, "cli")
    sentinel.assert_untouched()
    assert agent._memory_service.disposition is MemoryDisposition.STATELESS


def test_legacy_native_files_in_a_restored_archive_stay_dormant(tmp_path, monkeypatch, fake_backend):
    """One sentinel spans BOTH the import and the session that follows it (§9.10 L1669)."""
    from hermes_cli.config import load_config
    agent = _agent()
    with native_memory_sentinel(tmp_path / "dst" / "memories") as sentinel:
        target = _restore_legacy(tmp_path, monkeypatch)
        _init_memory(agent, load_config(), False, "cli")
    sentinel.assert_untouched()
    assert not (target / "memories").exists()
    binds = [req for op, req in fake_backend.calls if op == "bind_session"]
    assert len(binds) == 1 and binds[0].bind_intent == "new_session"
    assert agent._memory_store is None


def test_archive_restore_reconnect_is_not_a_bind_intent():  # ledger V-23; §9.3 L1186, L1229
    ctx = {  # built exactly as tests/agent/memory_service/test_wire.py L389-401 does
        "principal_id": "ethan", "profile_id": "default", "logical_session_id": "sess-1",
        "platform": "cli", "org_id": None, "project_id": None, "repo_id": None,
        "workspace_id": None, "resolution_source": "directory",
        "canonical_directory": "C:" + chr(92) + "work" + chr(92) + "repo",
    }
    req = {"expected_provider_epoch": "ep-1", "binding_request_id": "b1",
           "bind_intent": "new_session", "requested_context": ctx, "prior_identity": None}
    assert w.BindRequest.from_wire(req).bind_intent == "new_session"
    with pytest.raises(w.WireError):
        w.BindRequest.from_wire({**req, "bind_intent": "archive_restore_reconnect"})
