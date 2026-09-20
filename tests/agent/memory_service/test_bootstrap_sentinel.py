"""R37 gate: authoritative agent init touches no native file (\u00a79.10 L1685).

Proven with a filesystem sentinel, not inferred from output.
"""

import os
from types import SimpleNamespace

import pytest

from agent.agent_init import _init_memory
from agent.memory_service.service import MemoryDisposition
from tests.agent.memory_service.fake_backend import (
    FakeAuthoritativeBackend, FakeProviderStore, FakeRegistry,
)
from tests.agent.memory_service.native_sentinel import native_memory_sentinel


@pytest.fixture
def native_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.memory_tool import get_memory_dir
    return get_memory_dir()


def _agent():
    return SimpleNamespace(enabled_toolsets=None, disabled_toolsets=None,
                           session_id="sess-1", tools=None)


def _authoritative_cfg(tmp_path, **extra):
    exe = tmp_path / "p.exe"
    exe.write_bytes(b"MZ")
    section = {"provider": "example", "provider_mode": "authoritative",
               "provider_executable": str(exe), "principal_id": "ethan"}
    section.update(extra)
    return {"memory": section}


@pytest.fixture
def fake_backend(monkeypatch, tmp_path):
    """Inject the R36 fake at the REAL discovery seam, with the cwd registered.

    ``service._default_backend_factory`` does ``from plugins.memory import
    load_authoritative_backend_factory`` *inside* the function, so this patches
    exactly where production reads, at call time, and still exercises the
    discovery path instead of bypassing it with a test-only global.

    The registry matters here. ``_init_memory`` passes no ``working_directory``,
    so ``build_requested_context`` derives it from ``os.getcwd()`` --  and
    ``FakeRegistry.resolve`` is an EXACT dict lookup on
    ``canonical_directory`` (``fake_backend.py:219``), not a prefix match. The
    fake's default registry knows only the literal ``C:\\work\\repo``, so an
    unregistered cwd is a hard ``invalid_request`` bind failure. Registering the
    realpath'd cwd is what lets these tests exercise the real derivation instead
    of hardcoding a path production would never produce.
    """
    monkeypatch.chdir(tmp_path)
    registry = FakeRegistry(directories={os.path.realpath(os.getcwd()): ("repo-1", "proj-1", None)})
    backend = FakeAuthoritativeBackend(FakeProviderStore(registry=registry), provider="example")
    monkeypatch.setattr("plugins.memory.load_authoritative_backend_factory",
                        lambda name: (lambda cfg: backend))
    return backend


def test_clean_start_touches_nothing(tmp_path, native_dir, fake_backend):
    """\u00a79.10: authoritative mode touches no native file on clean start."""
    assert not native_dir.exists()
    agent = _agent()
    with native_memory_sentinel(native_dir) as sentinel:
        _init_memory(agent, _authoritative_cfg(tmp_path), False, "cli")
    sentinel.assert_untouched()
    assert agent._memory_store is None
    assert agent._memory_service.disposition is MemoryDisposition.AUTHORITATIVE


def test_existing_file_start_touches_nothing(tmp_path, native_dir, fake_backend):
    """Dormant files MAY remain on disk; their existence MUST be observationally irrelevant."""
    native_dir.mkdir(parents=True)
    (native_dir / "MEMORY.md").write_text("stale native memory\n", encoding="utf-8")
    (native_dir / "USER.md").write_text("stale native profile\n", encoding="utf-8")
    agent = _agent()
    with native_memory_sentinel(native_dir) as sentinel:
        _init_memory(agent, _authoritative_cfg(tmp_path), False, "cli")
    sentinel.assert_untouched()
    assert agent._memory_store is None
    assert (native_dir / "MEMORY.md").read_text(encoding="utf-8") == "stale native memory\n"


def test_provider_failure_fail_closed_never_falls_back_to_native(tmp_path, native_dir, monkeypatch):
    """I1 + \u00a79.10: a provider failure must not produce a native fallback."""
    native_dir.mkdir(parents=True)
    (native_dir / "MEMORY.md").write_text("stale\n", encoding="utf-8")

    def broken(cfg):
        raise RuntimeError("provider unreachable")

    monkeypatch.setattr("plugins.memory.load_authoritative_backend_factory", lambda name: broken)
    agent = _agent()
    with native_memory_sentinel(native_dir) as sentinel:
        with pytest.raises(Exception):
            _init_memory(agent, _authoritative_cfg(tmp_path), False, "cli")
    sentinel.assert_untouched()
    assert getattr(agent, "_memory_store", None) is None


def test_provider_failure_stateless_degrades_without_touching_native(tmp_path, native_dir, monkeypatch):
    native_dir.mkdir(parents=True)
    (native_dir / "MEMORY.md").write_text("stale\n", encoding="utf-8")
    store = FakeProviderStore()
    store.fail_transport("negotiate")  # the knob lives on the STORE, not the backend
    backend = FakeAuthoritativeBackend(store, provider="example")
    monkeypatch.setattr("plugins.memory.load_authoritative_backend_factory",
                        lambda name: (lambda cfg: backend))
    agent = _agent()
    with native_memory_sentinel(native_dir) as sentinel:
        _init_memory(agent, _authoritative_cfg(tmp_path, authoritative_failure_policy="stateless"),
                     False, "cli")
    sentinel.assert_untouched()
    assert agent._memory_service.disposition is MemoryDisposition.STATELESS
    assert agent._memory_store is None


def test_additive_still_loads_natively(tmp_path, native_dir):
    """\u00a79.10 first bullet: absent provider_mode retains additive behaviour."""
    native_dir.mkdir(parents=True)
    (native_dir / "MEMORY.md").write_text("- a real memory\n", encoding="utf-8")
    agent = _agent()
    with native_memory_sentinel(native_dir) as sentinel:
        _init_memory(agent, {"memory": {}}, False, "cli")
    assert sentinel.accesses, "additive mode must still read the native store"
    assert agent._memory_store is not None
    assert agent._memory_service.disposition is MemoryDisposition.BUILTIN
