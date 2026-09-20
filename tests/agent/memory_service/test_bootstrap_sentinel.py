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


def test_disabled_additive_store_touches_nothing(tmp_path, native_dir):
    """Regression (fix-round defect 1): a fully-disabled additive config must
    leave agent._memory_store None and touch no native file.

    The pre-router code only built a MemoryStore when ``memory_enabled or
    user_profile_enabled``; a naive router wiring builds (and loads) one
    unconditionally inside store_factory(), which would leave a live but
    disabled store on agent._memory_store. That matters beyond this call:
    agent/system_prompt.py's post-compression reload does ``if
    agent._memory_store: agent._memory_store.load_from_disk()`` with no flag
    check, so a non-None disabled store would start touching the native
    directory on every later compression event in a session that had memory
    fully disabled -- something a memory-*enabled* sentinel test can never
    catch, since all the other tests in this file run with at least one flag
    on.
    """
    agent = _agent()
    cfg = {"memory": {"memory_enabled": False, "user_profile_enabled": False}}
    with native_memory_sentinel(native_dir) as sentinel:
        _init_memory(agent, cfg, False, "cli")
    sentinel.assert_untouched()
    assert agent._memory_store is None
    assert agent._memory_service.disposition is MemoryDisposition.BUILTIN


def test_malformed_additive_store_factory_failure_degrades_silently(tmp_path, native_dir):
    """Regression (fix-round defect 2): an unexpected exception raised inside
    store_factory() itself -- not a MemoryConfigurationError from config
    resolution, which init_memory_service already handles -- must degrade
    silently in additive mode, matching the pre-router "memory is optional --
    don't break agent init" contract, rather than crash agent construction.

    ``nudge_interval: "not-a-number"`` makes _build_native_store's
    ``int(mem_config.get("nudge_interval", 10))`` raise ValueError; nothing
    upstream of store_factory() validates that field. The failure happens
    inside BuiltinMemoryService(cfg, store_factory())'s own construction, so
    there is no partial service to salvage -- agent._memory_service stays
    None, same as when init_memory_service returns (None, exc) for a
    malformed config. That's the existing, accepted "degrade to no memory"
    outcome; only "does it raise" is this defect's contract.
    """
    agent = _agent()
    cfg = {"memory": {"nudge_interval": "not-a-number"}}
    _init_memory(agent, cfg, False, "cli")  # must not raise
    assert agent._memory_store is None
    assert agent._memory_service is None


def test_malformed_config_under_authoritative_mode_still_raises(tmp_path, native_dir, monkeypatch):
    """Paired case for the regression above: the re-raise decision in
    _init_memory is scoped by requests_authoritative_mode(_agent_cfg), not by
    whether store_factory() happened to run.

    nudge_interval is additive-only -- MemoryServiceConfig carries no such
    field, and store_factory() is never invoked on the authoritative branch
    (service.py:252-255) -- so it cannot by itself raise here the way it does
    above; a real authoritative-path failure (the same broken-backend-factory
    shape as test_provider_failure_fail_closed_never_falls_back_to_native) is
    what exercises the exception path. Carrying the same malformed
    nudge_interval field alongside it proves that field's presence doesn't
    accidentally influence which branch of the new try/except fires.
    """
    def broken(cfg):
        raise RuntimeError("provider unreachable")

    monkeypatch.setattr("plugins.memory.load_authoritative_backend_factory", lambda name: broken)
    agent = _agent()
    cfg = _authoritative_cfg(tmp_path, nudge_interval="not-a-number")
    with pytest.raises(Exception):
        _init_memory(agent, cfg, False, "cli")
