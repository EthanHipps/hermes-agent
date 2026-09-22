"""R37: host facts -> RequestedContext -> MemoryService, before any native store exists."""

import os

import pytest

from agent.memory_service.bootstrap import build_requested_context
from agent.memory_service.config import resolve_memory_service_config


def _authoritative(tmp_path, **extra):
    exe = tmp_path / "p.exe"
    exe.write_bytes(b"MZ")
    section = {"provider": "example", "provider_mode": "authoritative",
               "provider_executable": str(exe), "principal_id": "ethan"}
    section.update(extra)
    return resolve_memory_service_config({"memory": section})


def test_context_carries_host_facts_and_validates(tmp_path):
    cfg = _authoritative(tmp_path)
    ctx = build_requested_context(
        cfg, logical_session_id="sess-1", platform="cli",
        profile_id="default", working_directory=str(tmp_path),
    )
    ctx.validate()  # raises WireError if the shape is wrong
    assert ctx.principal_id == "ethan"
    assert ctx.logical_session_id == "sess-1"
    assert ctx.platform == "cli"
    assert ctx.profile_id == "default"


def test_scopes_are_provider_resolved_not_host_guessed(tmp_path):
    """D-R37-1 / §4.2: Hermes names the directory; the provider resolves org/project/repo."""
    cfg = _authoritative(tmp_path)
    ctx = build_requested_context(
        cfg, logical_session_id="s", platform="cli",
        profile_id="default", working_directory=str(tmp_path),
    )
    assert ctx.resolution_source == "directory"
    assert (ctx.org_id, ctx.project_id, ctx.repo_id, ctx.workspace_id) == (None, None, None, None)


def test_canonical_directory_is_fully_resolved(tmp_path):
    """§4.2 matches the longest canonical registered repo path; a relative or
    unnormalized path could match the wrong registration or none at all."""
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    cfg = _authoritative(tmp_path)
    ctx = build_requested_context(
        cfg, logical_session_id="s", platform="cli", profile_id="default",
        working_directory=str(tmp_path / "a" / ".." / "a" / "b"),
    )
    assert ctx.canonical_directory == os.path.realpath(str(nested))


def test_explicit_directory_overrides_the_runtime_directory(tmp_path, monkeypatch):
    from agent.runtime_cwd import clear_session_cwd, set_session_cwd

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setenv("TERMINAL_CWD", str(runtime))
    set_session_cwd(str(runtime))
    try:
        ctx = build_requested_context(
            _authoritative(tmp_path), logical_session_id="s", platform="cli",
            profile_id="default", working_directory=str(tmp_path),
        )
        assert ctx.canonical_directory == os.path.realpath(tmp_path)
    finally:
        clear_session_cwd()


def test_working_directory_defaults_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _authoritative(tmp_path)
    ctx = build_requested_context(cfg, logical_session_id="s", platform="cli", profile_id="default")
    assert ctx.canonical_directory == os.path.realpath(str(tmp_path))


def test_profile_id_defaults_to_the_active_profile(tmp_path):
    cfg = _authoritative(tmp_path)
    ctx = build_requested_context(cfg, logical_session_id="s", platform="cli",
                                  working_directory=str(tmp_path))
    assert ctx.profile_id  # non-empty; get_active_profile_name() never returns ""


from agent.memory_service.bootstrap import init_memory_service
from agent.memory_service.config import MemoryConfigurationError
from agent.memory_service.errors import MemoryBlockedError
from agent.memory_service.service import MemoryDisposition
from tests.agent.memory_service.fake_backend import FakeAuthoritativeBackend, FakeProviderStore, FakeRegistry


class _StoreSpy:
    def __init__(self):
        self.built = 0

    def __call__(self):
        self.built += 1
        from tools.memory_tool import MemoryStore
        return MemoryStore()


def _raw(tmp_path, **extra):
    exe = tmp_path / "p.exe"
    exe.write_bytes(b"MZ")
    section = {"provider": "example", "provider_mode": "authoritative",
               "provider_executable": str(exe), "principal_id": "ethan"}
    section.update(extra)
    return {"memory": section}


def _fake_factory(directory):
    registry = FakeRegistry(directories={os.path.realpath(directory): ("repo-1", "proj-1", None)})
    store = FakeProviderStore(registry=registry)
    return lambda cfg: FakeAuthoritativeBackend(store, provider="example")


def test_additive_selects_builtin_and_builds_the_store_once(tmp_path):
    spy = _StoreSpy()
    service, swallowed = init_memory_service(
        {}, logical_session_id="s", platform="cli", store_factory=spy, working_directory=str(tmp_path))
    assert service.disposition is MemoryDisposition.BUILTIN
    assert spy.built == 1 and swallowed is None


def test_authoritative_selects_provider_and_never_builds_the_store(tmp_path):
    spy = _StoreSpy()
    service, swallowed = init_memory_service(
        _raw(tmp_path), logical_session_id="s", platform="cli", store_factory=spy,
        working_directory=str(tmp_path), backend_factory=_fake_factory(tmp_path))
    assert service.disposition is MemoryDisposition.AUTHORITATIVE
    assert spy.built == 0 and swallowed is None
    assert service.identity is not None and service.identity.principal_id == "ethan"


def test_authoritative_config_error_propagates(tmp_path):
    """§9.1 L944: Hermes MUST NOT reinterpret a bad authoritative config as additive."""
    spy = _StoreSpy()
    raw = _raw(tmp_path)
    del raw["memory"]["principal_id"]
    with pytest.raises(MemoryConfigurationError, match="principal_id"):
        init_memory_service(raw, logical_session_id="s", platform="cli", store_factory=spy,
                            working_directory=str(tmp_path), backend_factory=_fake_factory(tmp_path))
    assert spy.built == 0


def test_additive_config_error_is_swallowed_not_raised(tmp_path):
    """§9.10 first bullet: absent provider_mode retains today's degrade-and-boot behaviour."""
    spy = _StoreSpy()
    service, swallowed = init_memory_service(
        {"memory": "not-a-mapping"}, logical_session_id="s", platform="cli",
        store_factory=spy, working_directory=str(tmp_path))
    assert service is None
    assert isinstance(swallowed, MemoryConfigurationError)
    assert spy.built == 0


def test_provider_failure_fail_closed_raises_and_never_falls_back(tmp_path):
    """I1: the session does not become additive because the provider failed."""
    spy = _StoreSpy()

    def broken(cfg):
        raise RuntimeError("provider unreachable")

    with pytest.raises(Exception) as exc:
        init_memory_service(_raw(tmp_path), logical_session_id="s", platform="cli",
                            store_factory=spy, working_directory=str(tmp_path),
                            backend_factory=broken)
    assert not isinstance(exc.value, MemoryConfigurationError)
    assert spy.built == 0


def test_provider_failure_stateless_degrades_and_never_falls_back(tmp_path):
    spy = _StoreSpy()
    store = FakeProviderStore()
    store.fail_transport("negotiate")  # the knob lives on the STORE, not the backend
    backend = FakeAuthoritativeBackend(store, provider="example")
    service, swallowed = init_memory_service(
        _raw(tmp_path, authoritative_failure_policy="stateless"),
        logical_session_id="s", platform="cli", store_factory=spy,
        working_directory=str(tmp_path), backend_factory=lambda cfg: backend)
    assert service.disposition is MemoryDisposition.STATELESS
    assert service.prompt_block("memory") is None
    assert spy.built == 0 and swallowed is None
