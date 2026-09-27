"""S6 inventory suite A for R41 (§13.1 L2211; §14.1 row 41 L2311).

The non-agent memory command path, CLI/TUI/Gateway /memory, setup/status/off, doctor and CLI reset
route through MemoryService, or fail closed before any provider contact where the approval flow is
not yet available (ruling R41-1); nothing touches the native directory; gateway principals map
explicitly (I3, §13.1 L2191). Helpers are local (reconciliation rule 10: shared test support is
frozen)."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agent.memory_service import wire as w
from agent.memory_service.errors import MemoryBlockedError
from tests.agent.memory_service.fake_backend import FakeAuthoritativeBackend, FakeProviderStore, FakeRegistry

EPOCH_A, EPOCH_B = "EPOCHVALUEAAAA", "EPOCHVALUEBBBB"
REPO = w.ScopeRef(kind="repository", id="repo-1")


def _tool_defs(*names):
    return [{"type": "function", "function": {"name": n, "description": f"{n} tool",
                                              "parameters": {"type": "object", "properties": {}}}} for n in names]


@pytest.fixture
def native_dir():
    from tools.memory_tool import get_memory_dir
    return get_memory_dir()


@pytest.fixture
def authoritative_env(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    store = FakeProviderStore(epoch=EPOCH_A, registry=FakeRegistry(
        directories={os.path.realpath(os.getcwd()): ("repo-1", "proj-1", None)}))
    backends = []

    def factory(cfg):
        backends.append(FakeAuthoritativeBackend(store, provider="example"))
        return backends[-1]

    monkeypatch.setattr("plugins.memory.load_authoritative_backend_factory", lambda name: factory)
    exe = tmp_path / "provider.exe"
    exe.write_bytes(b"MZ")
    memory = {"provider": "example", "provider_mode": "authoritative", "provider_executable": str(exe),
              "principal_id": "ethan"}
    return SimpleNamespace(store=store, backends=backends, memory=memory, factory=factory)


def _write_config(memory_section, home=None):
    from hermes_constants import get_hermes_home
    path = (home or get_hermes_home()) / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"memory": memory_section}), encoding="utf-8")
    return path


def _agent(memory_section, **kwargs):
    from run_agent import AIAgent
    _write_config(memory_section)
    with patch("model_tools.get_tool_definitions", return_value=_tool_defs("memory", "web_search")), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=False, **kwargs)
    agent.client = MagicMock()
    return agent


def _context():
    return w.RequestedContext(principal_id="ethan", profile_id="default", logical_session_id="s-1", platform="cli",
                              org_id=None, project_id=None, repo_id=None, workspace_id=None,
                              resolution_source="directory", canonical_directory=os.path.realpath(os.getcwd()))


def _no_opaque(text, service=None):
    """§9.3 L987: neither fixture epoch nor the binding handle may appear."""
    assert EPOCH_A not in text and EPOCH_B not in text, text
    handle = getattr(getattr(service, "identity", None), "opaque_binding_b64url", None)
    assert not handle or handle not in text, text


def _cfg(memory):
    from agent.memory_service.config import resolve_memory_service_config
    return resolve_memory_service_config({"memory": memory})


def test_epoch_values_never_reach_a_blocked_message_or_warning(authoritative_env):
    """Checkpoint B carry-forward (D-R41-1): reachable from /context, /compress and the TUI persist log."""
    from agent.memory_service.authoritative import ProviderAuthoritativeMemoryService
    from agent.memory_service.service import select_memory_service
    cfg = _cfg(authoritative_env.memory)
    bound = ProviderAuthoritativeMemoryService(cfg, FakeAuthoritativeBackend(authoritative_env.store))
    bound.start(_context())
    state = bound.session_state
    authoritative_env.store.envelope_epoch_override("load_curated", EPOCH_B)
    with pytest.raises(MemoryBlockedError) as loaded:
        bound.load_curated("memory")
    authoritative_env.store.set_epoch(EPOCH_B)
    view = ProviderAuthoritativeMemoryService(cfg, FakeAuthoritativeBackend(authoritative_env.store))
    with pytest.raises(MemoryBlockedError) as resumed:
        view.resume(state)
    stateless = select_memory_service({"memory": {**authoritative_env.memory, "authoritative_failure_policy": "stateless"}},
                                      store_factory=lambda: None, session_state=state,
                                      backend_factory=lambda c: FakeAuthoritativeBackend(authoritative_env.store))
    for text in (str(loaded.value), str(resumed.value), bound.degraded_warning() or "", stateless.degraded_warning() or ""):
        _no_opaque(text)


def test_negotiate_only_negotiates_and_nothing_else(authoritative_env):
    from agent.memory_service.authoritative import ProviderAuthoritativeMemoryService
    backend = FakeAuthoritativeBackend(authoritative_env.store)
    service = ProviderAuthoritativeMemoryService(_cfg(authoritative_env.memory), backend)
    negotiation = service.negotiate_only()
    assert negotiation.provider == "example" and negotiation.selected_api_version == 1
    assert [op for op, _ in backend.calls] == ["negotiate"]
    assert service.identity is None and not service.blocked
