"""Authoritative selection never starts an additive provider (section 9.9 L1668).

``memory.provider`` names a provider plugin and the mode is authoritative.  The
named plugin is the authoritative backend only; it is never also loaded as a
legacy additive sidecar.  Sentinels on ``plugins.memory.load_memory_provider``
and on ``MemoryManager`` record any construction, and a sentinel manager
attached after init records initialize, mirroring and tool dispatch.
"""

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agent.memory_service.errors import MemoryBlockedError
from agent.memory_service.service import MemoryDisposition
from tests.agent.memory_service.fake_backend import FakeAuthoritativeBackend, FakeProviderStore, FakeRegistry
from tests.agent.memory_service.native_sentinel import native_memory_sentinel

ADD = {"action": "add", "target": "memory", "content": "uses pnpm", "memory_admission": "declarative"}


class AdditiveStarted(AssertionError):
    """Raised by a sentinel: an additive provider was constructed, initialized or mirrored to."""


class SentinelManager:
    """Any attribute access means the host drove an additive provider; it is recorded, then raised."""

    def __init__(self, events):
        object.__setattr__(self, "_events", events)

    def __getattr__(self, name):
        self._events.append(("SentinelManager", name))
        raise AdditiveStarted(f"MemoryManager.{name} used in an authoritative session")

    def __bool__(self):
        return True


@pytest.fixture
def sentinels(monkeypatch):
    events = []

    def load_memory_provider(name, **_):
        events.append(("load_memory_provider", name))
        raise AdditiveStarted(f"load_memory_provider({name!r})")

    class Manager:
        def __init__(self, *args, **kwargs):
            events.append(("MemoryManager", args, kwargs))
            raise AdditiveStarted("MemoryManager constructed")

    monkeypatch.setattr("plugins.memory.load_memory_provider", load_memory_provider)
    monkeypatch.setattr("agent.memory_manager.MemoryManager", Manager)
    return events


@pytest.fixture
def authoritative_env(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    store = FakeProviderStore(registry=FakeRegistry(directories={os.path.realpath(os.getcwd()): ("repo-1", "proj-1", None)}))
    monkeypatch.setattr("plugins.memory.load_authoritative_backend_factory",
                        lambda name: (lambda cfg: FakeAuthoritativeBackend(store, provider="example")))
    exe = tmp_path / "provider.exe"
    exe.write_bytes(b"MZ")
    memory = {"provider": "example", "provider_mode": "authoritative", "provider_executable": str(exe),
              "principal_id": "ethan"}
    return SimpleNamespace(store=store, memory=memory)


def _agent(memory_section):
    from hermes_constants import get_hermes_home

    path = get_hermes_home() / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"memory": memory_section}), encoding="utf-8")
    from run_agent import AIAgent  # imported after config.yaml exists: the import initializes the home

    tools = [{"type": "function", "function": {"name": "memory", "description": "memory tool",
                                               "parameters": {"type": "object", "properties": {}}}}]
    with patch("model_tools.get_tool_definitions", return_value=tools), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=False)
    agent.client = MagicMock()
    return agent


@pytest.fixture
def native_dir():
    from tools.memory_tool import get_memory_dir
    return get_memory_dir()


@pytest.mark.parametrize("policy", ["fail_closed", "stateless"])
def test_authoritative_selection_never_starts_an_additive_provider(authoritative_env, sentinels, native_dir, policy):
    """Healthy authoritative and degraded-to-stateless starts alike: the named plugin is never loaded as a sidecar."""
    stateless = policy == "stateless"
    if stateless:
        authoritative_env.store.fail_transport("negotiate")
    with native_memory_sentinel(native_dir) as native:
        agent = _agent({**authoritative_env.memory, "authoritative_failure_policy": policy})
        expected = MemoryDisposition.STATELESS if stateless else MemoryDisposition.AUTHORITATIVE
        assert agent._memory_service.disposition is expected
        assert sentinels == [] and agent._memory_manager is None

        # A manager that appears later (a double for any initialize, mirroring or dispatch path) is never driven.
        agent._memory_manager = SentinelManager(sentinels)
        out = json.loads(agent._invoke_tool("memory", ADD, "task-1"))
    native.assert_untouched()
    if stateless:
        assert out["success"] is False and "stateless" in out["error"]
    else:
        assert out["success"] is True
    assert sentinels == []


def test_an_authoritative_failure_is_not_retried_as_an_additive_sidecar(authoritative_env, sentinels):
    """A fail_closed startup failure raises; the plugin name is not then loaded as a sidecar either."""
    authoritative_env.store.fail_transport("negotiate")
    with pytest.raises(MemoryBlockedError) as caught:
        _agent(authoritative_env.memory)
    assert not isinstance(caught.value, AdditiveStarted)
    assert sentinels == []
