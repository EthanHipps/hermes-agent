"""§9.1: provider-defined curated-memory tools are suppressed in authoritative (and stateless) sessions."""

from types import SimpleNamespace

from agent.memory_manager import MemoryManager, inject_memory_provider_tools, memory_provider_tools_exposed
from agent.memory_service.config import resolve_memory_service_config
from agent.memory_service.service import MemoryDisposition, StatelessMemoryService
from tests.agent.test_memory_provider import FakeMemoryProvider


def _agent_with_provider_tools(service):
    manager = MemoryManager()
    provider = FakeMemoryProvider(name="ext", tools=[{"name": "ext_memory_search", "description": "x", "parameters": {"type": "object", "properties": {}}}])
    manager.add_provider(provider)
    return SimpleNamespace(_memory_manager=manager, tools=[], enabled_toolsets=None, disabled_toolsets=None, _memory_service=service)


def test_additive_session_keeps_provider_tools():
    agent = _agent_with_provider_tools(SimpleNamespace(disposition=MemoryDisposition.BUILTIN))
    assert memory_provider_tools_exposed(agent)
    assert inject_memory_provider_tools(agent) == 1
    agent = _agent_with_provider_tools(None)
    assert inject_memory_provider_tools(agent) == 1


def test_authoritative_session_withholds_provider_tools():
    agent = _agent_with_provider_tools(SimpleNamespace(disposition=MemoryDisposition.AUTHORITATIVE))
    assert not memory_provider_tools_exposed(agent)
    assert inject_memory_provider_tools(agent) == 0
    assert agent.tools == []


def test_stateless_session_withholds_provider_tools():
    service = StatelessMemoryService(resolve_memory_service_config({}), reason="test")
    agent = _agent_with_provider_tools(service)
    assert not memory_provider_tools_exposed(agent)
    assert inject_memory_provider_tools(agent) == 0
