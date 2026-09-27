"""S6 Honcho native reads (R46; spec §9.7 "Honcho provider native reads", §9.1 L950, §9.10 L1691).

Honcho uploads MEMORY.md/USER.md only through SessionMigrationMixin.migrate_memory_files, from its
own session init or the ``hermes honcho migrate`` CLI, and it is initialized only as the ADDITIVE
``memory.provider``. Ruling R46-5: in an authoritative home that holds by construction.
``_init_memory`` sets the provider config only on the additive branch, and naming an additive
provider as the authoritative one is a configuration error. So this suite proves it with the
sentinel and a mutation check instead of adding a guard.
"""

from unittest.mock import MagicMock, patch

import pytest
import yaml

from agent.memory_service.config import MemoryConfigurationError
from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.session_migration import SessionMigrationMixin
from tests.agent.memory_service.native_sentinel import native_memory_sentinel


@pytest.fixture
def native_dir():
    from tools.memory_tool import get_memory_dir
    native = get_memory_dir()   # under conftest's isolated HERMES_HOME
    native.mkdir(parents=True, exist_ok=True)
    (native / "MEMORY.md").write_text("dormant native memory\n", encoding="utf-8")
    (native / "USER.md").write_text("dormant native profile\n", encoding="utf-8")
    return native


@pytest.fixture
def honcho_calls(monkeypatch):
    """Spies on the real Honcho class; every provider lookup by name returns an instance of it."""
    calls = {"initialize": [], "migrate": []}
    monkeypatch.setattr(HonchoMemoryProvider, "is_available", lambda self: True)
    monkeypatch.setattr(HonchoMemoryProvider, "initialize",
                        lambda self, session_id, **kw: calls["initialize"].append(session_id))
    monkeypatch.setattr(SessionMigrationMixin, "migrate_memory_files",
                        lambda self, key, directory: calls["migrate"].append(directory) or False)
    # _init_memory imports load_memory_provider inside the function: patch where production reads.
    monkeypatch.setattr("plugins.memory.load_memory_provider",
                        lambda name, **kw: HonchoMemoryProvider() if name == "honcho" else None)
    return calls


def _tool_defs(*names):
    return [{"type": "function", "function": {"name": n, "description": f"{n} tool",
                                              "parameters": {"type": "object", "properties": {}}}} for n in names]


def _write_config(memory_section):
    from hermes_constants import get_hermes_home
    path = get_hermes_home() / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"memory": memory_section}), encoding="utf-8")


def _agent(memory_section):
    from run_agent import AIAgent
    _write_config(memory_section)
    with patch("model_tools.get_tool_definitions", return_value=_tool_defs("memory", "web_search")), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=False)
    agent.client = MagicMock()
    return agent


def _files(native):
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(native.iterdir())}


def test_additive_honcho_is_initialized(honcho_calls):
    """Non-vacuity: with the same spies, an additive home does initialize Honcho."""
    agent = _agent({"provider": "honcho"})
    assert honcho_calls["initialize"] == [agent.session_id]
    assert agent._memory_manager is not None


@pytest.mark.parametrize("policy", [None, "stateless"])
def test_authoritative_home_never_initializes_honcho_or_reads_native_files(tmp_path, native_dir, honcho_calls, policy):
    exe = tmp_path / "provider.exe"
    exe.write_bytes(b"MZ")
    section = {"provider": "honcho", "provider_mode": "authoritative",
               "provider_executable": str(exe), "principal_id": "ethan"}
    if policy:
        section["authoritative_failure_policy"] = policy
    before = _files(native_dir)
    with native_memory_sentinel(native_dir) as sentinel:
        with pytest.raises(MemoryConfigurationError):    # Honcho exports no create_authoritative_backend
            _agent(section)
    sentinel.assert_untouched()
    assert honcho_calls == {"initialize": [], "migrate": []}
    assert _files(native_dir) == before
