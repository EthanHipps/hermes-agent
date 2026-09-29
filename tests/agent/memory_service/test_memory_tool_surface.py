"""S6 surface suite for R38 (§13.1 L2204; §14.1 row 38 L2301): both memory-tool dispatch sites
route through MemoryService, with no store argument, no mirroring and no native access."""

import json
import os
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agent.memory_manager import MemoryManager
from agent.memory_service import wire as w
from agent.memory_service.service import MemoryDisposition
from tests.agent.memory_service.fake_backend import FakeAuthoritativeBackend, FakeProviderStore, FakeRegistry
from tests.agent.memory_service.native_sentinel import native_memory_sentinel

REPO = w.ScopeRef(kind="repository", id="repo-1")
ADD = {"action": "add", "target": "memory", "content": "uses pnpm"}


def _tool_defs(*names):
    return [{"type": "function", "function": {"name": n, "description": f"{n} tool",
                                              "parameters": {"type": "object", "properties": {}}}} for n in names]


def _tool_call(name, args, call_id):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def _assistant(tool_calls):
    return SimpleNamespace(content="", tool_calls=tool_calls)


@pytest.fixture
def native_dir():
    from tools.memory_tool import get_memory_dir
    return get_memory_dir()  # under conftest's isolated HERMES_HOME


@pytest.fixture
def authoritative_env(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    store = FakeProviderStore(registry=FakeRegistry(directories={os.path.realpath(os.getcwd()): ("repo-1", "proj-1", None)}))
    backends = []

    def factory(cfg):
        backends.append(FakeAuthoritativeBackend(store, provider="example"))
        return backends[-1]

    monkeypatch.setattr("plugins.memory.load_authoritative_backend_factory", lambda name: factory)
    exe = tmp_path / "provider.exe"
    exe.write_bytes(b"MZ")
    memory = {"provider": "example", "provider_mode": "authoritative", "provider_executable": str(exe), "principal_id": "ethan"}
    return SimpleNamespace(store=store, backends=backends, memory=memory)


def _write_config(memory_section):
    from hermes_constants import get_hermes_home
    path = get_hermes_home() / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"memory": memory_section}), encoding="utf-8")


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


def _spy_memory_tool(monkeypatch):
    import tools.memory_tool as mt
    real, seen = mt.memory_tool, []

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(mt, "memory_tool", spy)
    return seen


class _RecordingManager(MemoryManager):
    def __init__(self):
        super().__init__()
        self.notified, self.written = [], []

    def has_tool(self, tool_name):
        return False

    def notify_memory_tool_write(self, *args, **kwargs):
        self.notified.append(args)

    def on_memory_write(self, *args, **kwargs):
        self.written.append(args)


def _repo_texts(env):
    return sorted(r.text for r in env.store.records_for(REPO, "memory"))


def test_sequential_site_commits_through_the_service_only(authoritative_env, native_dir, monkeypatch):
    agent = _agent(authoritative_env.memory)
    assert agent._memory_service.disposition is MemoryDisposition.AUTHORITATIVE and agent._memory_store is None
    seen, manager = _spy_memory_tool(monkeypatch), _RecordingManager()
    agent._memory_manager = manager
    backend = authoritative_env.backends[-1]
    messages = []
    with native_memory_sentinel(native_dir) as sentinel:
        agent._execute_tool_calls_sequential(_assistant([_tool_call("memory", ADD, "mem-seq")]), messages, "task-1")
    sentinel.assert_untouched()
    assert messages[-1]["tool_call_id"] == "mem-seq" and '"success": true' in messages[-1]["content"]
    assert _repo_texts(authoritative_env) == ["uses pnpm"] and backend.count("commit_curated") == 1
    assert "store" not in seen[-1] and seen[-1]["service"] is agent._memory_service
    assert manager.notified == [] and manager.written == []


def test_concurrent_site_commits_through_the_service_only(authoritative_env, native_dir, monkeypatch):
    agent = _agent(authoritative_env.memory)
    seen, manager = _spy_memory_tool(monkeypatch), _RecordingManager()
    agent._memory_manager = manager
    with native_memory_sentinel(native_dir) as sentinel:
        out = json.loads(agent._invoke_tool("memory", ADD, "task-1", tool_call_id="mem-conc"))
    sentinel.assert_untouched()
    assert out["success"] is True and _repo_texts(authoritative_env) == ["uses pnpm"]
    assert "store" not in seen[-1] and manager.notified == [] and manager.written == []


def test_provider_tools_are_neither_exposed_nor_dispatchable(authoritative_env, native_dir):
    """Ruling R38-14 (a): §9.7 L1614 registration and dispatch, §9.10 L1678."""
    agent = _agent(authoritative_env.memory)
    assert agent._memory_manager is None
    assert {t["function"]["name"] for t in agent.tools} <= {"memory", "web_search"}
    with patch("model_tools.handle_function_call", return_value=json.dumps({"error": "Unknown tool"})) as registry:
        agent._invoke_tool("ext_memory_search", {}, "task-1")
    registry.assert_called_once()  # fell through to the registry: no provider dispatch path exists


def test_stateless_session_refuses_without_native_access(authoritative_env, native_dir):
    authoritative_env.store.fail_transport("negotiate")
    agent = _agent({**authoritative_env.memory, "authoritative_failure_policy": "stateless"})
    assert agent._memory_service.disposition is MemoryDisposition.STATELESS
    with native_memory_sentinel(native_dir) as sentinel:
        out = json.loads(agent._invoke_tool("memory", ADD, "task-1"))
    sentinel.assert_untouched()
    assert out["success"] is False and "stateless" in out["error"]


def test_cron_and_background_review_are_refused_before_any_load(authoritative_env, native_dir):
    """Ruling R38-3 (b): §9.6 L1583 "reject before read/write"."""
    from tools.skill_provenance import reset_current_write_origin, set_current_write_origin
    agent = _agent(authoritative_env.memory)
    backend = authoritative_env.backends[-1]
    loads = backend.count("load_curated")
    agent.platform = "cron"
    assert json.loads(agent._invoke_tool("memory", ADD, "task-1"))["success"] is False
    agent.platform = "cli"
    token = set_current_write_origin("background_review")
    try:
        assert json.loads(agent._invoke_tool("memory", ADD, "task-1"))["success"] is False
    finally:
        reset_current_write_origin(token)
    assert backend.count("load_curated") == loads and _repo_texts(authoritative_env) == []


def test_approval_required_write_leaves_no_native_pending_record(authoritative_env, native_dir):
    """R38 refused here until R39. R39 stages it as a handle-only approval under the session (ruling R39-5)."""
    from agent.memory_service.approval_store import list_pending_approvals
    from tools import write_approval as wa
    agent = _agent(authoritative_env.memory)
    with native_memory_sentinel(native_dir) as sentinel:
        out = json.loads(agent._invoke_tool("memory", {"action": "add", "target": "user", "content": "terse"}, "task-1"))
    sentinel.assert_untouched()
    assert out["staged"] is True and out["approval_required"] == ["target_user"] and wa.list_pending(wa.MEMORY) == []
    assert [pid for pid, _ in list_pending_approvals()] == [out["pending_id"]]


def test_registry_path_for_memory_touches_nothing(authoritative_env, native_dir):
    import model_tools
    _agent(authoritative_env.memory)
    backend = authoritative_env.backends[-1]
    stages = backend.count("stage_curated")
    with native_memory_sentinel(native_dir) as sentinel:
        out = json.loads(model_tools.handle_function_call("memory", ADD, "task-1"))
    sentinel.assert_untouched()
    assert "agent loop" in out["error"] and backend.count("stage_curated") == stages


def test_additive_agent_dispatch_is_unchanged(tmp_path, monkeypatch):
    """§9.10 L1668: store=, and mirroring, exactly as at 5c583156f7."""
    monkeypatch.chdir(tmp_path)
    agent = _agent({"memory_enabled": True, "user_profile_enabled": True})
    assert agent._memory_service.disposition is MemoryDisposition.BUILTIN
    seen, manager = _spy_memory_tool(monkeypatch), _RecordingManager()
    agent._memory_manager = manager
    out = json.loads(agent._invoke_tool("memory", ADD, "task-1"))
    assert out["success"] is True
    assert seen[-1]["store"] is agent._memory_store and "service" not in seen[-1]
    assert len(manager.notified) == 1


def test_consolidation_cap_is_per_turn(authoritative_env):
    from agent.turn_context import _reset_per_turn_agent_state
    agent = _agent(authoritative_env.memory)

    def miss():
        return json.loads(agent._invoke_tool("memory", {"action": "remove", "target": "memory",
                                                        "old_text": f"absent {uuid.uuid4().hex}"}, "task-1"))

    outs = [miss() for _ in range(4)]
    assert outs[-1].get("done") is True  # the 4th zero-match is terminal (#42405)
    _reset_per_turn_agent_state(agent)
    assert "done" not in miss()
