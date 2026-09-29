"""S6 inventory suite C (§13.1 L2211; §14.1 row 43 L2313): background review, /btw, cron-platform agents,
delegated children and memory-less workers receive MemoryService only as an explicit dependency
(§9.7 L1610, L1613, L1625; §9.6 L1585; contract C6b-7)."""

import json
import os
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agent.memory_service import wire as w
from agent.memory_service.host_state import host_state_dir
from agent.memory_service.service import MemoryDisposition
from agent.memory_service.view import (LimitedMemoryView, UnboundMemoryService, capture_parent_memory,
                                       is_injected, open_fork_view)
from tests.agent.memory_service.fake_backend import FakeAuthoritativeBackend, FakeProviderStore, FakeRegistry
from tests.agent.memory_service.native_sentinel import native_memory_sentinel

REPO = w.ScopeRef(kind="repository", id="repo-1")
ADD = {"action": "add", "target": "memory", "content": "uses pnpm"}


def _tool_defs(*names):
    return [{"type": "function", "function": {"name": n, "description": f"{n} tool",
                                              "parameters": {"type": "object", "properties": {}}}} for n in names]


@contextmanager
def _construction_patches(*tools):
    with patch("model_tools.get_tool_definitions", return_value=_tool_defs(*tools)), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI"):
        yield


def _native_dir():
    from tools.memory_tool import get_memory_dir
    return get_memory_dir()


def _records():
    root = host_state_dir()
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.glob("*.json")} if root.exists() else {}


@contextmanager
def _origin(origin="background_review", attended=False):
    from tools.skill_provenance import (reset_current_write_origin, reset_review_attended,
                                        set_current_write_origin, set_review_attended)
    t_origin, t_att = set_current_write_origin(origin), set_review_attended(attended)
    try:
        yield
    finally:
        reset_review_attended(t_att)
        reset_current_write_origin(t_origin)


class Env:
    def __init__(self, tmp_path, monkeypatch, *, additive=False, **memory_extra):
        repo = tmp_path / "repo"
        repo.mkdir()
        monkeypatch.chdir(repo)
        self.store = FakeProviderStore(registry=FakeRegistry(
            directories={os.path.realpath(os.getcwd()): ("repo-1", "proj-1", None)}))
        self.backends, self.factory_calls = [], 0
        monkeypatch.setattr("plugins.memory.load_authoritative_backend_factory", lambda name: self._factory)
        exe = tmp_path / "provider.exe"
        exe.write_bytes(b"MZ")
        self.memory = dict(memory_extra) if additive else {
            "provider": "example", "provider_mode": "authoritative", "provider_executable": str(exe),
            "principal_id": "ethan", **memory_extra}
        from hermes_constants import get_hermes_home
        config_path = get_hermes_home() / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump({"memory": self.memory}), encoding="utf-8")
        from hermes_state import SessionDB
        self.db = SessionDB(db_path=tmp_path / "state.db")

    def _factory(self, cfg):
        self.factory_calls += 1
        self.backends.append(FakeAuthoritativeBackend(self.store, provider=cfg.provider))
        return self.backends[-1]

    def count(self, op):
        return sum(b.count(op) for b in self.backends)

    def agent(self, session_id="parent-1", *, tools=("memory", "web_search"), **kwargs):
        from run_agent import AIAgent
        kwargs.setdefault("enabled_toolsets", ["memory"])
        with _construction_patches(*tools):
            agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                            skip_context_files=True, session_id=session_id, session_db=self.db, **kwargs)
        agent.client = MagicMock()
        agent._use_prompt_caching = agent.save_trajectories = agent.compression_enabled = False
        return agent

    def fork(self, parent, **kwargs):
        from agent.background_review import build_cache_parity_fork
        with _construction_patches("memory", "web_search"):
            fork, _rt, _routed = build_cache_parity_fork(parent, {}, max_iterations=2, **kwargs)
        return fork


# ---- A. the seam (Task 3) -------------------------------------------------------------------------

def test_injected_service_is_installed_without_resolving_anything(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    parent = env.agent()
    view = open_fork_view(capture_parent_memory(parent), surface="background_review")
    binds, calls, records = env.count("bind_session"), env.factory_calls, _records()
    with native_memory_sentinel(_native_dir()) as sentinel:
        agent = env.agent("injected-1", memory_service=view)
    sentinel.assert_untouched()
    assert agent._memory_service is view and is_injected(agent)
    assert agent._memory_store is None and agent._memory_manager is None
    assert env.count("bind_session") == binds and env.factory_calls == calls and _records() == records


def test_a_builtin_service_cannot_be_injected(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch, additive=True)
    additive = env.agent("additive-1")
    with pytest.raises(ValueError):
        env.agent("injected-2", memory_service=additive._memory_service)


def test_an_injected_service_is_never_re_resolved_or_recorded(tmp_path, monkeypatch):
    from agent.memory_service.lifecycle import follow_session_binding, record_curated_prompt
    env = Env(tmp_path, monkeypatch)
    parent = env.agent()
    view = open_fork_view(capture_parent_memory(parent), surface="background_review")
    agent = env.agent("injected-3", memory_service=view)
    agent.session_id = parent.session_id                    # a fork shares the parent's session id
    calls, records = env.factory_calls, _records()
    follow_session_binding(agent)
    record_curated_prompt(agent, "a fork's own prompt")
    assert agent._memory_service is view and agent._memory_session_key == parent.session_id
    assert env.factory_calls == calls and _records() == records   # the parent's record is untouched


@pytest.mark.parametrize(("platform", "session_id"), [("cron", "cron_job_1_20260927_000000"),
                                                      ("subagent", "subagent-1")])
def test_a_cron_or_subagent_platform_agent_without_a_service_is_rejected_before_provider_contact(
        tmp_path, monkeypatch, platform, session_id):
    env = Env(tmp_path, monkeypatch)
    with native_memory_sentinel(_native_dir()) as sentinel:
        agent = env.agent(session_id, platform=platform)
    sentinel.assert_untouched()
    assert isinstance(agent._memory_service, UnboundMemoryService)
    assert env.factory_calls == 0 and _records() == {}


def test_a_provider_managed_agent_reads_the_configured_review_cadence(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch, nudge_interval=3)
    assert env.agent()._memory_nudge_interval == 3


def test_additive_init_is_unchanged(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch, additive=True, memory_enabled=True, user_profile_enabled=True)
    agent = env.agent("additive-2")
    assert agent._memory_service.disposition is MemoryDisposition.BUILTIN and not is_injected(agent)
    assert agent._memory_store is not None and env.factory_calls == 0


# ---- B. the memory-tool gates (Task 4) ------------------------------------------------------------

def _injected_fork_agent(env, parent, surface="background_review"):
    agent = env.agent("fork-ctor-1", memory_service=open_fork_view(capture_parent_memory(parent), surface=surface))
    agent.session_id = parent.session_id
    return agent


def test_an_unattended_review_may_add(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    parent = env.agent()
    fork = _injected_fork_agent(env, parent)
    with _origin(attended=False), native_memory_sentinel(_native_dir()) as sentinel:
        out = json.loads(fork._invoke_tool("memory", ADD, "task-1"))
    sentinel.assert_untouched()
    assert out["success"] is True
    assert [r.text for r in env.store.records_for(REPO, "memory")] == ["uses pnpm"]


def test_an_unattended_review_cannot_replace_or_remove(tmp_path, monkeypatch):
    from tools import write_approval as wa
    env = Env(tmp_path, monkeypatch)
    env.store.seed_record(REPO, "memory", "old rule")
    parent = env.agent()
    fork = _injected_fork_agent(env, parent)
    view_backend = env.backends[-1]
    loads = view_backend.count("load_curated")
    for args in ({"action": "replace", "target": "memory", "old_text": "old rule", "content": "new"},
                 {"action": "remove", "target": "memory", "old_text": "old rule"},
                 {"operations": [{"action": "add", "content": "x"}, {"action": "remove", "old_text": "old rule"}]}):
        with _origin(attended=False):
            out = json.loads(fork._invoke_tool("memory", args, "task-1"))
        assert out["success"] is False and "'add' is still available" in out["error"]
    assert view_backend.count("load_curated") == loads and wa.list_pending(wa.MEMORY) == []
    assert [r.text for r in env.store.records_for(REPO, "memory")] == ["old rule"]


def test_an_attended_refine_keeps_the_full_operation_set(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    env.store.seed_record(REPO, "memory", "old rule")
    parent = env.agent()
    fork = _injected_fork_agent(env, parent)
    with _origin(attended=True):
        out = json.loads(fork._invoke_tool("memory", {"action": "replace", "target": "memory",
                                                       "old_text": "old rule", "content": "new rule"}, "task-1"))
    assert out["success"] is True
    assert [r.text for r in env.store.records_for(REPO, "memory")] == ["new rule"]


def test_a_read_only_view_refuses_before_any_load(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    parent = env.agent()
    fork = _injected_fork_agent(env, parent, surface="side_question")
    view_backend = env.backends[-1]
    with _origin("side_question"):
        out = json.loads(fork._invoke_tool("memory", ADD, "task-1"))
    assert out["success"] is False and "read-only" in out["error"]
    assert view_backend.count("load_curated") == 0 and view_backend.count("stage_curated") == 0


def test_an_unbound_service_names_the_missing_identity(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    agent = env.agent("cron_job_2_20260927_000000", platform="cron")
    out = json.loads(agent._invoke_tool("memory", ADD, "task-1"))
    assert out["success"] is False and "no explicit memory identity" in out["error"] and env.factory_calls == 0


# ---- C. background review and /btw (Task 5) -------------------------------------------------------

def test_a_review_fork_gets_a_view_of_the_parent_identity_and_binds_nothing(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    parent = env.agent()
    binds, records = env.count("bind_session"), _records()
    with native_memory_sentinel(_native_dir()) as sentinel:
        fork = env.fork(parent)
    sentinel.assert_untouched()
    view = fork._memory_service
    assert isinstance(view, LimitedMemoryView) and view.surface == "background_review" and view.mutations
    assert view.identity == parent._memory_service.identity and is_injected(fork)
    assert env.count("bind_session") == binds and _records() == records
    assert env.backends[-1].count("validate_session") == 1 and env.backends[-1].count("bind_session") == 0


def test_a_review_fork_commits_under_the_parent_identity_with_its_surface(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    parent = env.agent()
    fork = env.fork(parent)
    records = _records()
    with _origin(attended=False), native_memory_sentinel(_native_dir()) as sentinel:
        out = json.loads(fork._invoke_tool("memory", ADD, "task-1"))
    sentinel.assert_untouched()
    stage = [req for op, req in env.backends[-1].calls if op == "stage_curated"][-1]
    assert out["success"] is True and stage.provenance.initiating_surface == "background_review"
    assert stage.frozen_identity == parent._memory_service.identity.to_wire() and _records() == records


def test_the_review_identity_is_frozen_when_the_review_is_spawned(tmp_path, monkeypatch):
    from agent.memory_service.lifecycle import follow_session_binding
    env = Env(tmp_path, monkeypatch)
    parent = env.agent("session-a")
    started = []
    monkeypatch.setattr("run_agent.threading.Thread",
                        lambda target=None, daemon=None, name=None: type("T", (), {"start": lambda self: started.append(target)})())
    monkeypatch.setattr("run_agent._review_should_defer", lambda agent, cfg: False)
    parent._spawn_background_review(messages_snapshot=[{"role": "user", "content": "hi"}], review_memory=True)
    run = parent._background_review_run
    old_identity = parent._memory_service.identity
    assert started and run.memory_parent.state.identity == old_identity
    parent.session_id = "session-b"                  # /new before the (deferred) review runs
    follow_session_binding(parent)
    assert parent._memory_service.identity != old_identity
    fork = env.fork(parent, memory_parent=run.memory_parent)
    assert fork._memory_service.identity == old_identity


def test_side_question_fork_is_read_only_and_gates_through_its_view(tmp_path, monkeypatch):
    from agent.memory_service.lifecycle import curated_request_gate, ensure_session_binding
    env = Env(tmp_path, monkeypatch)
    parent = env.agent()
    fork = env.fork(parent, write_origin="side_question")
    view_backend = env.backends[-1]
    assert isinstance(fork._memory_service, LimitedMemoryView) and fork._memory_service.mutations is False
    ensure_session_binding(fork)
    assert curated_request_gate(fork) is None
    assert view_backend.count("load_curated") == 2           # §9.6 L1566: both enabled targets, own transport


def test_a_released_fork_closes_its_view_once(tmp_path, monkeypatch):
    from agent.background_review import _release_fork_clients
    env = Env(tmp_path, monkeypatch)
    fork = env.fork(env.agent())
    _release_fork_clients(fork)
    _release_fork_clients(fork)
    assert env.backends[-1].shutdown_calls == 1


def test_a_stateless_parent_forks_stateless_without_provider_contact(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch, authoritative_failure_policy="stateless")
    env.store.fail_transport("negotiate")
    parent = env.agent()
    calls = env.factory_calls
    fork = env.fork(parent)
    assert fork._memory_service.disposition is MemoryDisposition.STATELESS and env.factory_calls == calls


def test_an_additive_fork_is_built_exactly_as_before(tmp_path, monkeypatch):
    import run_agent
    env = Env(tmp_path, monkeypatch, additive=True, memory_enabled=True, user_profile_enabled=True)
    parent = env.agent("additive-3")
    real, seen = run_agent.AIAgent, []
    monkeypatch.setattr(run_agent, "AIAgent", lambda **kw: (seen.append(kw), real(**kw))[1])
    fork = env.fork(parent)
    assert "memory_service" not in seen[-1] and not is_injected(fork)
    assert fork._memory_store is parent._memory_store


def test_an_authoritative_parent_runs_memory_review_on_its_interval(tmp_path, monkeypatch):
    from agent.turn_context import _tick_memory_nudge
    env = Env(tmp_path, monkeypatch, nudge_interval=2)
    parent = env.agent()
    assert [_tick_memory_nudge(parent) for _ in range(2)] == [False, True]


def test_the_review_whitelist_follows_the_fork_service(tmp_path, monkeypatch):
    from agent.background_review import _review_tool_whitelist
    env = Env(tmp_path, monkeypatch)
    whitelist, _ = _review_tool_whitelist(env.fork(env.agent()), None, review_memory=True)
    assert "memory" in whitelist


def test_a_stateless_fork_gets_no_memory_in_its_whitelist(tmp_path, monkeypatch):
    from agent.background_review import _review_tool_whitelist
    env = Env(tmp_path, monkeypatch, authoritative_failure_policy="stateless")
    env.store.fail_transport("negotiate")
    whitelist, _ = _review_tool_whitelist(env.fork(env.agent()), None, review_memory=True)
    assert "memory" not in whitelist


# ---- D. delegation and memory-less workers (Task 7) -----------------------------------------------

def test_a_delegate_child_touches_no_memory(tmp_path, monkeypatch):
    """Ruling R43-9 (a): children stay memory-less; R40-4b's resolver never inherits via _delegate_from."""
    from tools import delegate_tool as dt
    import tools.delegate_tool_config as dtc
    env = Env(tmp_path, monkeypatch)
    monkeypatch.setattr(dt, "_load_config", lambda: {})
    monkeypatch.setattr(dtc, "_load_config", lambda: {})
    parent = env.agent(enabled_toolsets=["memory", "file"], tools=("memory", "read_file"))
    counts = {op: env.count(op) for op in ("negotiate", "bind_session", "validate_session", "load_curated")}
    records = _records()
    with native_memory_sentinel(_native_dir()) as sentinel, _construction_patches("read_file"):
        child = dt._build_child_agent(task_index=0, goal="goal", context=None, toolsets=["file"], model=None,
                                      max_iterations=4, task_count=1, parent_agent=parent)
    sentinel.assert_untouched()
    try:
        assert child._memory_service is None and child._memory_store is None and not is_injected(child)
        assert "memory" not in (child.valid_tool_names or set())
        assert {op: env.count(op) for op in counts} == counts and _records() == records
    finally:
        child.close()


def test_a_skip_memory_worker_agent_has_no_service(tmp_path, monkeypatch):
    """Curator, batch and Feishu comment agents: skip_memory=True with no memory toolset (R43-11)."""
    env = Env(tmp_path, monkeypatch)
    worker = env.agent("worker-1", platform="curator", skip_memory=True, enabled_toolsets=["skills"], tools=("skills_list",))
    assert worker._memory_service is None and env.factory_calls == 0 and _records() == {}
