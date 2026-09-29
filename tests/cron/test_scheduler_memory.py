"""Cron receives an explicit memory identity or none (S6 inventory suite C gate; §9.7 L1613, L1625;
§9.6 L1585; rulings R43-6, R43-7, R43-8; contract C6b-8)."""

import json
import os
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agent.memory_service import wire as w
from agent.memory_service.host_state import host_state_dir
from agent.memory_service.view import UnboundMemoryService, is_injected
from cron.scheduler import run_job
from cron.scheduler_memory import resolve_cron_memory_service
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


class Env:
    """An authoritative (or additive) memory section in the isolated HERMES_HOME, and the R36 fake behind
    the real discovery seam; no provider executable is ever launched."""

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

    def _factory(self, cfg):
        self.factory_calls += 1
        self.backends.append(FakeAuthoritativeBackend(self.store, provider=cfg.provider))
        return self.backends[-1]


def _spy():
    """An AIAgent stand-in whose run returns a normal result, as tests/cron/test_scheduler.py's harness does."""
    spy = MagicMock()
    spy.return_value.run_conversation.return_value = {"final_response": "ok"}
    return spy


@contextmanager
def _run_job_patches(tmp_path, agent_cls):
    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state_registry.acquire", return_value=MagicMock()), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
             "api_key": "test-key", "base_url": "https://example.invalid/v1", "provider": "openrouter",
             "api_mode": "chat_completions"}), \
         patch("run_agent.AIAgent", agent_cls):
        yield


def _setup():
    from cron.scheduler import _CronAgentSetup
    return _CronAgentSetup(model="m", runtime={"api_key": "test-key-1234567890", "provider": "openrouter",
                                               "base_url": "https://openrouter.ai/api/v1", "api_mode": "chat_completions"},
                           max_iterations=4)


def _cron_agent(session_id):
    from cron.scheduler import _construct_cron_agent
    from run_agent import AIAgent
    with _construction_patches("memory", "web_search"):
        return _construct_cron_agent(AIAgent, {"id": "j1", "prompt": "p"}, {}, _setup(), workdir=None,
                                     session_id=session_id, session_db=None)


def test_additive_cron_is_unchanged(tmp_path, monkeypatch):
    Env(tmp_path, monkeypatch, additive=True)
    assert resolve_cron_memory_service(session_id="cron_j1_1") is None
    spy = _spy()
    with _run_job_patches(tmp_path, spy):
        run_job({"id": "j1", "name": "t", "prompt": "hello"})
    assert "memory_service" not in spy.call_args.kwargs and spy.call_args.kwargs["skip_memory"] is False


def test_cron_binds_its_explicit_scope_and_never_the_process_directory(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch, cron_scope={"project_id": "proj-1", "repo_id": "repo-1"})
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)                                             # an unregistered cwd
    with native_memory_sentinel(_native_dir()) as sentinel:
        agent = _cron_agent("cron_j1_20260927_000000")
        out = json.loads(agent._invoke_tool("memory", ADD, "task-1"))
    sentinel.assert_untouched()
    bind = [req for op, req in env.backends[0].calls if op == "bind_session"][-1]
    assert bind.requested_context.resolution_source == "explicit_ids"
    assert bind.requested_context.canonical_directory is None and bind.requested_context.platform == "cron"
    assert is_injected(agent) and out["success"] is True
    assert [r.text for r in env.store.records_for(REPO, "memory")] == ["uses pnpm"]
    assert _records() == {}                                                  # ruling R43-7


def test_cron_without_a_scope_is_rejected_before_any_read_or_write(tmp_path, monkeypatch):
    from agent.memory_service.lifecycle import curated_prompt_parts
    env = Env(tmp_path, monkeypatch)
    with native_memory_sentinel(_native_dir()) as sentinel:
        agent = _cron_agent("cron_j1_20260927_000001")
        out = json.loads(agent._invoke_tool("memory", ADD, "task-1"))
    sentinel.assert_untouched()
    assert isinstance(agent._memory_service, UnboundMemoryService) and env.factory_calls == 0
    assert out["success"] is False and "no explicit memory identity" in out["error"]
    assert curated_prompt_parts(agent) == []


def test_an_empty_scope_is_explicit_principal_global(tmp_path, monkeypatch):
    Env(tmp_path, monkeypatch, cron_scope={})
    agent = _cron_agent("cron_j1_20260927_000002")
    out = json.loads(agent._invoke_tool("memory", ADD, "task-1"))
    assert out["success"] is False and out.get("code") == "scope_unresolved"


def test_a_malformed_scope_fails_the_job_before_any_agent_exists(tmp_path, monkeypatch):
    Env(tmp_path, monkeypatch, cron_scope=["repo-1"])
    spy = _spy()
    with _run_job_patches(tmp_path, spy):
        ok, _output, _final, error = run_job({"id": "j2", "name": "t", "prompt": "hello"})
    assert ok is False and "memory.cron_scope" in error and not spy.called


def test_a_fail_closed_outage_fails_the_job_before_the_model(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch, cron_scope={"project_id": "proj-1", "repo_id": "repo-1"})
    env.store.fail_transport("negotiate")
    spy = _spy()
    with _run_job_patches(tmp_path, spy):
        ok, _output, _final, error = run_job({"id": "j3", "name": "t", "prompt": "hello"})
    assert ok is False and "MemoryBlockedError" in error and not spy.called


def test_a_failed_construction_shuts_the_cron_service_down(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch, cron_scope={"project_id": "proj-1", "repo_id": "repo-1"})
    from cron.scheduler import _construct_cron_agent

    def boom(**kwargs):
        raise RuntimeError("constructor failed")

    with pytest.raises(RuntimeError):
        _construct_cron_agent(boom, {"id": "j4", "prompt": "p"}, {}, _setup(), workdir=None,
                              session_id="cron_j4_1", session_db=None)
    assert env.backends[-1].shutdown_calls == 1
