"""R41: ``hermes memory status|setup|off|reset`` and the dashboard provider switch in authoritative mode
(§9.7 L1618, L1620; rulings R41-7, R41-8, R41-21). Helpers are local (shared test support is frozen)."""

import os
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from tests.agent.memory_service.fake_backend import FakeAuthoritativeBackend, FakeProviderStore, FakeRegistry


def _tool_defs(*names):
    return [{"type": "function", "function": {"name": n, "description": f"{n} tool",
                                              "parameters": {"type": "object", "properties": {}}}} for n in names]


@pytest.fixture
def authoritative_env(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    store = FakeProviderStore(epoch="EPOCHVALUEAAAA", registry=FakeRegistry(
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
    _write_config(memory_section)   # before the first run_agent import (ensure_hermes_home reads it)
    from run_agent import AIAgent
    with patch("model_tools.get_tool_definitions", return_value=_tool_defs("memory", "web_search")), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=False, **kwargs)
    agent.client = MagicMock()
    return agent


def test_status_shows_and_validates_the_authoritative_config(authoritative_env, capsys):
    from hermes_cli.memory_authoritative import status_command
    code = status_command({"memory": {**authoritative_env.memory, "gateway_principals": {"telegram:1": "ethan"}}},
                          Namespace(session=None), backend_factory=authoritative_env.factory)
    out = capsys.readouterr().out
    assert code == 0 and "example" in out and "fail_closed" in out and "provider-managed" in out
    assert [op for op, _ in authoritative_env.backends[-1].calls] == ["negotiate"]
    assert authoritative_env.store.epoch not in out


def test_status_fails_on_an_unhealthy_provider_or_bad_config(authoritative_env, capsys):
    from hermes_cli.memory_authoritative import status_command
    authoritative_env.store.fail_transport("negotiate")
    assert status_command({"memory": authoritative_env.memory}, None, backend_factory=authoritative_env.factory) == 1
    assert status_command({"memory": {**authoritative_env.memory, "principal_id": ""}}, None) == 1


def test_status_of_a_persisted_session_shows_its_identity_without_binding(authoritative_env, capsys):
    from hermes_cli.memory_authoritative import status_command
    agent = _agent(authoritative_env.memory, session_id="persisted-1")
    capsys.readouterr()
    code = status_command({"memory": authoritative_env.memory}, Namespace(session="persisted-1"),
                          backend_factory=authoritative_env.factory)
    out = capsys.readouterr().out
    assert code == 0 and agent._memory_service.identity.binding_revision in out and "repository:repo-1" in out
    assert authoritative_env.backends[-1].count("bind_session") == 0
    assert agent._memory_service.identity.opaque_binding_b64url not in out


def test_setup_writes_nothing_and_never_opens_the_picker(authoritative_env, monkeypatch):
    from hermes_cli import memory_setup
    path = _write_config(authoritative_env.memory)
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *a, **k: 1 / 0)
    memory_setup.memory_command(Namespace(memory_command="setup", provider=None))
    memory_setup.memory_command(Namespace(memory_command="setup", provider="honcho"))
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_off_refuses_and_writes_nothing(authoritative_env):
    from hermes_cli.main_agent_cmds import _cmd_memory_off
    path = _write_config(authoritative_env.memory)
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    with pytest.raises(SystemExit) as refused:
        _cmd_memory_off()
    assert refused.value.code == 2 and (path.read_bytes(), path.stat().st_mtime_ns) == before


def _client():
    """Copied from tests/hermes_cli/test_dashboard_admin_endpoints.py L16-30 (never import a test module)."""
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    hermes_state.DEFAULT_DB_PATH = get_hermes_home() / "state.db"
    return client, _SESSION_HEADER_NAME


def test_dashboard_provider_switch_refuses_in_authoritative_mode(authoritative_env, _isolate_hermes_home):
    """Ruling R41-21 (R42-13): the provider switch never changes authority; config.yaml is untouched."""
    client, _ = _client()
    path = _write_config(authoritative_env.memory)
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    response = client.put("/api/memory/provider", json={"provider": "built-in"})
    assert response.status_code == 409 and (path.read_bytes(), path.stat().st_mtime_ns) == before


REPO = {"organization": None, "project": "proj-1", "repository": "repo-1"}


def _reset_args(**overrides):
    return Namespace(**{"target": "memory", "yes": False, "scope": ["repository:repo-1"], **REPO, **overrides})


@pytest.fixture
def terminal(monkeypatch):
    """An interactive stdin whose answers the test scripts; an exception stands for EOF or Ctrl-C."""
    answers = []

    def answer(_question):
        reply = answers.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", answer)
    return answers


def _seed(authoritative_env):
    from agent.memory_service import wire as w
    gone = authoritative_env.store.seed_record(w.ScopeRef(kind="repository", id="repo-1"), "memory", "reset me")
    kept = authoritative_env.store.seed_record(w.ScopeRef(kind="project", id="proj-1"), "memory", "keep me")
    return gone, kept


def test_an_approved_cli_reset_retires_exactly_its_scopes(authoritative_env, terminal, capsys):
    from hermes_cli.memory_authoritative import reset_command
    gone, kept = _seed(authoritative_env)
    terminal.append("y")
    assert reset_command({"memory": authoritative_env.memory}, _reset_args()) == 0
    assert "reset me" in capsys.readouterr().out                 # the inspection was shown before deciding
    records = authoritative_env.store.records
    assert records[gone.id].lifecycle == "retired" and records[kept.id].lifecycle == "active"


@pytest.mark.parametrize("reply", ["n", EOFError(), KeyboardInterrupt()])
def test_a_declined_or_interrupted_reset_changes_nothing_and_never_waits_for_approval(authoritative_env, terminal, reply):
    from agent.memory_service.approval_store import list_pending_approvals
    from hermes_cli.memory_authoritative import reset_command
    gone, _ = _seed(authoritative_env)
    terminal.append(reply)
    assert reset_command({"memory": authoritative_env.memory}, _reset_args()) == 0
    assert authoritative_env.store.records[gone.id].lifecycle == "active" and list_pending_approvals() == []


def test_an_unknown_cli_reset_outcome_keeps_the_approved_request_for_the_identical_retry(authoritative_env, terminal,
                                                                                         capsys):
    from agent.memory_service.approval_replay import ReplayStatus, replay_pending_approval
    from agent.memory_service.approval_store import list_pending_approvals, load_pending_approval
    from hermes_cli.memory_authoritative import reset_command
    gone, _ = _seed(authoritative_env)
    authoritative_env.store.fail_transport("commit_curated", times=2)
    terminal.append("y")
    assert reset_command({"memory": authoritative_env.memory}, _reset_args()) == 1
    [(pid, record)] = list_pending_approvals()
    assert record.decision == "approved" and record.outcome == "unknown" and pid in capsys.readouterr().out
    assert replay_pending_approval(pid, raw_config={"memory": authoritative_env.memory},
                                   principal=None).status is ReplayStatus.COMMITTED
    assert authoritative_env.store.records[gone.id].lifecycle == "retired" and load_pending_approval(pid) is None


def test_a_non_interactive_reset_is_refused_before_any_provider_contact(authoritative_env, monkeypatch, capsys):
    from hermes_cli.memory_authoritative import reset_command
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: False))
    assert reset_command({"memory": authoritative_env.memory}, _reset_args(target="user", scope=["global:ethan"])) == 2
    assert "interactive terminal" in capsys.readouterr().out and authoritative_env.backends == []


def test_reset_refuses_a_malformed_scope_with_the_encoding_hint(authoritative_env, capsys):
    from hermes_cli.memory_authoritative import reset_command
    code = reset_command({"memory": authoritative_env.memory}, _reset_args(scope=["repo:x"]))
    assert code == 2 and "repository:<id>" in capsys.readouterr().out


def test_a_repository_reset_without_its_chain_says_how_to_name_the_chain(authoritative_env, terminal, capsys):
    from hermes_cli.memory_authoritative import reset_command
    code = reset_command({"memory": authoritative_env.memory}, _reset_args(project=None, repository=None))
    assert code == 2 and "--repository" in capsys.readouterr().out
