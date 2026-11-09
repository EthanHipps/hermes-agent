"""Compositions of provider-managed memory features that were proven only one at a time: the review fork with
approvals (R39 x R43), the gateway with approvals (R39 x R41), the kanban worker marker (EDD-106) and the
archive block for a configured provider (K-10).

Most worlds use a full org -> project -> repo chain and principal ``ethan``, reached through the same
``plugins.memory.load_authoritative_backend_factory`` seam as the neighbouring suites."""

import asyncio
import json
import os
import subprocess

import pytest

from agent.memory_service import wire as w
from tests.agent.memory_service.native_sentinel import native_memory_sentinel
from tests.agent.memory_service.test_s6_inventory_a import _Gateway, _event, gateway_home  # noqa: F401  (fixture)
from tests.agent.memory_service.test_s6_inventory_c import (
    Env, _deferred_review_across_new, _fork_user_add, _native_dir, _records)
from tests.cron.test_scheduler_memory import _cron_agent

ORG = w.ScopeRef(kind="organization", id="org-1")
USER_SCOPE = w.ScopeRef(kind="principal_global", id="ethan")
GATEWAY = {"telegram:111": "ethan"}
MULTI = {**GATEWAY, "telegram:222": "other"}
NOT_AVAILABLE = "Curated memory is not available in this chat."
TEXT = "prefers terse replies"


def _org_chain(env):
    """The fixture registry's repo carries no organization; give it one so the world is org -> project -> repo."""
    registry = env.store.registry
    (directory,) = registry.directories
    registry.directories[directory] = ("repo-1", "proj-1", "org-1")
    registry.projects["proj-1"] = "org-1"
    return env


def _gateway_world(tmp_path, monkeypatch, table, session_id):
    env = _org_chain(Env(tmp_path, monkeypatch, gateway_principals=table))
    parent = env.agent(session_id, platform="telegram", user_id="111")
    return env, parent


def _stage_review_write(env, parent):
    """What a background review leaves behind: an ``[auto]`` pending user-profile add under the parent's session."""
    out = _fork_user_add(env.fork(parent))
    assert out["staged"] is True and out["approval_required"] == ["target_user"]
    return out["pending_id"]


def _say(runner, args, **event):
    return asyncio.run(runner._handle_memory_command(_event(args, **event)))


def _commits(env):
    return env.count("commit_curated")


# ---- R39 x R43: the review fork with approvals -----------------------------------------------------

def test_review_fork_defers_auto_then_memory_approve_replays(tmp_path, monkeypatch):
    from agent.memory_service.approval_store import list_pending_approvals
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    env = _org_chain(Env(tmp_path, monkeypatch))
    parent, memory_parent = _deferred_review_across_new(env, monkeypatch)
    old_identity = memory_parent.state.identity
    assert parent._memory_service.identity != old_identity                 # the parent has moved on to session-b
    fork = env.fork(parent, memory_parent=memory_parent)
    out = _fork_user_add(fork)
    assert out["staged"] is True and out["approval_required"] == ["target_user"]
    pid = out["pending_id"]
    assert [r.session_id for _, r in list_pending_approvals()] == ["session-a"]    # the reviewed session, not the live one
    with native_memory_sentinel(_native_dir()) as sentinel:
        listing = handle_pending_subcommand(wa.MEMORY, ["pending"], memory_principal=None)
        assert f"{pid} [auto]" in listing and TEXT in listing
        assert env.store.records_for(USER_SCOPE, "user") == [] and _commits(env) == 0
        result = handle_pending_subcommand(wa.MEMORY, ["approve", pid], memory_principal=None)
    sentinel.assert_untouched()
    assert result == f"{pid}: approved and saved."
    commit = [req for b in env.backends for op, req in b.calls if op == "commit_curated"][-1]
    assert commit.frozen_identity == old_identity.to_wire()               # replayed under the reviewed session's identity
    assert [r.text for r in env.store.records_for(USER_SCOPE, "user")] == [TEXT]
    assert list_pending_approvals() == []


def test_cron_approval_required_write_fails_closed_before_stage(tmp_path, monkeypatch):
    from agent.memory_service.approval_store import list_pending_approvals
    env = _org_chain(Env(tmp_path, monkeypatch, cron_scope={"org_id": "org-1", "project_id": "proj-1",
                                                           "repo_id": "repo-1"}))
    agent = _cron_agent("cron_j1_20261105_000000")
    assert agent._memory_service.identity.org_id == "org-1"               # the explicit chain bound
    stages = env.count("stage_curated")
    with native_memory_sentinel(_native_dir()) as sentinel:
        out = json.loads(agent._invoke_tool("memory", {"action": "add", "target": "user", "content": TEXT}, "task-1"))
    sentinel.assert_untouched()
    assert out["success"] is False and out["approval_required"] == ["target_user"]
    assert env.count("stage_curated") == stages and list_pending_approvals() == []
    assert env.store.records_for(USER_SCOPE, "user") == [] and _records() == {}    # cron has no host record to wait under


# ---- R39 x R41: the gateway with approvals ---------------------------------------------------------

def test_mapped_gateway_user_reaches_approve(tmp_path, monkeypatch, gateway_home):
    from agent.memory_service.approval_store import list_pending_approvals
    env, parent = _gateway_world(tmp_path, monkeypatch, GATEWAY, "gw-1")
    pid = _stage_review_write(env, parent)
    runner = _Gateway(parent)
    listing = _say(runner, "pending")
    assert f"{pid} [auto]" in listing and TEXT in listing
    assert _say(runner, f"approve {pid}") == f"{pid}: approved and saved."
    assert [r.text for r in env.store.records_for(USER_SCOPE, "user")] == [TEXT] and list_pending_approvals() == []


def test_unmapped_gateway_user_never_reaches_it(tmp_path, monkeypatch, gateway_home):
    from agent.memory_service.approval_store import list_pending_approvals
    env, parent = _gateway_world(tmp_path, monkeypatch, GATEWAY, "gw-2")
    pid = _stage_review_write(env, parent)
    runner, commits = _Gateway(parent), _commits(env)
    for args in ("pending", f"approve {pid}", "approve all", f"reject {pid}", "reject all"):
        assert _say(runner, args, user_id="999") == NOT_AVAILABLE
    assert [p for p, _ in list_pending_approvals()] == [pid]
    assert _commits(env) == commits and env.store.records_for(USER_SCOPE, "user") == []


def test_multi_principal_gateway_reviews_only_the_callers_writes(tmp_path, monkeypatch, gateway_home):
    """With two mapped principals a shared chat sees the change named but never its text, and the other
    principal's user cannot reach the record at all."""
    from agent.memory_service.approval_store import list_pending_approvals
    env = Env(tmp_path, monkeypatch, gateway_principals=MULTI)
    parent = env.agent("gw-3", platform="telegram", user_id="111")
    pid = _stage_review_write(env, parent)
    runner, commits = _Gateway(parent), _commits(env)
    group = _say(runner, "pending", chat_type="group", chat_id="g1")
    assert f"{pid} [auto]" in group and TEXT not in group
    for args in ("pending", f"approve {pid}", f"reject {pid}"):
        assert _say(runner, args, user_id="222", chat_type="dm", chat_id="c2") == NOT_AVAILABLE
    assert [p for p, _ in list_pending_approvals()] == [pid] and _commits(env) == commits
    assert TEXT in _say(runner, "pending", chat_type="dm", chat_id="c1")      # the owner's private chat still shows it


# ---- Kanban workers (EDD-106) ----------------------------------------------------------------------

def _dispatcher_env(tmp_path, monkeypatch):
    """The environment ``_default_spawn`` hands a worker, captured from a patched ``Popen`` (nothing is launched)."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd
    captured = {}

    class _Proc:
        pid = 12345

    def fake_popen(cmd, *args, **kwargs):
        captured["env"] = kwargs.get("env", {})
        return _Proc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    kb.create_board("spawntest")
    task = kb.Task(id="t_0123abcd", title="worker", body=None, assignee="teknium", status="ready", priority=0,
                   created_by="user", created_at=0, started_at=None, completed_at=None, workspace_kind="scratch",
                   workspace_path=None, claim_lock=None, claim_expires=None, tenant=None)
    kbd._default_spawn(task, str(tmp_path / "ws"), board="spawntest")
    return captured["env"]


def test_dispatcher_spawn_env_carries_the_worker_marker(tmp_path, monkeypatch):
    from agent.memory_service.bootstrap import agent_spawned_surface
    env = _dispatcher_env(tmp_path, monkeypatch)
    assert env["HERMES_KANBAN_TASK"] == "t_0123abcd"
    assert agent_spawned_surface() is None                              # the dispatcher's own process is not a worker
    monkeypatch.setenv("HERMES_KANBAN_TASK", env["HERMES_KANBAN_TASK"])
    assert agent_spawned_surface() == "kanban"                          # the marker it sets is the one the guard reads


def test_kanban_worker_never_contacts_the_provider(tmp_path, monkeypatch):
    from agent.memory_service.host_state import load_host_state
    from agent.memory_service.view import AgentSpawnedMemoryService, UnboundMemoryService
    env = Env(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_KANBAN_TASK", _dispatcher_env(tmp_path, monkeypatch)["HERMES_KANBAN_TASK"])
    with native_memory_sentinel(_native_dir()) as sentinel:
        worker = env.agent("kanban-worker-9")
        out = json.loads(worker._invoke_tool("memory", {"action": "add", "target": "memory", "content": "x",
                                                         "memory_admission": "declarative"}, "task-1"))
    sentinel.assert_untouched()
    assert isinstance(worker._memory_service, AgentSpawnedMemoryService) and isinstance(worker._memory_service, UnboundMemoryService)
    assert out["success"] is False and env.factory_calls == 0 and env.backends == []     # zero backend calls
    assert load_host_state("kanban-worker-9").disposition == "stateless"


def test_kanban_worker_refused_admin_and_replay(tmp_path, monkeypatch):
    """A worker cannot review, approve or reject the owner's pending review write, nor open the admin identity."""
    from agent.memory_service.admin import AdminContext, admin_service
    from agent.memory_service.approval_replay import (
        RejectStatus, ReplayStatus, reject_pending_approval, replay_pending_approval)
    from agent.memory_service.approval_store import list_pending_approvals, load_pending_approval
    from agent.memory_service.errors import MemoryBlockedError
    from hermes_cli.memory_command import provider_memory_command
    env = _org_chain(Env(tmp_path, monkeypatch))
    parent = env.agent("owner-1")
    pid = _stage_review_write(env, parent)
    saved, commits = load_pending_approval(pid), _commits(env)
    monkeypatch.setenv("HERMES_KANBAN_TASK", _dispatcher_env(tmp_path, monkeypatch)["HERMES_KANBAN_TASK"])
    worker = env.agent("kanban-worker-10")
    with native_memory_sentinel(_native_dir()) as sentinel:
        with pytest.raises(MemoryBlockedError), admin_service({"memory": env.memory}, context=AdminContext()):
            pass
        for args in (["pending"], ["approve", pid], ["reject", pid]):
            reply = provider_memory_command(args, agent=worker, set_mode_fn=None)
            assert "unavailable in agent-started processes" in reply and TEXT not in reply
        raw = {"memory": env.memory}
        assert replay_pending_approval(pid, raw_config=raw, principal=None).status is ReplayStatus.UNAVAILABLE
        assert reject_pending_approval(pid, principal=None) is RejectStatus.UNAVAILABLE
    sentinel.assert_untouched()
    assert load_pending_approval(pid) == saved and [p for p, _ in list_pending_approvals()] == [pid]
    assert _commits(env) == commits and env.store.records_for(USER_SCOPE, "user") == []


# ---- K-10: the archive block for a configured provider ---------------------------------------------

#: Spec section 9.8, verbatim: the block as written there, for the value ``ygg``.
SPEC_BLOCK_TEXT = (
    "curated_memory:\n"
    "  authority: provider\n"
    "  provider: ygg\n"
    "  provider_api: 1\n"
    "  included: false\n"
    "  disposition: provider-managed\n"
    "  restore_action: reconnect-provider\n")
SPEC_SENTENCE = "Hermes archive complete; authoritative ygg memory is provider-managed and not included"


def test_archive_block_is_byte_identical_for_a_configured_provider():
    from agent.memory_service.archive import archive_disposition
    disposition = archive_disposition({"memory": {"provider": "ygg", "provider_mode": "authoritative"}})
    assert disposition.yaml_block().encode("utf-8") == SPEC_BLOCK_TEXT.encode("utf-8")
    assert disposition.record_text().endswith(SPEC_BLOCK_TEXT)
    assert disposition.archive_message() == SPEC_SENTENCE
