"""S6 approval suite for R39 (§13.1 L2211; §14.1 row 39 L2309): write-approval staging and replay.

Every "never" is proven: the native directory with ``native_memory_sentinel``, the
persisted sinks (the approval store and the native pending store) with
byte-and-mtime snapshots, and provider tokens by searching every user- and
model-facing string. Contracts C6b-1..C6b-4.
"""

import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agent.memory_service import wire as w
from agent.memory_service.approval_store import approvals_dir, list_pending_approvals, load_pending_approval
from hermes_cli.write_approval_commands import handle_pending_subcommand
from tests.agent.memory_service.fake_backend import FakeAuthoritativeBackend, FakeClock, FakeProviderStore, FakeRegistry
from tests.agent.memory_service.native_sentinel import native_memory_sentinel
from tools import write_approval as wa
from tools.memory_tool import memory_tool
from tools.memory_tool_curated import ConsolidationBudget

EPOCH = "ep-opaque-7f3a"
GLOBAL = w.ScopeRef(kind="principal_global", id="ethan")
USER_ADD = {"action": "add", "target": "user", "content": "prefers terse replies"}
MEMORY_ADD = {"action": "add", "target": "memory", "content": "uses pnpm"}


def _tool_defs(*names):
    return [{"type": "function", "function": {"name": n, "description": f"{n} tool",
                                              "parameters": {"type": "object", "properties": {}}}} for n in names]


def _tool_call(name, args, call_id):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def _assistant(tool_calls):
    return SimpleNamespace(content="", tool_calls=tool_calls)


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


@pytest.fixture
def native_dir():
    from tools.memory_tool import get_memory_dir
    return get_memory_dir()


@pytest.fixture
def env(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    # A clock at real time keeps approved_at <= the stage's expires_at, as ygg's authorize requires.
    store = FakeProviderStore(epoch=EPOCH, clock=FakeClock(datetime.now(timezone.utc)),
                              registry=FakeRegistry(directories={os.path.realpath(os.getcwd()): ("repo-1", "proj-1", None)}))
    backends = []

    def factory(cfg):
        backends.append(FakeAuthoritativeBackend(store, provider="example"))
        return backends[-1]

    monkeypatch.setattr("plugins.memory.load_authoritative_backend_factory", lambda name: factory)
    exe = tmp_path / "provider.exe"
    exe.write_bytes(b"MZ")
    memory = {"provider": "example", "provider_mode": "authoritative", "provider_executable": str(exe),
              "principal_id": "ethan"}
    return SimpleNamespace(store=store, backends=backends, memory=memory)


@pytest.fixture
def answer():
    from tools.terminal_tool import set_approval_callback
    seen = []

    def install(choice):
        set_approval_callback(lambda command, description, **kw: seen.append((command, description)) or choice)
        return seen
    yield install
    set_approval_callback(None)


def _calls(env, op):
    return [req for b in env.backends for o, req in b.calls if o == op]


def _home():
    from hermes_constants import get_hermes_home
    return get_hermes_home()


def _tree(root):
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in (sorted(root.rglob("*")) if root.exists() else ()) if p.is_file()}


def _sinks():
    home = _home()
    return {sub: _tree(home / sub) for sub in ("memory_service/approvals", "pending", "memories")}


def _dispatch(agent, site, args):
    if site == "sequential":
        messages = []
        agent._execute_tool_calls_sequential(_assistant([_tool_call("memory", args, "m-seq")]), messages, "task-1")
        return json.loads(messages[-1]["content"])
    return json.loads(agent._invoke_tool("memory", args, "task-1", tool_call_id="m-conc"))


def test_pending_review_then_approve_replays_the_byte_free_commit(env, native_dir):
    agent = _agent(env.memory)
    pid = json.loads(agent._invoke_tool("memory", USER_ADD, "task-1"))["pending_id"]
    record = load_pending_approval(pid)
    with native_memory_sentinel(native_dir) as sentinel:
        listing = handle_pending_subcommand(wa.MEMORY, ["pending"])
        result = handle_pending_subcommand(wa.MEMORY, ["approve", pid])
    sentinel.assert_untouched()
    assert pid in listing and "prefers terse replies" in listing and "target_user" in listing
    assert result == f"{pid}: approved and saved."
    commit = _calls(env, "commit_curated")[-1]
    assert (commit.request_id, commit.stage_handle_b64url, commit.approval_binding_sha256,
            commit.authorized_write_scopes) == (record.request_id, record.stage_handle_b64url,
                                                record.approval_binding_sha256, record.requested_write_scopes)
    assert commit.frozen_identity == agent._memory_service.identity.to_wire()
    assert b"prefers terse replies" not in w.canonical_json(commit.to_wire())
    assert sorted(r.text for r in env.store.records_for(GLOBAL, "user")) == ["prefers terse replies"]
    assert list_pending_approvals() == [] and wa.list_pending(wa.MEMORY) == []


def test_native_pending_records_stay_dormant(env, native_dir):
    """Ruling R39-13: neither listed, applied nor deleted in an authoritative home (§9.1 L950)."""
    _write_config(env.memory)
    native = wa.stage_write(wa.MEMORY, {"action": "add", "target": "memory", "content": "native body"},
                            summary="native", origin="foreground")
    before = _tree(_home() / "pending")
    with native_memory_sentinel(native_dir) as sentinel:
        outputs = [handle_pending_subcommand(wa.MEMORY, args, memory_store=None) for args in
                   ([], ["pending"], ["approve", "all"], ["reject", "all"], ["approve", native["id"]], ["reject", native["id"]])]
    sentinel.assert_untouched()
    assert _tree(_home() / "pending") == before and [r["id"] for r in wa.list_pending(wa.MEMORY)] == [native["id"]]
    assert not any("native body" in out for out in outputs)


def test_an_approval_required_write_waits_as_a_handle_only_record(env, native_dir):
    agent = _agent(env.memory)
    with native_memory_sentinel(native_dir) as sentinel:
        out = json.loads(agent._invoke_tool("memory", USER_ADD, "task-1"))
    sentinel.assert_untouched()
    assert out["success"] is True and out["staged"] is True and out["approval_required"] == ["target_user"]
    assert wa.list_pending(wa.MEMORY) == []
    [(pid, record)] = list_pending_approvals()
    assert pid == out["pending_id"] and record.decision == "pending" and record.session_id == agent.session_id
    raw = (approvals_dir() / f"{pid}.json").read_bytes()
    assert b"prefers terse replies" not in raw
    assert all(h.canonical_sha256.encode() not in raw
               for h in env.store.stages[record.stage_handle_b64url].result.candidate_hashes)
    assert _calls(env, "inspect_staged") == [] and _calls(env, "commit_curated") == []   # ruling R39-6


@pytest.mark.parametrize("site", ["sequential", "concurrent"])
def test_inline_approval_commits_the_inspected_stage_at_both_dispatch_sites(env, native_dir, answer, site):
    agent = _agent(env.memory)
    seen = answer("once")
    with native_memory_sentinel(native_dir) as sentinel:
        out = _dispatch(agent, site, USER_ADD)
    sentinel.assert_untouched()
    assert out["success"] is True and sorted(r.text for r in env.store.records_for(GLOBAL, "user")) == ["prefers terse replies"]
    command, description = seen[-1]
    assert "prefers terse replies" in command and "target_user" in command and "Save to memory" in description
    [commit] = _calls(env, "commit_curated")
    assert len(_calls(env, "inspect_staged")) == 1 and commit.authorization.kind == "approved"
    assert commit.authorization.approved_by_principal_id == "ethan"
    assert commit.authorization.approval_binding_sha256 == commit.approval_binding_sha256
    assert list_pending_approvals() == []                           # the write-ahead record was settled


def test_inline_denial_drops_the_handle_and_persists_nothing(env, native_dir, answer):
    agent = _agent(env.memory)
    answer("deny")
    before = _sinks()
    with native_memory_sentinel(native_dir) as sentinel:
        out = json.loads(agent._invoke_tool("memory", USER_ADD, "task-1"))
    sentinel.assert_untouched()
    assert out["success"] is False and "denied" in out["error"]
    assert _sinks() == before and not approvals_dir().exists()
    assert _calls(env, "commit_curated") == [] and env.store.records_for(GLOBAL, "user") == []


def test_an_unknown_outcome_is_reported_until_a_typed_outcome(env, native_dir):
    agent = _agent(env.memory)
    pid = json.loads(agent._invoke_tool("memory", USER_ADD, "task-1"))["pending_id"]
    env.store.fail_transport("commit_curated", phase="after_publish", times=2)
    first = handle_pending_subcommand(wa.MEMORY, ["approve", pid])
    assert "did not confirm" in first and load_pending_approval(pid).outcome == "unknown"
    assert handle_pending_subcommand(wa.MEMORY, ["approve", pid]) == f"{pid}: approved and saved."
    commits = [w.canonical_json(c.to_wire()) for c in _calls(env, "commit_curated")]
    assert len(commits) == 3 and len(set(commits)) == 1            # §9.5 L1562: only the identical request
    assert list_pending_approvals() == []


def test_an_expired_stage_is_reported_and_dropped_on_review(env, native_dir):
    agent = _agent(env.memory)
    json.loads(agent._invoke_tool("memory", USER_ADD, "task-1"))
    env.store.clock.advance(3601)
    listing = handle_pending_subcommand(wa.MEMORY, ["pending"])
    assert "expired before approval; nothing was saved" in listing and list_pending_approvals() == []


def test_reject_persists_nothing_and_calls_no_provider(env, native_dir):
    agent = _agent(env.memory)
    pid = json.loads(agent._invoke_tool("memory", USER_ADD, "task-1"))["pending_id"]
    views = len(env.backends)
    assert handle_pending_subcommand(wa.MEMORY, ["reject", pid]) == f"Rejected pending memory write '{pid}'. Nothing was saved."
    assert len(env.backends) == views and list_pending_approvals() == [] and _calls(env, "commit_curated") == []


def test_a_mandatory_approval_fails_closed_without_any_channel(env, native_dir):
    """No inline callback and no persisted session to wait under: refused before staging (§9.5 L1538)."""
    agent = _agent(env.memory)
    stages = len(_calls(env, "stage_curated"))
    out = json.loads(memory_tool(**USER_ADD, service=agent._memory_service, budget=ConsolidationBudget()))
    assert out["success"] is False and out["approval_required"] == ["target_user"]
    assert len(_calls(env, "stage_curated")) == stages and list_pending_approvals() == []


def test_write_approval_adds_a_requirement_and_an_unrecognized_value_counts_as_on(env, native_dir):
    agent = _agent({**env.memory, "write_approval": "maybe"})
    out = json.loads(agent._invoke_tool("memory", MEMORY_ADD, "task-1"))
    assert out["staged"] is True and out["approval_required"] == ["memory.write_approval"]


def test_write_approval_off_never_removes_a_mandatory_approval(env, native_dir):
    agent = _agent({**env.memory, "write_approval": False})
    out = json.loads(agent._invoke_tool("memory", USER_ADD, "task-1"))
    assert out["staged"] is True and out["approval_required"] == ["target_user"]


def test_no_provider_token_reaches_the_model_or_the_ui(env, native_dir, answer):
    """§9.3 L987 / D-R39-3."""
    agent = _agent(env.memory)
    tool_out = agent._invoke_tool("memory", USER_ADD, "task-1")
    pid = json.loads(tool_out)["pending_id"]
    record = load_pending_approval(pid)
    listing = handle_pending_subcommand(wa.MEMORY, ["pending"])
    approved = handle_pending_subcommand(wa.MEMORY, ["approve", pid])
    seen = answer("once")
    inline_out = agent._invoke_tool("memory", {"action": "add", "target": "user", "content": "likes tea"}, "task-2")
    handle = agent._memory_service.identity.opaque_binding_b64url
    tokens = {record.stage_handle_b64url, record.approval_binding_sha256, record.request_id, EPOCH, handle,
              env.store.token_for(handle, "user")} | {c.authorization.approval_id for c in _calls(env, "commit_curated")}
    surfaces = [tool_out, listing, approved, inline_out] + [c + d for c, d in seen]
    assert not any(token and token in text for token in tokens for text in surfaces)
