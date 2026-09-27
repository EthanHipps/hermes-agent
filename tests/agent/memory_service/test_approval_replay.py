"""R39: deferred approvals — review, byte-free replay and reject (§9.4 steps 7–9; §9.5 L1542, L1562; §4.1 L250;
contract C6b-4)."""

import os
import shutil
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from agent.memory_service import wire as w
from agent.memory_service.approval import ApprovalChannel
from agent.memory_service.approval_replay import (
    RejectStatus, ReplayStatus, ReviewState, reject_pending_approval, replay_pending_approval,
    review_pending_approvals)
from agent.memory_service.approval_store import list_pending_approvals, load_pending_approval
from agent.memory_service.bootstrap import init_memory_service
from agent.memory_service.host_state import host_state_dir
from agent.memory_service.mutation import MutationStatus, PlannedMutation, run_curated_mutation
from tests.agent.memory_service.fake_backend import FakeAuthoritativeBackend, FakeClock, FakeProviderStore, FakeRegistry
from tests.agent.memory_service.native_sentinel import native_memory_sentinel

EPOCH = "ep-opaque-7f3a"
GLOBAL = w.ScopeRef(kind="principal_global", id="ethan")


def _never_built():
    raise AssertionError("native store built")


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    repo = tmp_path / "repo"
    repo.mkdir()
    store = FakeProviderStore(epoch=EPOCH, clock=FakeClock(datetime(2026, 9, 27, tzinfo=timezone.utc)),
                              registry=FakeRegistry(directories={os.path.realpath(repo): ("repo-1", "proj-1", None)}))
    backends = []

    def factory(cfg):
        backends.append(FakeAuthoritativeBackend(store, provider="example"))
        return backends[-1]

    exe = tmp_path / "provider.exe"
    exe.write_bytes(b"MZ")
    raw = {"memory": {"provider": "example", "provider_mode": "authoritative", "provider_executable": str(exe),
                      "principal_id": "ethan"}}
    service, _ = init_memory_service(raw, logical_session_id="sess-1", platform="cli", store_factory=_never_built,
                                     profile_id="default", working_directory=str(repo), backend_factory=factory)
    from tools.memory_tool import get_memory_dir
    return SimpleNamespace(store=store, backends=backends, factory=factory, raw=raw, service=service,
                           native_dir=get_memory_dir())


def _add(snapshot, text):
    cand = w.CandidateEntry(client_ref="c1", text=text, destination_scope=snapshot.default_write_scope,
                            target=snapshot.target, proposed_policy_key=None, import_source_identity=None)
    return PlannedMutation(intent=w.MutationIntent(kind="add"),
                           mutation_delta=(w.MutationDeltaItem(action="add", client_ref="c1"),),
                           candidate_entries=(cand,), requested_write_scopes=(snapshot.default_write_scope,),
                           projected_texts=(text,), message="Entry added.")


def _defer(env, text="prefers terse replies"):
    outcome = run_curated_mutation(env.service, "user", lambda s: _add(s, text),
                                   approval=ApprovalChannel(session_id="sess-1", clock=env.store.clock.now))
    assert outcome.status is MutationStatus.PENDING_APPROVAL
    return outcome.pending_id


def _replay(env, pid):
    with native_memory_sentinel(env.native_dir) as sentinel:
        report = replay_pending_approval(pid, raw_config=env.raw, backend_factory=env.factory, clock=env.store.clock.now)
    sentinel.assert_untouched()
    return report


def _view_calls(env, op):
    return [req for b in env.backends[1:] for o, req in b.calls if o == op]   # backends[0] is the live service


def _user_texts(env):
    return sorted(r.text for r in env.store.records_for(GLOBAL, "user"))


def test_approve_replays_the_byte_free_commit_from_the_persisted_identity(env):
    pid = _defer(env)
    record = load_pending_approval(pid)
    report = _replay(env, pid)
    assert report.status is ReplayStatus.COMMITTED and _user_texts(env) == ["prefers terse replies"]
    assert [op for b in env.backends[1:] for op, _ in b.calls] == ["negotiate", "validate_session", "inspect_staged",
                                                                  "commit_curated"]
    [commit] = _view_calls(env, "commit_curated")
    assert (commit.request_id, commit.stage_handle_b64url, commit.approval_binding_sha256,
            commit.authorized_write_scopes) == (record.request_id, record.stage_handle_b64url,
                                                record.approval_binding_sha256, record.requested_write_scopes)
    assert commit.frozen_identity == env.service.identity.to_wire() and commit.authorization.kind == "approved"
    assert b"prefers terse replies" not in w.canonical_json(commit.to_wire())
    assert list_pending_approvals() == []


@pytest.mark.parametrize("phase, outcome", [("after_publish", "idempotent_replay"), ("before", "committed_audit_clean")])
def test_an_unknown_replay_is_retried_only_with_the_identical_request(env, phase, outcome):
    pid = _defer(env)
    env.store.fail_transport("commit_curated", phase=phase, times=2)
    first = _replay(env, pid)
    assert first.status is ReplayStatus.UNKNOWN and load_pending_approval(pid).outcome == "unknown"
    second = _replay(env, pid)
    assert second.status is ReplayStatus.COMMITTED and second.code == outcome
    commits = [w.canonical_json(c.to_wire()) for c in _view_calls(env, "commit_curated")]
    assert len(commits) == 3 and len(set(commits)) == 1          # §9.5 L1562
    assert _user_texts(env) == ["prefers terse replies"] and list_pending_approvals() == []


def test_an_expired_stage_is_dropped_as_not_saved(env):
    pid = _defer(env)
    env.store.clock.advance(3601)
    assert _replay(env, pid).status is ReplayStatus.EXPIRED
    assert _view_calls(env, "commit_curated") == [] and list_pending_approvals() == [] and _user_texts(env) == []


def test_a_conflict_is_dropped_and_publishes_nothing(env):
    pid = _defer(env)
    env.store.external_write(GLOBAL, "user", "written elsewhere")
    assert _replay(env, pid).status is ReplayStatus.CONFLICT
    assert _user_texts(env) == ["written elsewhere"] and list_pending_approvals() == []


def test_a_missing_host_state_record_voids_the_approval_without_a_provider_call(env):
    """§4.1 L250: approval replay fails binding_invalid when the persisted identity is missing."""
    pid = _defer(env)
    shutil.rmtree(host_state_dir())
    views = len(env.backends)
    report = _replay(env, pid)
    assert report.status is ReplayStatus.VOID and report.code == "binding_invalid"
    assert len(env.backends) == views and list_pending_approvals() == []


@pytest.mark.parametrize("void", [lambda env: env.store.set_epoch("ep-2"),
                                  lambda env: env.store.revoke(env.service.identity.opaque_binding_b64url)])
def test_an_epoch_change_or_revoked_binding_voids_the_approval(env, void):
    """§9.5 L1562 / D-R39-8."""
    pid = _defer(env)
    void(env)
    assert _replay(env, pid).status is ReplayStatus.VOID and list_pending_approvals() == []


@pytest.mark.parametrize("operation", ["negotiate", "validate_session"])
def test_an_unreachable_provider_keeps_the_record(env, operation):
    pid = _defer(env)
    env.store.fail_transport(operation)
    assert _replay(env, pid).status is ReplayStatus.UNAVAILABLE
    assert [p for p, _ in list_pending_approvals()] == [pid]


def test_reject_drops_the_record_and_calls_no_provider(env):
    pid = _defer(env)
    views = len(env.backends)
    assert reject_pending_approval(pid) is RejectStatus.DROPPED
    assert len(env.backends) == views and list_pending_approvals() == []
    assert reject_pending_approval(pid) is RejectStatus.NOT_FOUND
    assert reject_pending_approval("../escape") is RejectStatus.NOT_FOUND


def test_reject_refuses_an_approved_record_whose_result_is_unconfirmed(env):
    pid = _defer(env)
    env.store.fail_transport("commit_curated", phase="after_publish", times=2)
    _replay(env, pid)
    assert reject_pending_approval(pid) is RejectStatus.REFUSED and load_pending_approval(pid) is not None


def test_review_renders_live_inspections_and_settles_terminal_states(env):
    expired = _defer(env, "old ask")
    env.store.clock.advance(1800)
    live = _defer(env, "new ask")
    env.store.clock.advance(1801)                              # the first stage's TTL (3600 s) has passed
    with native_memory_sentinel(env.native_dir) as sentinel:
        reviews = review_pending_approvals(raw_config=env.raw, backend_factory=env.factory)
    sentinel.assert_untouched()
    states = {r.pending_id: r for r in reviews}
    assert states[expired].state is ReviewState.EXPIRED and states[live].state is ReviewState.AWAITING
    assert "new ask" in states[live].text and "target_user" in states[live].text
    assert [p for p, _ in list_pending_approvals()] == [live]
    assert _view_calls(env, "commit_curated") == []


def test_review_of_a_committed_unknown_record_reports_it_saved(env):
    pid = _defer(env)
    env.store.fail_transport("commit_curated", phase="after_publish", times=2)
    _replay(env, pid)
    [review] = review_pending_approvals(raw_config=env.raw, backend_factory=env.factory)
    assert review.state is ReviewState.COMMITTED and list_pending_approvals() == []
