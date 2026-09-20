"""Stateful service contracts that a scripted backend cannot prove."""

from dataclasses import replace
from threading import Event, Thread

import pytest

from agent.memory_service import wire as w
from agent.memory_service.errors import MemoryBlockedError, ProviderError, ProviderTransportError
from agent.memory_service.service import ContinuityCapture, InspectRequest
from tests.agent.memory_service.fake_backend import set_key
from tests.agent.memory_service.test_failure_matrix import (
    PROVENANCE, REPO, _context, _intent, _mutation, _recall_query, _select, _store,
)


@pytest.mark.parametrize("operation", [
    "load_curated", "stage_curated", "commit_curated", "recall_context",
    "inspect_staged", "capture_continuity",
])
def test_envelope_epoch_precedence_over_correlation(tmp_path, operation):
    store = _store()
    service, factory = _select(tmp_path, store)
    snapshot = service.load_curated("memory")
    staged = service.stage_curated(_mutation(snapshot))
    actions = {
        "load_curated": lambda: service.load_curated("memory"),
        "stage_curated": lambda: service.stage_curated(_mutation(snapshot, "r2")),
        "commit_curated": lambda: service.commit_curated(_intent(staged)),
        "recall_context": lambda: service.recall_context(_recall_query(snapshot)),
        "inspect_staged": lambda: service.inspect_staged(InspectRequest("memory", "r1", staged.stage_handle_b64url)),
        "capture_continuity": lambda: service.capture_continuity(ContinuityCapture("cr1", "compression_snapshot", "context", "compression")),
    }
    path = {
        "load_curated": "frozen_identity.logical_session_id",
        "stage_curated": "request_id", "commit_curated": "request_id",
        "recall_context": "frozen_identity.logical_session_id",
        "inspect_staged": "summary.request_id", "capture_continuity": "request_id",
    }[operation]
    store.corrupt_result(operation, set_key(path, "other-session-or-request"))
    store.envelope_epoch_override(operation, "ep-other")
    with pytest.raises(MemoryBlockedError, match="epoch"):
        actions[operation]()
    assert service.epoch_changed and service.blocked
    calls = len(factory.backend.calls)
    with pytest.raises(MemoryBlockedError, match="rebind"):
        service.load_curated("memory")
    assert len(factory.backend.calls) == calls
    if operation == "commit_curated":
        assert [record.text for record in store.records_for(REPO, "memory")] == ["x"]
        assert store.stage_state("r1") == "committed"
    # An envelope override simulates a response; actually changing the store
    # epoch also invalidates its old receipts and binding handles (§9.5).
    old_identity = service.identity
    store.set_epoch("ep-other")
    service.start(_context("sess-2"), bind_intent="explicit_rebind", prior_identity=old_identity)
    assert not service.epoch_changed and not service.blocked
    for action in (
        lambda: service.inspect_staged(InspectRequest("memory", "r1", staged.stage_handle_b64url)),
        lambda: service.commit_curated(_intent(staged)),
    ):
        with pytest.raises(ProviderError) as exc:
            action()
        assert exc.value.code == "stage_not_found" and exc.value.details is None


def test_malformed_commit_reply_after_publication_is_unknown_then_replayable(tmp_path):
    store = _store()
    service, _ = _select(tmp_path, store)
    staged = service.stage_curated(_mutation(service.load_curated("memory")))
    intent = _intent(staged)
    store.corrupt_result("commit_curated", set_key("snapshot.target", "user"))
    with pytest.raises(ProviderTransportError) as exc:
        service.commit_curated(intent)
    assert exc.value.mutation_outcome_unknown and service.blocked
    assert [record.text for record in store.records_for(REPO, "memory")] == ["x"]
    service.load_curated("memory")
    with pytest.raises(ProviderError) as inspected:
        service.inspect_staged(InspectRequest("memory", "r1", staged.stage_handle_b64url))
    assert inspected.value.code == "stage_not_found"
    assert inspected.value.details["state"] == "committed"
    original_tx_id = inspected.value.details["tx_id"]
    replay = service.commit_curated(intent)
    assert replay.outcome == "idempotent_replay" and replay.tx_id == original_tx_id
    assert [record.text for record in store.records_for(REPO, "memory")] == ["x"]


@pytest.mark.parametrize("text", ["x", "different body"])
def test_two_services_one_store_share_the_request_id_namespace(tmp_path, text):
    store = _store()
    first, _ = _select(tmp_path, store)
    second, _ = _select(tmp_path, store, session="sess-2")
    first.stage_curated(_mutation(first.load_curated("memory")))
    with pytest.raises(ProviderError) as exc:
        second.stage_curated(_mutation(second.load_curated("memory"), text=text))
    assert exc.value.code == "idempotency_mismatch"
    assert store.stage_state("r1") == "live"
    assert store.records_for(REPO, "memory") == []


def test_cross_session_commit_race_has_one_winner(tmp_path):
    store = _store()
    first, _ = _select(tmp_path, store)
    second, _ = _select(tmp_path, store, session="sess-2")
    snapshot_a = first.load_curated("memory")
    snapshot_b = second.load_curated("memory")
    assert snapshot_a.revision == snapshot_b.revision
    staged_a = first.stage_curated(_mutation(snapshot_a, "r-a", text="A"))
    staged_b = second.stage_curated(_mutation(snapshot_b, "r-b", text="B", provenance=replace(PROVENANCE, logical_session_id="sess-2")))
    entered = Event()
    outcomes = []

    def commit_b():
        entered.set()
        try:
            outcomes.append(second.commit_curated(_intent(staged_b)))
        except ProviderError as exc:
            outcomes.append(exc)

    contender = Thread(target=commit_b)

    def start_contender():
        store.before_publish = None
        contender.start()
        assert entered.wait(5), "contender did not enter commit"

    store.before_publish = start_contender
    try:
        winner = first.commit_curated(_intent(staged_a))
    finally:
        contender.join(5)
    assert not contender.is_alive()
    assert winner.outcome == "committed_audit_clean"
    assert len(outcomes) == 1 and isinstance(outcomes[0], ProviderError)
    assert outcomes[0].code == "version_conflict"
    conflict = w.decode_error_details("version_conflict", outcomes[0].details)
    assert conflict.current_snapshot.revision == winner.snapshot.revision
    assert [record.text for record in store.records_for(REPO, "memory")] == ["A"]
    fresh = second.load_curated("memory")
    restaged = second.stage_curated(_mutation(fresh, "r-b-retry", text="B", provenance=replace(PROVENANCE, logical_session_id="sess-2")))
    assert second.commit_curated(_intent(restaged)).outcome == "committed_audit_clean"
    assert [record.text for record in store.records_for(REPO, "memory")] == ["A", "B"]
