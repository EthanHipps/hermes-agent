"""§9.6 failure matrix, fail_closed (F) and stateless (S) columns, rows 1–16,
proven through the real ``ProviderAuthoritativeMemoryService`` over the
stateful fake (R36).

Row 17 (backup/status surfaces) is R41/R44's: only the fault knobs are
exercised here. The Additive column is R48's except row 1 (ruling R36-D).
Row 7 asserts ruling R36-A = (b): optional recall failures do not latch.
"""

import logging
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from agent.memory_service import wire as w
from agent.memory_service.authoritative import ProviderAuthoritativeMemoryService
from agent.memory_service.config import MemoryConfigurationError, resolve_memory_service_config
from agent.memory_service.errors import CapabilityUnavailableError, MemoryBlockedError, ProviderError, ProviderTransportError, StatelessSessionError
from agent.memory_service.identity import FrozenMemoryIdentity, HostSessionState
from agent.memory_service.service import CommitIntent, ContinuityCapture, InspectRequest, MemoryDisposition, MutationRequest, RecallQuery, StatelessMemoryService, select_memory_service

from tests.agent.memory_service.fake_backend import FakeAuthoritativeBackend, FakeClock, FakeProviderStore, drop_key, fake_backend_factory, set_key

CLOCK_START = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)
REPO = w.ScopeRef(kind="repository", id="repo-1")
PROJ = w.ScopeRef(kind="project", id="proj-1")
PG = w.ScopeRef(kind="principal_global", id="ethan")
PROVENANCE = w.MutationProvenance(actor_kind="hermes", principal_id="ethan", logical_session_id="sess-1", initiating_surface="memory_tool", source_entry_ids=(), source_commit=None, threat_decision_id=None)
POLICIES = ["fail_closed", "stateless"]


def _never_built():
    raise AssertionError("native store built")


def _store(**kwargs) -> FakeProviderStore:
    kwargs.setdefault("clock", FakeClock(CLOCK_START))
    return FakeProviderStore(**kwargs)


def _config(tmp_path, **memory):
    exe = tmp_path / "provider.exe"
    exe.write_bytes(b"MZ")
    base = {"provider": "example", "provider_mode": "authoritative", "provider_executable": str(exe)}
    base.update(memory)
    return {"memory": base}


def _context(session="sess-1", *, degraded=False) -> w.RequestedContext:
    if degraded:
        return w.RequestedContext(principal_id="ethan", profile_id="default", logical_session_id=session, platform="cli", org_id=None, project_id=None, repo_id=None, workspace_id=None, resolution_source="explicit_ids", canonical_directory=None)
    return w.RequestedContext(principal_id="ethan", profile_id="default", logical_session_id=session, platform="cli", org_id=None, project_id=None, repo_id=None, workspace_id=None, resolution_source="directory", canonical_directory="C:\\work\\repo")


class _Factory:
    """``fake_backend_factory`` plus a record of every backend it built."""

    def __init__(self, store, **backend_kwargs):
        self.inner = fake_backend_factory(store, **backend_kwargs)
        self.backends = []

    def __call__(self, cfg):
        backend = self.inner(cfg)
        self.backends.append(backend)
        return backend

    @property
    def backend(self) -> FakeAuthoritativeBackend:
        assert len(self.backends) == 1
        return self.backends[0]


def _select(tmp_path, store, *, policy="fail_closed", session="sess-1", degraded=False, factory=None, **memory):
    factory = factory or _Factory(store)
    service = select_memory_service(_config(tmp_path, authoritative_failure_policy=policy, **memory), store_factory=_never_built, requested_context=_context(session, degraded=degraded), backend_factory=factory)
    return service, factory


def _resume(tmp_path, store, session_state, *, policy="fail_closed", factory=None):
    factory = factory or _Factory(store)
    service = select_memory_service(_config(tmp_path, authoritative_failure_policy=policy), store_factory=_never_built, session_state=session_state, backend_factory=factory)
    return service, factory


def _mutation(snapshot, request_id="r1", text="x", *, scope=None, target="memory", intent=None, delta=None, provenance=PROVENANCE) -> MutationRequest:
    scope = scope or snapshot.default_write_scope
    candidates = (w.CandidateEntry(client_ref="c1", text=text, destination_scope=scope, target=target, proposed_policy_key=None, import_source_identity=None),)
    return MutationRequest(target=target, request_id=request_id, expected_revision=snapshot.revision, hidden_preservation_state=snapshot.hidden_preservation_state, requested_write_scopes=(scope,), intent=intent or w.MutationIntent(kind="add"), mutation_delta=delta if delta is not None else (w.MutationDeltaItem(action="add", client_ref="c1"),), candidate_entries=candidates, provenance=provenance)


def _intent(staged, *, scopes=None, authorization=None, binding=None, handle=None, target=None) -> CommitIntent:
    return CommitIntent(target or staged.target, staged.request_id, handle or staged.stage_handle_b64url, binding or staged.approval_binding_sha256, tuple(scopes if scopes is not None else staged.requested_write_scopes), authorization or w.ApprovalAuthorization(kind="not_required"))


def _approved(staged, *, by="ethan", expires_at=None) -> w.ApprovalAuthorization:
    return w.ApprovalAuthorization(kind="approved", approval_id="ap-1", approved_by_principal_id=by, approved_at="2026-09-17T12:00:00Z", expires_at=expires_at or staged.expires_at, approval_binding_sha256=staged.approval_binding_sha256)


def _recall_query(snapshot, query="q") -> RecallQuery:
    return RecallQuery("memory", snapshot.revision, query, ("general", "hermes_memory"), (), w.RecallBudget(1000, 3000, 20))


def _stateless_records(caplog):
    return [r for r in caplog.records if r.name == "agent.memory_service.service" and r.levelno == logging.WARNING]


# --- Row 1: additive never constructs the backend (ruling R36-D) -------------


class _NativeStore:
    """Stand-in for the deferred native store factory (as test_selection._Store)."""

    def __init__(self):
        self.constructed = 0

    def factory(self):
        self.constructed += 1
        from tools.memory_tool import MemoryStore

        return MemoryStore()


@pytest.mark.parametrize("config", [{}, {"memory": {"provider": "example"}}])
def test_additive_never_builds_the_backend(config):
    store = _NativeStore()
    factory = _Factory(_store())
    service = select_memory_service(config, store_factory=store.factory, backend_factory=factory)
    assert service.disposition is MemoryDisposition.BUILTIN
    assert factory.backends == [] and store.constructed == 1


# --- Row 2: configuration errors --------------------------------------------


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("case,backend_kwargs,match", [
    ("api_version", {"api_version": 2}, "API version"),
    ("missing_operation", {"operations": ["bind_session", "validate_session", "load_curated", "stage_curated", "inspect_staged"]}, "commit_curated"),
    ("provider_mismatch", {"provider": "other"}, "provider"),
    ("capability_without_operation", {"operations": ["bind_session", "validate_session", "load_curated", "stage_curated", "inspect_staged", "commit_curated"], "recall": True}, "recall_context"),
])
def test_row2_negotiation_rejection_is_a_configuration_error(tmp_path, policy, case, backend_kwargs, match):
    factory = _Factory(_store(), **backend_kwargs)
    with pytest.raises(MemoryConfigurationError, match=match):
        _select(tmp_path, None, policy=policy, factory=factory)
    assert factory.backend.shutdown_calls == 1
    assert [op for op, _ in factory.backend.calls] == ["negotiate"]


# --- Row 3: startup transport / malformed ------------------------------------


_ROW3_CORRUPTIONS = {"negotiate": "limits", "bind_session": "user_write_scope", "validate_session": "visible_scopes", "load_curated": "revision"}


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("fault", ["transport", "malformed"])
@pytest.mark.parametrize("operation", ["negotiate", "bind_session", "validate_session", "load_curated"])
def test_row3_startup_failure_blocks_or_starts_stateless(tmp_path, caplog, policy, fault, operation):
    store = _store()
    session_state = None
    if operation == "validate_session":
        first, _ = _select(tmp_path, store)
        session_state = first.session_state
    if fault == "transport":
        store.fail_transport(operation)
    else:
        store.corrupt_result(operation, drop_key(_ROW3_CORRUPTIONS[operation]))
    factory = _Factory(store)
    boot = (lambda: _resume(tmp_path, store, session_state, policy=policy, factory=factory)) if session_state else (lambda: _select(tmp_path, store, policy=policy, factory=factory))
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        if policy == "fail_closed":
            with pytest.raises(MemoryBlockedError):
                boot()
            backend = factory.backend
            assert backend.shutdown_calls == 1
            calls = len(backend.calls)
            return
        service, _ = boot()
    backend = factory.backend
    assert isinstance(service, StatelessMemoryService) and service.disposition is MemoryDisposition.STATELESS
    assert len(_stateless_records(caplog)) == 1  # M8: one record for the stateless transition
    assert backend.shutdown_calls == 1
    assert "stateless" in service.degraded_warning().lower() and service.degraded_warning() == service.degraded_warning()
    calls = len(backend.calls)
    with pytest.raises(StatelessSessionError):
        service.stage_curated(MutationRequest(target="memory", request_id="r", expected_revision=None, hidden_preservation_state=None, requested_write_scopes=(), intent=None, mutation_delta=(), candidate_entries=(), provenance=None))
    with pytest.raises(StatelessSessionError):
        service.load_curated("memory")
    assert len(backend.calls) == calls and service.prompt_block("memory") is None


# --- Row 4: snapshot integrity ----------------------------------------------


_ROW4_CORRUPTIONS = {
    "identity": drop_key("frozen_identity"),
    "target": drop_key("target"),
    "record_channel": set_key("mutation_entries[0].record_channel", "general"),
    "completeness": drop_key("complete_for_scopes"),
    "preservation": drop_key("hidden_preservation_state"),
    "revision": drop_key("revision"),
}


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("member", list(_ROW4_CORRUPTIONS))
def test_row4_startup_snapshot_integrity(tmp_path, policy, member):
    store = _store()
    store.seed_record(REPO, "memory", "existing")
    store.corrupt_result("load_curated", _ROW4_CORRUPTIONS[member])
    factory = _Factory(store)
    if policy == "fail_closed":
        with pytest.raises(MemoryBlockedError, match="malformed provider output"):
            _select(tmp_path, store, factory=factory)
        assert factory.backend.shutdown_calls == 1
        return
    service, _ = _select(tmp_path, store, policy=policy, factory=factory)
    assert isinstance(service, StatelessMemoryService)
    assert "malformed provider output" in service.degraded_warning()


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("member", list(_ROW4_CORRUPTIONS))
def test_row4_mid_session_snapshot_integrity_blocks_until_a_clean_load(tmp_path, policy, member):
    store = _store()
    store.seed_record(REPO, "memory", "existing")
    service, factory = _select(tmp_path, store, policy=policy)
    assert isinstance(service, ProviderAuthoritativeMemoryService)  # started authoritative under either policy
    snap = service.load_curated("memory")
    store.corrupt_result("load_curated", _ROW4_CORRUPTIONS[member])
    with pytest.raises(MemoryBlockedError, match="malformed provider output"):
        service.load_curated("memory")
    assert service.blocked and service.disposition is MemoryDisposition.AUTHORITATIVE  # never switches (§9.6 L1563)
    calls = len(factory.backend.calls)
    with pytest.raises(MemoryBlockedError):
        service.stage_curated(_mutation(snap))
    assert len(factory.backend.calls) == calls
    assert service.load_curated("memory").revision == snap.revision
    assert not service.blocked
    assert service.stage_curated(_mutation(snap)).request_id == "r1"


# --- Row 5: repository/project unresolved -----------------------------------


@pytest.mark.parametrize("policy", POLICIES)
def test_row5_degraded_registry_context(tmp_path, policy):
    store = _store()
    store.seed_general(PG, "global policy")
    store.seed_record(REPO, "memory", "repo evidence")
    service, factory = _select(tmp_path, store, policy=policy, degraded=True)
    assert isinstance(service, ProviderAuthoritativeMemoryService) and service.disposition is MemoryDisposition.AUTHORITATIVE
    memory = service.load_curated("memory")
    assert memory.status == "degraded_global_only" and memory.default_write_scope is None and memory.mutation_entries == ()
    assert [e.text for e in memory.delivery_entries] == ["global policy"]
    with pytest.raises(ProviderError) as exc:
        service.stage_curated(_mutation(memory, scope=PG))
    assert exc.value.code == "scope_unresolved" and exc.value.outcome == "not_committed"
    assert service.blocked is False
    user = service.load_curated("user")
    assert user.status == "ok" and user.default_write_scope == PG
    staged = service.stage_curated(_mutation(user, scope=PG, target="user", text="profile"))
    assert staged.approval_requirements == ("target_user",)
    with pytest.raises(ProviderError) as exc:
        service.commit_curated(_intent(staged))
    assert exc.value.code == "approval_required"
    assert service.commit_curated(_intent(staged, authorization=_approved(staged))).outcome == "committed_audit_clean"


# --- Row 6: provider fails after prior success ------------------------------


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("operation", ["load_curated", "inspect_staged"])
def test_row6_mid_session_transport_failure_blocks_until_fresh_load(tmp_path, policy, operation):
    store = _store()
    service, factory = _select(tmp_path, store, policy=policy)
    snap = service.load_curated("memory")
    staged = service.stage_curated(_mutation(snap))
    store.fail_transport(operation)
    with pytest.raises(MemoryBlockedError):
        if operation == "load_curated":
            service.load_curated("memory")
        else:
            service.inspect_staged(InspectRequest("memory", "r1", staged.stage_handle_b64url))
    assert service.blocked and service.disposition is MemoryDisposition.AUTHORITATIVE
    calls = len(factory.backend.calls)
    with pytest.raises(MemoryBlockedError):
        service.stage_curated(_mutation(snap, "r2"))
    with pytest.raises(MemoryBlockedError):
        service.commit_curated(_intent(staged))
    with pytest.raises(MemoryBlockedError):
        service.inspect_staged(InspectRequest("memory", "r1", staged.stage_handle_b64url))
    assert len(factory.backend.calls) == calls
    fresh = service.load_curated("memory")
    assert not service.blocked and fresh.revision == snap.revision
    assert service.commit_curated(_intent(staged)).outcome == "committed_audit_clean"


def test_row6_a_session_that_started_stateless_stays_stateless(tmp_path):
    store = _store()
    store.fail_transport("bind_session")
    service, factory = _select(tmp_path, store, policy="stateless")
    assert isinstance(service, StatelessMemoryService)
    calls = len(factory.backend.calls)
    with pytest.raises(StatelessSessionError):
        service.load_curated("memory")  # the fake is healthy again; the session is not
    assert len(factory.backend.calls) == calls


# --- Row 7: optional recall unavailable after required load (ruling R36-A) --


def test_row7_recall_transport_failure_does_not_latch_the_block(tmp_path):
    """Ruling R36-A = (b): the transport error propagates, `blocked` stays
    false, and an already-approved mutation still goes through. The
    host-visible warning half of §9.6 L1573 is R40's."""
    store = _store()
    service, factory = _select(tmp_path, store)
    snap = service.load_curated("memory")
    staged = service.stage_curated(_mutation(snap, scope=PROJ))
    store.fail_transport("recall_context")
    with pytest.raises(ProviderTransportError) as exc:
        service.recall_context(_recall_query(snap))
    assert exc.value.operation == "recall_context" and exc.value.mutation_outcome_unknown is False
    assert service.blocked is False and service.degraded_warning() is None
    committed = service.commit_curated(_intent(staged, authorization=_approved(staged)))
    assert committed.outcome == "committed_audit_clean"
    assert service.recall_context(_recall_query(committed.snapshot)).target == "memory"


def test_row7_stateless_recall_is_refused_before_any_call(tmp_path):
    store = _store()
    store.fail_transport("load_curated")
    service, factory = _select(tmp_path, store, policy="stateless")
    calls = len(factory.backend.calls)
    with pytest.raises(StatelessSessionError):
        service.recall_context(RecallQuery("memory", w.CompositeRevision("ep-1", "1", ()), "q", ("general",), (), w.RecallBudget(0, 0, 0)))
    assert len(factory.backend.calls) == calls


# --- Row 8: required load is ambiguous_policy -------------------------------


@pytest.mark.parametrize("policy", POLICIES)
def test_row8_startup_ambiguous_policy(tmp_path, policy):
    store = _store()
    store.seed_general(REPO, "instruction body", policy_key="k")
    store.add_policy_conflict("k", 2)
    factory = _Factory(store)
    if policy == "fail_closed":
        with pytest.raises(MemoryBlockedError) as exc:
            _select(tmp_path, store, factory=factory)
        cause = exc.value.__cause__
        assert isinstance(cause, MemoryBlockedError) and cause.code == "ambiguous_policy"
        details = w.decode_error_details("ambiguous_policy", cause.provider_error.details)
        assert details.ambiguities[0].policy_key == "k" and "instruction body" not in str(cause.provider_error.details)
        assert factory.backend.shutdown_calls == 1
        return
    service, _ = _select(tmp_path, store, policy=policy, factory=factory)
    assert isinstance(service, StatelessMemoryService) and "ambiguous_policy" in service.degraded_warning()


@pytest.mark.parametrize("policy", POLICIES)
def test_row8_later_ambiguous_policy_blocks_the_request_with_no_partial_memory(tmp_path, policy):
    store = _store()
    store.seed_general(REPO, "instruction body", policy_key="k")
    service, factory = _select(tmp_path, store, policy=policy)
    assert service.prompt_block("memory") is not None
    store.add_policy_conflict("k", 2)
    with pytest.raises(MemoryBlockedError) as exc:
        service.prompt_block("memory")
    assert exc.value.code == "ambiguous_policy" and isinstance(exc.value.provider_error, ProviderError)
    assert w.decode_error_details("ambiguous_policy", exc.value.provider_error.details).ambiguities[0].candidate_count == 2
    assert service.blocked
    store.policy_conflicts.clear()
    assert service.load_curated("memory").status == "ok" and not service.blocked


# --- Row 9: recall is ambiguous_policy ---------------------------------------


def test_row9_recall_ambiguous_policy_is_typed_and_unlatched(tmp_path):
    store = _store()
    service, factory = _select(tmp_path, store)
    snap = service.load_curated("memory")
    store.add_policy_conflict("k", 2)
    with pytest.raises(ProviderError) as exc:
        service.recall_context(_recall_query(snap))
    assert exc.value.code == "ambiguous_policy" and exc.value.outcome == "not_applicable"
    assert not isinstance(exc.value, (CapabilityUnavailableError, ProviderTransportError))
    assert service.blocked is False
    store.policy_conflicts.clear()
    assert service.recall_context(_recall_query(snap)).source_revision == snap.revision


# --- Row 10: typed mutation refusals ----------------------------------------


_ROW10_CODES = ["scope_unresolved", "unauthorized_scope", "approval_required", "approval_invalid", "approval_expired", "secret_rejected", "limit_exceeded", "stage_not_found", "stage_expired", "store_blocked:git_dirty", "store_blocked:maintenance", "store_blocked:restore_cutover"]


@pytest.mark.parametrize("code", _ROW10_CODES)
def test_row10_typed_refusals_publish_nothing_and_keep_reads_working(tmp_path, code):
    store = _store(secret_detector=lambda t: "aws_access_key" if "AKIA" in t else None)
    store.seed_record(REPO, "memory", "existing")
    degraded = code == "scope_unresolved"
    service, factory = _select(tmp_path, store, degraded=degraded)
    snap = service.load_curated("memory")
    revision_before = snap.revision
    records_before = [r.id for r in store.records.values()]
    expected_code, _, reason = code.partition(":")
    if reason:
        store.block_store(reason)
    if code == "scope_unresolved":
        action = lambda: service.stage_curated(_mutation(snap, scope=PG))  # noqa: E731
    elif code == "unauthorized_scope":
        action = lambda: service.stage_curated(_mutation(snap, scope=PG))  # noqa: E731
    elif code == "approval_required":
        staged = service.stage_curated(_mutation(snap, scope=PROJ))
        action = lambda: service.commit_curated(_intent(staged))  # noqa: E731
    elif code == "approval_invalid":
        staged = service.stage_curated(_mutation(snap, scope=PROJ))
        action = lambda: service.commit_curated(_intent(staged, authorization=_approved(staged, by="someone-else")))  # noqa: E731
    elif code == "approval_expired":
        staged = service.stage_curated(_mutation(snap, scope=PROJ))
        action = lambda: service.commit_curated(_intent(staged, authorization=_approved(staged, expires_at="2026-09-17T11:00:00Z")))  # noqa: E731
    elif code == "secret_rejected":
        action = lambda: service.stage_curated(_mutation(snap, text="AKIAIOSFODNN7EXAMPLE"))  # noqa: E731
    elif code == "limit_exceeded":
        action = lambda: service.stage_curated(_mutation(snap, text="x" * 5000))  # noqa: E731
    elif code == "stage_not_found":
        action = lambda: service.commit_curated(CommitIntent("memory", "r-none", "AAAA", "c" * 64, (REPO,), w.ApprovalAuthorization(kind="not_required")))  # noqa: E731
    elif code == "stage_expired":
        staged = service.stage_curated(_mutation(snap))
        store.clock.advance(3601)
        action = lambda: service.commit_curated(_intent(staged))  # noqa: E731
    else:
        action = lambda: service.stage_curated(_mutation(snap))  # noqa: E731
    with pytest.raises(ProviderError) as exc:
        action()
    assert exc.value.code == expected_code and exc.value.outcome == "not_committed"
    if reason:
        assert exc.value.details == {"reason": reason}
    assert service.blocked is False and service.degraded_warning() is None
    assert [r.id for r in store.records.values()] == records_before
    fresh = service.load_curated("memory")
    assert fresh.revision == revision_before  # reads keep working, nothing published
    if reason:
        assert service.recall_context(_recall_query(fresh)).target == "memory"  # reads keep working while blocked
        store.unblock()
    if degraded:
        fresh = service.load_curated("user")  # preservation tokens are target-bound
    assert service.stage_curated(_mutation(fresh, "r-after", scope=PG if degraded else None, target="user" if degraded else "memory")).request_id == "r-after"


def test_row10_stateless_refuses_every_mutation(tmp_path):
    store = _store()
    store.fail_transport("negotiate")
    service, factory = _select(tmp_path, store, policy="stateless")
    calls = len(factory.backend.calls)
    with pytest.raises(StatelessSessionError):
        service.stage_curated(MutationRequest(target="memory", request_id="r", expected_revision=None, hidden_preservation_state=None, requested_write_scopes=(), intent=None, mutation_delta=(), candidate_entries=(), provenance=None))
    with pytest.raises(StatelessSessionError):
        service.inspect_staged(InspectRequest("memory", "r", "AAAA"))
    with pytest.raises(StatelessSessionError):
        service.commit_curated(CommitIntent("memory", "r", "AAAA", "c" * 64, (), w.ApprovalAuthorization(kind="not_required")))
    with pytest.raises(StatelessSessionError):
        service.capture_continuity(ContinuityCapture("cr", "compression_snapshot", "t", "compression"))
    assert len(factory.backend.calls) == calls


# --- Row 11: version_conflict ------------------------------------------------


@pytest.mark.parametrize("change", ["external_write", "hidden_change", "visibility_change"])
def test_row11_native_version_conflict_on_stage(tmp_path, change):
    store = _store()
    service, factory = _select(tmp_path, store)
    stale = service.load_curated("memory")
    if change == "external_write":
        store.external_write(REPO, "memory", "concurrent")
    elif change == "hidden_change":
        store.hidden_change(REPO, "memory")
    else:
        store.visibility_change()
    with pytest.raises(ProviderError) as exc:
        service.stage_curated(_mutation(stale))
    assert exc.value.code == "version_conflict" and exc.value.outcome == "not_committed"
    current = w.decode_error_details("version_conflict", exc.value.details).current_snapshot
    assert current.revision != stale.revision and current.frozen_identity == service.identity.to_wire()
    assert store.stage_state("r1") == "absent" and service.blocked is False
    fresh = service.load_curated("memory")
    assert fresh.revision == current.revision
    staged = service.stage_curated(_mutation(fresh, "r2"))
    assert service.commit_curated(_intent(staged)).outcome == "committed_audit_clean"


def test_row11_native_version_conflict_on_commit_requires_a_new_request_id(tmp_path):
    store = _store()
    service, factory = _select(tmp_path, store)
    snap = service.load_curated("memory")
    staged = service.stage_curated(_mutation(snap))
    store.external_write(REPO, "memory", "concurrent")
    with pytest.raises(ProviderError) as exc:
        service.commit_curated(_intent(staged))
    assert exc.value.code == "version_conflict" and exc.value.outcome == "not_committed"
    current = w.decode_error_details("version_conflict", exc.value.details).current_snapshot
    assert [e.text for e in current.mutation_entries] == ["concurrent"]
    assert [r.text for r in store.records_for(REPO, "memory")] == ["concurrent"] and service.blocked is False
    fresh = service.load_curated("memory")
    with pytest.raises(ProviderError) as exc:  # same request_id at the new revision is a changed replay
        service.stage_curated(_mutation(fresh))
    assert exc.value.code == "idempotency_mismatch"
    restaged = service.stage_curated(_mutation(fresh, "r2"))
    assert service.commit_curated(_intent(restaged)).outcome == "committed_audit_clean"
    assert [r.text for r in store.records_for(REPO, "memory")] == ["concurrent", "x"]


# --- Row 12: unknown commit outcome -----------------------------------------


@pytest.mark.parametrize("fault", ["before", "during", "after_publish", "corrupt"])
def test_row12_unknown_commit_outcome_then_exact_retry(tmp_path, fault):
    store = _store()
    service, factory = _select(tmp_path, store)
    snap = service.load_curated("memory")
    staged = service.stage_curated(_mutation(snap))
    if fault == "corrupt":
        store.corrupt_result("commit_curated", set_key("request_id", "other"))
    else:
        store.fail_transport("commit_curated", phase=fault)
    with pytest.raises(ProviderTransportError) as exc:
        service.commit_curated(_intent(staged))
    assert exc.value.mutation_outcome_unknown is True and service.blocked
    landed = fault != "before"
    assert ([r.text for r in store.records_for(REPO, "memory")] == ["x"]) is landed
    calls = len(factory.backend.calls)
    with pytest.raises(MemoryBlockedError):
        service.commit_curated(_intent(staged))  # refused locally until a fresh load
    assert len(factory.backend.calls) == calls
    service.load_curated("memory")
    assert not service.blocked
    if landed:
        with pytest.raises(ProviderError) as changed:
            service.commit_curated(_intent(staged, scopes=(REPO, PROJ)))
        assert changed.value.code == "idempotency_mismatch"
    else:
        with pytest.raises(ProviderError) as changed:
            service.commit_curated(_intent(staged, binding="d" * 64))
        assert changed.value.code == "approval_invalid"
    retry = service.commit_curated(_intent(staged))
    assert retry.outcome == ("idempotent_replay" if landed else "committed_audit_clean")
    assert [r.text for r in store.records_for(REPO, "memory")] == ["x"]


# --- Row 13: binding revoked / invalid ---------------------------------------


def test_row13_real_revocation_and_registry_change(tmp_path):
    store = _store()
    service, factory = _select(tmp_path, store)
    old = service.identity
    other, _ = _select(tmp_path, store, session="sess-2")
    other.start(replace(_context("sess-3"), resolution_source="explicit_ids", canonical_directory=None, repo_id="repo-1", project_id="proj-1"), bind_intent="session_reset", prior_identity=old)
    with pytest.raises(MemoryBlockedError) as exc:
        service.load_curated("memory")
    assert exc.value.code == "binding_revoked" and exc.value.provider_error.code == "binding_revoked"
    assert "rebind" in service.degraded_warning() and "binding_revoked" in service.degraded_warning()
    calls = len(factory.backend.calls)
    with pytest.raises(MemoryBlockedError, match="rebind"):
        service.load_curated("memory")
    assert len(factory.backend.calls) == calls
    service.start(_context("sess-4"), bind_intent="explicit_rebind", prior_identity=old)  # only start() recovers
    assert service.load_curated("memory").status == "ok" and not service.blocked
    # registry change: the known handle at the older revision is binding_revoked (contract C7, §7.3 L786)
    current = service.identity
    store.registry_change()
    with pytest.raises(MemoryBlockedError) as exc:
        service.load_curated("memory")
    assert exc.value.code == "binding_revoked"
    tampered = FrozenMemoryIdentity.from_wire(replace(current.to_wire(), logical_session_id="tampered"))
    with pytest.raises(MemoryBlockedError) as exc:
        _resume(tmp_path, store, HostSessionState(provider_epoch="ep-1", identity=tampered))
    assert exc.value.__cause__.code == "binding_invalid"
    unknown = FrozenMemoryIdentity.from_wire(replace(current.to_wire(), opaque_binding_b64url="A" * 43))
    with pytest.raises(MemoryBlockedError) as exc:
        _resume(tmp_path, store, HostSessionState(provider_epoch="ep-1", identity=unknown))
    assert exc.value.__cause__.code == "binding_invalid"
    service.start(_context("sess-5"), bind_intent="explicit_rebind", prior_identity=current)
    assert service.identity.binding_revision == "rev-2" and service.load_curated("memory").status == "ok"


# --- Row 14: provider epoch change ------------------------------------------


@pytest.mark.parametrize("operation", ["load_curated", "stage_curated", "commit_curated"])
def test_row14_typed_epoch_change_voids_state_until_rebind(tmp_path, operation):
    store = _store()
    service, factory = _select(tmp_path, store)
    snap = service.load_curated("memory")
    staged = service.stage_curated(_mutation(snap))
    old = service.identity
    store.set_epoch("ep-2")
    actions = {
        "load_curated": lambda: service.load_curated("memory"),
        "stage_curated": lambda: service.stage_curated(_mutation(snap, "r2")),
        "commit_curated": lambda: service.commit_curated(_intent(staged)),
    }
    with pytest.raises(MemoryBlockedError) as exc:
        actions[operation]()
    assert exc.value.code == "provider_epoch_changed"
    assert w.decode_error_details("provider_epoch_changed", exc.value.provider_error.details).current_provider_epoch == "ep-2"
    assert service.epoch_changed and service.blocked
    calls = len(factory.backend.calls)
    for action in actions.values():
        with pytest.raises(MemoryBlockedError, match="rebind"):
            action()
    assert len(factory.backend.calls) == calls
    service.start(_context("sess-2"), bind_intent="explicit_rebind", prior_identity=old)
    assert service.session_state.provider_epoch == "ep-2" and not service.epoch_changed and not service.blocked
    with pytest.raises(ProviderError) as exc:  # the old stage handle is void at the new epoch
        service.commit_curated(_intent(staged))
    assert exc.value.code == "stage_not_found" and exc.value.details is None
    with pytest.raises(ProviderError) as exc:  # and so is the old token/revision
        service.stage_curated(_mutation(snap, "r3"))
    assert exc.value.code == "version_conflict"
    fresh = service.load_curated("memory")
    assert fresh.revision.provider_epoch == "ep-2" and service.stage_curated(_mutation(fresh, "r3")).request_id == "r3"


def test_row14_envelope_epoch_change_latches_the_same_way(tmp_path):
    store = _store()
    service, factory = _select(tmp_path, store)
    store.envelope_epoch_override("load_curated", "ep-other")
    with pytest.raises(MemoryBlockedError, match="epoch"):
        service.load_curated("memory")
    assert service.epoch_changed and service.blocked
    with pytest.raises(MemoryBlockedError, match="rebind"):
        service.load_curated("memory")


# --- Row 15: provider later recovers ----------------------------------------


def test_row15_fail_closed_session_resumes_on_the_same_binding(tmp_path):
    store = _store()
    service, factory = _select(tmp_path, store)
    snap = service.load_curated("memory")
    store.fail_transport("load_curated", times=2)
    for _ in range(2):
        with pytest.raises(MemoryBlockedError):
            service.load_curated("memory")
    assert service.blocked
    identity = service.identity
    fresh = service.load_curated("memory")  # faults cleared: same binding, fresh load
    assert not service.blocked and service.identity is identity and fresh.frozen_identity == identity.to_wire()
    assert service.commit_curated(_intent(service.stage_curated(_mutation(fresh)))).outcome == "committed_audit_clean"


def test_row15_stateless_never_recovers_but_a_new_session_can(tmp_path):
    store = _store()
    store.fail_transport("negotiate")
    service, factory = _select(tmp_path, store, policy="stateless")
    assert isinstance(service, StatelessMemoryService)
    with pytest.raises(StatelessSessionError):
        service.load_curated("memory")
    fresh, _ = _select(tmp_path, store, policy="stateless", session="sess-2")
    assert isinstance(fresh, ProviderAuthoritativeMemoryService) and fresh.load_curated("memory").status == "ok"
    with pytest.raises(StatelessSessionError):
        service.load_curated("memory")


# --- Row 16: no explicit frozen identity -------------------------------------


def test_row16_unbound_service_refuses_locally_with_zero_backend_calls(tmp_path):
    store = _store()
    backend = FakeAuthoritativeBackend(store)
    service = ProviderAuthoritativeMemoryService(resolve_memory_service_config(_config(tmp_path)), backend)
    revision = w.CompositeRevision("ep-1", "1", ())
    actions = [
        lambda: service.load_curated("memory"),
        lambda: service.prompt_block("memory"),
        lambda: service.stage_curated(MutationRequest(target="memory", request_id="r", expected_revision=revision, hidden_preservation_state=None, requested_write_scopes=(), intent=None, mutation_delta=(), candidate_entries=(), provenance=None)),
        lambda: service.inspect_staged(InspectRequest("memory", "r", "AAAA")),
        lambda: service.commit_curated(CommitIntent("memory", "r", "AAAA", "c" * 64, (), w.ApprovalAuthorization(kind="not_required"))),
        lambda: service.recall_context(RecallQuery("memory", revision, "q", ("general",), (), w.RecallBudget(0, 0, 0))),
        lambda: service.capture_continuity(ContinuityCapture("cr", "compression_snapshot", "t", "compression")),
    ]
    for action in actions:
        with pytest.raises(MemoryBlockedError):
            action()
    assert backend.calls == [] and service.identity is None


# --- Row 17: knob only (the surfaces are R41/R44's) --------------------------


@pytest.mark.parametrize("operation", ["negotiate", "validate_session"])
def test_row17_resume_fault_knobs_exist(tmp_path, operation):
    store = _store()
    first, _ = _select(tmp_path, store)
    store.fail_transport(operation)
    with pytest.raises(MemoryBlockedError):
        _resume(tmp_path, store, first.session_state)
