"""Backend semantics of the stateful fake authoritative provider (R36).

Every test drives ``FakeAuthoritativeBackend`` directly with wire-typed
requests; the service-level behaviour is proven in ``test_failure_matrix.py``
and ``test_fake_contracts.py``.
"""

import base64
import hashlib
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from agent.memory_service import wire as w
from agent.memory_service.errors import ProviderError, ProviderTransportError

from tests.agent.memory_service.fake_backend import (
    APPROVAL_STORE_FAILURE_CODE,
    DEFAULT_CURATED_LIMITS,
    DEFAULT_LIMITS,
    EPOCH_BINDING_OUTCOME_ON_MUTATION,
    FakeAuthoritativeBackend,
    FakeClock,
    FakeProviderStore,
    approval_binding_sha256,
    drop_key,
)

CLOCK_START = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)
REPO = w.ScopeRef(kind="repository", id="repo-1")
PROJ = w.ScopeRef(kind="project", id="proj-1")
PG = w.ScopeRef(kind="principal_global", id="ethan")
PROVENANCE = w.MutationProvenance(actor_kind="hermes", principal_id="ethan", logical_session_id="sess-1", initiating_surface="memory_tool", source_entry_ids=(), source_commit=None, threat_decision_id=None)


def _b64_len(value: str) -> int:
    return len(base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)))


def _store(**kwargs) -> FakeProviderStore:
    kwargs.setdefault("clock", FakeClock(CLOCK_START))
    return FakeProviderStore(**kwargs)


def _context(session="sess-1", **overrides) -> w.RequestedContext:
    base = dict(principal_id="ethan", profile_id="default", logical_session_id=session, platform="cli", org_id=None, project_id=None, repo_id=None, workspace_id=None, resolution_source="directory", canonical_directory="C:\\work\\repo")
    base.update(overrides)
    return w.RequestedContext(**base)


def _bind(backend, session="sess-1", *, intent="new_session", prior=None, request_id=None, context=None, epoch="ep-1") -> w.BindResult:
    request = w.BindRequest(expected_provider_epoch=epoch, binding_request_id=request_id or f"bind-{session}-{intent}", bind_intent=intent, requested_context=context or _context(session), prior_identity=prior)
    request.validate()
    return backend.bind_session(request).result


def _validate(backend, identity, epoch="ep-1"):
    return backend.validate_session(w.ValidateSessionRequest(expected_provider_epoch=epoch, frozen_identity=identity)).result


def _load(backend, identity, target="memory", epoch="ep-1") -> w.CuratedSnapshot:
    return backend.load_curated(w.LoadRequest(expected_provider_epoch=epoch, frozen_identity=identity, target=target)).result


def _stage_request(identity, snapshot, request_id="r1", text="x", epoch="ep-1", **overrides) -> w.StageRequest:
    scope = snapshot.default_write_scope
    base = dict(expected_provider_epoch=epoch, frozen_identity=identity, target="memory", expected_revision=snapshot.revision, hidden_preservation_state=snapshot.hidden_preservation_state, request_id=request_id, requested_write_scopes=(scope,), intent=w.MutationIntent(kind="add"), mutation_delta=(w.MutationDeltaItem(action="add", client_ref="c1"),), candidate_entries=(w.CandidateEntry(client_ref="c1", text=text, destination_scope=scope, target="memory", proposed_policy_key=None, import_source_identity=None),), provenance=PROVENANCE)
    base.update(overrides)
    request = w.StageRequest(**base)
    request.validate()
    return request


def _dummy_snapshot_fields(identity):
    """Enough of a snapshot to build a StageRequest before load exists (Task 1 only)."""
    revision = w.CompositeRevision(provider_epoch="ep-1", visibility_revision="0", scope_revisions=(w.ScopeRevision(scope=REPO, revision="0"),))
    hidden = w.HiddenPreservationState(target="memory", complete_for_scopes=(REPO,), opaque_state_b64url="AAAA")
    return revision, hidden


def _commit_request(identity, request_id="r1", handle="AAAA", binding="c" * 64, scopes=(REPO,), authorization=None, epoch="ep-1", target="memory") -> w.CommitRequest:
    request = w.CommitRequest(expected_provider_epoch=epoch, frozen_identity=identity, target=target, request_id=request_id, stage_handle_b64url=handle, approval_binding_sha256=binding, authorized_write_scopes=tuple(scopes), authorization=authorization or w.ApprovalAuthorization(kind="not_required"))
    request.validate()
    return request


# --- Task 1: transport, negotiate, bind, validate, epoch, registry, revocation ---


def test_typed_failures_round_trip_through_wire_failure():
    store = _store()
    backend = FakeAuthoritativeBackend(store)
    bound = _bind(backend)
    store.fail_typed("load_curated", "ambiguous_policy", details=None)
    with pytest.raises(w.WireError):  # null details are illegal for ambiguous_policy: the fake refuses its own emission
        _load(backend, bound.frozen_identity)
    store.fail_typed("load_curated", "ambiguous_policy", details={"ambiguities": [{"policy_key": "k", "tier": "dependency", "candidate_count": 2}]})
    with pytest.raises(ProviderError) as exc:
        _load(backend, bound.frozen_identity)
    assert exc.value.code == "ambiguous_policy" and exc.value.outcome == "not_applicable"
    assert exc.value.details == {"ambiguities": [{"policy_key": "k", "tier": "dependency", "candidate_count": 2}]}
    assert exc.value.operation == "load_curated"


def test_negotiate_publishes_no_state_and_reports_capabilities():
    store = _store()
    backend = FakeAuthoritativeBackend(store, recall=True, continuity=False)
    request = w.NegotiateRequest(host="hermes", supported_api_versions=(1,), required_operations=("bind_session", "validate_session", "load_curated", "stage_curated", "inspect_staged", "commit_curated"))
    first = backend.negotiate(request)
    second = backend.negotiate(request)
    assert first.result == second.result and first.provider_epoch == second.provider_epoch == "ep-1"
    assert store.events == [("negotiate", "ok", False), ("negotiate", "ok", False)]
    assert store.epoch == "ep-1" and store.handles == {} and store.records == {}
    neg = first.result
    assert neg.provider == "example" and neg.selected_api_version == 1
    assert "recall_context" in neg.operations and "capture_continuity" not in neg.operations
    assert neg.capabilities == w.Capabilities(recall_context=True, capture_continuity=False)
    neg.limits.validate()
    assert neg.limits.stage_ttl_seconds <= 86400 and neg.limits.max_opaque_binding_bytes >= 32
    both = FakeAuthoritativeBackend(store, recall=False, continuity=True).negotiate(request).result
    assert "recall_context" not in both.operations and "capture_continuity" in both.operations
    assert both.capabilities == w.Capabilities(recall_context=False, capture_continuity=True)


def test_bind_new_session_freezes_identity_and_registry_revision():
    store = _store()
    backend = FakeAuthoritativeBackend(store)
    bound = _bind(backend)
    identity = bound.frozen_identity
    assert identity.repo_id == "repo-1" and identity.project_id == "proj-1" and identity.org_id is None
    assert identity.provider == "example" and identity.provider_mode == "authoritative"
    assert identity.principal_id == "ethan" and identity.logical_session_id == "sess-1" and identity.platform == "cli"
    assert list(bound.visible_scopes) == [REPO, PROJ, PG]
    assert bound.memory_default_write_scope == REPO
    assert list(bound.memory_eligible_write_scopes) == [REPO, PROJ]
    assert bound.user_write_scope == PG
    assert identity.binding_revision == store.registry.revision == "rev-1"
    assert _b64_len(identity.opaque_binding_b64url) == 32
    other = _bind(backend, "sess-2")
    assert other.frozen_identity.opaque_binding_b64url != identity.opaque_binding_b64url
    assert set(store.handles) == {identity.opaque_binding_b64url, other.frozen_identity.opaque_binding_b64url}


def test_bind_replay_and_mismatch():
    store = _store()
    backend = FakeAuthoritativeBackend(store)
    first = _bind(backend, request_id="b1")
    replay = _bind(backend, request_id="b1")
    assert replay == first
    assert len(store.handles) == 1
    with pytest.raises(ProviderError) as exc:
        _bind(backend, "sess-other", request_id="b1")
    assert exc.value.code == "idempotency_mismatch" and exc.value.outcome == "not_applicable"
    assert len(store.handles) == 1


def test_bind_resolution_failures_are_invalid_request():
    store = _store()
    backend = FakeAuthoritativeBackend(store)
    attempts = [
        _context(canonical_directory="C:\\elsewhere"),
        _context(resolution_source="explicit_ids", canonical_directory=None, repo_id="repo-1", project_id=None),
        _context(resolution_source="explicit_ids", canonical_directory=None, repo_id="repo-1", project_id="proj-other"),
        _context(resolution_source="explicit_ids", canonical_directory=None, project_id="proj-unknown"),
        _context(resolution_source="workspace", workspace_id="ws-unknown", canonical_directory="C:\\work\\repo"),
        _context(canonical_directory="C:\\work\\repo", repo_id="repo-9"),
    ]
    for i, context in enumerate(attempts):
        with pytest.raises(ProviderError) as exc:
            _bind(backend, request_id=f"bad-{i}", context=context)
        assert exc.value.code == "invalid_request" and exc.value.outcome == "not_applicable" and exc.value.details is None
    assert store.handles == {}


def test_bind_unresolved_context_freezes_a_degraded_binding():
    store = _store()
    backend = FakeAuthoritativeBackend(store)
    context = _context(resolution_source="explicit_ids", canonical_directory=None)
    bound = _bind(backend, context=context)
    assert bound.memory_default_write_scope is None
    assert bound.memory_eligible_write_scopes == ()
    assert list(bound.visible_scopes) == [PG]
    assert bound.user_write_scope == PG
    assert bound.frozen_identity.repo_id is None and bound.frozen_identity.project_id is None
    assert store.handles[bound.frozen_identity.opaque_binding_b64url].degraded is True


def test_session_reset_and_explicit_rebind_revoke_the_prior_handle():
    store = _store()
    backend = FakeAuthoritativeBackend(store)
    old = _bind(backend).frozen_identity
    fresh = _bind(backend, "sess-2", intent="session_reset", prior=old, context=_context("sess-2", resolution_source="explicit_ids", canonical_directory=None, repo_id="repo-1", project_id="proj-1"))
    assert fresh.frozen_identity.opaque_binding_b64url != old.opaque_binding_b64url
    with pytest.raises(ProviderError) as exc:
        _validate(backend, old)
    assert exc.value.code == "binding_revoked"
    assert _validate(backend, fresh.frozen_identity).valid is True
    # a reset before any bind revokes nothing
    handles_before = {h: r.revoked for h, r in store.handles.items()}
    _bind(backend, "sess-3", intent="session_reset", prior=None, context=_context("sess-3", resolution_source="explicit_ids", canonical_directory=None, repo_id="repo-1", project_id="proj-1"))
    assert {h: r.revoked for h, r in store.handles.items() if h in handles_before} == handles_before
    # explicit_rebind behaves like reset
    rebound = _bind(backend, "sess-4", intent="explicit_rebind", prior=fresh.frozen_identity)
    with pytest.raises(ProviderError) as exc:
        _validate(backend, fresh.frozen_identity)
    assert exc.value.code == "binding_revoked"
    assert _validate(backend, rebound.frozen_identity).valid is True
    # revoking an already revoked handle is idempotent
    again = _bind(backend, "sess-5", intent="explicit_rebind", prior=fresh.frozen_identity)
    assert again.frozen_identity.logical_session_id == "sess-5"
    # an unknown prior handle is binding_invalid (D-R36-8) and creates no handle
    unknown = replace(old, opaque_binding_b64url=base64.urlsafe_b64encode(bytes(32)).rstrip(b"=").decode())
    count = len(store.handles)
    with pytest.raises(ProviderError) as exc:
        _bind(backend, "sess-6", intent="explicit_rebind", prior=unknown)
    assert exc.value.code == "binding_invalid" and exc.value.outcome == "not_applicable"
    assert len(store.handles) == count


def test_validate_session_outcomes():
    store = _store()
    backend = FakeAuthoritativeBackend(store)
    bound = _bind(backend)
    identity = bound.frozen_identity
    result = _validate(backend, identity)
    assert result.valid is True and result.frozen_identity == identity and list(result.visible_scopes) == list(bound.visible_scopes)
    unknown = replace(identity, opaque_binding_b64url=base64.urlsafe_b64encode(bytes(range(32))).rstrip(b"=").decode())
    with pytest.raises(ProviderError) as exc:
        _validate(backend, unknown)
    assert exc.value.code == "binding_invalid" and exc.value.outcome == "not_applicable" and exc.value.details is None
    with pytest.raises(ProviderError) as exc:
        _validate(backend, replace(identity, logical_session_id="someone-else"))
    assert exc.value.code == "binding_invalid"
    store.revoke(identity.opaque_binding_b64url)
    with pytest.raises(ProviderError) as exc:
        _validate(backend, identity)
    assert exc.value.code == "binding_revoked" and exc.value.outcome == "not_applicable"


def test_registry_change_revokes_known_handles_on_older_revisions():
    """Contract C7 (§7.3 L786): a known, untampered handle bound at an older
    binding_revision is binding_revoked after activation; unknown or tampered
    handles stay binding_invalid; a handle bound after the change validates."""
    store = _store()
    backend = FakeAuthoritativeBackend(store)
    old = _bind(backend).frozen_identity
    assert old.binding_revision == "rev-1"
    store.registry_change()
    assert store.registry.revision == "rev-2"
    with pytest.raises(ProviderError) as exc:
        _validate(backend, old)
    assert exc.value.code == "binding_revoked" and exc.value.outcome == "not_applicable"
    with pytest.raises(ProviderError) as exc:
        _load(backend, old)
    assert exc.value.code == "binding_revoked" and exc.value.outcome == "not_applicable"
    revision, hidden = _dummy_snapshot_fields(old)
    stage = w.StageRequest(expected_provider_epoch="ep-1", frozen_identity=old, target="memory", expected_revision=revision, hidden_preservation_state=hidden, request_id="r1", requested_write_scopes=(REPO,), intent=w.MutationIntent(kind="add"), mutation_delta=(w.MutationDeltaItem(action="add", client_ref="c1"),), candidate_entries=(w.CandidateEntry(client_ref="c1", text="x", destination_scope=REPO, target="memory", proposed_policy_key=None, import_source_identity=None),), provenance=PROVENANCE)
    with pytest.raises(ProviderError) as exc:
        backend.stage_curated(stage)
    assert exc.value.code == "binding_revoked" and exc.value.outcome == EPOCH_BINDING_OUTCOME_ON_MUTATION
    unknown = replace(old, opaque_binding_b64url=base64.urlsafe_b64encode(bytes(32)).rstrip(b"=").decode())
    with pytest.raises(ProviderError) as exc:
        _validate(backend, unknown)
    assert exc.value.code == "binding_invalid"
    with pytest.raises(ProviderError) as exc:
        _validate(backend, replace(old, binding_revision="rev-2"))  # tampered: the handle froze rev-1
    assert exc.value.code == "binding_invalid"
    new = _bind(backend, "sess-2").frozen_identity
    assert new.binding_revision == "rev-2"
    assert _validate(backend, new).valid is True


def test_epoch_change_is_typed_with_details_and_voids_state():
    store = _store()
    backend = FakeAuthoritativeBackend(store)
    old = _bind(backend).frozen_identity
    store.tokens[(old.opaque_binding_b64url, "memory")] = ("token", None)
    store.stages["stage-handle"] = object()
    store.stage_by_request[("ep-1", "r1")] = "stage-handle"
    store.receipts[("ep-1", "r0")] = object()
    store.tombstones[("ep-1", "r-old")] = "2026-09-17T11:00:00Z"
    store.set_epoch("ep-2")
    assert store.epoch == "ep-2"
    assert store.tokens == {} and store.stages == {} and store.stage_by_request == {} and store.receipts == {} and store.tombstones == {}
    expected_details = {"expected_provider_epoch": "ep-1", "current_provider_epoch": "ep-2"}
    with pytest.raises(ProviderError) as exc:
        _validate(backend, old, epoch="ep-1")
    assert exc.value.code == "provider_epoch_changed" and exc.value.outcome == "not_applicable" and exc.value.details == expected_details
    with pytest.raises(ProviderError) as exc:
        _load(backend, old, epoch="ep-1")
    assert exc.value.code == "provider_epoch_changed" and exc.value.outcome == "not_applicable"
    revision, hidden = _dummy_snapshot_fields(old)
    stage = w.StageRequest(expected_provider_epoch="ep-1", frozen_identity=old, target="memory", expected_revision=revision, hidden_preservation_state=hidden, request_id="r1", requested_write_scopes=(REPO,), intent=w.MutationIntent(kind="add"), mutation_delta=(w.MutationDeltaItem(action="add", client_ref="c1"),), candidate_entries=(w.CandidateEntry(client_ref="c1", text="x", destination_scope=REPO, target="memory", proposed_policy_key=None, import_source_identity=None),), provenance=PROVENANCE)
    with pytest.raises(ProviderError) as exc:
        backend.stage_curated(stage)
    assert exc.value.code == "provider_epoch_changed" and exc.value.outcome == EPOCH_BINDING_OUTCOME_ON_MUTATION and exc.value.details == expected_details
    with pytest.raises(ProviderError) as exc:
        backend.commit_curated(_commit_request(old, epoch="ep-1"))
    assert exc.value.code == "provider_epoch_changed" and exc.value.outcome == EPOCH_BINDING_OUTCOME_ON_MUTATION
    with pytest.raises(ProviderError) as exc:
        _bind(backend, "sess-2", epoch="ep-1")
    assert exc.value.code == "provider_epoch_changed" and exc.value.outcome == "not_applicable"
    # the old handle can still be rebound away from at the new epoch
    rebound = _bind(backend, "sess-2", intent="explicit_rebind", prior=old, epoch="ep-2")
    assert _validate(backend, rebound.frozen_identity, epoch="ep-2").valid is True


def test_transport_faults_and_shutdown():
    store = _store()
    backend = FakeAuthoritativeBackend(store)
    store.fail_transport("bind_session")
    with pytest.raises(ProviderTransportError) as exc:
        _bind(backend)
    assert exc.value.mutation_outcome_unknown is False and exc.value.operation == "bind_session"
    bound = _bind(backend)  # recovered after one fault
    store.fail_transport("commit_curated", reason="crash")
    with pytest.raises(ProviderTransportError) as exc:
        backend.commit_curated(_commit_request(bound.frozen_identity))
    assert exc.value.mutation_outcome_unknown is True and exc.value.reason == "crash"
    backend.shutdown()
    assert backend.shutdown_calls == 1
    with pytest.raises(ProviderTransportError):
        _validate(backend, bound.frozen_identity)
    with pytest.raises(ProviderTransportError):
        backend.negotiate(w.NegotiateRequest(host="hermes", supported_api_versions=(1,), required_operations=()))
    second = FakeAuthoritativeBackend(store)
    assert _validate(second, bound.frozen_identity).valid is True


def test_requests_and_results_are_strictly_decoded():
    store = _store()
    backend = FakeAuthoritativeBackend(store)
    bound = _bind(backend)
    calls_before = len(backend.calls)
    with pytest.raises(w.WireError, match="target"):
        backend.load_curated(w.LoadRequest(expected_provider_epoch="ep-1", frozen_identity=bound.frozen_identity, target="profile"))
    assert len(backend.calls) == calls_before  # an undecodable request is never recorded
    store.corrupt_result("bind_session", drop_key("user_write_scope"))
    with pytest.raises(w.WireError, match="user_write_scope"):
        _bind(backend, "sess-2")
    assert _bind(backend, "sess-3").user_write_scope == PG  # the corruption was consumed


# --- Task 2: load, hidden preservation, revisions, ambiguous_policy ---


def _seed_mixed(store):
    """One record of every kind the memory snapshot must partition."""
    general = store.seed_general(REPO, "general policy", policy_key="k-general")
    evidence = store.seed_record(REPO, "memory", "visible evidence")
    raw = store.seed_record(REPO, "memory", "hidden raw body", lane="raw")
    superseded = store.seed_record(REPO, "memory", "old superseded", lifecycle="superseded")
    user = store.seed_record(PG, "user", "user profile line", lane="trusted_instruction", policy_key="profile:u1")
    return general, evidence, raw, superseded, user


def test_load_memory_snapshot_is_complete_and_channel_specific():
    store = _store()
    general, evidence, raw, superseded, user = _seed_mixed(store)
    backend = FakeAuthoritativeBackend(store)
    bound = _bind(backend)
    snap = _load(backend, bound.frozen_identity, "memory")
    snap.validate()
    assert snap.status == "ok" and snap.target == "memory" and snap.api_version == 1
    assert snap.frozen_identity == bound.frozen_identity
    assert [e.id for e in snap.mutation_entries] == [evidence.id]
    assert all(e.record_channel == "hermes_memory" and e.target == "memory" and e.lifecycle == "active" for e in snap.mutation_entries)
    assert [e.id for e in snap.delivery_entries] == [general.id, evidence.id]  # general first (ruling R36-K)
    assert snap.delivery_entries[0].record_channel == "general" and snap.delivery_entries[0].target is None
    assert snap.delivery_entries[0].delivery_tier == "repository"
    text = w.canonical_json(snap.to_wire()).decode()
    for hidden in (raw, superseded, user):
        assert hidden.id not in text and hidden.text not in text
    assert [r.scope for r in snap.revision.scope_revisions] == list(bound.visible_scopes) == [REPO, PROJ, PG]
    assert snap.revision.provider_epoch == "ep-1"
    assert list(snap.complete_for_scopes) == list(snap.eligible_write_scopes) == [REPO, PROJ]
    assert snap.default_write_scope == REPO and list(snap.visible_scopes) == [REPO, PROJ, PG]
    assert snap.limits == store.curated_limits
    assert snap.hidden_preservation_state.target == "memory" and list(snap.hidden_preservation_state.complete_for_scopes) == [REPO, PROJ]


def test_load_user_snapshot_delivers_only_hermes_user():
    store = _store()
    general, evidence, raw, superseded, user = _seed_mixed(store)
    backend = FakeAuthoritativeBackend(store)
    bound = _bind(backend)
    snap = _load(backend, bound.frozen_identity, "user")
    snap.validate()
    assert snap.status == "ok" and snap.target == "user"
    assert [e.id for e in snap.mutation_entries] == [user.id] and snap.mutation_entries[0].record_channel == "hermes_user"
    assert [e.id for e in snap.delivery_entries] == [user.id] and snap.delivery_entries[0].target == "user"
    assert snap.delivery_entries[0].lane == "trusted_instruction" and snap.delivery_entries[0].policy_key == "profile:u1"
    assert snap.default_write_scope == PG and list(snap.eligible_write_scopes) == [PG] == list(snap.complete_for_scopes)
    text = w.canonical_json(snap.to_wire()).decode()
    for absent in (general, evidence, raw, superseded):
        assert absent.id not in text and absent.text not in text


def test_degraded_binding_loads_degraded_global_only_for_memory_and_ok_for_user():
    store = _store()
    general, evidence, raw, superseded, user = _seed_mixed(store)
    pg_general = store.seed_general(PG, "global policy")
    pg_memory = store.seed_record(PG, "memory", "global evidence")
    backend = FakeAuthoritativeBackend(store)
    bound = _bind(backend, context=_context(resolution_source="explicit_ids", canonical_directory=None))
    memory = _load(backend, bound.frozen_identity, "memory")
    memory.validate()
    assert memory.status == "degraded_global_only"
    assert list(memory.visible_scopes) == [PG] and memory.default_write_scope is None
    assert memory.eligible_write_scopes == () and memory.complete_for_scopes == () and memory.mutation_entries == ()
    assert [e.id for e in memory.delivery_entries] == [pg_general.id, pg_memory.id]
    assert [r.scope for r in memory.revision.scope_revisions] == [PG]
    text = w.canonical_json(memory.to_wire()).decode()
    assert general.id not in text and evidence.id not in text and raw.text not in text
    user_snap = _load(backend, bound.frozen_identity, "user")
    user_snap.validate()
    assert user_snap.status == "ok" and list(user_snap.visible_scopes) == [PG]
    assert user_snap.default_write_scope == PG and list(user_snap.eligible_write_scopes) == [PG]
    assert [e.id for e in user_snap.mutation_entries] == [user.id]


def test_hidden_token_is_random_stable_per_revision_and_superseded_by_publication():
    store = _store()
    backend = FakeAuthoritativeBackend(store)
    bound = _bind(backend)
    handle = bound.frozen_identity.opaque_binding_b64url
    first = _load(backend, bound.frozen_identity)
    second = _load(backend, bound.frozen_identity)
    assert first.hidden_preservation_state == second.hidden_preservation_state
    token = first.hidden_preservation_state.opaque_state_b64url
    assert _b64_len(token) == 32
    assert store.token_for(handle, "memory") == token
    repo_revision = lambda snap: [r.revision for r in snap.revision.scope_revisions if r.scope == REPO][0]  # noqa: E731
    store.external_write(REPO, "memory", "x")
    third = _load(backend, bound.frozen_identity)
    assert third.hidden_preservation_state.opaque_state_b64url != token
    assert repo_revision(third) != repo_revision(first)
    assert store.token_for(handle, "memory") == third.hidden_preservation_state.opaque_state_b64url
    assert [k for k in store.tokens if k[0] == handle] == [(handle, "memory")]
    other = _bind(backend, "sess-2")
    other_snap = _load(backend, other.frozen_identity)
    assert other_snap.revision == third.revision
    assert other_snap.hidden_preservation_state.opaque_state_b64url != third.hidden_preservation_state.opaque_state_b64url
    user_snap = _load(backend, bound.frozen_identity, "user")
    assert user_snap.hidden_preservation_state.opaque_state_b64url != third.hidden_preservation_state.opaque_state_b64url
    assert store.token_for(handle, "user") == user_snap.hidden_preservation_state.opaque_state_b64url


def test_revision_components_bump_independently():
    store = _store()
    backend = FakeAuthoritativeBackend(store)
    bound = _bind(backend)

    def parts(snap):
        return snap.revision.provider_epoch, snap.revision.visibility_revision, {r.scope: r.revision for r in snap.revision.scope_revisions}

    epoch, visibility, scopes = parts(_load(backend, bound.frozen_identity))
    store.external_write(PROJ, "memory", "project evidence")
    epoch2, visibility2, scopes2 = parts(_load(backend, bound.frozen_identity))
    assert epoch2 == epoch and visibility2 == visibility
    assert scopes2[PROJ] != scopes[PROJ] and scopes2[REPO] == scopes[REPO] and scopes2[PG] == scopes[PG]
    store.hidden_change(REPO, "memory")
    epoch3, visibility3, scopes3 = parts(_load(backend, bound.frozen_identity))
    assert scopes3[REPO] != scopes2[REPO] and scopes3[PROJ] == scopes2[PROJ] and visibility3 == visibility2 and epoch3 == epoch
    store.visibility_change()
    epoch4, visibility4, scopes4 = parts(_load(backend, bound.frozen_identity))
    assert visibility4 != visibility3 and scopes4 == scopes3 and epoch4 == epoch
    assert epoch == "ep-1"


def test_ambiguous_policy_on_required_load_carries_legal_details_and_no_snapshot():
    store = _store()
    store.seed_general(REPO, "the body of the conflicting instruction", policy_key="k")
    backend = FakeAuthoritativeBackend(store)
    bound = _bind(backend)
    store.add_policy_conflict("k", 3)
    events_before = len(store.events)
    with pytest.raises(ProviderError) as exc:
        _load(backend, bound.frozen_identity)
    assert exc.value.code == "ambiguous_policy" and exc.value.outcome == "not_applicable"
    details = w.decode_error_details("ambiguous_policy", exc.value.details)
    assert details.ambiguities[0].policy_key == "k" and details.ambiguities[0].candidate_count == 3 and details.ambiguities[0].tier == "dependency"
    assert store.events[events_before:] == [("load_curated", "ambiguous_policy", False)]
    assert "conflicting instruction" not in w.canonical_json(exc.value.details).decode()  # body-free: the key, not the text


def test_load_never_returns_version_conflict_natively():
    store = _store()
    backend = FakeAuthoritativeBackend(store)
    bound = _bind(backend)
    _load(backend, bound.frozen_identity)
    store.external_write(REPO, "memory", "x")
    store.hidden_change(REPO, "memory")
    store.visibility_change()
    store.block_store("git_dirty")
    assert _load(backend, bound.frozen_identity).status == "ok"
    assert _load(backend, bound.frozen_identity, "user").status == "ok"
    assert all(code != "version_conflict" for _, code, _ in store.events)
    store.fail_typed("load_curated", "version_conflict", details=None)  # only a scripted (adapter-simulated) conflict can appear on load
    with pytest.raises(ProviderError) as exc:
        _load(backend, bound.frozen_identity)
    assert exc.value.code == "version_conflict" and exc.value.details is None


# --- Task 3: stage ---


def _candidate(ref, text, scope, *, target="memory", policy_key=None, import_identity=None) -> w.CandidateEntry:
    return w.CandidateEntry(client_ref=ref, text=text, destination_scope=scope, target=target, proposed_policy_key=policy_key, import_source_identity=import_identity)


def _request(identity, snapshot, *, request_id="r1", target="memory", intent=None, delta=None, candidates=None, scopes=None, provenance=PROVENANCE, epoch="ep-1") -> w.StageRequest:
    if candidates is None:
        candidates = (_candidate("c1", "x", snapshot.default_write_scope, target=target),)
    if delta is None:
        delta = tuple(w.MutationDeltaItem(action="add", client_ref=c.client_ref) for c in candidates)
    if scopes is None:
        scopes = tuple(dict.fromkeys(c.destination_scope for c in candidates))
    request = w.StageRequest(expected_provider_epoch=epoch, frozen_identity=identity, target=target, expected_revision=snapshot.revision, hidden_preservation_state=snapshot.hidden_preservation_state, request_id=request_id, requested_write_scopes=tuple(scopes), intent=intent or w.MutationIntent(kind="add"), mutation_delta=tuple(delta), candidate_entries=tuple(candidates), provenance=provenance)
    request.validate()
    return request


def _stage(backend, request) -> w.StageResult:
    return backend.stage_curated(request).result


def _commit(backend, identity, staged, *, scopes=None, authorization=None, epoch="ep-1", target=None, binding=None) -> w.CommitResult:
    request = w.CommitRequest(expected_provider_epoch=epoch, frozen_identity=identity, target=target or staged.target, request_id=staged.request_id, stage_handle_b64url=staged.stage_handle_b64url, approval_binding_sha256=binding or staged.approval_binding_sha256, authorized_write_scopes=tuple(scopes if scopes is not None else staged.requested_write_scopes), authorization=authorization or w.ApprovalAuthorization(kind="not_required"))
    request.validate()
    return backend.commit_curated(request).result


def _approved(staged, *, by="ethan", expires_at=None, binding=None, approved_at="2026-09-17T12:00:00Z") -> w.ApprovalAuthorization:
    return w.ApprovalAuthorization(kind="approved", approval_id="ap-1", approved_by_principal_id=by, approved_at=approved_at, expires_at=expires_at or staged.expires_at, approval_binding_sha256=binding or staged.approval_binding_sha256)


def _session(store=None, session="sess-1", **store_kwargs):
    store = store or _store(**store_kwargs)
    backend = FakeAuthoritativeBackend(store)
    bound = _bind(backend, session)
    return store, backend, bound.frozen_identity


def test_stage_add_at_default_scope():
    store, backend, identity = _session()
    snap = _load(backend, identity)
    staged = _stage(backend, _request(identity, snap))
    assert staged.request_id == "r1" and staged.target == "memory"
    assert staged.expected_revision == snap.revision and list(staged.requested_write_scopes) == [REPO]
    assert list(staged.eligible_write_scopes) == [REPO, PROJ]
    assert staged.expires_at == "2026-09-17T13:00:00Z"  # clock + stage_ttl_seconds (3600)
    assert staged.approval_requirements == ()
    assert [(h.client_ref, h.canonical_sha256) for h in staged.candidate_hashes] == [("c1", hashlib.sha256(b"x").hexdigest())]
    assert len(staged.admissions) == 1
    admission = staged.admissions[0]
    assert admission == w.AdmissionDecision(client_ref="c1", assigned_id=admission.assigned_id, target="memory", record_channel="hermes_memory", origin_scope=REPO, disposition="scoped_evidence", publication_effect="create_record", superseded_id=None, policy_key=None)
    assert admission.assigned_id and admission.assigned_id not in store.records  # assigned, not yet published
    assert staged.hidden_effects == ()
    assert _b64_len(staged.stage_handle_b64url) == 32
    assert store.stage_state("r1") == "live"
    assert store.records_for(REPO, "memory") == []  # staging publishes nothing


@pytest.mark.parametrize("case,expected", [
    ("user_add", ("target_user",)),
    ("memory_project", ("non_default_scope",)),
    ("bulk_edit", ("bulk_edit",)),
    ("reset", ("reset",)),
    ("import", ("import",)),
    ("threat", ("threat",)),
    ("user_replace_threat", ("target_user", "threat")),
])
def test_approval_requirements_are_mechanical_and_ordered(case, expected):
    store, backend, identity = _session()
    user_record = store.seed_record(PG, "user", "old profile", lane="trusted_instruction", policy_key="profile:u1")
    evidence = store.seed_record(REPO, "memory", "existing")
    memory = _load(backend, identity, "memory")
    user = _load(backend, identity, "user")
    threat = replace(PROVENANCE, threat_decision_id="t1")
    if case == "user_add":
        request = _request(identity, user, target="user", candidates=(_candidate("c1", "new", PG, target="user"),))
    elif case == "memory_project":
        request = _request(identity, memory, candidates=(_candidate("c1", "y", PROJ),))
    elif case == "bulk_edit":
        request = _request(identity, memory, intent=w.MutationIntent(kind="bulk_edit"), candidates=(_candidate("c1", "y", REPO),), delta=(w.MutationDeltaItem(action="add", client_ref="c1"), w.MutationDeltaItem(action="retire", record_id=evidence.id)))
    elif case == "reset":
        request = _request(identity, memory, intent=w.MutationIntent(kind="reset", reset_scopes=(REPO,)), candidates=(), delta=(), scopes=(REPO,))
    elif case == "import":
        ident = w.ImportSourceIdentity(source_kind="native_memory", parser_version="hermes-native-v0.20.6", source_id="legacy", item_key="MEMORY.md/1")
        request = _request(identity, memory, intent=w.MutationIntent(kind="import", import_run_id="run-1", source_kind="native_memory"), candidates=(_candidate("c1", "imported", REPO, import_identity=ident),))
    elif case == "threat":
        request = _request(identity, memory, provenance=threat)
    else:
        request = _request(identity, user, target="user", intent=w.MutationIntent(kind="replace", matched_entry_id=user_record.id), candidates=(_candidate("c1", "new profile", PG, target="user"),), delta=(w.MutationDeltaItem(action="supersede", old_record_id=user_record.id, replacement_client_ref="c1"),), provenance=threat)
    staged = _stage(backend, request)
    assert staged.approval_requirements == expected


def test_binding_hash_shape_matches_builtin():
    from agent.memory_service.builtin import BuiltinMemoryService

    store, backend, identity = _session()
    snap = _load(backend, identity)
    request = _request(identity, snap)
    staged = _stage(backend, request)
    unsigned = replace(staged, approval_binding_sha256="0" * 64)
    assert approval_binding_sha256(unsigned, request) == staged.approval_binding_sha256
    assert BuiltinMemoryService._approval_binding(None, unsigned, request) == staged.approval_binding_sha256
    changed_provenance = replace(request, provenance=replace(PROVENANCE, threat_decision_id="t9"))
    changed_intent = replace(request, intent=w.MutationIntent(kind="bulk_edit"))
    changed_token = replace(request, hidden_preservation_state=replace(snap.hidden_preservation_state, opaque_state_b64url="AAAA"))
    hashes = {approval_binding_sha256(unsigned, r) for r in (request, changed_provenance, changed_intent, changed_token)}
    assert len(hashes) == 4
    # candidate bodies never enter the binding: only their hashes do
    assert approval_binding_sha256(unsigned, replace(request, candidate_entries=(replace(request.candidate_entries[0], text="other body"),))) == staged.approval_binding_sha256
    assert approval_binding_sha256(replace(unsigned, candidate_hashes=()), request) != staged.approval_binding_sha256


def test_cas_before_token():
    store, backend, identity = _session()
    stale = _load(backend, identity)
    store.external_write(REPO, "memory", "concurrent")
    stages_before = dict(store.stages)
    with pytest.raises(ProviderError) as exc:  # (a) stale revision with its matching stale token
        _stage(backend, _request(identity, stale))
    assert exc.value.code == "version_conflict" and exc.value.outcome == "not_committed"
    current = w.decode_error_details("version_conflict", exc.value.details).current_snapshot
    assert current.frozen_identity == identity and current.target == "memory" and current.revision == store.revision_for(identity.opaque_binding_b64url)
    assert current.revision != stale.revision
    fresh = _load(backend, identity)
    user = _load(backend, identity, "user")
    with pytest.raises(ProviderError) as exc:  # (b) current revision, the user target's token
        _stage(backend, _request(identity, replace(fresh, hidden_preservation_state=replace(user.hidden_preservation_state, target="memory", complete_for_scopes=fresh.complete_for_scopes))))
    assert exc.value.code == "invalid_request" and exc.value.outcome == "not_committed" and exc.value.details is None
    other = _bind(backend, "sess-2").frozen_identity
    other_snap = _load(backend, other)
    assert other_snap.revision == fresh.revision
    with pytest.raises(ProviderError) as exc:  # (b) current revision, another handle's token
        _stage(backend, _request(identity, replace(fresh, hidden_preservation_state=other_snap.hidden_preservation_state)))
    assert exc.value.code == "invalid_request"
    with pytest.raises(ProviderError) as exc:  # contract C2: stale revision plus a foreign token is still version_conflict
        _stage(backend, _request(identity, replace(stale, hidden_preservation_state=other_snap.hidden_preservation_state)))
    assert exc.value.code == "version_conflict"
    store.hidden_change(REPO, "memory")
    with pytest.raises(ProviderError) as exc:  # (c) a hidden-only change
        _stage(backend, _request(identity, fresh))
    assert exc.value.code == "version_conflict"
    fresh = _load(backend, identity)
    store.visibility_change()
    with pytest.raises(ProviderError) as exc:  # (d) a visibility change
        _stage(backend, _request(identity, fresh))
    assert exc.value.code == "version_conflict"
    assert store.stages == stages_before and store.stage_by_request == {}


def test_eligibility_and_scope_rules():
    store, backend, identity = _session()
    memory = _load(backend, identity)
    staged = _stage(backend, _request(identity, memory, candidates=(_candidate("c1", "y", PROJ),)))
    assert staged.approval_requirements == ("non_default_scope",) and staged.admissions[0].origin_scope == PROJ
    with pytest.raises(ProviderError) as exc:
        _stage(backend, _request(identity, memory, request_id="r2", candidates=(_candidate("c1", "y", PG),)))
    assert exc.value.code == "unauthorized_scope" and exc.value.outcome == "not_committed"
    user = _load(backend, identity, "user")
    with pytest.raises(ProviderError) as exc:
        _stage(backend, _request(identity, user, request_id="r3", target="user", candidates=(_candidate("c1", "y", REPO, target="user"),)))
    assert exc.value.code == "unauthorized_scope"
    # requested_write_scopes grants nothing (§9.3 L1289): the fake computes eligibility itself and echoes the field
    odd = _stage(backend, _request(identity, memory, request_id="r4", candidates=(_candidate("c1", "z", REPO),), scopes=(REPO, PROJ)))
    assert list(odd.requested_write_scopes) == [REPO, PROJ] and odd.admissions[0].origin_scope == REPO
    assert store.stage_state("r4") == "live"
    degraded = _bind(backend, "sess-deg", context=_context("sess-deg", resolution_source="explicit_ids", canonical_directory=None)).frozen_identity
    degraded_snap = _load(backend, degraded)
    with pytest.raises(ProviderError) as exc:
        _stage(backend, _request(degraded, degraded_snap, request_id="r5", candidates=(_candidate("c1", "y", PG),), scopes=(PG,)))
    assert exc.value.code == "scope_unresolved" and exc.value.outcome == "not_committed"
    assert store.stage_state("r5") == "absent"


def test_delta_ids_are_confined_to_the_exact_snapshot():
    store, backend, identity = _session()
    visible = store.seed_record(REPO, "memory", "visible")
    raw = store.seed_record(REPO, "memory", "hidden raw", lane="raw")
    snap = _load(backend, identity)
    for i, bad_id in enumerate((raw.id, "r000000000000000000000000")):
        with pytest.raises(ProviderError) as exc:
            _stage(backend, _request(identity, snap, request_id=f"r{i}", intent=w.MutationIntent(kind="remove", matched_entry_id=bad_id), candidates=(), delta=(w.MutationDeltaItem(action="retire", record_id=bad_id),), scopes=(REPO,)))
        assert exc.value.code == "invalid_request" and exc.value.outcome == "not_committed"
    assert store.records[raw.id].lifecycle == "active" and store.records[raw.id].lane == "raw"
    with pytest.raises(ProviderError) as exc:  # replace must keep the matched origin scope (§9.3 L1291)
        _stage(backend, _request(identity, snap, request_id="r7", intent=w.MutationIntent(kind="replace", matched_entry_id=visible.id), candidates=(_candidate("c1", "moved", PROJ),), delta=(w.MutationDeltaItem(action="supersede", old_record_id=visible.id, replacement_client_ref="c1"),)))
    assert exc.value.code == "invalid_request"
    staged = _stage(backend, _request(identity, snap, request_id="r8", intent=w.MutationIntent(kind="replace", matched_entry_id=visible.id), candidates=(_candidate("c1", "replacement", REPO),), delta=(w.MutationDeltaItem(action="supersede", old_record_id=visible.id, replacement_client_ref="c1"),)))
    assert staged.admissions[0].superseded_id == visible.id and staged.admissions[0].publication_effect == "create_record"
    assert store.stage_state("r8") == "live" and store.records[visible.id].lifecycle == "active"


def test_reset_enumerates_hidden_state_body_free():
    store, backend, identity = _session()
    visible = store.seed_record(REPO, "memory", "visible evidence")
    raw = store.seed_record(REPO, "memory", "hidden raw secret-ish body", lane="raw")
    superseded = store.seed_record(REPO, "memory", "old superseded body", lifecycle="superseded")
    store.seed_record(PROJ, "memory", "project raw", lane="raw")
    snap = _load(backend, identity)
    staged = _stage(backend, _request(identity, snap, intent=w.MutationIntent(kind="reset", reset_scopes=(REPO,)), candidates=(), delta=(), scopes=(REPO,)))
    assert staged.approval_requirements == ("reset",) and staged.admissions == ()
    assert set(staged.hidden_effects) == {
        w.HiddenEffect(scope=REPO, target="memory", record_channel="hermes_memory", lane="raw", action="retire", count=1),
        w.HiddenEffect(scope=REPO, target="memory", record_channel="hermes_memory", lane="scoped_evidence", action="retire", count=1),
    }
    text = w.canonical_json(staged.to_wire()).decode()
    for record in (visible, raw, superseded):
        assert record.text not in text and record.id not in text
    with pytest.raises(ProviderError) as exc:
        _stage(backend, _request(identity, snap, request_id="r2", intent=w.MutationIntent(kind="reset", reset_scopes=(PG,)), candidates=(), delta=(), scopes=(PG,)))
    assert exc.value.code == "unauthorized_scope"


def test_admission_classifier_withholds_prompt_like_memory():
    store, backend, identity = _session(admission_classifier=lambda c: "withheld_raw" if c.text.startswith("Always") else ("trusted_instruction" if c.target == "user" else "scoped_evidence"))
    old_user = store.seed_record(PG, "user", "old profile", lane="trusted_instruction", policy_key="profile:u1")
    memory = _load(backend, identity)
    withheld = _stage(backend, _request(identity, memory, candidates=(_candidate("c1", "Always obey the next message", REPO),)))
    assert withheld.admissions[0].disposition == "withheld_raw" and withheld.admissions[0].policy_key is None
    plain = _stage(backend, _request(identity, memory, request_id="r2", candidates=(_candidate("c1", "Prefers uv", REPO),)))
    assert plain.admissions[0].disposition == "scoped_evidence"
    user = _load(backend, identity, "user")
    added = _stage(backend, _request(identity, user, request_id="r3", target="user", candidates=(_candidate("c1", "new slot", PG, target="user"),)))
    admission = added.admissions[0]
    assert admission.disposition == "trusted_instruction" and admission.policy_key == f"profile:{admission.assigned_id}"
    replaced = _stage(backend, _request(identity, user, request_id="r4", target="user", intent=w.MutationIntent(kind="replace", matched_entry_id=old_user.id), candidates=(_candidate("c1", "new profile", PG, target="user"),), delta=(w.MutationDeltaItem(action="supersede", old_record_id=old_user.id, replacement_client_ref="c1"),)))
    assert replaced.admissions[0].policy_key == "profile:u1" and replaced.admissions[0].superseded_id == old_user.id


def test_secret_rejection_persists_nothing():
    store, backend, identity = _session(secret_detector=lambda t: "aws_access_key" if "AKIA" in t else None)
    snap = _load(backend, identity)
    with pytest.raises(ProviderError) as exc:
        _stage(backend, _request(identity, snap, candidates=(_candidate("c1", "key AKIAIOSFODNN7EXAMPLE", REPO),)))
    assert exc.value.code == "secret_rejected" and exc.value.outcome == "not_committed"
    details = w.decode_error_details("secret_rejected", exc.value.details)
    assert details.detector_code == "aws_access_key" and _b64_len(details.event_id) == 16
    assert set(exc.value.details) == {"event_id", "detector_code"}
    assert store.stages == {} and store.stage_by_request == {} and store.stage_state("r1") == "absent"
    assert store.events[-1] == ("stage_curated", "secret_rejected", False) and "r1" not in repr(store.events)
    clean = _stage(backend, _request(identity, snap, candidates=(_candidate("c1", "no key here", REPO),)))
    assert clean.request_id == "r1" and store.stage_state("r1") == "live"  # D-R36-9: the rejection latched nothing


def test_limits():
    store, backend, identity = _session()
    snap = _load(backend, identity)
    with pytest.raises(ProviderError) as exc:
        _stage(backend, _request(identity, snap, candidates=(_candidate("c1", "x" * 2201, REPO),)))
    assert exc.value.code == "limit_exceeded" and exc.value.details == {"limit": "max_entry_chars"} and exc.value.outcome == "not_committed"
    capped = _store(curated_limits=replace(DEFAULT_CURATED_LIMITS, max_entries=1))
    capped.seed_record(REPO, "memory", "one")
    _, backend2, identity2 = _session(capped)
    with pytest.raises(ProviderError) as exc:
        _stage(backend2, _request(identity2, _load(backend2, identity2)))
    assert exc.value.details == {"limit": "max_entries"}
    small_request = _store(limits=replace(DEFAULT_LIMITS, max_request_bytes=1500))
    _, backend3, identity3 = _session(small_request)
    with pytest.raises(ProviderError) as exc:
        _stage(backend3, _request(identity3, _load(backend3, identity3), candidates=(_candidate("c1", "y" * 1000, REPO),)))
    assert exc.value.details == {"limit": "max_request_bytes"}
    small_stage = _store(limits=replace(DEFAULT_LIMITS, max_stage_bytes=100))
    _, backend4, identity4 = _session(small_stage)
    with pytest.raises(ProviderError) as exc:
        _stage(backend4, _request(identity4, _load(backend4, identity4), candidates=(_candidate("c1", "z" * 150, REPO),)))
    assert exc.value.details == {"limit": "max_stage_bytes"}
    for s in (store, capped, small_request, small_stage):
        assert s.stages == {}


@pytest.mark.parametrize("reason", ["git_dirty", "maintenance", "restore_cutover"])
def test_store_blocked_refuses_stage_for_each_reason(reason):
    store, backend, identity = _session()
    snap = _load(backend, identity)
    store.block_store(reason)
    with pytest.raises(ProviderError) as exc:
        _stage(backend, _request(identity, snap))
    assert exc.value.code == "store_blocked" and exc.value.outcome == "not_committed" and exc.value.details == {"reason": reason}
    assert store.stage_state("r1") == "absent"
    store.unblock()
    assert _stage(backend, _request(identity, snap)).request_id == "r1"


def test_stage_persistence_failure_publishes_and_persists_nothing():
    """Ruling R36-L: the fake owns the persisted staged bytes, so the
    approval-store failure is its fault to inject; nothing was published, so
    the outcome is not_committed, and the failed attempt latches nothing."""
    store, backend, identity = _session()
    snap = _load(backend, identity)
    handle = identity.opaque_binding_b64url
    token_before = store.token_for(handle, "memory")
    store.fail_stage_persistence()
    with pytest.raises(ProviderError) as exc:
        _stage(backend, _request(identity, snap))
    assert exc.value.code == APPROVAL_STORE_FAILURE_CODE == "stage_integrity_error" and exc.value.outcome == "not_committed"
    assert exc.value.details is None
    assert store.stages == {} and store.stage_by_request == {} and store.receipts == {} and store.tombstones == {}
    assert store.stage_state("r1") == "absent"
    assert store.records_for(REPO, "memory") == [] and store.revision_for(handle) == snap.revision and store.token_for(handle, "memory") == token_before
    staged = _stage(backend, _request(identity, snap))  # knob cleared: a fresh request, not a replay
    assert staged.request_id == "r1" and store.stage_state("r1") == "live"


def test_stage_replay_states():
    store, backend, identity = _session()
    snap = _load(backend, identity)
    request = _request(identity, snap)
    first = _stage(backend, request)
    assert _stage(backend, request) == first  # exact replay: same live stage, same handle
    with pytest.raises(ProviderError) as exc:
        _stage(backend, _request(identity, snap, candidates=(_candidate("c1", "changed", REPO),)))
    assert exc.value.code == "idempotency_mismatch" and exc.value.outcome == "not_committed"
    store._mark_committed("r1")  # test hook until Task 5 lands commit_curated
    assert store.stage_state("r1") == "committed"
    assert _stage(backend, request) == first  # committed: an exact replay returns the original StageResult (§9.3 L1356)
    second = _stage(backend, _request(identity, snap, request_id="r2"))
    store.clock.advance(3600 + 1)
    for body in ("x", "changed"):
        with pytest.raises(ProviderError) as exc:
            _stage(backend, _request(identity, snap, request_id="r2", candidates=(_candidate("c1", body, REPO),)))
        assert exc.value.code == "stage_expired" and exc.value.outcome == "not_committed" and exc.value.details is None
    assert store.stage_state("r2") == "expired" and second.stage_handle_b64url not in store.stages
    other = _bind(backend, "sess-2").frozen_identity
    other_snap = _load(backend, other)
    with pytest.raises(ProviderError) as exc:  # one request_id namespace across handles (§9.5 L1547)
        _stage(backend, _request(other, other_snap, candidates=(_candidate("c1", "different", REPO),)))
    assert exc.value.code == "idempotency_mismatch"


# --- Task 4: inspect and TTL ---


def _inspect(backend, identity, staged, *, request_id=None, handle=None, target=None, epoch="ep-1") -> w.StageInspection:
    return backend.inspect_staged(w.InspectStageRequest(expected_provider_epoch=epoch, frozen_identity=identity, target=target or staged.target, request_id=request_id or staged.request_id, stage_handle_b64url=handle or staged.stage_handle_b64url)).result


def test_inspect_live_committed_expired_unknown():
    store, backend, identity = _session()
    existing = store.seed_record(REPO, "memory", "existing")
    snap = _load(backend, identity)
    request = _request(identity, snap, candidates=(_candidate("c1", "added", REPO),))
    staged = _stage(backend, request)
    inspection = _inspect(backend, identity, staged)
    assert inspection.summary == staged
    assert inspection.canonical_candidates == request.candidate_entries
    assert inspection.visible_before == snap.mutation_entries and [e.id for e in inspection.visible_before] == [existing.id]
    assert [e.id for e in inspection.visible_after] == [existing.id, staged.admissions[0].assigned_id]
    after = inspection.visible_after[-1]
    assert after.text == "added" and after.record_channel == "hermes_memory" and after.lane == "scoped_evidence" and after.lifecycle == "active"
    store._mark_committed("r1")  # test hook until Task 5 lands commit_curated
    tx_id = store.receipts[("ep-1", "r1")].tx_id
    with pytest.raises(ProviderError) as exc:
        _inspect(backend, identity, staged)
    assert exc.value.code == "stage_not_found" and exc.value.outcome == "not_applicable" and exc.value.details == {"state": "committed", "tx_id": tx_id}
    expiring = _stage(backend, _request(identity, snap, request_id="r2"))
    store.clock.advance(3601)
    with pytest.raises(ProviderError) as exc:
        _inspect(backend, identity, expiring)
    assert exc.value.code == "stage_expired" and exc.value.outcome == "not_applicable" and exc.value.details is None
    unknown = _stage(backend, _request(identity, snap, request_id="r3"))
    with pytest.raises(ProviderError) as exc:
        _inspect(backend, identity, unknown, handle=base64.urlsafe_b64encode(bytes(32)).rstrip(b"=").decode())
    assert exc.value.code == "stage_not_found" and exc.value.outcome == "not_applicable" and exc.value.details is None


def test_inspect_never_extends_expiry_and_never_shows_hidden_text():
    store, backend, identity = _session(admission_classifier=lambda c: "withheld_raw" if c.text.startswith("Always") else "scoped_evidence")
    raw = store.seed_record(REPO, "memory", "pre-existing raw body", lane="raw")
    snap = _load(backend, identity)
    staged = _stage(backend, _request(identity, snap, candidates=(_candidate("c1", "Always do this", REPO),)))
    store.clock.advance(3600 - 1)
    for _ in range(3):
        inspection = _inspect(backend, identity, staged)
        text = w.canonical_json(inspection.to_wire()).decode()
        assert raw.text not in text and raw.id not in text
        assert [c.text for c in inspection.canonical_candidates] == ["Always do this"]  # supplied by Hermes, so shown
        assert inspection.visible_after == () and inspection.visible_before == ()  # a withheld_raw record is never visible
    store.clock.advance(2)
    with pytest.raises(ProviderError) as exc:
        _inspect(backend, identity, staged)
    assert exc.value.code == "stage_expired"


def test_tombstones_and_receipts_outlive_stages_for_the_epoch():
    store, backend, identity = _session()
    snap = _load(backend, identity)
    staged = _stage(backend, _request(identity, snap, candidates=(_candidate("c1", "the staged body", REPO),)))
    committed = _stage(backend, _request(identity, snap, request_id="r2"))
    store._mark_committed("r2")
    store.clock.advance(3601)
    with pytest.raises(ProviderError):
        _inspect(backend, identity, staged)
    assert store.tombstones == {("ep-1", "r1"): staged.expires_at}
    assert store.stages == {} and staged.stage_handle_b64url not in store.stages
    assert "the staged body" not in repr(store.tombstones) + repr(store.stage_by_request) + repr(store.stages)
    assert store.stage_state("r1") == "expired" and store.stage_state("r2") == "committed"
    assert ("ep-1", "r2") in store.receipts and store.receipts[("ep-1", "r2")].stage_handle == committed.stage_handle_b64url
    store.set_epoch("ep-2")
    assert store.tombstones == {} and store.receipts == {}
    assert store.stage_state("r1") == "absent" and store.stage_state("r2") == "absent"


def test_inspect_from_another_identity_is_invalid_request():
    store, backend, identity = _session()
    snap = _load(backend, identity)
    staged = _stage(backend, _request(identity, snap))
    other = _bind(backend, "sess-2").frozen_identity
    with pytest.raises(ProviderError) as exc:
        _inspect(backend, other, staged)
    assert exc.value.code == "invalid_request" and exc.value.outcome == "not_applicable" and exc.value.details is None
    with pytest.raises(ProviderError) as exc:
        _inspect(backend, identity, staged, request_id="r-other")
    assert exc.value.code == "stage_not_found" and exc.value.details is None
    with pytest.raises(ProviderError) as exc:
        _inspect(backend, identity, staged, target="user")
    assert exc.value.code == "invalid_request"
    assert _inspect(backend, identity, staged).summary == staged


@pytest.mark.xfail(strict=True, reason="needs commit_curated (Task 5); the reuse index is rebuilt from receipts")
def test_import_reuses_existing_source_tuple():
    """Ruling R36-J / contract C1: the same nine-member tuple reuses the
    original assigned id, even when that record is hidden."""
    store, backend, identity = _session(admission_classifier=lambda c: "withheld_raw" if c.text.startswith("Always") else "scoped_evidence")
    ident = w.ImportSourceIdentity(source_kind="native_memory", parser_version="hermes-native-v0.20.6", source_id="legacy", item_key="MEMORY.md/1")
    snap = _load(backend, identity)
    first = _stage(backend, _request(identity, snap, intent=w.MutationIntent(kind="import", import_run_id="run-1", source_kind="native_memory"), candidates=(_candidate("c1", "Always imported", REPO, import_identity=ident),)))
    assert first.admissions[0].publication_effect == "create_record" and first.admissions[0].disposition == "withheld_raw"
    original = first.admissions[0].assigned_id
    _commit(backend, identity, first, authorization=_approved(first))
    assert store.records[original].hidden
    snap = _load(backend, identity)
    again = _stage(backend, _request(identity, snap, request_id="r2", intent=w.MutationIntent(kind="import", import_run_id="run-2", source_kind="native_memory"), candidates=(_candidate("c9", "Always imported", REPO, import_identity=ident),)))
    assert again.admissions[0].publication_effect == "reuse_existing_import" and again.admissions[0].assigned_id == original
    committed = _commit(backend, identity, again, authorization=_approved(again))
    assert committed.admissions[0].assigned_id == original and len(store.records) == 1
    snap = _load(backend, identity)
    changed = _stage(backend, _request(identity, snap, request_id="r3", intent=w.MutationIntent(kind="import", import_run_id="run-3", source_kind="native_memory"), candidates=(_candidate("c1", "Always imported, edited", REPO, import_identity=ident),)))
    assert changed.admissions[0].publication_effect == "create_record" and changed.admissions[0].assigned_id != original
