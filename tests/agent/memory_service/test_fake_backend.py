"""Backend semantics of the stateful fake authoritative provider (R36).

Every test drives ``FakeAuthoritativeBackend`` directly with wire-typed
requests; the service-level behaviour is proven in ``test_failure_matrix.py``
and ``test_fake_contracts.py``.
"""

import base64
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from agent.memory_service import wire as w
from agent.memory_service.errors import ProviderError, ProviderTransportError

from tests.agent.memory_service.fake_backend import (
    EPOCH_BINDING_OUTCOME_ON_MUTATION,
    FakeAuthoritativeBackend,
    FakeClock,
    FakeProviderStore,
    FakeRegistry,
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
