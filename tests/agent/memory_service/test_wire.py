"""§9.3 wire schemas: strict decode, exact encode, invariants."""

import base64
import dataclasses
import json

import pytest

from agent.memory_service import wire as w

PG = {"kind": "principal_global", "id": "ethan"}
REPO = {"kind": "repository", "id": "repo-1"}
HANDLE = base64.urlsafe_b64encode(bytes(range(32))).rstrip(b"=").decode()
IDENTITY = {
    "provider": "example",
    "provider_mode": "authoritative",
    "principal_id": "ethan",
    "profile_id": "default",
    "logical_session_id": "sess-1",
    "org_id": None,
    "project_id": "proj-1",
    "repo_id": "repo-1",
    "workspace_id": None,
    "platform": "cli",
    "binding_revision": "rev-7",
    "opaque_binding_b64url": HANDLE,
}
PROVENANCE = {
    "actor_kind": "hermes",
    "principal_id": "ethan",
    "logical_session_id": "sess-1",
    "surface": "memory_tool",
    "source_entry_ids": [],
    "source_commit": None,
    "transaction_id": None,
}
ENTRY = {
    "id": "e1",
    "text": "Prefer tabs",
    "origin_scope": REPO,
    "target": "memory",
    "record_channel": "hermes_memory",
    "lane": "scoped_evidence",
    "lifecycle": "active",
    "policy_key": None,
    "provenance": PROVENANCE,
}
DELIVERED = {**{k: v for k, v in ENTRY.items() if k != "lifecycle"}, "delivery_tier": "repository"}
REVISION = {
    "provider_epoch": "ep-1",
    "visibility_revision": "vis-1",
    "scope_revisions": [{"scope": REPO, "revision": "r1"}, {"scope": PG, "revision": "r0"}],
}
LIMITS = {"memory_chars": 2200, "user_chars": 1375, "initial_general_chars": 3000, "max_entry_chars": 2200, "max_entries": 100}
HIDDEN = {"target": "memory", "complete_for_scopes": [REPO], "opaque_state_b64url": HANDLE}
SNAPSHOT = {
    "api_version": 1,
    "status": "ok",
    "frozen_identity": IDENTITY,
    "target": "memory",
    "visible_scopes": [REPO, PG],
    "default_write_scope": REPO,
    "eligible_write_scopes": [REPO],
    "complete_for_scopes": [REPO],
    "revision": REVISION,
    "limits": LIMITS,
    "mutation_entries": [ENTRY],
    "delivery_entries": [DELIVERED],
    "hidden_preservation_state": HIDDEN,
}


def test_scope_ref_round_trip_and_strictness():
    ref = w.ScopeRef.from_wire(REPO)
    assert ref.kind == "repository" and ref.id == "repo-1"
    assert ref.to_wire() == REPO
    with pytest.raises(w.WireError, match="unknown field"):
        w.ScopeRef.from_wire({**REPO, "extra": 1})
    with pytest.raises(w.WireError, match="missing"):
        w.ScopeRef.from_wire({"kind": "repository"})
    with pytest.raises(w.WireError, match="one of"):
        w.ScopeRef.from_wire({"kind": "repo", "id": "x"})
    with pytest.raises(w.WireError, match="string"):
        w.ScopeRef.from_wire({"kind": "repository", "id": 7})


def test_identity_wire_validates_handle_and_mode():
    ident = w.FrozenIdentityWire.from_wire(IDENTITY)
    assert ident.to_wire() == IDENTITY
    with pytest.raises(w.WireError, match="32 bytes"):
        w.FrozenIdentityWire.from_wire({**IDENTITY, "opaque_binding_b64url": "AAAA"})
    with pytest.raises(w.WireError, match="base64url"):
        w.FrozenIdentityWire.from_wire({**IDENTITY, "opaque_binding_b64url": HANDLE[:-1] + "="})
    with pytest.raises(w.WireError, match="one of"):
        w.FrozenIdentityWire.from_wire({**IDENTITY, "provider_mode": "additive"})
    with pytest.raises(w.WireError, match="platform"):
        w.FrozenIdentityWire.from_wire({**IDENTITY, "platform": "CLI"})
    with pytest.raises(w.WireError, match="null"):
        w.FrozenIdentityWire.from_wire({**IDENTITY, "principal_id": None})


def test_integers_reject_bools_floats_and_out_of_range():
    ok = w.CuratedLimits.from_wire(LIMITS)
    assert ok.memory_chars == 2200
    with pytest.raises(w.WireError, match="integer"):
        w.CuratedLimits.from_wire({**LIMITS, "memory_chars": True})
    with pytest.raises(w.WireError, match="integer"):
        w.CuratedLimits.from_wire({**LIMITS, "memory_chars": 2200.0})
    with pytest.raises(w.WireError, match="2\\^53"):
        w.CuratedLimits.from_wire({**LIMITS, "memory_chars": 2**53})
    with pytest.raises(w.WireError, match="non-negative"):
        w.CuratedLimits.from_wire({**LIMITS, "memory_chars": -1})


def test_negotiation_limits_invariants():
    neg = {
        "provider": "example",
        "selected_api_version": 1,
        "selected_contract_revision": 2,
        "operations": ["bind_session", "validate_session", "load_curated", "stage_curated", "inspect_staged", "commit_curated"],
        "capabilities": {"recall_context": False, "capture_continuity": True},
        "limits": {
            "max_request_bytes": 1_000_000,
            "max_response_bytes": 4_000_000,
            "max_stage_bytes": 200_000,
            "stage_ttl_seconds": 86_400,
            "max_opaque_binding_bytes": 32,
            "max_continuity_bytes": 65_536,
        },
    }
    parsed = w.Negotiation.from_wire(neg)
    assert parsed.capabilities.capture_continuity is True
    assert parsed.to_wire() == neg
    with pytest.raises(w.WireError, match="86400"):
        w.Negotiation.from_wire({**neg, "limits": {**neg["limits"], "stage_ttl_seconds": 86_401}})
    with pytest.raises(w.WireError, match=">= 32"):
        w.Negotiation.from_wire({**neg, "limits": {**neg["limits"], "max_opaque_binding_bytes": 31}})
    with pytest.raises(w.WireError, match="positive"):
        w.Negotiation.from_wire({**neg, "limits": {**neg["limits"], "max_stage_bytes": 0}})
    with pytest.raises(w.WireError, match="one of"):
        w.Negotiation.from_wire({**neg, "operations": ["bind_session", "shutdown"]})
    # §9.1 makes an incompatible provider API a CONFIGURATION error, so the
    # codec must let the host see the version it selected and report it; only a
    # non-positive or non-integer version is a wire violation.
    assert w.Negotiation.from_wire({**neg, "selected_api_version": 7}).selected_api_version == 7
    with pytest.raises(w.WireError, match="selected_api_version"):
        w.Negotiation.from_wire({**neg, "selected_api_version": 0})
    with pytest.raises(w.WireError, match="integer"):
        w.Negotiation.from_wire({**neg, "selected_api_version": True})


def test_snapshot_invariants():
    snap = w.CuratedSnapshot.from_wire(SNAPSHOT)
    assert snap.to_wire() == SNAPSHOT
    assert snap.revision.scope_revisions[0].scope.id == "repo-1"
    with pytest.raises(w.WireError, match="api_version"):
        w.CuratedSnapshot.from_wire({**SNAPSHOT, "api_version": 2})
    with pytest.raises(w.WireError, match="complete_for_scopes"):
        w.CuratedSnapshot.from_wire({**SNAPSHOT, "complete_for_scopes": []})
    with pytest.raises(w.WireError, match="scope_revisions"):
        w.CuratedSnapshot.from_wire({**SNAPSHOT, "revision": {**REVISION, "scope_revisions": REVISION["scope_revisions"][:1]}})
    with pytest.raises(w.WireError, match="hidden_preservation_state"):
        w.CuratedSnapshot.from_wire({**SNAPSHOT, "hidden_preservation_state": {**HIDDEN, "target": "user"}})
    with pytest.raises(w.WireError, match="mutation_entries"):
        w.CuratedSnapshot.from_wire({**SNAPSHOT, "mutation_entries": [{**ENTRY, "record_channel": "general", "target": None}]})
    with pytest.raises(w.WireError, match="delivery_entries"):
        w.CuratedSnapshot.from_wire({**SNAPSHOT, "delivery_entries": [{**DELIVERED, "record_channel": "hermes_user", "target": "user"}]})


def test_degraded_snapshot_shape():
    degraded = {
        **SNAPSHOT,
        "status": "degraded_global_only",
        "visible_scopes": [PG],
        "default_write_scope": None,
        "eligible_write_scopes": [],
        "complete_for_scopes": [],
        "revision": {**REVISION, "scope_revisions": [{"scope": PG, "revision": "r0"}]},
        "mutation_entries": [],
        "delivery_entries": [],
        "hidden_preservation_state": {**HIDDEN, "complete_for_scopes": []},
    }
    assert w.CuratedSnapshot.from_wire(degraded).status == "degraded_global_only"
    with pytest.raises(w.WireError, match="degraded_global_only"):
        w.CuratedSnapshot.from_wire({**degraded, "default_write_scope": REPO})
    two_scopes = {**REVISION, "scope_revisions": [{"scope": PG, "revision": "r0"}, {"scope": REPO, "revision": "r1"}]}
    with pytest.raises(w.WireError, match="degraded_global_only"):
        w.CuratedSnapshot.from_wire({**degraded, "visible_scopes": [PG, REPO], "revision": two_scopes})


def _stage_request(**over):
    base = {
        "expected_provider_epoch": "ep-1",
        "frozen_identity": IDENTITY,
        "target": "memory",
        "expected_revision": REVISION,
        "hidden_preservation_state": HIDDEN,
        "request_id": "req-1",
        "requested_write_scopes": [REPO],
        "intent": {"kind": "add"},
        "mutation_delta": [{"action": "add", "client_ref": "c1"}],
        "candidate_entries": [
            {
                "client_ref": "c1",
                "text": "Use uv",
                "destination_scope": REPO,
                "target": "memory",
                "proposed_policy_key": None,
                "import_source_identity": None,
                "memory_admission": None,
            }
        ],
        "provenance": {
            "actor_kind": "hermes",
            "principal_id": "ethan",
            "logical_session_id": "sess-1",
            "initiating_surface": "memory_tool",
            "source_entry_ids": [],
            "source_commit": None,
            "threat_decision_id": None,
        },
    }
    base.update(over)
    return base


def test_mutation_intent_and_delta_variants():
    assert w.MutationIntent.from_wire({"kind": "replace", "matched_entry_id": "e1"}).matched_entry_id == "e1"
    assert w.MutationIntent.from_wire({"kind": "reset", "reset_scopes": [REPO]}).reset_scopes[0].id == "repo-1"
    imp = w.MutationIntent.from_wire({"kind": "import", "import_run_id": "run-1", "source_kind": "native_memory"})
    assert imp.to_wire() == {"kind": "import", "import_run_id": "run-1", "source_kind": "native_memory"}
    with pytest.raises(w.WireError, match="matched_entry_id"):
        w.MutationIntent.from_wire({"kind": "replace"})
    with pytest.raises(w.WireError, match="unknown field"):
        w.MutationIntent.from_wire({"kind": "add", "matched_entry_id": "e1"})
    with pytest.raises(w.WireError, match="one of"):
        w.MutationIntent.from_wire({"kind": "merge"})
    sup = w.MutationDeltaItem.from_wire({"action": "supersede", "old_record_id": "e1", "replacement_client_ref": "c1"})
    assert sup.to_wire() == {"action": "supersede", "old_record_id": "e1", "replacement_client_ref": "c1"}
    with pytest.raises(w.WireError, match="record_id"):
        w.MutationDeltaItem.from_wire({"action": "retire"})


def test_a_validator_never_raises_anything_but_wire_error():
    """A missing or empty variant field must fail typed, not crash a consumer
    with TypeError (sorted over a None) or KeyError (a None record id)."""
    empty_ref = _stage_request(
        intent={"kind": "bulk_edit"},
        mutation_delta=[{"action": "add", "client_ref": ""}, {"action": "retire", "record_id": "e1"}],
    )
    with pytest.raises(w.WireError, match="client_ref"):
        w.StageRequest.from_wire(empty_ref)
    with pytest.raises(w.WireError, match="client_ref"):
        w.MutationDeltaItem.from_wire({"action": "add", "client_ref": ""})
    # directly constructed (never decoded): StageRequest.validate must still
    # reach the item and the intent before any cardinality expression does.
    decoded = w.StageRequest.from_wire(_stage_request())
    broken_item = dataclasses.replace(decoded, mutation_delta=(w.MutationDeltaItem(action="retire"),), candidate_entries=(), intent=w.MutationIntent(kind="bulk_edit"))
    with pytest.raises(w.WireError, match="record_id"):
        broken_item.validate()
    broken_intent = dataclasses.replace(decoded, intent=w.MutationIntent(kind="replace"))
    with pytest.raises(w.WireError, match="matched_entry_id"):
        broken_intent.validate()


def test_stage_request_cardinality_rules():
    assert w.StageRequest.from_wire(_stage_request()).intent.kind == "add"
    second = dict(_stage_request()["candidate_entries"][0], client_ref="c2", text="Use ruff")
    with pytest.raises(w.WireError, match="add requires"):
        w.StageRequest.from_wire(
            _stage_request(
                candidate_entries=[_stage_request()["candidate_entries"][0], second],
                mutation_delta=[{"action": "add", "client_ref": "c1"}, {"action": "add", "client_ref": "c2"}],
            )
        )
    with pytest.raises(w.WireError, match="replace"):
        w.StageRequest.from_wire(_stage_request(intent={"kind": "replace", "matched_entry_id": "e1"}))
    ok_replace = _stage_request(
        intent={"kind": "replace", "matched_entry_id": "e1"},
        mutation_delta=[{"action": "supersede", "old_record_id": "e1", "replacement_client_ref": "c1"}],
    )
    assert w.StageRequest.from_wire(ok_replace).mutation_delta[0].old_record_id == "e1"
    with pytest.raises(w.WireError, match="remove"):
        w.StageRequest.from_wire(_stage_request(intent={"kind": "remove", "matched_entry_id": "e1"}))
    ok_remove = _stage_request(
        intent={"kind": "remove", "matched_entry_id": "e1"},
        mutation_delta=[{"action": "retire", "record_id": "e1"}],
        candidate_entries=[],
    )
    assert w.StageRequest.from_wire(ok_remove).candidate_entries == ()
    with pytest.raises(w.WireError, match="bulk_edit"):
        w.StageRequest.from_wire(_stage_request(intent={"kind": "bulk_edit"}))
    with pytest.raises(w.WireError, match="referenced"):
        w.StageRequest.from_wire(_stage_request(mutation_delta=[{"action": "add", "client_ref": "zz"}]))
    with pytest.raises(w.WireError, match="target"):
        cand = dict(_stage_request()["candidate_entries"][0], target="user")
        w.StageRequest.from_wire(_stage_request(candidate_entries=[cand]))
    with pytest.raises(w.WireError, match="import_source_identity"):
        cand = dict(
            _stage_request()["candidate_entries"][0],
            import_source_identity={
                "source_kind": "native_memory",
                "parser_version": "hermes-native-v0.20.6",
                "source_id": "memory-md",
                "item_key": "MEMORY.md/1",
            },
        )
        w.StageRequest.from_wire(_stage_request(candidate_entries=[cand]))
    with pytest.raises(w.WireError, match="reset"):
        w.StageRequest.from_wire(_stage_request(intent={"kind": "reset", "reset_scopes": [REPO]}))


def test_import_source_identity_grammar():
    good = {"source_kind": "legacy_archive", "parser_version": "hermes-legacy-archive-v1", "source_id": "archive-2024", "item_key": "notes/a_b.md"}
    assert w.ImportSourceIdentity.from_wire(good).item_key == "notes/a_b.md"
    for bad_key in ["/abs", "a//b", "a/./b", "a/../b", "", "x" * 257, "spa ce"]:
        with pytest.raises(w.WireError, match="item_key"):
            w.ImportSourceIdentity.from_wire({**good, "item_key": bad_key})
    for bad_id in ["", "UPPER", "a" * 65, "with_underscore"]:
        with pytest.raises(w.WireError, match="source_id"):
            w.ImportSourceIdentity.from_wire({**good, "source_id": bad_id})


def test_stage_result_and_approval_authorization():
    result = {
        "stage_handle_b64url": HANDLE,
        "request_id": "req-1",
        "target": "memory",
        "expected_revision": REVISION,
        "requested_write_scopes": [REPO],
        "eligible_write_scopes": [REPO],
        "expires_at": "2026-09-04T00:00:00Z",
        "approval_binding_sha256": "a" * 64,
        "approval_requirements": [],
        "candidate_hashes": [{"client_ref": "c1", "canonical_sha256": "b" * 64}],
        "admissions": [
            {
                "client_ref": "c1",
                "assigned_id": "e2",
                "target": "memory",
                "record_channel": "hermes_memory",
                "origin_scope": REPO,
                "disposition": "scoped_evidence",
                "publication_effect": "create_record",
                "superseded_id": None,
                "policy_key": None,
            }
        ],
        "hidden_effects": [],
    }
    parsed = w.StageResult.from_wire(result)
    assert parsed.to_wire() == result
    with pytest.raises(w.WireError, match="sha256"):
        w.StageResult.from_wire({**result, "approval_binding_sha256": "A" * 64})
    with pytest.raises(w.WireError, match="approval_requirements"):
        w.StageResult.from_wire({**result, "approval_requirements": ["reset", "reset"]})
    with pytest.raises(w.WireError, match="expires_at"):
        w.StageResult.from_wire({**result, "expires_at": "tomorrow"})
    assert w.ApprovalAuthorization.from_wire({"kind": "not_required"}).kind == "not_required"
    approved = {
        "kind": "approved",
        "approval_id": "ap-1",
        "approved_by_principal_id": "ethan",
        "approved_at": "2026-09-03T21:00:00Z",
        "expires_at": "2026-09-03T22:00:00Z",
        "approval_binding_sha256": "a" * 64,
    }
    assert w.ApprovalAuthorization.from_wire(approved).to_wire() == approved
    with pytest.raises(w.WireError, match="unknown field"):
        w.ApprovalAuthorization.from_wire({"kind": "not_required", "approval_id": "x"})


def test_timestamps_must_be_utc_and_valid():
    approved = {
        "kind": "approved",
        "approval_id": "ap-1",
        "approved_by_principal_id": "ethan",
        "approved_at": "2026-09-03T21:00:00Z",
        "expires_at": "2026-09-03T22:00:00Z",
        "approval_binding_sha256": "a" * 64,
    }
    with pytest.raises(w.WireError, match="UTC"):
        w.ApprovalAuthorization.from_wire({**approved, "approved_at": "2026-09-03T21:00:00+05:00"})
    with pytest.raises(w.WireError, match="valid"):
        w.ApprovalAuthorization.from_wire({**approved, "approved_at": "9999-99-99T99:99:99Z"})
    assert w.ApprovalAuthorization.from_wire({**approved, "approved_at": "2026-09-03T21:00:00Z"}).approved_at == "2026-09-03T21:00:00Z"
    assert w.ApprovalAuthorization.from_wire({**approved, "approved_at": "2026-09-03T21:00:00+00:00"}).approved_at == "2026-09-03T21:00:00+00:00"


def test_bind_request_intent_rules():
    ctx = {
        "principal_id": "ethan",
        "profile_id": "default",
        "logical_session_id": "sess-1",
        "platform": "cli",
        "org_id": None,
        "project_id": None,
        "repo_id": None,
        "workspace_id": None,
        "resolution_source": "directory",
        "canonical_directory": "C:\\work\\repo",
    }
    req = {"expected_provider_epoch": "ep-1", "binding_request_id": "b1", "bind_intent": "new_session", "requested_context": ctx, "prior_identity": None}
    assert w.BindRequest.from_wire(req).bind_intent == "new_session"
    with pytest.raises(w.WireError, match="prior_identity"):
        w.BindRequest.from_wire({**req, "prior_identity": IDENTITY})
    with pytest.raises(w.WireError, match="prior_identity"):
        w.BindRequest.from_wire({**req, "bind_intent": "explicit_rebind"})
    with pytest.raises(w.WireError, match="directory"):
        w.BindRequest.from_wire({**req, "bind_intent": "session_reset"})
    explicit = {**ctx, "resolution_source": "explicit_ids", "canonical_directory": None, "repo_id": "repo-1", "project_id": "proj-1"}
    assert w.BindRequest.from_wire({**req, "bind_intent": "session_reset", "requested_context": explicit}).prior_identity is None
    with pytest.raises(w.WireError, match="canonical_directory"):
        w.BindRequest.from_wire({**req, "requested_context": {**ctx, "resolution_source": "explicit_ids"}})


def test_recall_request_channels_per_target():
    base = {
        "expected_provider_epoch": "ep-1",
        "frozen_identity": IDENTITY,
        "target": "memory",
        "source_revision": REVISION,
        "query": "tabs",
        "include_channels": ["general", "hermes_memory"],
        "exclude_entry_ids": ["e1"],
        "budget": {"trusted_chars": 0, "evidence_chars": 3000, "max_entries": 20},
    }
    assert w.RecallRequest.from_wire(base).budget.evidence_chars == 3000
    with pytest.raises(w.WireError, match="include_channels"):
        w.RecallRequest.from_wire({**base, "include_channels": ["hermes_user"]})
    with pytest.raises(w.WireError, match="include_channels"):
        w.RecallRequest.from_wire({**base, "include_channels": []})
    with pytest.raises(w.WireError, match="include_channels"):
        w.RecallRequest.from_wire({**base, "target": "user", "include_channels": ["general"]})
    assert w.RecallRequest.from_wire({**base, "target": "user", "include_channels": ["hermes_user"]}).target == "user"


def test_envelopes_and_error_details():
    success = {"wire_version": 1, "call_id": "c1", "operation": "load_curated", "provider_epoch": "ep-1", "status": "ok", "result": SNAPSHOT}
    env = w.WireSuccess.from_wire(success)
    assert w.RESULT_TYPES["load_curated"].from_wire(env.result).target == "memory"
    failure = {
        "wire_version": 1,
        "call_id": "c1",
        "operation": "stage_curated",
        "provider_epoch": "ep-1",
        "status": "error",
        "error": {"code": "store_blocked", "outcome": "not_committed", "details": {"reason": "git_dirty"}},
    }
    fail = w.WireFailure.from_wire(failure)
    assert w.decode_error_details("store_blocked", fail.error.details).reason == "git_dirty"
    with pytest.raises(w.WireError, match="one of"):
        w.WireFailure.from_wire({**failure, "error": {**failure["error"], "code": "disk_full"}})
    with pytest.raises(w.WireError, match="wire_version"):
        w.WireSuccess.from_wire({**success, "wire_version": 2})
    assert w.decode_error_details("binding_invalid", None) is None
    with pytest.raises(w.WireError, match="null"):
        w.decode_error_details("binding_invalid", {"x": 1})
    with pytest.raises(w.WireError, match="details"):
        w.decode_error_details("store_blocked", None)
    epoch = w.decode_error_details("provider_epoch_changed", {"expected_provider_epoch": "ep-1", "current_provider_epoch": "ep-2"})
    assert epoch.current_provider_epoch == "ep-2"
    amb = w.decode_error_details("ambiguous_policy", {"ambiguities": [{"policy_key": "k", "tier": "dependency", "candidate_count": 2}]})
    assert amb.ambiguities[0].candidate_count == 2
    with pytest.raises(w.WireError, match="candidate_count"):
        w.decode_error_details("ambiguous_policy", {"ambiguities": [{"policy_key": "k", "tier": "dependency", "candidate_count": 1}]})
    assert w.decode_error_details("version_conflict", None) is None
    assert w.decode_error_details("version_conflict", {"current_snapshot": SNAPSHOT}).current_snapshot.target == "memory"
    # §9.3 lists exactly these twenty codes; pin the set, not just its size.
    assert set(w.ERROR_CODES) == {
        "incompatible_api", "invalid_request", "unavailable",
        "provider_epoch_changed", "binding_invalid", "binding_revoked",
        "ambiguous_policy",
        "scope_unresolved", "unauthorized_scope", "store_blocked",
        "approval_required", "approval_invalid", "approval_expired",
        "secret_rejected",
        "stage_not_found", "stage_expired", "stage_integrity_error",
        "version_conflict", "idempotency_mismatch", "limit_exceeded",
    }
    assert len(w.ERROR_CODES) == 20
    assert set(w.REQUEST_TYPES) == set(w.OPERATIONS) == set(w.RESULT_TYPES)


def test_validate_session_result_requires_a_real_boolean_true():
    """§9.3 ``valid: true`` is a boolean literal: 1 is not true."""
    result = {"valid": True, "frozen_identity": IDENTITY, "visible_scopes": [REPO, PG]}
    assert w.ValidateSessionResult.from_wire(result).valid is True
    with pytest.raises(w.WireError, match="valid"):
        w.ValidateSessionResult.from_wire({**result, "valid": 1})
    with pytest.raises(w.WireError, match="valid"):
        w.ValidateSessionResult.from_wire({**result, "valid": False})


def test_canonical_json_matches_rfc8785_for_the_subset():
    obj = {"b": [1, {"y": None, "x": "é\n\"\\"}], "a": True, "": 0}
    assert w.canonical_json(obj) == '{"":0,"a":true,"b":[1,{"x":"é\\n\\"\\\\","y":null}]}'.encode("utf-8")
    assert json.loads(w.canonical_json(obj)) == obj
    with pytest.raises(ValueError):
        w.canonical_json({"f": 1.5})
    with pytest.raises(ValueError):
        w.canonical_json({"é": 1})


def _contract_vectors(kind):
    """Every sealed vector has exactly one executed contract boundary."""
    from collections import Counter
    from pathlib import Path

    vectors = json.loads((Path(__file__).parents[2] / "fixtures/provider-contract-v2/vectors.json").read_bytes())
    boundaries = {
        "codec": (70, "test_revision_two_shared_codec_contract"),
        "negotiation": (14, "test_revision_two_shared_negotiation_vectors"),
        "lane": (18, "test_revision_two_shared_contextual_lanes"),
        "lane_batch": (1, "test_revision_two_shared_contextual_batch"),
        "binding": (3, "test_revision_two_shared_binding_retry_identity"),
    }
    assert Counter(v["kind"] for v in vectors) == {key: value[0] for key, value in boundaries.items()}
    ids = [v["id"] for v in vectors]
    assert len(ids) == len(set(ids)) == 106
    assert all(isinstance(vector_id, str) and vector_id for vector_id in ids)
    assert kind in boundaries
    return [vector for vector in vectors if vector["kind"] == kind]


def test_revision_two_shared_codec_contract():
    """Accepted H2/OR-W12-1 structural contract, at direct and serialized boundaries."""
    vectors = _contract_vectors("codec")
    for vector in vectors:
        schema = getattr(w, vector["input"]["schema"])
        value = vector["input"]["value"]
        if vector["expected"]["accepted"]:
            assert schema.from_wire(value).to_wire() == value, vector["id"]
            assert schema.from_json(json.dumps(value)).to_wire() == value, vector["id"]
        else:
            with pytest.raises(w.WireError):
                schema.from_wire(value)
            with pytest.raises(w.WireError):
                schema.from_json(json.dumps(value))
    candidate = next(v["input"]["value"] for v in vectors if v["id"] == "candidate-memory-declarative")
    duplicate = json.dumps(candidate).replace('"memory_admission": "declarative"', '"memory_admission": null, "memory_admission": "declarative"')
    with pytest.raises(w.WireError, match="duplicate"):
        w.CandidateEntry.from_json(duplicate)
    parsed = w.CandidateEntry.from_wire(candidate)
    with pytest.raises(w.WireError, match="memory_admission"):
        dataclasses.replace(parsed, memory_admission=True).validate()
    with pytest.raises(TypeError):
        w.CandidateEntry(**{k: getattr(parsed, k) for k in candidate if k != "memory_admission"})


    # Rust command_argv uses Unicode White_Space, unlike Python's extra ASCII separators.
    registry = next(v["input"]["value"] for v in vectors if v["id"] == "delivery-registry")
    whitespace_cases = [(first, True) for first in "\x1c\x1d\x1e\x1f"]
    whitespace_cases += [(first, False) for first in
                         "\t\n\v\f\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"]
    for first, accepted in whitespace_cases:
        value = {**registry, "argv": [first]}
        if accepted:
            assert w.RegistryVerifyEntry.from_wire(value).to_wire() == value
            assert w.RegistryVerifyEntry.from_json(json.dumps(value)).to_wire() == value
        else:
            with pytest.raises(w.WireError, match="argv"):
                w.RegistryVerifyEntry.from_wire(value)



def test_revision_two_shared_negotiation_vectors():
    """Shared requests reach the fake protocol boundary; no provider conformance claim."""
    from agent.memory_service.errors import ProviderError
    from tests.agent.memory_service.fake_backend import FakeAuthoritativeBackend, FakeProviderStore
    vectors = _contract_vectors("negotiation")
    for vector in vectors:
        store = FakeProviderStore()
        backend = FakeAuthoritativeBackend(store)
        try:
            request = w.NegotiateRequest.from_wire(vector["input"]["value"])
            result = backend.negotiate(request).result
        except w.WireError:
            assert vector["expected"].get("error_code") == "invalid_request", vector["id"]
        except ProviderError as exc:
            assert exc.code == vector["expected"]["error_code"], vector["id"]
        else:
            assert vector["expected"]["accepted"], vector["id"]
            assert result.selected_contract_revision == vector["expected"]["selected_contract_revision"]
        finally:
            backend.shutdown()
        assert not store.handles and not store.stages


def _contextual_contract_plan(snapshot, candidates, source_kind=None):
    from agent.memory_service.mutation import PlannedMutation

    imported = source_kind is not None
    entries = tuple(w.CandidateEntry(
        client_ref=candidate["client_ref"], text="contract " + candidate["client_ref"],
        destination_scope=snapshot.default_write_scope, target=snapshot.target,
        proposed_policy_key=None,
        import_source_identity=w.ImportSourceIdentity(
            source_kind=source_kind,
            parser_version={"native_memory": "hermes-native-v0.20.6", "legacy_archive": "hermes-legacy-archive-v1"}[source_kind],
            source_id="shared-corpus", item_key="item-" + str(index),
        ) if imported else None,
        memory_admission=candidate["memory_admission"],
    ) for index, candidate in enumerate(candidates))
    intent = (w.MutationIntent(kind="import", import_run_id="shared-import", source_kind=source_kind)
              if imported else w.MutationIntent(kind="bulk_edit" if len(entries) > 1 else "add"))
    return PlannedMutation(
        intent=intent, candidate_entries=entries,
        mutation_delta=tuple(w.MutationDeltaItem(action="add", client_ref=c.client_ref) for c in entries),
        requested_write_scopes=(snapshot.default_write_scope,),
        projected_texts=tuple(c.text for c in entries), message="contract fixture",
    )


@pytest.mark.parametrize("vector", _contract_vectors("lane"), ids=lambda vector: vector["id"])
def test_revision_two_shared_contextual_lanes(tmp_path, vector):
    """Context refusals precede fake staging; accepted source lanes remain distinct."""
    from agent.memory_service.errors import ProviderError
    from agent.memory_service.mutation import _mutation_request
    from tests.agent.memory_service.test_curated_mutation import _service, _store

    store = _store()
    service, backend = _service(tmp_path, store)
    try:
        snapshot = service.load_curated(vector["input"]["target"])
        declaration = {"client_ref": "shared-candidate", "memory_admission": vector["input"]["memory_admission"]}
        plan = _contextual_contract_plan(snapshot, [declaration], vector["input"]["source_kind"])
        request = _mutation_request(service, snapshot, plan, vector["id"], "hermes", "memory_tool")
        if vector["expected"]["accepted"]:
            result = service.stage_curated(request)
            assert [(a.client_ref, a.disposition, a.record_channel) for a in result.admissions] == [
                (declaration["client_ref"], vector["expected"]["disposition"], vector["expected"]["record_channel"])]
        else:
            with pytest.raises(ProviderError) as exc:
                service.stage_curated(request)
            assert (exc.value.code, exc.value.outcome) == (vector["expected"]["code"], "not_committed")
            assert not store.stages and not store.stage_by_request
        assert backend.count("stage_curated") == 1
        assert not store.records and not store.import_index and not store.receipts
    finally:
        backend.shutdown()


@pytest.mark.parametrize("vector", _contract_vectors("lane_batch"), ids=lambda vector: vector["id"])
def test_revision_two_shared_contextual_batch(tmp_path, vector):
    """The real host builder and serialized fake preserve per-client-ref lane order."""
    from agent.memory_service.mutation import _mutation_request
    from tests.agent.memory_service.test_curated_mutation import _service, _store

    store = _store()
    service, backend = _service(tmp_path, store)
    try:
        snapshot = service.load_curated(vector["input"]["target"])
        plan = _contextual_contract_plan(snapshot, vector["input"]["candidates"], vector["input"]["source_kind"])
        request = _mutation_request(service, snapshot, plan, vector["id"], "hermes", "memory_tool")
        result = service.stage_curated(request)
        assert vector["expected"]["accepted"]
        assert [{key: getattr(a, key) for key in ("client_ref", "disposition", "record_channel")}
                for a in result.admissions] == vector["expected"]["admissions"]
        assert [c.to_wire()["memory_admission"] for c in backend.calls[-1][1].candidate_entries] == [
            c["memory_admission"] for c in vector["input"]["candidates"]]
        assert not store.records and not store.receipts
    finally:
        backend.shutdown()


@pytest.mark.parametrize("vector", _contract_vectors("binding"), ids=lambda vector: vector["id"])
def test_revision_two_shared_binding_retry_identity(tmp_path, vector):
    """Host ordered projection/retry identity, not the production ygg binding hash."""
    from agent.memory_service.errors import ProviderError
    from agent.memory_service.mutation import _mutation_request, _with_exact_retry
    from tests.agent.memory_service.test_curated_mutation import _service, _store

    store = _store()
    service, backend = _service(tmp_path, store)
    try:
        assert service._negotiation.selected_contract_revision == vector["input"]["contract_revision"] == 2
        assert next(req for op, req in backend.calls if op == "negotiate").supported_contract_revisions == (2,)
        snapshot = service.load_curated("memory")
        original = _contextual_contract_plan(snapshot, vector["input"]["original"])
        request = _mutation_request(service, snapshot, original, vector["id"], "hermes", "memory_tool")
        # Reach a written stage and lose the reply, then use the real immutable retry helper.
        store.fail_transport("stage_curated", phase="during")
        staged = _with_exact_retry(service, "memory", lambda: service.stage_curated(request))
        stages = [req for op, req in backend.calls if op == "stage_curated"]
        assert len(stages) == 2 and w.canonical_json(stages[0].to_wire()) == w.canonical_json(stages[1].to_wire())
        assert list(store.stages) == [staged.stage_handle_b64url]
        assert store.stages[staged.stage_handle_b64url].result == staged

        changed = _contextual_contract_plan(snapshot, vector["input"]["retry"])
        retry_plan = dataclasses.replace(original, candidate_entries=changed.candidate_entries)
        retry = _mutation_request(service, snapshot, retry_plan, vector["id"], "hermes", "memory_tool")
        if vector["expected"]["same_binding"]:
            replay = service.stage_curated(retry)
            assert replay == staged
            assert replay.approval_binding_sha256 == staged.approval_binding_sha256
        else:
            with pytest.raises(ProviderError) as exc:
                service.stage_curated(retry)
            assert (exc.value.code, exc.value.outcome) == ("idempotency_mismatch", "not_committed")

        stages = [req.to_wire() for op, req in backend.calls if op == "stage_curated"]
        projection = lambda body: [{key: c[key] for key in ("client_ref", "memory_admission")}
                                   for c in body["candidate_entries"]]
        assert projection(stages[0]) == vector["input"]["original"]
        assert projection(stages[-1]) == vector["input"]["retry"]
        assert (projection(stages[0]) == projection(stages[-1])) is vector["expected"]["same_projection"]
        assert (w.canonical_json(stages[0]) == w.canonical_json(stages[-1])) is vector["expected"]["same_binding"]
        assert list(store.stages) == [staged.stage_handle_b64url]
        assert store.stages[staged.stage_handle_b64url].result == staged
        assert not store.records and not store.receipts
    finally:
        backend.shutdown()
