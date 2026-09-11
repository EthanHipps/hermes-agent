"""Built-in (additive) service: native semantics over complete state and explicit delta."""

from datetime import datetime, timedelta, timezone
from dataclasses import replace
import subprocess
import sys

import pytest

from agent.memory_service import builtin as builtin_module
from agent.memory_service import wire as w
from agent.memory_service.builtin import BuiltinMemoryService, BuiltinStoreError
from agent.memory_service.config import resolve_memory_service_config
from agent.memory_service.errors import ProviderError, TargetDisabledError
from agent.memory_service.service import CommitIntent, InspectRequest, MemoryDisposition, MutationRequest
from tools.memory_tool import MemoryStore, get_memory_dir
from tools.memory_tool_store import MemoryFileUnreadableError

PROVENANCE = w.MutationProvenance(
    actor_kind="hermes",
    principal_id="local",
    logical_session_id="sess-1",
    initiating_surface="memory_tool",
    source_entry_ids=(),
    source_commit=None,
    threat_decision_id=None,
)


def _service(**memory):
    cfg = resolve_memory_service_config({"memory": memory})
    store = MemoryStore(
        memory_char_limit=memory.get("memory_char_limit", 2200),
        user_char_limit=memory.get("user_char_limit", 1375),
        memory_enabled=cfg.memory_enabled,
        user_profile_enabled=cfg.user_profile_enabled,
    )
    store.load_from_disk()
    return BuiltinMemoryService(cfg, store, principal_id="local", profile_id="default", logical_session_id="sess-1", platform="cli")


def _add(service, target, text, request_id="r1"):
    snap = service.load_curated(target)
    scope = snap.default_write_scope
    req = MutationRequest(
        target=target,
        request_id=request_id,
        expected_revision=snap.revision,
        hidden_preservation_state=snap.hidden_preservation_state,
        requested_write_scopes=(scope,),
        intent=w.MutationIntent(kind="add"),
        mutation_delta=(w.MutationDeltaItem(action="add", client_ref="c1"),),
        candidate_entries=(
            w.CandidateEntry(client_ref="c1", text=text, destination_scope=scope, target=target, proposed_policy_key=None, import_source_identity=None),
        ),
        provenance=PROVENANCE,
    )
    staged = service.stage_curated(req)
    return staged, service.commit_curated(
        CommitIntent(
            target=target,
            request_id=request_id,
            stage_handle_b64url=staged.stage_handle_b64url,
            approval_binding_sha256=staged.approval_binding_sha256,
            authorized_write_scopes=staged.requested_write_scopes,
            authorization=w.ApprovalAuthorization(kind="not_required"),
        )
    )


def test_disposition_identity_and_empty_snapshot():
    service = _service()
    assert service.disposition is MemoryDisposition.BUILTIN
    assert service.identity.provider == "builtin" and service.identity.provider_mode == "additive"
    snap = service.load_curated("memory")
    assert snap.status == "ok" and snap.target == "memory"
    assert [s.kind for s in snap.visible_scopes] == ["principal_global"]
    assert snap.default_write_scope.kind == "principal_global"
    assert snap.mutation_entries == () and snap.delivery_entries == ()
    assert snap.limits.memory_chars == 2200 and snap.limits.user_chars == 1375
    assert snap.revision.provider_epoch == "builtin"
    assert service.capabilities.recall_context is False
    assert service.prompt_block("memory") is None


def test_add_stage_inspect_commit_writes_memory_md():
    service = _service()
    staged, committed = _add(service, "memory", "Prefer uv over pip")
    assert staged.approval_requirements == ()
    assert staged.admissions[0].disposition == "scoped_evidence"
    assert staged.admissions[0].record_channel == "hermes_memory"
    assert committed.outcome == "committed_audit_clean"
    assert [e.text for e in committed.snapshot.mutation_entries] == ["Prefer uv over pip"]
    assert committed.snapshot.mutation_entries[0].id == staged.admissions[0].assigned_id
    assert "Prefer uv over pip" in (get_memory_dir() / "MEMORY.md").read_text(encoding="utf-8")
    assert service.load_curated("memory").revision == committed.snapshot.revision


def test_inspect_shows_before_and_after_without_mutating():
    service = _service()
    snap = service.load_curated("user")
    scope = snap.default_write_scope
    req = MutationRequest(
        target="user",
        request_id="r-user",
        expected_revision=snap.revision,
        hidden_preservation_state=snap.hidden_preservation_state,
        requested_write_scopes=(scope,),
        intent=w.MutationIntent(kind="add"),
        mutation_delta=(w.MutationDeltaItem(action="add", client_ref="c1"),),
        candidate_entries=(w.CandidateEntry(client_ref="c1", text="  Name: Ethan  ", destination_scope=scope, target="user", proposed_policy_key=None, import_source_identity=None),),
        provenance=PROVENANCE,
    )
    staged = service.stage_curated(req)
    inspection = service.inspect_staged(InspectRequest(target="user", request_id="r-user", stage_handle_b64url=staged.stage_handle_b64url))
    assert inspection.visible_before == ()
    assert [e.text for e in inspection.visible_after] == ["Name: Ethan"]
    assert inspection.canonical_candidates[0].text == "Name: Ethan"
    assert staged.admissions[0].disposition == "trusted_instruction"
    assert service.load_curated("user").mutation_entries == ()
    assert not (get_memory_dir() / "USER.md").exists()


def test_replace_and_remove_use_exact_ids():
    service = _service()
    _, committed = _add(service, "memory", "alpha", "r1")
    alpha_id = committed.snapshot.mutation_entries[0].id
    _add(service, "memory", "alphabet", "r2")
    snap = service.load_curated("memory")
    scope = snap.default_write_scope
    replace = MutationRequest(
        target="memory",
        request_id="r3",
        expected_revision=snap.revision,
        hidden_preservation_state=snap.hidden_preservation_state,
        requested_write_scopes=(scope,),
        intent=w.MutationIntent(kind="replace", matched_entry_id=alpha_id),
        mutation_delta=(w.MutationDeltaItem(action="supersede", old_record_id=alpha_id, replacement_client_ref="c1"),),
        candidate_entries=(w.CandidateEntry(client_ref="c1", text="beta", destination_scope=scope, target="memory", proposed_policy_key=None, import_source_identity=None),),
        provenance=PROVENANCE,
    )
    staged = service.stage_curated(replace)
    assert staged.admissions[0].superseded_id == alpha_id
    result = service.commit_curated(CommitIntent("memory", "r3", staged.stage_handle_b64url, staged.approval_binding_sha256, staged.requested_write_scopes, w.ApprovalAuthorization(kind="not_required")))
    assert sorted(e.text for e in result.snapshot.mutation_entries) == ["alphabet", "beta"]
    snap = result.snapshot
    alphabet_id = next(e.id for e in snap.mutation_entries if e.text == "alphabet")
    remove = MutationRequest(
        target="memory",
        request_id="r4",
        expected_revision=snap.revision,
        hidden_preservation_state=snap.hidden_preservation_state,
        requested_write_scopes=(scope,),
        intent=w.MutationIntent(kind="remove", matched_entry_id=alphabet_id),
        mutation_delta=(w.MutationDeltaItem(action="retire", record_id=alphabet_id),),
        candidate_entries=(),
        provenance=PROVENANCE,
    )
    staged = service.stage_curated(remove)
    result = service.commit_curated(CommitIntent("memory", "r4", staged.stage_handle_b64url, staged.approval_binding_sha256, staged.requested_write_scopes, w.ApprovalAuthorization(kind="not_required")))
    assert [e.text for e in result.snapshot.mutation_entries] == ["beta"]


def test_reset_retires_every_entry_of_the_target():
    """§9.3: reset means "retire every current-epoch record of this target in
    the exact approved reset scopes" -- it must never report success while
    leaving the entries in place."""
    service = _service()
    _add(service, "memory", "alpha", "r1")
    _add(service, "memory", "beta", "r2")
    snap = service.load_curated("memory")
    assert sorted(e.text for e in snap.mutation_entries) == ["alpha", "beta"]
    scope = snap.default_write_scope
    reset = MutationRequest("memory", "r3", snap.revision, snap.hidden_preservation_state, (scope,), w.MutationIntent(kind="reset", reset_scopes=(scope,)), (), (), PROVENANCE)
    staged = service.stage_curated(reset)
    # nothing is admitted and nothing is hidden: the native store has no hidden
    # raw records, so the whole effect is the visible after-state.
    assert staged.admissions == () and staged.candidate_hashes == () and staged.hidden_effects == ()
    inspection = service.inspect_staged(InspectRequest("memory", "r3", staged.stage_handle_b64url))
    assert sorted(e.text for e in inspection.visible_before) == ["alpha", "beta"]
    assert inspection.visible_after == ()  # the after-state is visible BEFORE commit
    assert sorted(e.text for e in service.load_curated("memory").mutation_entries) == ["alpha", "beta"]
    committed = service.commit_curated(CommitIntent("memory", "r3", staged.stage_handle_b64url, staged.approval_binding_sha256, staged.requested_write_scopes, w.ApprovalAuthorization(kind="not_required")))
    assert committed.outcome == "committed_audit_clean"
    assert committed.snapshot.mutation_entries == ()
    assert service.load_curated("memory").mutation_entries == ()
    on_disk = (get_memory_dir() / "MEMORY.md").read_text(encoding="utf-8")
    assert "alpha" not in on_disk and "beta" not in on_disk


def test_reset_outside_the_single_native_scope_is_unauthorized():
    service = _service()
    _add(service, "memory", "alpha", "r1")
    snap = service.load_curated("memory")
    scope = snap.default_write_scope
    other = w.ScopeRef(kind="repository", id="repo-1")
    wrong_reset_scope = MutationRequest("memory", "r2", snap.revision, snap.hidden_preservation_state, (scope,), w.MutationIntent(kind="reset", reset_scopes=(other,)), (), (), PROVENANCE)
    with pytest.raises(ProviderError) as exc:
        service.stage_curated(wrong_reset_scope)
    assert exc.value.code == "unauthorized_scope" and exc.value.outcome == "not_committed"
    no_write_scope = MutationRequest("memory", "r3", snap.revision, snap.hidden_preservation_state, (), w.MutationIntent(kind="reset", reset_scopes=(scope,)), (), (), PROVENANCE)
    with pytest.raises(ProviderError) as exc:
        service.stage_curated(no_write_scope)
    assert exc.value.code == "unauthorized_scope"
    assert [e.text for e in service.load_curated("memory").mutation_entries] == ["alpha"]


def test_import_is_refused_because_there_is_no_import_source_index():
    """§9.3 import needs an acknowledgement-derived source index for reuse
    detection; the native store has none, so the intent is refused at stage
    rather than reported as create_record and deduped away at commit."""
    service = _service()
    _add(service, "memory", "already here", "r0")
    snap = service.load_curated("memory")
    scope = snap.default_write_scope
    identity = w.ImportSourceIdentity(source_kind="native_memory", parser_version="hermes-native-v0.20.6", source_id="memory-md", item_key="MEMORY.md/1")
    req = MutationRequest(
        "memory",
        "r1",
        snap.revision,
        snap.hidden_preservation_state,
        (scope,),
        w.MutationIntent(kind="import", import_run_id="run-1", source_kind="native_memory"),
        (w.MutationDeltaItem(action="add", client_ref="c1"),),
        (w.CandidateEntry("c1", "already here", scope, "memory", None, identity),),
        PROVENANCE,
    )
    with pytest.raises(ProviderError) as exc:
        service.stage_curated(req)
    assert exc.value.code == "invalid_request" and exc.value.outcome == "not_committed"
    assert [e.text for e in service.load_curated("memory").mutation_entries] == ["already here"]


def test_expired_stage_answers_stage_expired_on_every_door(monkeypatch):
    """§9.3: an expired stage fails ``stage_expired`` at inspect, commit, and a
    stage replay -- not ``stage_not_found`` at two of the three doors."""
    service = _service()
    snap = service.load_curated("memory")
    scope = snap.default_write_scope
    req = MutationRequest("memory", "r1", snap.revision, snap.hidden_preservation_state, (scope,), w.MutationIntent(kind="add"), (w.MutationDeltaItem(action="add", client_ref="c1"),), (w.CandidateEntry("c1", "x", scope, "memory", None, None),), PROVENANCE)
    staged = service.stage_curated(req)
    later = datetime.now(timezone.utc) + timedelta(hours=2)
    monkeypatch.setattr(builtin_module, "_now", lambda: later)
    with pytest.raises(ProviderError) as exc:
        service.inspect_staged(InspectRequest("memory", "r1", staged.stage_handle_b64url))
    assert exc.value.code == "stage_expired" and exc.value.details is None
    with pytest.raises(ProviderError) as exc:
        service.commit_curated(CommitIntent("memory", "r1", staged.stage_handle_b64url, staged.approval_binding_sha256, staged.requested_write_scopes, w.ApprovalAuthorization(kind="not_required")))
    assert exc.value.code == "stage_expired" and exc.value.details is None
    with pytest.raises(ProviderError) as exc:
        service.stage_curated(req)
    assert exc.value.code == "stage_expired" and exc.value.details is None
    assert service.load_curated("memory").mutation_entries == ()


def test_stale_revision_is_a_version_conflict_with_current_snapshot():
    service = _service()
    stale = service.load_curated("memory")
    _add(service, "memory", "first", "r1")
    scope = stale.default_write_scope
    req = MutationRequest("memory", "r2", stale.revision, stale.hidden_preservation_state, (scope,), w.MutationIntent(kind="add"), (w.MutationDeltaItem(action="add", client_ref="c1"),), (w.CandidateEntry("c1", "second", scope, "memory", None, None),), PROVENANCE)
    with pytest.raises(ProviderError) as exc:
        service.stage_curated(req)
    assert exc.value.code == "version_conflict" and exc.value.outcome == "not_committed"
    assert exc.value.details["current_snapshot"]["mutation_entries"][0]["text"] == "first"


def test_unknown_record_id_and_mismatched_token_are_invalid_request():
    service = _service()
    snap = service.load_curated("memory")
    scope = snap.default_write_scope
    bad_id = MutationRequest("memory", "r1", snap.revision, snap.hidden_preservation_state, (scope,), w.MutationIntent(kind="remove", matched_entry_id="bzzz"), (w.MutationDeltaItem(action="retire", record_id="bzzz"),), (), PROVENANCE)
    with pytest.raises(ProviderError) as exc:
        service.stage_curated(bad_id)
    assert exc.value.code == "invalid_request"
    other_token = w.HiddenPreservationState(target="memory", complete_for_scopes=snap.hidden_preservation_state.complete_for_scopes, opaque_state_b64url="AAAA")
    req = MutationRequest("memory", "r2", snap.revision, other_token, (scope,), w.MutationIntent(kind="add"), (w.MutationDeltaItem(action="add", client_ref="c1"),), (w.CandidateEntry("c1", "x", scope, "memory", None, None),), PROVENANCE)
    with pytest.raises(ProviderError) as exc:
        service.stage_curated(req)
    assert exc.value.code == "invalid_request"


def test_stage_replay_and_commit_replay_are_idempotent():
    service = _service()
    snap = service.load_curated("memory")
    scope = snap.default_write_scope
    req = MutationRequest("memory", "r1", snap.revision, snap.hidden_preservation_state, (scope,), w.MutationIntent(kind="add"), (w.MutationDeltaItem(action="add", client_ref="c1"),), (w.CandidateEntry("c1", "x", scope, "memory", None, None),), PROVENANCE)
    first = service.stage_curated(req)
    assert service.stage_curated(req) == first
    changed = MutationRequest("memory", "r1", snap.revision, snap.hidden_preservation_state, (scope,), w.MutationIntent(kind="add"), (w.MutationDeltaItem(action="add", client_ref="c1"),), (w.CandidateEntry("c1", "y", scope, "memory", None, None),), PROVENANCE)
    with pytest.raises(ProviderError) as exc:
        service.stage_curated(changed)
    assert exc.value.code == "idempotency_mismatch"
    intent = CommitIntent("memory", "r1", first.stage_handle_b64url, first.approval_binding_sha256, first.requested_write_scopes, w.ApprovalAuthorization(kind="not_required"))
    committed = service.commit_curated(intent)
    replay = service.commit_curated(intent)
    assert replay.outcome == "idempotent_replay" and replay.tx_id == committed.tx_id
    assert replay.snapshot == service.load_curated("memory")
    with pytest.raises(ProviderError) as exc:
        service.inspect_staged(InspectRequest("memory", "r1", first.stage_handle_b64url))
    assert exc.value.code == "stage_not_found" and exc.value.details == {"state": "committed", "tx_id": committed.tx_id}


def test_stage_replay_after_commit_reports_the_committed_state():
    service = _service()
    snap = service.load_curated("memory")
    scope = snap.default_write_scope
    req = MutationRequest("memory", "r1", snap.revision, snap.hidden_preservation_state, (scope,), w.MutationIntent(kind="add"), (w.MutationDeltaItem(action="add", client_ref="c1"),), (w.CandidateEntry("c1", "x", scope, "memory", None, None),), PROVENANCE)
    staged = service.stage_curated(req)
    committed = service.commit_curated(CommitIntent("memory", "r1", staged.stage_handle_b64url, staged.approval_binding_sha256, staged.requested_write_scopes, w.ApprovalAuthorization(kind="not_required")))
    with pytest.raises(ProviderError) as exc:
        service.stage_curated(req)
    assert exc.value.code == "stage_not_found"
    assert exc.value.details == {"state": "committed", "tx_id": committed.tx_id}
    assert [e.text for e in service.load_curated("memory").mutation_entries] == ["x"]


def test_commit_checks_binding_scopes_and_handle():
    service = _service()
    snap = service.load_curated("memory")
    scope = snap.default_write_scope
    req = MutationRequest("memory", "r1", snap.revision, snap.hidden_preservation_state, (scope,), w.MutationIntent(kind="add"), (w.MutationDeltaItem(action="add", client_ref="c1"),), (w.CandidateEntry("c1", "x", scope, "memory", None, None),), PROVENANCE)
    staged = service.stage_curated(req)
    with pytest.raises(ProviderError) as exc:
        service.commit_curated(CommitIntent("memory", "r1", staged.stage_handle_b64url, "0" * 64, staged.requested_write_scopes, w.ApprovalAuthorization(kind="not_required")))
    assert exc.value.code == "approval_invalid"
    with pytest.raises(ProviderError) as exc:
        service.commit_curated(CommitIntent("memory", "r1", staged.stage_handle_b64url, staged.approval_binding_sha256, (), w.ApprovalAuthorization(kind="not_required")))
    assert exc.value.code == "unauthorized_scope"
    with pytest.raises(ProviderError) as exc:
        service.commit_curated(CommitIntent("memory", "r-other", staged.stage_handle_b64url, staged.approval_binding_sha256, staged.requested_write_scopes, w.ApprovalAuthorization(kind="not_required")))
    assert exc.value.code == "stage_not_found" and exc.value.details is None


def test_disabled_target_is_never_loaded_rendered_or_mutated():
    service = _service(user_profile_enabled=False)
    assert not service.target_enabled("user")
    with pytest.raises(TargetDisabledError):
        service.load_curated("user")
    with pytest.raises(TargetDisabledError):
        service.prompt_block("user")
    snap = service.load_curated("memory")
    req = MutationRequest("user", "r1", snap.revision, snap.hidden_preservation_state, (snap.default_write_scope,), w.MutationIntent(kind="add"), (w.MutationDeltaItem(action="add", client_ref="c1"),), (w.CandidateEntry("c1", "x", snap.default_write_scope, "user", None, None),), PROVENANCE)
    with pytest.raises(TargetDisabledError):
        service.stage_curated(req)


def test_over_budget_commit_surfaces_the_store_error_without_writing():
    service = _service(memory_char_limit=40)
    snap = service.load_curated("memory")
    scope = snap.default_write_scope
    req = MutationRequest("memory", "r1", snap.revision, snap.hidden_preservation_state, (scope,), w.MutationIntent(kind="add"), (w.MutationDeltaItem(action="add", client_ref="c1"),), (w.CandidateEntry("c1", "x" * 50, scope, "memory", None, None),), PROVENANCE)
    staged = service.stage_curated(req)
    with pytest.raises(BuiltinStoreError) as exc:
        service.commit_curated(CommitIntent("memory", "r1", staged.stage_handle_b64url, staged.approval_binding_sha256, staged.requested_write_scopes, w.ApprovalAuthorization(kind="not_required")))
    assert exc.value.response["success"] is False
    assert service.load_curated("memory").mutation_entries == ()


def test_apply_exact_delta_is_all_or_nothing():
    store = MemoryStore()
    store.load_from_disk()
    assert store.apply_exact_delta("memory", retire=[], add=["one", "two"])["success"]
    assert store.memory_entries == ["one", "two"]
    bad = store.apply_exact_delta("memory", retire=["missing"], add=["three"])
    assert bad["success"] is False and store.memory_entries == ["one", "two"]
    ok = store.apply_exact_delta("memory", retire=["one"], add=["three", "two"])
    assert ok["success"] and store.memory_entries == ["two", "three"]
    fresh = MemoryStore()
    fresh.load_from_disk()
    assert fresh.memory_entries == ["two", "three"]


def test_snapshot_identity_is_a_wire_artifact_not_the_host_identity():
    """CuratedSnapshot.frozen_identity is a wire-typed artifact (its schema has no
    "additive" provider_mode slot); hosts must read service.identity / service.disposition
    for the real built-in identity, not snapshot.frozen_identity.provider_mode."""
    service = _service()
    snap = service.load_curated("memory")
    assert snap.frozen_identity.provider == "builtin"
    assert service.identity.provider == "builtin"
    assert service.identity.provider_mode == "additive"
    assert service.disposition is MemoryDisposition.BUILTIN


def test_unreadable_native_file_raises_instead_of_false_completeness(monkeypatch):
    service = _service()
    monkeypatch.setattr(service._store, "_read_raw_checked", lambda path: ("", False))
    with pytest.raises(BuiltinStoreError) as exc:
        service.load_curated("memory")
    assert exc.value.response["success"] is False


def _mutation_request(service, target, kind="add", texts=("new",), request_id="transaction"):
    snapshot = service.load_curated(target)
    scope = snapshot.default_write_scope
    candidates = tuple(
        w.CandidateEntry(f"c{i}", text, scope, target, None, None)
        for i, text in enumerate(texts)
    )
    if kind == "reset":
        intent = w.MutationIntent(kind="reset", reset_scopes=(scope,))
        delta, candidates = (), ()
    elif kind == "remove":
        record_id = snapshot.mutation_entries[0].id
        intent = w.MutationIntent(kind="remove", matched_entry_id=record_id)
        delta, candidates = (w.MutationDeltaItem(action="retire", record_id=record_id),), ()
    elif kind == "replace":
        record_id = snapshot.mutation_entries[0].id
        intent = w.MutationIntent(kind="replace", matched_entry_id=record_id)
        delta = (w.MutationDeltaItem(action="supersede", old_record_id=record_id,
                                     replacement_client_ref="c0"),)
    else:
        intent = w.MutationIntent(kind="bulk_edit" if len(candidates) > 1 else "add")
        delta = tuple(w.MutationDeltaItem(action="add", client_ref=c.client_ref) for c in candidates)
    return MutationRequest(target, request_id, snapshot.revision, snapshot.hidden_preservation_state,
                           (scope,), intent, delta, candidates, PROVENANCE)


@pytest.mark.parametrize("target,other", [("memory", "user"), ("user", "memory")])
@pytest.mark.parametrize("operation", ["inspect", "commit", "replay"])
def test_stage_target_remains_bound_through_inspection_commit_and_replay(target, other, operation):
    service = _service()
    staged = service.stage_curated(_mutation_request(service, target))
    intended = _commit_intent(staged)
    if operation == "replay":
        committed = service.commit_curated(intended)
    before = {t: service.load_curated(t) for t in (target, other)}
    with pytest.raises(ProviderError) as exc:
        if operation == "inspect":
            service.inspect_staged(InspectRequest(other, staged.request_id, staged.stage_handle_b64url))
        else:
            service.commit_curated(replace(intended, target=other))
    assert (exc.value.code, exc.value.outcome) == ("invalid_request", "not_committed")
    assert {t: service.load_curated(t) for t in (target, other)} == before
    # A refused cross-target request neither consumes nor invalidates the
    # correctly bound stage/receipt, even when the two files held equal text.
    correct = service.commit_curated(intended)
    assert correct.snapshot.target == target
    assert {entry.target for entry in correct.admissions} == {target}
    assert service.load_curated(other) == before[other]
    if operation == "replay":
        assert (correct.tx_id, correct.admissions) == (committed.tx_id, committed.admissions)


def _commit_intent(staged):
    return CommitIntent(staged.target, staged.request_id, staged.stage_handle_b64url,
                        staged.approval_binding_sha256, staged.requested_write_scopes,
                        w.ApprovalAuthorization(kind="not_required"))


@pytest.mark.parametrize("target", ["memory", "user"])
@pytest.mark.parametrize("kind", ["reset", "add", "remove", "replace"])
def test_commit_rejects_another_session_write_at_the_native_lock(monkeypatch, target, kind):
    service = _service()
    _add(service, target, "alpha", "seed")
    staged = service.stage_curated(_mutation_request(service, target, kind))
    service.inspect_staged(InspectRequest(target, staged.request_id, staged.stage_handle_b64url))
    other = MemoryStore()
    other.load_from_disk()
    frozen_prompt = service.prompt_block(target)
    original_apply = service._store.apply_exact_delta

    def interleave(*args, **kwargs):
        # A separate process publishes between the service and native mutation.
        subprocess.run([
            sys.executable, "-c",
            "import sys; from tools.memory_tool_store import MemoryStore; "
            "result = MemoryStore().add(sys.argv[1], 'concurrent'); assert result['success'], result",
            target,
        ], check=True, capture_output=True, text=True, timeout=15)
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(service._store, "apply_exact_delta", interleave)
    with pytest.raises(ProviderError) as exc:
        service.commit_curated(_commit_intent(staged))
    assert (exc.value.code, exc.value.outcome) == ("version_conflict", "not_committed")
    observed = w.CuratedSnapshot.from_wire(exc.value.details["current_snapshot"])
    assert [entry.text for entry in observed.mutation_entries] == other.read_entries(target) == ["alpha", "concurrent"]
    assert service.prompt_block(target) == frozen_prompt


@pytest.mark.parametrize("target", ["memory", "user"])
@pytest.mark.parametrize("after_write", ["read_failure", "another_write"])
def test_commit_receipt_describes_publication_and_replays_without_writing(monkeypatch, target, after_write):
    service = _service()
    staged = service.stage_curated(_mutation_request(service, target))
    intent = _commit_intent(staged)
    original_apply = service._store.apply_exact_delta
    original_read = service._store.read_entries
    other = MemoryStore()
    writes = []

    def fail_read(_target):
        raise MemoryFileUnreadableError(service._store._path_for(_target))

    def publish_then_interleave(*args, **kwargs):
        response = original_apply(*args, **kwargs)
        writes.append(response)
        if after_write == "read_failure":
            monkeypatch.setattr(service._store, "read_entries", fail_read)
        else:
            assert other.add(target, "later")["success"]
            # A subsequent read must not mutate the captured publication result.
            original_read(target)
        return response

    monkeypatch.setattr(service._store, "apply_exact_delta", publish_then_interleave)
    committed = service.commit_curated(intent)
    assert committed.outcome == "committed_audit_clean"
    assert [entry.text for entry in committed.snapshot.mutation_entries] == ["new"]
    if after_write == "read_failure":
        with pytest.raises(BuiltinStoreError):
            service.commit_curated(intent)
        monkeypatch.setattr(service._store, "read_entries", original_read)
    replay = service.commit_curated(intent)
    assert replay.outcome == "idempotent_replay"
    assert replay.tx_id == committed.tx_id and replay.admissions == committed.admissions
    assert replay.snapshot == service.load_curated(target)
    assert len(writes) == 1


@pytest.mark.parametrize("target", ["memory", "user"])
@pytest.mark.parametrize("kind", ["add", "replace"])
@pytest.mark.parametrize("text", ["alpha\n§\nbeta", "alpha\r\n§\r\nbeta", "\ufeffalpha"])
def test_unrepresentable_candidate_is_rejected_before_stage_or_write(target, kind, text):
    service = _service()
    _add(service, target, "seed", "seed")
    if kind == "add":
        # Put a BOM candidate first as well as covering replacement of the first entry.
        service._store.remove(target, "seed")
    before = service.load_curated(target)
    path = service._store._path_for(target)
    before_bytes = path.read_bytes()
    request = _mutation_request(service, target, kind, (text,))
    with pytest.raises(ProviderError) as exc:
        service.stage_curated(request)
    assert (exc.value.code, exc.value.outcome) == ("invalid_request", "not_committed")
    assert service.load_curated(target) == before and path.read_bytes() == before_bytes
    with pytest.raises(ProviderError) as retry:
        service.stage_curated(request)
    assert retry.value.code == "invalid_request"
    response = service._store.apply_exact_delta(target, retire=[e.text for e in before.mutation_entries], add=[text])
    assert response["success"] is False and path.read_bytes() == before_bytes


@pytest.mark.parametrize("target", ["memory", "user"])
@pytest.mark.parametrize("texts, expected", [
    (("alpha\n§", "beta"), None),
    (("alpha\n§",), ["alpha\n§"]),
    (("alpha\nsecond line", "inline § symbol"), ["alpha\nsecond line", "inline § symbol"]),
    (("alpha\r\nsecond\rthird",), ["alpha\nsecond\nthird"]),
    (("same\r\nentry", "same\nentry"), ["same\nentry"]),
])
def test_serialization_preserves_whole_record_list_and_admission_ids(target, texts, expected):
    service = _service()
    request = _mutation_request(service, target, texts=texts)
    if expected is None:
        with pytest.raises(ProviderError) as exc:
            service.stage_curated(request)
        assert exc.value.code == "invalid_request"
        result = service._store.apply_exact_delta(target, retire=[], add=list(texts))
        assert result["success"] is False
        assert service.load_curated(target).mutation_entries == ()
        return
    staged = service.stage_curated(request)
    inspected = service.inspect_staged(InspectRequest(target, staged.request_id, staged.stage_handle_b64url))
    committed = service.commit_curated(_commit_intent(staged))
    disk = service.load_curated(target)
    assert [entry.text for entry in disk.mutation_entries] == expected
    assert inspected.visible_after == committed.snapshot.mutation_entries == disk.mutation_entries
    assert {a.assigned_id for a in staged.admissions} == {e.id for e in disk.mutation_entries}
    assert [c.text for c in inspected.canonical_candidates] == [text.replace("\r\n", "\n").replace("\r", "\n") for text in texts]
