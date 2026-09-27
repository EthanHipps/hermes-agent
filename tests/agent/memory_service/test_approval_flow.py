"""R39: approval inside the curated mutation flow (§9.4 steps 6–9; §9.5; rulings R39-1, R39-5..R39-11;
contracts C6b-1, C6b-2; ruling X6b-2)."""

import dataclasses
from datetime import datetime, timezone

import pytest

from agent.memory_service import wire as w
from agent.memory_service.approval import ApprovalChannel, build_authorization, render_stage_inspection
from agent.memory_service.approval_store import approvals_dir, list_pending_approvals, load_pending_approval
from agent.memory_service.mutation import MutationStatus, PlannedMutation, run_curated_mutation
from agent.memory_service.service import InspectRequest, MutationRequest, select_memory_service
from tests.agent.memory_service.fake_backend import FakeClock, FakeProviderStore, fake_backend_factory, set_key
from tests.agent.memory_service.native_sentinel import native_memory_sentinel

EPOCH = "ep-opaque-7f3a"          # distinctive, so "never rendered" can be checked

REPO = w.ScopeRef(kind="repository", id="repo-1")
PROJECT = w.ScopeRef(kind="project", id="proj-1")
GLOBAL = w.ScopeRef(kind="principal_global", id="ethan")
PROVENANCE = w.MutationProvenance(actor_kind="hermes", principal_id="ethan", logical_session_id="sess-1",
                                  initiating_surface="memory_tool", source_entry_ids=(), source_commit=None,
                                  threat_decision_id=None)


def _never_built():
    raise AssertionError("native store built")


def _config(tmp_path, **memory):
    exe = tmp_path / "provider.exe"
    exe.write_bytes(b"MZ")
    section = {"provider": "example", "provider_mode": "authoritative",
               "provider_executable": str(exe), "principal_id": "ethan"}
    section.update(memory)
    return {"memory": section}


def _context():
    return w.RequestedContext(principal_id="ethan", profile_id="default", logical_session_id="sess-1",
                              platform="cli", org_id=None, project_id=None, repo_id=None, workspace_id=None,
                              resolution_source="directory", canonical_directory="C:\\work\\repo")


def _store(**kwargs):
    kwargs.setdefault("clock", FakeClock(datetime(2026, 9, 22, tzinfo=timezone.utc)))
    kwargs.setdefault("epoch", EPOCH)
    return FakeProviderStore(**kwargs)


def _service(tmp_path, store, **memory):
    backends = []
    inner = fake_backend_factory(store)

    def factory(cfg):
        backends.append(inner(cfg))
        return backends[-1]

    service = select_memory_service(_config(tmp_path, **memory), store_factory=_never_built,
                                    requested_context=_context(), backend_factory=factory)
    return service, backends[-1]


def _add_plan(snapshot, text="a fact"):
    cand = w.CandidateEntry(client_ref="c1", text=text, destination_scope=snapshot.default_write_scope,
                            target=snapshot.target, proposed_policy_key=None, import_source_identity=None)
    return PlannedMutation(intent=w.MutationIntent(kind="add"),
                           mutation_delta=(w.MutationDeltaItem(action="add", client_ref="c1"),),
                           candidate_entries=(cand,), requested_write_scopes=(snapshot.default_write_scope,),
                           projected_texts=tuple(e.text for e in snapshot.mutation_entries) + (text,),
                           message="Entry added.")


def _replace_plan(snapshot, entry, text):
    cand = w.CandidateEntry(client_ref="c1", text=text, destination_scope=entry.origin_scope,
                            target=snapshot.target, proposed_policy_key=None, import_source_identity=None)
    return PlannedMutation(intent=w.MutationIntent(kind="replace", matched_entry_id=entry.id),
                           mutation_delta=(w.MutationDeltaItem(action="supersede", old_record_id=entry.id,
                                                               replacement_client_ref="c1"),),
                           candidate_entries=(cand,), requested_write_scopes=(entry.origin_scope,),
                           projected_texts=(text,), message="Entry replaced.")


def _request(snapshot, plan, request_id="r-predict"):
    return MutationRequest(target=snapshot.target, request_id=request_id, expected_revision=snapshot.revision,
                           hidden_preservation_state=snapshot.hidden_preservation_state,
                           requested_write_scopes=plan.requested_write_scopes, intent=plan.intent,
                           mutation_delta=plan.mutation_delta, candidate_entries=plan.candidate_entries,
                           provenance=PROVENANCE)


def _inspect(service, stage):
    return service.inspect_staged(InspectRequest(target=stage.target, request_id=stage.request_id,
                                                 stage_handle_b64url=stage.stage_handle_b64url))


def test_render_shows_every_member_the_approval_ui_needs_and_no_token(tmp_path):
    """§9.5 L1546: target, scopes, operation, visible diff, disposition, create/reuse, keys, hidden counts."""
    store = _store()
    old = store.seed_record(GLOBAL, "user", "likes coffee", lane="trusted_instruction")
    service, _ = _service(tmp_path, store)
    user = service.load_curated("user")
    entry = next(e for e in user.mutation_entries if e.id == old.id)
    stage = service.stage_curated(_request(user, _replace_plan(user, entry, "likes tea"), request_id="r-1"))
    text = render_stage_inspection(_inspect(service, stage), intent_kind="replace", requirements=("target_user",))
    for fragment in ("user profile", "replace", "principal_global:ethan", "target_user", stage.expires_at,
                     "~ likes coffee", "-> likes tea", stage.admissions[0].disposition,
                     stage.admissions[0].publication_effect, f"key {stage.admissions[0].policy_key}"):
        assert fragment in text
    hidden = service.load_curated("user").hidden_preservation_state.opaque_state_b64url
    for token in (stage.stage_handle_b64url, stage.approval_binding_sha256, stage.request_id, EPOCH, hidden,
                  service.identity.opaque_binding_b64url):
        assert token not in text


def test_render_reports_hidden_reset_effects_as_counts_only(tmp_path):
    store = _store()
    store.seed_record(REPO, "memory", "visible fact")
    store.seed_record(REPO, "memory", "raw body never shown", lane="raw")
    service, _ = _service(tmp_path, store)
    memory = service.load_curated("memory")
    stage = service.stage_curated(MutationRequest(
        target="memory", request_id="r-reset", expected_revision=memory.revision,
        hidden_preservation_state=memory.hidden_preservation_state, requested_write_scopes=(REPO,),
        intent=w.MutationIntent(kind="reset", reset_scopes=(REPO,)), mutation_delta=(), candidate_entries=(),
        provenance=PROVENANCE))
    text = render_stage_inspection(_inspect(service, stage), intent_kind="reset", requirements=("reset",))
    assert "- visible fact" in text and "raw body never shown" not in text
    assert "hidden: retire 1 raw record(s) in repository:repo-1 (hermes_memory)" in text


def test_authorization_binds_the_stage_and_never_outlives_it(tmp_path):
    service, _ = _service(tmp_path, _store())
    user = service.load_curated("user")
    stage = service.stage_curated(_request(user, _add_plan(user, "terse"), request_id="r-a"))
    now = datetime(2026, 9, 22, 0, 5, 7, 999, tzinfo=timezone.utc)
    auth = build_authorization(stage, principal_id="ethan", now=now)
    assert (auth.kind, auth.approved_by_principal_id, auth.approved_at, auth.expires_at, auth.approval_binding_sha256) == (
        "approved", "ethan", "2026-09-22T00:05:07Z", stage.expires_at, stage.approval_binding_sha256)
    assert set(auth.approval_id) <= set("0123456789abcdef")        # R39-8: passes ygg's token + secret scan
    assert build_authorization(stage, principal_id="ethan", now=now).approval_id != auth.approval_id
    assert w.ApprovalAuthorization.from_wire(auth.to_wire()) == auth
