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


@pytest.fixture
def native_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from tools.memory_tool import get_memory_dir
    return get_memory_dir()


def _user_add(text="prefers terse replies"):
    return lambda snapshot: _add_plan(snapshot, text)


def _run(service, planner, native_dir, target="user", **kwargs):
    with native_memory_sentinel(native_dir) as sentinel:
        outcome = run_curated_mutation(service, target, planner, **kwargs)
    sentinel.assert_untouched()
    return outcome


def _ops(backend):
    return [op for op, _ in backend.calls]


def _commits(backend):
    return [req for op, req in backend.calls if op == "commit_curated"]


def _sinks(native_dir):
    """Byte-and-mtime snapshot of every sink an approval could write (brief §4)."""
    home = native_dir.parent
    out = {}
    for sub in ("memory_service/approvals", "pending", "memories"):
        root = home / sub
        for p in (sorted(root.rglob("*")) if root.exists() else ()):
            if p.is_file():
                out[str(p.relative_to(home))] = (p.read_bytes(), p.stat().st_mtime_ns)
    return out


def _answer(value, seen=None):
    def prompt(p):
        if seen is not None:
            seen.append(p)
        return value
    return prompt


def _persist(service, session_id="sess-1"):
    """X6b-2: the session a deferral names must hold the staging identity's C2 record."""
    from agent.memory_service.host_state import HostStateRecord, save_host_state
    save_host_state(HostStateRecord(session_id, "provider_authoritative", service.session_state))


def test_no_usable_channel_refuses_before_staging(tmp_path, native_dir):
    """R38-1 preserved for callers with no approval channel (§9.5 L1538)."""
    for approval in (None, ApprovalChannel()):
        service, backend = _service(tmp_path, _store())
        outcome = _run(service, _user_add(), native_dir, approval=approval)
        assert outcome.status is MutationStatus.APPROVAL_UNAVAILABLE and outcome.approval_requirements == ("target_user",)
        assert "stage_curated" not in _ops(backend)


def test_inline_approval_inspects_then_commits_the_exact_stage(tmp_path, native_dir):
    store, seen = _store(), []
    service, backend = _service(tmp_path, store)
    outcome = _run(service, _user_add(), native_dir, approval=ApprovalChannel(prompt=_answer(True, seen), clock=store.clock.now))
    assert outcome.status is MutationStatus.COMMITTED
    assert _ops(backend)[-6:] == ["load_curated", "stage_curated", "inspect_staged", "commit_curated",
                                  "load_curated", "load_curated"]
    [prompt] = seen
    assert prompt.inspection.summary == prompt.stage and prompt.requirements == ("target_user",)
    assert prompt.text == render_stage_inspection(prompt.inspection, intent_kind="add", requirements=("target_user",))
    commit = _commits(backend)[-1]
    assert commit.authorization.kind == "approved" and commit.authorization.approved_by_principal_id == "ethan"
    assert commit.authorization.expires_at == prompt.stage.expires_at
    assert commit.approval_binding_sha256 == commit.authorization.approval_binding_sha256 == prompt.stage.approval_binding_sha256
    assert not approvals_dir(native_dir.parent).exists()           # no session: nothing written


def test_inline_approval_with_a_session_writes_ahead_and_settles(tmp_path, native_dir, monkeypatch):
    store = _store()
    service, backend = _service(tmp_path, store)
    _persist(service)
    real, during = service.commit_curated, []

    def commit(intent):
        during.extend(record for _, record in list_pending_approvals())
        return real(intent)

    monkeypatch.setattr(service, "commit_curated", commit)
    outcome = _run(service, _user_add(), native_dir,
                   approval=ApprovalChannel(prompt=_answer(True), session_id="sess-1", clock=store.clock.now))
    assert outcome.status is MutationStatus.COMMITTED and list_pending_approvals() == []
    [record] = during
    assert record.decision == "approved" and record.authorization == _commits(backend)[-1].authorization


def test_denial_persists_nothing_and_never_commits(tmp_path, native_dir):
    """§9.5 L1542: a host denial persists no approval and drops the handle."""
    service, backend = _service(tmp_path, _store())
    _persist(service)
    before = _sinks(native_dir)
    outcome = _run(service, _user_add(), native_dir, approval=ApprovalChannel(prompt=_answer(False), session_id="sess-1"))
    assert outcome.status is MutationStatus.DENIED and _commits(backend) == []
    assert _sinks(native_dir) == before and not approvals_dir(native_dir.parent).exists()


def test_unanswered_prompt_defers_when_a_session_can_wait(tmp_path, native_dir):
    service, backend = _service(tmp_path, _store())
    _persist(service)
    outcome = _run(service, _user_add(), native_dir, approval=ApprovalChannel(prompt=_answer(None), session_id="sess-1"))
    assert outcome.status is MutationStatus.PENDING_APPROVAL and _commits(backend) == []
    record = load_pending_approval(outcome.pending_id)
    assert (record.decision, record.authorization, record.stage_handle_b64url) == ("pending", None, outcome.stage.stage_handle_b64url)


def test_unanswered_prompt_without_a_session_fails_closed(tmp_path, native_dir):
    service, backend = _service(tmp_path, _store())
    outcome = _run(service, _user_add(), native_dir, approval=ApprovalChannel(prompt=_answer(None)))
    assert outcome.status is MutationStatus.APPROVAL_UNAVAILABLE and _commits(backend) == []
    assert list_pending_approvals() == []


def test_deferral_without_a_prompt_does_not_inspect(tmp_path, native_dir):
    """Ruling R39-6: inspection precedes a decision; a deferral is not one."""
    service, backend = _service(tmp_path, _store())
    _persist(service)
    outcome = _run(service, _user_add(), native_dir, approval=ApprovalChannel(session_id="sess-1"))
    assert outcome.status is MutationStatus.PENDING_APPROVAL and "inspect_staged" not in _ops(backend)


def test_the_record_holds_no_candidate_body_entry_body_or_candidate_hash(tmp_path, native_dir):
    store = _store()
    old = store.seed_record(GLOBAL, "user", "likes coffee", lane="trusted_instruction")
    service, _ = _service(tmp_path, store)
    _persist(service)
    outcome = _run(service, lambda s: _replace_plan(s, next(e for e in s.mutation_entries if e.id == old.id), "likes tea"),
                   native_dir, approval=ApprovalChannel(session_id="sess-1"))
    raw = (approvals_dir(native_dir.parent) / f"{outcome.pending_id}.json").read_bytes()
    assert b"likes coffee" not in raw and b"likes tea" not in raw
    assert all(h.canonical_sha256.encode() not in raw for h in outcome.stage.candidate_hashes)


def test_a_host_only_requirement_is_approved_not_waived(tmp_path, native_dir):
    """R39-9: memory.write_approval adds a requirement the provider does not derive (§9.5 L1538)."""
    store, seen = _store(), []
    service, backend = _service(tmp_path, store)
    outcome = _run(service, lambda s: _add_plan(s), native_dir, target="memory",
                   extra_approval="memory.write_approval",
                   approval=ApprovalChannel(prompt=_answer(True, seen), clock=store.clock.now))
    assert outcome.status is MutationStatus.COMMITTED and seen[0].stage.approval_requirements == ()
    assert seen[0].requirements == ("memory.write_approval",) and _commits(backend)[-1].authorization.kind == "approved"


def test_a_provider_requirement_outside_the_prediction_fails_closed_unasked(tmp_path, native_dir):
    """R39-10 / K-5."""
    store, seen = _store(), []
    store.corrupt_result("stage_curated", set_key("approval_requirements", ["target_user", "bulk_edit"]))
    service, backend = _service(tmp_path, store)
    _persist(service)
    outcome = _run(service, _user_add(), native_dir, approval=ApprovalChannel(prompt=_answer(True, seen), session_id="sess-1"))
    assert outcome.status is MutationStatus.APPROVAL_UNAVAILABLE and seen == [] and _commits(backend) == []
    assert list_pending_approvals() == []


def test_a_predicted_requirement_the_provider_omits_is_still_asked(tmp_path, native_dir):
    store, seen = _store(), []
    store.corrupt_result("stage_curated", set_key("approval_requirements", []))
    store.corrupt_result("inspect_staged", set_key("summary.approval_requirements", []))
    service, _ = _service(tmp_path, store)
    outcome = _run(service, _user_add(), native_dir, approval=ApprovalChannel(prompt=_answer(True, seen), clock=store.clock.now))
    assert outcome.status is MutationStatus.COMMITTED and seen[0].requirements == ("target_user",)


def test_an_inspection_that_differs_from_its_stage_is_never_shown(tmp_path, native_dir):
    store, seen = _store(), []
    store.corrupt_result("inspect_staged", set_key("summary.expires_at", "2030-01-01T00:00:00Z"))
    service, backend = _service(tmp_path, store)
    outcome = _run(service, _user_add(), native_dir, approval=ApprovalChannel(prompt=_answer(True, seen)))
    assert outcome.status is MutationStatus.UNAVAILABLE and outcome.error_code == "inspection_mismatch"
    assert seen == [] and _commits(backend) == []


def test_an_unknown_approved_commit_keeps_the_identical_request(tmp_path, native_dir):
    """R39-7, §9.5 L1562: after the one exact retry, the approved record waits with outcome unknown."""
    store = _store()
    store.fail_transport("commit_curated", phase="after_publish", times=2)
    service, backend = _service(tmp_path, store)
    _persist(service)
    outcome = _run(service, _user_add(), native_dir,
                   approval=ApprovalChannel(prompt=_answer(True), session_id="sess-1", clock=store.clock.now))
    assert outcome.status is MutationStatus.OUTCOME_UNKNOWN and outcome.pending_id
    record = load_pending_approval(outcome.pending_id)
    assert record.outcome == "unknown" and record.decision == "approved"
    first, second = _commits(backend)
    assert w.canonical_json(first.to_wire()) == w.canonical_json(second.to_wire())
    assert record.authorization == first.authorization


def test_a_typed_commit_error_after_approval_drops_the_record(tmp_path, native_dir):
    store = _store()
    store.fail_typed("commit_curated", "store_blocked", details={"reason": "maintenance"})
    service, _ = _service(tmp_path, store)
    _persist(service)
    outcome = _run(service, _user_add(), native_dir,
                   approval=ApprovalChannel(prompt=_answer(True), session_id="sess-1", clock=store.clock.now))
    assert outcome.status is MutationStatus.REJECTED and outcome.error_detail == "maintenance"
    assert list_pending_approvals() == []


def test_a_commit_conflict_after_approval_restages_and_reapproves(tmp_path, native_dir, monkeypatch):
    """§9.5 L1556: never edits an approved stage; a new request ID and a new approval."""
    store, seen = _store(), []
    service, backend = _service(tmp_path, store)
    _persist(service)
    real, fired = service.commit_curated, []

    def commit(intent):
        if not fired:
            fired.append(True)
            store.external_write(GLOBAL, "user", "written elsewhere")
        return real(intent)

    monkeypatch.setattr(service, "commit_curated", commit)
    outcome = _run(service, _user_add(), native_dir,
                   approval=ApprovalChannel(prompt=_answer(True, seen), session_id="sess-1", clock=store.clock.now))
    assert outcome.status is MutationStatus.COMMITTED and outcome.attempts == 2 and len(seen) == 2
    assert seen[0].stage.request_id != seen[1].stage.request_id and list_pending_approvals() == []


@pytest.mark.parametrize("prompt", [_answer(True), None])
def test_an_approval_store_failure_fails_closed_before_any_commit(tmp_path, native_dir, monkeypatch, prompt):
    """§9.10 L1689 'approval-store failure'."""
    def boom(*args, **kwargs):
        raise OSError("disk full")
    service, backend = _service(tmp_path, _store())
    _persist(service)          # before the patch: save_host_state also writes through atomic_json_write
    monkeypatch.setattr("utils.atomic_json_write", boom)
    outcome = _run(service, _user_add(), native_dir, approval=ApprovalChannel(prompt=prompt, session_id="sess-1"))
    assert outcome.status is MutationStatus.APPROVAL_UNAVAILABLE and outcome.error_code == "approval_store_unavailable"
    assert _commits(backend) == []


def test_a_threat_decision_makes_approval_mandatory(tmp_path, native_dir):
    """R39-11 (a): the generic path; the memory tool itself still refuses threat hits."""
    store, seen = _store(), []
    service, _ = _service(tmp_path, store)
    outcome = _run(service, lambda s: dataclasses.replace(_add_plan(s), threat_decision_id="t" + "0" * 31),
                   native_dir, target="memory", approval=ApprovalChannel(prompt=_answer(True, seen), clock=store.clock.now))
    assert outcome.status is MutationStatus.COMMITTED
    assert seen[0].requirements == ("threat",) and seen[0].stage.approval_requirements == ("threat",)


def test_a_session_without_a_matching_record_cannot_defer(tmp_path, native_dir):
    """Ruling X6b-2: nothing to replay from, so no stage and no record (§4.1 L250)."""
    service, backend = _service(tmp_path, _store())
    outcome = _run(service, _user_add(), native_dir, approval=ApprovalChannel(session_id="sess-1"))
    assert outcome.status is MutationStatus.APPROVAL_UNAVAILABLE and "stage_curated" not in _ops(backend)
    assert list_pending_approvals() == []


def test_an_inline_approval_without_a_matching_record_commits_without_write_ahead(tmp_path, native_dir):
    """Ruling X6b-2: an inline answer still commits, but nothing is written ahead without a replayable identity."""
    store = _store()
    service, _ = _service(tmp_path, store)
    outcome = _run(service, _user_add(), native_dir,
                   approval=ApprovalChannel(prompt=_answer(True), session_id="sess-1", clock=store.clock.now))
    assert outcome.status is MutationStatus.COMMITTED and not approvals_dir(native_dir.parent).exists()
