"""R38: the curated mutation flow over MemoryService (§9.4, §9.5, §6.3)."""

import dataclasses
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from agent.memory_service import wire as w
from agent.memory_service.config import resolve_memory_service_config
from agent.memory_service.errors import ProviderError
from agent.memory_service.mutation import (
    MAX_ATTEMPTS,
    MutationStatus,
    PlannedMutation,
    PlanShortCircuit,
    predict_approval_requirements,
    run_curated_mutation,
)
from agent.memory_service.service import (
    MemoryDisposition,
    MutationRequest,
    StatelessMemoryService,
    is_provider_managed,
    select_memory_service,
)
from tests.agent.memory_service.fake_backend import FakeClock, FakeProviderStore, fake_backend_factory, set_key
from tests.agent.memory_service.native_sentinel import native_memory_sentinel


def test_provider_managed_means_authoritative_or_stateless():
    assert is_provider_managed(SimpleNamespace(disposition=MemoryDisposition.AUTHORITATIVE))
    assert is_provider_managed(StatelessMemoryService(resolve_memory_service_config({}), reason="test"))
    assert is_provider_managed(SimpleNamespace(disposition="provider_authoritative"))


def test_missing_or_builtin_disposition_is_additive():
    """R37's reader rule: a missing service or disposition is additive."""
    assert not is_provider_managed(SimpleNamespace(disposition=MemoryDisposition.BUILTIN))
    assert not is_provider_managed(None)
    assert not is_provider_managed(SimpleNamespace())


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


def test_prediction_equals_the_providers_derivation(tmp_path):
    """§9.3 L1336 is mechanical, so Hermes can predict it; prove equality, not a table."""
    store = _store()
    store.seed_record(PROJECT, "memory", "project fact")
    service, _ = _service(tmp_path, store)
    memory, user = service.load_curated("memory"), service.load_curated("user")
    project_entry = next(e for e in memory.mutation_entries if e.origin_scope == PROJECT)
    for snapshot, plan in ((memory, _add_plan(memory)), (user, _add_plan(user)),
                           (memory, _replace_plan(memory, project_entry, "project fact v2"))):
        staged = service.stage_curated(_request(snapshot, plan, request_id=f"r-{snapshot.target}-{plan.intent.kind}"))
        assert predict_approval_requirements(snapshot, plan) == staged.approval_requirements


def test_prediction_covers_bulk_edit_and_threat(tmp_path):
    service, _ = _service(tmp_path, _store())
    memory = service.load_curated("memory")
    plan = _add_plan(memory)
    assert predict_approval_requirements(memory, plan) == ()
    assert predict_approval_requirements(memory, dataclasses.replace(plan, intent=w.MutationIntent(kind="bulk_edit"))) == ("bulk_edit",)
    assert predict_approval_requirements(memory, dataclasses.replace(plan, threat_decision_id="t-1")) == ("threat",)


# ---- the engine against the real service + fake (every run under the sentinel) -----------------

@pytest.fixture
def native_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from tools.memory_tool import get_memory_dir
    return get_memory_dir()


def _ops(backend):
    return [(op, getattr(req, "target", None)) for op, req in backend.calls]


def _stage_ids(backend):
    return [req.request_id for op, req in backend.calls if op == "stage_curated"]


def _texts(store, scope, target="memory"):
    return sorted(r.text for r in store.records_for(scope, target))


def _run(service, planner, native_dir, **kwargs):
    with native_memory_sentinel(native_dir) as sentinel:
        outcome = run_curated_mutation(service, "memory", planner, **kwargs)
    sentinel.assert_untouched()
    return outcome


def test_add_commits_once_then_reloads_both_enabled_targets(tmp_path, native_dir):
    store = _store()
    service, backend = _service(tmp_path, store)
    outcome = _run(service, _add_plan, native_dir)
    assert outcome.status is MutationStatus.COMMITTED and outcome.attempts == 1
    assert outcome.commit.outcome == "committed_audit_clean"
    assert _texts(store, REPO) == ["a fact"]
    assert _ops(backend)[-5:] == [("load_curated", "memory"), ("stage_curated", "memory"), ("commit_curated", "memory"),
                                  ("load_curated", "memory"), ("load_curated", "user")]
    assert set(outcome.reloaded) == {"memory", "user"} and not outcome.reload_failed


def test_disabled_target_is_never_reloaded(tmp_path, native_dir):
    """§9.1 L943: a disabled target MUST NOT be loaded."""
    service, backend = _service(tmp_path, _store(), user_profile_enabled=False)
    outcome = _run(service, _add_plan, native_dir)
    assert outcome.status is MutationStatus.COMMITTED and set(outcome.reloaded) == {"memory"}
    assert all(target != "user" for op, target in _ops(backend) if op == "load_curated")


def test_stage_conflict_reloads_replans_and_uses_a_new_request_id(tmp_path, native_dir):
    store = _store()
    service, backend = _service(tmp_path, store)
    seen = []

    def planner(snapshot):
        seen.append(snapshot.revision)
        if len(seen) == 1:
            store.external_write(REPO, "memory", "written elsewhere")  # stale before the stage
        return _add_plan(snapshot)

    outcome = _run(service, planner, native_dir)
    assert outcome.status is MutationStatus.COMMITTED and outcome.attempts == 2
    assert seen[0] != seen[1]
    ids = _stage_ids(backend)
    assert len(ids) == 2 and len(set(ids)) == 2
    assert _texts(store, REPO) == ["a fact", "written elsewhere"]


def test_commit_conflict_reloads_replans_and_uses_a_new_request_id(tmp_path, native_dir, monkeypatch):
    store = _store()
    service, backend = _service(tmp_path, store)
    real_commit, fired = service.commit_curated, []

    def commit_after_a_foreign_write(intent):
        if not fired:
            fired.append(True)
            store.external_write(PROJECT, "memory", "foreign")  # project is visible to memory
        return real_commit(intent)

    monkeypatch.setattr(service, "commit_curated", commit_after_a_foreign_write)
    outcome = _run(service, _add_plan, native_dir)
    assert outcome.status is MutationStatus.COMMITTED and outcome.attempts == 2
    assert len(set(_stage_ids(backend))) == 2
    assert _texts(store, REPO) == ["a fact"]


def test_conflict_replay_is_bounded_and_publishes_nothing(tmp_path, native_dir):
    store = _store()
    service, backend = _service(tmp_path, store)

    def always_stale(snapshot):
        store.external_write(PROJECT, "memory", f"foreign {len(_stage_ids(backend))}")
        return _add_plan(snapshot)

    outcome = _run(service, always_stale, native_dir)
    assert outcome.status is MutationStatus.CONFLICT_EXHAUSTED and outcome.attempts == MAX_ATTEMPTS
    assert _texts(store, REPO) == []


def test_predicted_approval_refuses_before_any_stage(tmp_path, native_dir):
    store = _store()
    store.seed_record(PROJECT, "memory", "project fact")
    service, backend = _service(tmp_path, store)
    outcome = run_curated_mutation(service, "user", _add_plan)  # outside _run, which fixes target="memory"
    assert outcome.status is MutationStatus.APPROVAL_UNAVAILABLE and outcome.approval_requirements == ("target_user",)

    def replace_project(snapshot):
        entry = next(e for e in snapshot.mutation_entries if e.origin_scope == PROJECT)
        return _replace_plan(snapshot, entry, "project fact v2")

    outcome = _run(service, replace_project, native_dir)
    assert outcome.approval_requirements == ("non_default_scope",)
    outcome = _run(service, _add_plan, native_dir, extra_approval="memory.write_approval")
    assert outcome.approval_requirements == ("memory.write_approval",)
    assert backend.count("stage_curated") == 0


def test_unexpected_stage_requirement_drops_the_stage_without_commit(tmp_path, native_dir):
    store = _store()
    service, backend = _service(tmp_path, store)
    store.corrupt_result("stage_curated", set_key("approval_requirements", ["bulk_edit"]))
    outcome = _run(service, _add_plan, native_dir)
    assert outcome.status is MutationStatus.APPROVAL_UNAVAILABLE and outcome.approval_requirements == ("bulk_edit",)
    assert backend.count("commit_curated") == 0 and _texts(store, REPO) == []


def test_unknown_commit_after_publish_is_retried_exactly_and_replays(tmp_path, native_dir):
    store = _store()
    service, backend = _service(tmp_path, store)
    store.fail_transport("commit_curated", phase="after_publish")
    outcome = _run(service, _add_plan, native_dir)
    assert outcome.status is MutationStatus.COMMITTED and outcome.commit.outcome == "idempotent_replay"
    commits = [req for op, req in backend.calls if op == "commit_curated"]
    assert len(commits) == 2 and commits[0] == commits[1]
    assert _texts(store, REPO) == ["a fact"]


def test_unknown_commit_before_publish_is_retried_and_commits(tmp_path, native_dir):
    store = _store()
    service, _ = _service(tmp_path, store)
    store.fail_transport("commit_curated")
    outcome = _run(service, _add_plan, native_dir)
    assert outcome.status is MutationStatus.COMMITTED and outcome.commit.outcome == "committed_audit_clean"


def test_unknown_stage_during_write_is_retried_with_the_same_request_id(tmp_path, native_dir):
    store = _store()
    service, backend = _service(tmp_path, store)
    store.fail_transport("stage_curated", phase="during")
    outcome = _run(service, _add_plan, native_dir)
    assert outcome.status is MutationStatus.COMMITTED
    ids = _stage_ids(backend)
    assert len(ids) == 2 and ids[0] == ids[1]


def test_second_unknown_outcome_is_reported_unknown(tmp_path, native_dir):
    store = _store()
    service, backend = _service(tmp_path, store)
    store.fail_transport("commit_curated", times=2)
    outcome = _run(service, _add_plan, native_dir)
    assert outcome.status is MutationStatus.OUTCOME_UNKNOWN
    assert store.stage_state(_stage_ids(backend)[0]) == "live" and _texts(store, REPO) == []


def _raises(exc):
    def _call(*args, **kwargs):
        raise exc
    return _call


def test_provider_error_with_unknown_outcome_is_never_rejected(tmp_path, native_dir, monkeypatch):
    """Correction R-5 / X-4: ygg itself can answer an unknown outcome (§9.5 L1560, §9.3 L1490)."""
    store = _store()
    service, _ = _service(tmp_path, store)
    monkeypatch.setattr(service, "commit_curated", _raises(ProviderError(
        code="unavailable", outcome="unknown", details=None, operation="commit_curated")))
    outcome = _run(service, _add_plan, native_dir)
    assert outcome.status is MutationStatus.OUTCOME_UNKNOWN


def test_stage_not_found_for_a_committed_request_is_never_rejected(tmp_path, native_dir, monkeypatch):
    """Correction R-5: `stage_not_found` with `state: committed` proves the opposite of "nothing published" (§9.5 L1558)."""
    service, _ = _service(tmp_path, _store())
    monkeypatch.setattr(service, "commit_curated", _raises(ProviderError(
        code="stage_not_found", outcome="not_committed", details={"state": "committed", "tx_id": "t-1"},
        operation="commit_curated")))
    outcome = _run(service, _add_plan, native_dir)
    assert outcome.status is MutationStatus.OUTCOME_UNKNOWN


def test_commit_reply_that_disagrees_with_its_stage_is_outcome_unknown(tmp_path, native_dir):
    """Carried R36-C: the tool holds the StageResult, so the tool checks the reply."""
    store = _store()
    service, _ = _service(tmp_path, store)
    store.corrupt_result("commit_curated", set_key("admissions[0].assigned_id", "r" + "0" * 24))
    outcome = _run(service, _add_plan, native_dir)
    assert outcome.status is MutationStatus.OUTCOME_UNKNOWN


def test_store_blocked_is_rejected_with_its_reason_and_reads_keep_working(tmp_path, native_dir):
    store = _store()
    service, _ = _service(tmp_path, store)
    store.block_store("git_dirty")
    outcome = _run(service, _add_plan, native_dir)
    assert (outcome.status, outcome.error_code, outcome.error_detail) == (MutationStatus.REJECTED, "store_blocked", "git_dirty")
    service.load_curated("memory")  # §9.6 L1577: reads keep working


def test_secret_rejection_carries_only_the_detector_code(tmp_path, native_dir):
    store = _store(secret_detector=lambda text: "test_marker" if "SECRET-MARKER-38" in text else None)
    service, backend = _service(tmp_path, store)
    outcome = _run(service, lambda s: _add_plan(s, "token SECRET-MARKER-38"), native_dir)
    assert (outcome.status, outcome.error_code, outcome.error_detail) == (MutationStatus.REJECTED, "secret_rejected", "test_marker")
    assert store.stage_state(_stage_ids(backend)[0]) == "absent"


def test_epoch_change_is_unavailable(tmp_path, native_dir):
    store = _store()
    service, _ = _service(tmp_path, store)
    store.set_epoch("ep-2")
    outcome = _run(service, _add_plan, native_dir)
    assert (outcome.status, outcome.error_code) == (MutationStatus.UNAVAILABLE, "provider_epoch_changed")


def test_short_circuit_makes_no_mutation_call(tmp_path, native_dir):
    service, backend = _service(tmp_path, _store())
    outcome = _run(service, lambda s: PlanShortCircuit({"success": True}), native_dir)
    assert outcome.status is MutationStatus.SHORT_CIRCUIT and backend.count("stage_curated") == 0
