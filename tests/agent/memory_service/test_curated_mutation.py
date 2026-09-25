"""R38: the curated mutation flow over MemoryService (§9.4, §9.5, §6.3)."""

import dataclasses
from datetime import datetime, timezone
from types import SimpleNamespace

from agent.memory_service import wire as w
from agent.memory_service.config import resolve_memory_service_config
from agent.memory_service.mutation import PlannedMutation, predict_approval_requirements
from agent.memory_service.service import (
    MemoryDisposition,
    MutationRequest,
    StatelessMemoryService,
    is_provider_managed,
    select_memory_service,
)
from tests.agent.memory_service.fake_backend import FakeClock, FakeProviderStore, fake_backend_factory


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
