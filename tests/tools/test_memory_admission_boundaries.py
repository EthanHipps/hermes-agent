"""Final authored producer boundaries; fake host witnesses, not adapter conformance."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
from agent.memory_service import wire as w
from tests.tools.test_memory_tool_curated import _service, _provider, REPO, native_dir
from tests.agent.test_learning_journey_s6 import env


@pytest.mark.parametrize("choice", [None, "DECLARATIVE", {"private-invalid": "value"}])
def test_human_old_or_malformed_client_gets_typed_fixed_refusal(choice, env):
    from hermes_cli.web_models import LearningNodeEdit
    from hermes_cli.web_routers.status import update_learning_node
    record = env.store.seed_record(REPO, "memory", "before")
    payload = {"id": f"memory:memory:{record.id}", "content": "private-body",
               "project_id": "proj-1", "repo_id": "repo-1"}
    if choice is not None:
        payload["memory_admission"] = choice
    body = LearningNodeEdit.model_validate_json(json.dumps(payload))
    with pytest.raises(HTTPException) as caught:
        asyncio.run(update_learning_node(body))
    assert caught.value.status_code == 400
    assert caught.value.detail == {"code": "memory_admission_required", "message":
        "An explicit memory admission declaration is required. Nothing was saved."}
    assert all(backend.count("stage_curated") == 0 for backend in env.backends)
    assert env.store.records[record.id].lifecycle == "active"


def test_duplicate_add_does_not_erase_surviving_replacement_identity(tmp_path, native_dir, monkeypatch):
    from agent.memory_service.approval import ApprovalChannel
    monkeypatch.setattr("tools.write_approval_curated.memory_tool_approval_channel",
                        lambda session: ApprovalChannel(prompt=lambda prompt: True))
    provider = _provider()
    old = provider.seed_record(REPO, "memory", "before")
    service, backend = _service(tmp_path, provider)
    agent = SimpleNamespace(_memory_service=service, _memory_store=None, _memory_manager=None)
    out = json.loads(INLINE_TOOL_EXECUTORS["memory"](agent, {"operations": [
        {"action": "replace", "old_text": "before", "content": "final", "memory_admission": "prompt_like"},
        {"action": "add", "content": "final", "memory_admission": "prompt_like"}]}, InlineToolContext("task")))
    assert out["success"] is True, out
    request = next(req for op, req in backend.calls if op == "stage_curated")
    assert request.mutation_delta == (w.MutationDeltaItem(action="supersede", old_record_id=old.id, replacement_client_ref="c1"),)
    assert provider.records[old.id].lifecycle == "superseded"


@pytest.mark.parametrize("mode", ["lost_stage", "lost_commit", "conflict", "pending"])
def test_authored_retries_and_approval_metadata(mode, tmp_path, native_dir, monkeypatch):
    from agent.memory_service.approval import ApprovalChannel
    from agent.memory_service.approval_store import approvals_dir, list_pending_approvals
    from tests.agent.memory_service.test_approval_flow import _persist
    monkeypatch.setattr("tools.write_approval.write_approval_enabled", lambda subsystem, **kw: True)
    provider = _provider()
    old = provider.seed_record(REPO, "memory", "before")
    service, backend = _service(tmp_path, provider)
    _persist(service)
    seen = []
    def approve(prompt):
        seen.append(prompt)
        return True
    monkeypatch.setattr("tools.write_approval_curated.memory_tool_approval_channel",
                        lambda session: ApprovalChannel(prompt=None if mode == "pending" else approve,
                            session_id="sess-1", clock=provider.clock.now))
    if mode in ("lost_stage", "lost_commit"):
        provider.fail_transport("stage_curated" if mode == "lost_stage" else "commit_curated",
                                phase="during" if mode == "lost_stage" else "after_publish")
    if mode == "conflict":
        real_commit = service.commit_curated
        fired = []
        def commit(intent):
            if not fired:
                fired.append(True)
                provider.external_write(REPO, "memory", "concurrent")
            return real_commit(intent)
        monkeypatch.setattr(service, "commit_curated", commit)
    agent = SimpleNamespace(_memory_service=service, _memory_store=None, _memory_manager=None)
    out = json.loads(INLINE_TOOL_EXECUTORS["memory"](agent, {"action": "replace", "old_text": "before",
        "content": "private-authored-body", "memory_admission": "prompt_like"}, InlineToolContext("task")))
    assert out["success"] is True, out
    stages = [req for op, req in backend.calls if op == "stage_curated"]
    commits = [req for op, req in backend.calls if op == "commit_curated"]
    assert all(req.candidate_entries[0].memory_admission == "prompt_like" for req in stages)
    assert all(req.provenance.threat_decision_id is None for req in stages)
    if mode == "lost_stage":
        assert len(stages) == 2 and w.canonical_json(stages[0].to_wire()) == w.canonical_json(stages[1].to_wire())
    elif mode == "lost_commit":
        assert len(commits) == 2 and w.canonical_json(commits[0].to_wire()) == w.canonical_json(commits[1].to_wire())
    elif mode == "conflict":
        assert len(stages) == 2 and stages[0].request_id != stages[1].request_id
        assert len(seen) == 2 and seen[0].stage.approval_binding_sha256 != seen[1].stage.approval_binding_sha256
    else:
        assert out["staged"] is True and len(list_pending_approvals()) == 1 and commits == []
        records = list(approvals_dir(native_dir.parent).glob("*.json"))
        assert records
        for path in records:
            persisted = path.read_bytes()
            assert b"private-authored-body" not in persisted
            assert b"prompt_like" not in persisted and b"memory_admission" not in persisted
        return
    assert provider.records[old.id].lifecycle == "superseded"
