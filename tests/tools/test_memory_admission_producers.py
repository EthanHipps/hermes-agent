"""H2 authored intent through real dispatch and human request boundaries; fake host proof only."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
from agent.memory_service import wire as w
from tests.tools.test_memory_tool_curated import _service, _provider, REPO, native_dir
from tests.agent.test_learning_journey_s6 import env, REPO_CTX
from tests.agent.memory_service.native_sentinel import native_memory_sentinel


@pytest.mark.parametrize("choice", ["declarative", "prompt_like"])
def test_authored_dispatch_reaches_serialized_candidate(choice, tmp_path, native_dir):
    provider = _provider()
    service, backend = _service(tmp_path, provider)
    agent = SimpleNamespace(_memory_service=service, _memory_store=None, _memory_manager=None)
    with native_memory_sentinel(native_dir) as sentinel:
        out = json.loads(INLINE_TOOL_EXECUTORS["memory"](agent, {
            "action": "add", "target": "memory", "new_text": "authored fact",
            "memory_admission": choice}, InlineToolContext("task")))
    sentinel.assert_untouched()
    assert out["success"] is True, out
    request = next(req for op, req in backend.calls if op == "stage_curated")
    decoded = w.StageRequest.from_json(w.canonical_json(request.to_wire()))
    assert decoded.candidate_entries[0].memory_admission == choice
    assert decoded.candidate_entries[0].text == "authored fact"
    snapshot = service.load_curated("memory")
    assert [e.text for e in snapshot.mutation_entries] == ([] if choice == "prompt_like" else ["authored fact"])
    assert backend.count("commit_curated") == 1


@pytest.mark.parametrize("choice", ["declarative", "prompt_like"])
def test_human_request_reaches_exact_id_edit(choice, env):
    from hermes_cli.web_models import LearningNodeEdit
    from hermes_cli.web_routers.status import update_learning_node
    record = env.store.seed_record(REPO, "memory", "before edit")
    payload = {"id": f"memory:memory:{record.id}", "content": "human authored",
               "project_id": "proj-1", "repo_id": "repo-1", "memory_admission": choice}
    body = LearningNodeEdit.model_validate_json(json.dumps(payload))
    with native_memory_sentinel(env.native) as sentinel:
        result = asyncio.run(update_learning_node(body))
    sentinel.assert_untouched()
    assert result["ok"] is True
    requests = [req for backend in env.backends for op, req in backend.calls if op == "stage_curated"]
    request = requests[0]
    assert request.intent.matched_entry_id == record.id
    assert request.candidate_entries[0].memory_admission == choice
    assert request.provenance.actor_kind == "human"
    assert env.store.records[record.id].lifecycle == "superseded"


@pytest.mark.parametrize("case", ["mixed", "final_replacement", "same_add", "same_replace", "declarative_noop", "normalized_conflict", "exact_conflict"])
def test_surviving_authorship_and_same_text_intent(case, tmp_path, native_dir, monkeypatch):
    from agent.memory_service.approval import ApprovalChannel
    monkeypatch.setattr("tools.write_approval_curated.memory_tool_approval_channel",
                        lambda session: ApprovalChannel(prompt=lambda prompt: True))
    provider = _provider()
    old = provider.seed_record(REPO, "memory", "visible")
    service, backend = _service(tmp_path, provider)
    agent = SimpleNamespace(_memory_service=service, _memory_store=None, _memory_manager=None)
    ops = {
        "mixed": [{"action": "add", "content": "first", "memory_admission": "declarative"},
                  {"action": "add", "new_text": "second", "memory_admission": "prompt_like"}],
        "final_replacement": [{"action": "add", "content": "draft", "memory_admission": "declarative"},
                              {"action": "replace", "old_text": "draft", "content": "final", "new_text": "unused", "memory_admission": "prompt_like"}],
        "normalized_conflict": [{"action": "add", "content": "a\nb", "memory_admission": "declarative"},
                                {"action": "add", "content": "a\r\nb", "memory_admission": "prompt_like"}],
        "exact_conflict": [{"action": "add", "content": "same", "memory_admission": "declarative"},
                           {"action": "add", "content": "same", "memory_admission": "prompt_like"}],
    }
    args = ({"operations": ops[case]} if case in ops else
            {"action": "replace" if case == "same_replace" else "add", "content": "visible",
             "old_text": "visible", "memory_admission": "declarative" if case == "declarative_noop" else "prompt_like"})
    with native_memory_sentinel(native_dir) as sentinel:
        out = json.loads(INLINE_TOOL_EXECUTORS["memory"](agent, args, InlineToolContext("task")))
    sentinel.assert_untouched()
    requests = [req for op, req in backend.calls if op == "stage_curated"]
    if case.endswith("conflict"):
        assert out["success"] is False and out["code"] == "memory_admission_conflict"
        assert requests == [] and len(provider.records) == 1
        return
    assert out["success"] is True, out
    if case == "declarative_noop":
        assert requests == [] and len(provider.records) == 1
        return
    assert len(requests) == 1
    request = w.StageRequest.from_json(w.canonical_json(requests[0].to_wire()))
    expected = {"mixed": [("first", "declarative"), ("second", "prompt_like")],
                "final_replacement": [("final", "prompt_like")],
                "same_add": [("visible", "prompt_like")], "same_replace": [("visible", "prompt_like")]}
    assert [(c.text, c.memory_admission) for c in request.candidate_entries] == expected[case]
    assert request.provenance.threat_decision_id is None
    assert provider.records[old.id].lifecycle == ("superseded" if case == "same_replace" else "active")
    assert any(r.lane == "raw" and r.lifecycle == "active" for r in provider.records.values())
    snapshot = service.load_curated("memory")
    assert all(e.lane != "raw" for e in snapshot.mutation_entries)
    # An ordinary later delta must preserve the hidden record.
    raw_ids = {r.id for r in provider.records.values() if r.lane == "raw"}
    later = json.loads(INLINE_TOOL_EXECUTORS["memory"](agent, {
        "action": "add", "content": "later", "memory_admission": "declarative"}, InlineToolContext("task")))
    assert later["success"] is True
    assert all(provider.records[rid].lifecycle == "active" for rid in raw_ids)


@pytest.mark.parametrize("choice,target,action", [(None,"memory","add"), ("", "memory", "add"),
    ("DECLARATIVE", "memory", "add"), (True,"memory","add"), ({"text":"private"},"memory","add"),
    ("declarative", "user", "add"), ("prompt_like", "memory", "remove")])
def test_bad_authorship_is_fixed_pre_stage_refusal(choice, target, action, tmp_path, native_dir):
    provider = _provider()
    provider.seed_record(REPO, "memory", "old")
    service, backend = _service(tmp_path, provider)
    agent = SimpleNamespace(_memory_service=service, _memory_store=None, _memory_manager=None)
    out = json.loads(INLINE_TOOL_EXECUTORS["memory"](agent, {
        "target": target, "action": action, "content": "private-body", "old_text": "old",
        "memory_admission": choice}, InlineToolContext("task")))
    assert out["success"] is False and out["code"] == "memory_admission_required"
    assert backend.count("stage_curated") == 0
    assert "private" not in json.dumps(out) and "DECLARATIVE" not in json.dumps(out)
