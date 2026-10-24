"""The memory tool over a provider-managed MemoryService (§9.7 L1598–1599; R38).

Native semantics — exact-match and ambiguity checks, normalization, no-op
detection, the per-target char quota, the per-turn consolidation cap (#42405),
the empty-batch refusal (#103419) and the threat scan — are the fork's own
MemoryStore code, run over one complete snapshot in memory by
:class:`SnapshotMemoryStore` (ruling R38-4). A replay of the same index logic
records which snapshot entry each surviving text came from, which turns the
native result into an explicit intent and an ID-based delta that
:func:`agent.memory_service.mutation.run_curated_mutation` stages and commits.
Nothing here opens, stats, locks or writes MEMORY.md/USER.md, and nothing
stages a native-shaped payload in the pending store.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from agent.memory_service import wire as w
from agent.memory_service.mutation import (
    MutationOutcome,
    MutationStatus,
    PlannedMutation,
    PlanShortCircuit,
    run_curated_mutation,
)
from agent.memory_service.service import MemoryDisposition, MemoryService
from tools.memory_tool_store import MemoryStore, _find_unique_match, _scan_memory_content
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_LABEL = {"memory": "memory", "user": "user profile"}
_STATELESS = {"success": False, "done": True, "error": (
    "Curated memory is unavailable in this session (stateless): the memory provider was not "
    "available at startup, so nothing can be saved until a new session starts.")}
_SCOPE_UNRESOLVED = {"success": False, "done": True, "code": "scope_unresolved", "error": (
    "No repository or project is resolved for this session, so memory writes are unavailable "
    "(scope_unresolved). Nothing was saved.")}


@dataclass
class ConsolidationBudget:
    """The per-turn failed-consolidation counter a native MemoryStore keeps on itself (#42405)."""

    failures: int = 0

    def reset_consolidation_failures(self) -> None:
        self.failures = 0


class SnapshotMemoryStore(MemoryStore):
    """A MemoryStore whose persistence is one snapshot's texts, in memory.

    ``_mutate`` is the only persistence path the native add/replace/remove/
    apply_batch methods use; here it applies the native closure to the
    snapshot's entry texts and captures the result instead of locking, reading
    and writing a file. ``_path_for`` names the target for the empty-batch
    refusal and never builds a native path.
    """

    def __init__(self, snapshot: w.CuratedSnapshot) -> None:
        super().__init__(memory_char_limit=snapshot.limits.memory_chars, user_char_limit=snapshot.limits.user_chars)
        self._set_entries(snapshot.target, [entry.text for entry in snapshot.mutation_entries])  # ruling R38-5 (a)
        self.captured: Optional[Tuple[str, ...]] = None
        self.captured_message = ""

    @staticmethod
    def _path_for(target: str) -> PurePosixPath:
        return PurePosixPath(_LABEL["user" if target == "user" else "memory"])

    def _mutate(self, target: str, mutate, *, skip_drift: bool = False) -> Dict[str, Any]:
        result = mutate(self._entries_for(target), self._char_limit(target))
        if isinstance(result, dict):
            return result
        self.captured, self.captured_message = tuple(result[0]), result[1]
        self._set_entries(target, list(result[0]))
        return self._success_response(target, result[1])


@dataclass(frozen=True)
class NativeCall:
    action: Optional[str]
    content: Optional[str]
    old_text: Optional[str]
    operations: Optional[List[Dict[str, Any]]]
    memory_admission: Optional[Literal["declarative", "prompt_like"]] = None


@dataclass(frozen=True)
class _NativeOp:
    action: Optional[str]
    content: str
    old_text: str
    memory_admission: Optional[Literal["declarative", "prompt_like"]]


@dataclass(frozen=True)
class _Item:
    text: str
    entry: Optional[w.StoredEntry]
    memory_admission: Optional[Literal["declarative", "prompt_like"]] = None
    raw_add: bool = False  # duplicate authored add preserves the visible snapshot entry
    conflicting_admission: bool = False


def _native_ops(call: NativeCall) -> List[_NativeOp]:
    """Selected authored new text, with native single/batch alias precedence."""
    if call.operations:
        return [_NativeOp((op or {}).get("action"),
                          ((op or {}).get("content") or (op or {}).get("new_text") or "").strip(),
                          ((op or {}).get("old_text") or "").strip(),
                          (op or {}).get("memory_admission")) for op in call.operations]
    return [_NativeOp(call.action, (call.content or "").strip(), (call.old_text or "").strip(),
                      call.memory_admission)]


def _replay_identity(snapshot: w.CuratedSnapshot, call: NativeCall) -> List[_Item]:
    """Replay native positions while retaining the final author's declaration."""
    items = [_Item(entry.text, entry) for entry in snapshot.mutation_entries]
    for op in _native_ops(call):
        texts = [item.text for item in items]
        if op.action == "add":
            if op.content not in texts:
                items.append(_Item(op.content, None, op.memory_admission))
            else:
                index = texts.index(op.content)
                item = items[index]
                conflict = (item.memory_admission is not None and item.memory_admission != op.memory_admission)
                items[index] = replace(item, memory_admission=op.memory_admission,
                                       raw_add=item.raw_add or (item.memory_admission is None and item.entry is not None
                                                                    and op.memory_admission == "prompt_like"),
                                       conflicting_admission=item.conflicting_admission or conflict)
            continue
        index, _ = _find_unique_match(texts, op.old_text)
        items[index:index + 1] = ([_Item(op.content, items[index].entry, op.memory_admission)]
                                  if op.action == "replace" else [])
    return items


def _admission_refusal() -> PlanShortCircuit:
    return PlanShortCircuit({"success": False, "code": "memory_admission_required", "error":
                             "An explicit memory admission declaration is required. Nothing was saved."})


def _valid_admission(target: str, op: _NativeOp) -> bool:
    if target == "memory" and op.action in ("add", "replace"):
        return type(op.memory_admission) is str and op.memory_admission in ("declarative", "prompt_like")
    return op.memory_admission is None


def _raw_texts(call: NativeCall) -> List[str]:
    if call.operations:
        return [str(v) for op in call.operations for v in ((op or {}).get("content"), (op or {}).get("new_text")) if v]
    return [call.content] if call.content else []


def _candidate_scan(call: NativeCall, candidates: List[_Item]) -> Optional[str]:
    """Ruling R38-12 (a): the strict scan on every candidate's raw and canonical text (§9.4 L1517)."""
    for text in _raw_texts(call) + [item.text for item in candidates] + [MemoryStore.normalize_entry(i.text) for i in candidates]:
        hit = _scan_memory_content(text)
        if hit:
            return hit
    return None


def _intent(delta: List[w.MutationDeltaItem]) -> w.MutationIntent:
    if len(delta) > 1:
        return w.MutationIntent(kind="bulk_edit")
    item = delta[0]
    single: Dict[str, Callable[[], w.MutationIntent]] = {
        "add": lambda: w.MutationIntent(kind="add"),
        "supersede": lambda: w.MutationIntent(kind="replace", matched_entry_id=item.old_record_id),
        "retire": lambda: w.MutationIntent(kind="remove", matched_entry_id=item.record_id),
    }
    return single[item.action]()


def _candidate(client_ref: str, item: _Item, scope: w.ScopeRef, target: str) -> w.CandidateEntry:
    return w.CandidateEntry(client_ref=client_ref, text=MemoryStore.normalize_entry(item.text), destination_scope=scope,
                            target=target, proposed_policy_key=None, import_source_identity=None, memory_admission=item.memory_admission)


def _to_plan(snapshot: w.CuratedSnapshot, items: List[_Item], message: str, response: Dict[str, Any],
             budget: ConsolidationBudget) -> Union[PlannedMutation, PlanShortCircuit]:
    key = MemoryStore.normalize_entry  # duplicates are judged on the candidate (stored) form
    choices: Dict[str, str] = {}
    for item in items:
        text = key(item.text)
        if item.conflicting_admission or (item.memory_admission is not None and text in choices
                                          and choices[text] != item.memory_admission):
            return PlanShortCircuit({"success": False, "code": "memory_admission_conflict", "error":
                                     "Conflicting memory admission declarations. Nothing was saved."})
        if item.memory_admission is not None:
            choices[text] = item.memory_admission
    unchanged = {key(i.text) for i in items if i.entry is not None and key(i.text) == key(i.entry.text)
                 and (i.memory_admission != "prompt_like" or i.raw_add)}
    kept_ids, changed, seen = set(), [], set()
    for item in items:
        text = key(item.text)
        if item.raw_add:
            kept_ids.add(item.entry.id)
            changed.append(item)
        elif item.entry is not None and text == key(item.entry.text) and item.memory_admission != "prompt_like":
            kept_ids.add(item.entry.id)
        elif (item.memory_admission == "prompt_like" or text not in unchanged) and text not in seen:  # native reload would de-duplicate the rest
            seen.add(text)
            changed.append(item)
    superseding = {item.entry.id: item for item in changed if item.entry is not None and not item.raw_add}
    delta: List[w.MutationDeltaItem] = []
    candidates: List[w.CandidateEntry] = []
    scopes: List[w.ScopeRef] = []
    for entry in snapshot.mutation_entries:
        if entry.id in kept_ids:
            continue
        replacement = superseding.get(entry.id)
        if replacement is None:
            delta.append(w.MutationDeltaItem(action="retire", record_id=entry.id))
        else:
            ref = f"c{len(candidates) + 1}"
            candidates.append(_candidate(ref, replacement, entry.origin_scope, snapshot.target))
            delta.append(w.MutationDeltaItem(action="supersede", old_record_id=entry.id, replacement_client_ref=ref))
        scopes.append(entry.origin_scope)
    for item in changed:
        if item.entry is not None and not item.raw_add:
            continue
        if snapshot.default_write_scope is None:  # §9.2 L971
            return PlanShortCircuit(dict(_SCOPE_UNRESOLVED))
        ref = f"c{len(candidates) + 1}"
        candidates.append(_candidate(ref, item, snapshot.default_write_scope, snapshot.target))
        delta.append(w.MutationDeltaItem(action="add", client_ref=ref))
        scopes.append(snapshot.default_write_scope)
    if not delta:  # a no-op: native would write the same entries and report success
        budget.failures = 0
        return PlanShortCircuit(response)
    return PlannedMutation(intent=_intent(delta), mutation_delta=tuple(delta), candidate_entries=tuple(candidates),
                           requested_write_scopes=tuple(dict.fromkeys(scopes)),
                           projected_texts=tuple(dict.fromkeys(item.text for item in items)), message=message)


def plan_native_call(snapshot: w.CuratedSnapshot, call: NativeCall,
                     budget: ConsolidationBudget) -> Union[PlannedMutation, PlanShortCircuit]:
    """Apply one native memory-tool call to ``snapshot`` in memory (§9.4 step 4)."""
    from tools import memory_tool as facade  # late import: facade + sibling rule (root AGENTS.md)

    target = snapshot.target
    if (call.operations and call.memory_admission is not None) or any(
            not _valid_admission(target, op) for op in _native_ops(call)):
        return _admission_refusal()
    view = SnapshotMemoryStore(snapshot)
    view._consolidation_failures = budget.failures
    if call.operations:
        response = view.apply_batch(target, call.operations)
    else:
        invalid = facade._validate_single_op(view, call.action, target, call.content, call.old_text)
        response = (json.loads(invalid) if invalid is not None
                    else facade._STORE_ACTIONS[call.action][0](view, target, call.content, call.old_text))
    raw_noop = (view.captured is None and response.get("success") is True
                and any(op.memory_admission == "prompt_like" for op in _native_ops(call)))
    if view.captured is None and not raw_noop:  # refusal, failure or declarative native no-op
        budget.failures = view._consolidation_failures
        return PlanShortCircuit(response)
    items = _replay_identity(snapshot, call)
    expected = view.captured if view.captured is not None else tuple(e.text for e in snapshot.mutation_entries)
    if tuple(item.text for item in items) != expected:
        logger.error("memory tool identity replay disagrees with the native result; refusing")
        return PlanShortCircuit({"success": False, "error": "Memory could not map this change to stored entries. Nothing was saved."})
    candidates = [item for item in items if item.entry is None or item.text != item.entry.text
                  or item.memory_admission == "prompt_like"]
    hit = _candidate_scan(call, candidates)
    if hit:
        return PlanShortCircuit({"success": False, "error": hit})
    return _to_plan(snapshot, items, view.captured_message or response.get("message", "Entry added."), response, budget)


# ---- model-facing responses -------------------------------------------------------------------

def _committed(outcome: MutationOutcome, budget: ConsolidationBudget) -> Dict[str, Any]:
    response = SnapshotMemoryStore(outcome.commit.snapshot)._success_response(outcome.target, outcome.plan.message)
    budget.failures = 0  # native: a successful write resets the per-turn cap
    if any(a.disposition == "withheld_raw" for a in outcome.commit.admissions):  # ruling R38-11 (a)
        response["withheld"] = True
        response["message"] += " The provider stored it as raw evidence, so it will not appear in memory delivery."
    if outcome.commit.outcome == "committed_audit_pending":
        response["warning"] = "Saved and published; the memory provider's audit trail is pending."
    return response


def _approval_unavailable(outcome: MutationOutcome, budget: ConsolidationBudget) -> Dict[str, Any]:
    reason = ("the approval could not be recorded" if outcome.error_code == "approval_store_unavailable"
              else "approval is not available for provider-managed memory in this session")
    return {"success": False, "done": True, "approval_required": list(outcome.approval_requirements), "error": (
        f"This change needs explicit approval ({', '.join(outcome.approval_requirements)}) and {reason}, "
        "so nothing was saved. Do not retry it this turn.")}


def _pending_approval(outcome: MutationOutcome, budget: ConsolidationBudget) -> Dict[str, Any]:
    return {"success": True, "done": True, "staged": True, "pending_id": outcome.pending_id,
            "approval_required": list(outcome.approval_requirements),
            "message": (f"Staged for approval ({', '.join(outcome.approval_requirements)}). Not yet saved — "
                        "the user reviews it with /memory pending.")}


def _denied(outcome: MutationOutcome, budget: ConsolidationBudget) -> Dict[str, Any]:
    return {"success": False, "done": True, "error": "Memory write denied by user. The change was not saved."}


def _outcome_unknown(outcome: MutationOutcome, budget: ConsolidationBudget) -> Dict[str, Any]:
    response = {"success": False, "done": True, "code": "outcome_unknown", "error": (
        "The memory provider did not confirm whether this change was saved. Do not repeat it this turn; "
        "if it was saved it will be visible on the next load.")}
    if outcome.pending_id:
        response["pending_id"] = outcome.pending_id
        response["error"] += " The user can settle it with /memory pending."
    return response


_REJECTIONS: Dict[str, Callable[[Optional[str]], str]] = {
    "store_blocked": lambda d: f"Memory is temporarily read-only ({d}). Nothing was saved.",
    "secret_rejected": lambda d: f"The memory provider rejected this content as a possible secret ({d}). Nothing was saved.",
    "limit_exceeded": lambda d: f"The memory provider refused this change: it would exceed the limit ({d}). Nothing was saved.",
    "scope_unresolved": lambda d: _SCOPE_UNRESOLVED["error"],
}


def _rejected(outcome: MutationOutcome, budget: ConsolidationBudget) -> Dict[str, Any]:
    text = _REJECTIONS.get(outcome.error_code, lambda d: f"The memory provider rejected this change ({outcome.error_code}). Nothing was saved.")
    return {"success": False, "done": True, "code": outcome.error_code, "error": text(outcome.error_detail)}


def _conflict_exhausted(outcome: MutationOutcome, budget: ConsolidationBudget) -> Dict[str, Any]:
    response = {"success": False, "error": ("Memory changed repeatedly while this update was being saved, so nothing "
                                            "was saved. Retry once against the current entries.")}
    budget.failures += 1
    return response


_RESPONSES: Dict[MutationStatus, Callable[[MutationOutcome, ConsolidationBudget], Dict[str, Any]]] = {
    MutationStatus.COMMITTED: _committed,
    MutationStatus.SHORT_CIRCUIT: lambda outcome, budget: dict(outcome.short_circuit.response),
    MutationStatus.APPROVAL_UNAVAILABLE: _approval_unavailable,
    MutationStatus.REJECTED: _rejected,
    MutationStatus.CONFLICT_EXHAUSTED: _conflict_exhausted,
    MutationStatus.UNAVAILABLE: lambda outcome, budget: {"success": False, "done": True, "code": outcome.error_code, "error": (
        f"Curated memory is unavailable for this session ({outcome.error_code}). Nothing was saved.")},
    MutationStatus.OUTCOME_UNKNOWN: _outcome_unknown,
    MutationStatus.DENIED: _denied,
    MutationStatus.PENDING_APPROVAL: _pending_approval,
}


def _host_write_approval() -> Optional[str]:
    from tools import write_approval as wa
    # Checkpoint B carry-forward; ruling R39-12: an unreadable or unrecognized value counts as on.
    return "memory.write_approval" if wa.write_approval_enabled(wa.MEMORY, fail_closed=True) else None


def curated_memory_tool(service: MemoryService, *, action: Optional[str] = None, target: Optional[str] = "memory",
                        content: Optional[str] = None, old_text: Optional[str] = None,
                        new_text: Optional[str] = None, operations: Any = None,
                        budget: Optional[ConsolidationBudget] = None,
                        approval_session_id: Optional[str] = None,
                        memory_admission: Optional[Literal["declarative", "prompt_like"]] = None) -> str:
    """The memory tool in a provider-managed session; returns the tool's JSON string.

    ``approval_session_id`` names the Hermes session whose persisted identity an approval may
    wait under (ruling R39-4; contract C6b-1).
    """
    from tools import memory_tool as facade
    from tools.write_approval_curated import memory_tool_approval_channel

    budget = budget if budget is not None else ConsolidationBudget()
    if content is None and new_text is not None:
        content = new_text
    target = "memory" if target is None else target
    if target not in _LABEL:
        return json.dumps(facade._memory_target_error(service, target))
    if service.disposition is MemoryDisposition.STATELESS:  # §9.1 L946; ruling R38-18 (a)
        return json.dumps(_STATELESS)
    if not service.target_enabled(target):  # §9.1 L943: never loaded
        return json.dumps({"success": False, "error": f"Writes to the {_LABEL[target]} target are disabled in memory config.",
                           "target": target})
    if operations:
        if not isinstance(operations, list):
            return tool_error("operations must be a list of {action, content?, old_text?} objects.", success=False)
    elif action not in facade._STORE_ACTIONS:
        return tool_error(f"Unknown action '{action}'. Use: add, replace, remove", success=False)
    call = NativeCall(action=action, content=content, old_text=old_text, operations=operations or None,
                      memory_admission=memory_admission)
    outcome = run_curated_mutation(service, target, lambda snapshot: plan_native_call(snapshot, call, budget),
                                   extra_approval=_host_write_approval(),
                                   approval=memory_tool_approval_channel(approval_session_id))
    return json.dumps(_RESPONSES[outcome.status](outcome, budget), ensure_ascii=False)
