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
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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


@dataclass(frozen=True)
class _Item:
    text: str
    entry: Optional[w.StoredEntry]  # the snapshot entry this text stands for; None for a new add


def _native_ops(call: NativeCall) -> List[Tuple[Any, str, str]]:
    """``(action, content, old_text)`` exactly as MemoryStore normalizes them."""
    if call.operations:
        return [((op or {}).get("action"), ((op or {}).get("content") or (op or {}).get("new_text") or "").strip(),
                 ((op or {}).get("old_text") or "").strip()) for op in call.operations]
    return [(call.action, (call.content or "").strip(), (call.old_text or "").strip())]


def _replay_identity(snapshot: w.CuratedSnapshot, call: NativeCall) -> List[_Item]:
    """The native index logic once more, on items that remember their snapshot entry."""
    items = [_Item(entry.text, entry) for entry in snapshot.mutation_entries]
    for action, content, old_text in _native_ops(call):
        texts = [item.text for item in items]
        if action == "add":
            if content not in texts:
                items.append(_Item(content, None))
            continue
        index, _ = _find_unique_match(texts, old_text)
        items[index:index + 1] = [_Item(content, items[index].entry)] if action == "replace" else []
    return items


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


def _candidate(client_ref: str, text: str, scope: w.ScopeRef, target: str) -> w.CandidateEntry:
    return w.CandidateEntry(client_ref=client_ref, text=MemoryStore.normalize_entry(text), destination_scope=scope,
                            target=target, proposed_policy_key=None, import_source_identity=None)


def _to_plan(snapshot: w.CuratedSnapshot, items: List[_Item], message: str, response: Dict[str, Any],
             budget: ConsolidationBudget) -> Union[PlannedMutation, PlanShortCircuit]:
    unchanged = {item.text for item in items if item.entry is not None and item.text == item.entry.text}
    kept_ids, changed, seen = set(), [], set()
    for item in items:
        if item.entry is not None and item.text == item.entry.text:
            kept_ids.add(item.entry.id)
        elif item.text not in unchanged and item.text not in seen:  # native reload would de-duplicate the rest
            seen.add(item.text)
            changed.append(item)
    superseding = {item.entry.id: item for item in changed if item.entry is not None}
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
            candidates.append(_candidate(ref, replacement.text, entry.origin_scope, snapshot.target))
            delta.append(w.MutationDeltaItem(action="supersede", old_record_id=entry.id, replacement_client_ref=ref))
        scopes.append(entry.origin_scope)
    for item in changed:
        if item.entry is not None:
            continue
        if snapshot.default_write_scope is None:  # §9.2 L971
            return PlanShortCircuit(dict(_SCOPE_UNRESOLVED))
        ref = f"c{len(candidates) + 1}"
        candidates.append(_candidate(ref, item.text, snapshot.default_write_scope, snapshot.target))
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
    view = SnapshotMemoryStore(snapshot)
    view._consolidation_failures = budget.failures
    if call.operations:
        response = view.apply_batch(target, call.operations)
    else:
        invalid = facade._validate_single_op(view, call.action, target, call.content, call.old_text)
        response = (json.loads(invalid) if invalid is not None
                    else facade._STORE_ACTIONS[call.action][0](view, target, call.content, call.old_text))
    if view.captured is None:  # refusal, failure or native no-op: no provider mutation
        budget.failures = view._consolidation_failures
        return PlanShortCircuit(response)
    items = _replay_identity(snapshot, call)
    if tuple(item.text for item in items) != view.captured:
        logger.error("memory tool identity replay disagrees with the native result; refusing")
        return PlanShortCircuit({"success": False, "error": "Memory could not map this change to stored entries. Nothing was saved."})
    candidates = [item for item in items if item.entry is None or item.text != item.entry.text]
    hit = _candidate_scan(call, candidates)
    if hit:
        return PlanShortCircuit({"success": False, "error": hit})
    return _to_plan(snapshot, items, view.captured_message, response, budget)


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
    return {"success": False, "done": True, "approval_required": list(outcome.approval_requirements), "error": (
        f"This change needs explicit approval ({', '.join(outcome.approval_requirements)}) and "
        "approval is not available for provider-managed memory in this session, so nothing was "
        "saved. Do not retry it this turn.")}


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
    MutationStatus.OUTCOME_UNKNOWN: lambda outcome, budget: {"success": False, "done": True, "code": "outcome_unknown", "error": (
        "The memory provider did not confirm whether this change was saved. Do not repeat it this turn; "
        "if it was saved it will be visible on the next load.")},
}


def _host_write_approval() -> Optional[str]:
    from tools import write_approval as wa
    return "memory.write_approval" if wa.write_approval_enabled(wa.MEMORY) else None


def curated_memory_tool(service: MemoryService, *, action: Optional[str] = None, target: Optional[str] = "memory",
                        content: Optional[str] = None, old_text: Optional[str] = None,
                        new_text: Optional[str] = None, operations: Any = None,
                        budget: Optional[ConsolidationBudget] = None) -> str:
    """The memory tool in a provider-managed session; returns the tool's JSON string."""
    from tools import memory_tool as facade

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
    call = NativeCall(action=action, content=content, old_text=old_text, operations=operations or None)
    outcome = run_curated_mutation(service, target, lambda snapshot: plan_native_call(snapshot, call, budget),
                                   extra_approval=_host_write_approval())
    return json.dumps(_RESPONSES[outcome.status](outcome, budget), ensure_ascii=False)
