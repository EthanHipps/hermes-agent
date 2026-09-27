"""Journey memory over provider-managed curated memory (§9.7 L1608-L1609; rulings R42-4, R42-5, R42-6).

In authoritative mode the journey ("learning graph") never reads, stats, rewrites or deletes
``MEMORY.md``/``USER.md``. Cards come from each enabled target's ``mutation_entries`` -- the
addressable stored entries, each with a stable entry ID and its origin scope (§9.3 L1237) -- read
through the explicit administrative identity (``agent/memory_service/admin.py``, contract C6b-5),
never the process directory. Node ids keep the vocabulary every front end parses,
``memory:<memory|profile>:<entry-id>``; the stable entry ID replaces the positional index, and
opaque provider values (handles, revisions, tokens) never reach a card (§9.3 L989).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from agent.memory_service import wire as w
from agent.memory_service.admin import (ADMIN_FAILURES, AdminContext, admin_service, format_scope, outcome_payload,
                                        unavailable_payload)
from agent.memory_service.mutation import PlannedMutation, PlanShortCircuit, Planner, run_curated_mutation
from tools.memory_tool_store import ENTRY_DELIMITER, MemoryStore, _scan_memory_content

logger = logging.getLogger(__name__)

_SOURCE = {"memory": "memory", "user": "profile"}
_TARGET = {source: target for target, source in _SOURCE.items()}
JOURNEY_SURFACE = "journey"
#: The native journey's own wording (agent/learning_mutations.py) for an id that no longer resolves.
_STALE = {"ok": False, "message": "memory node id is stale — refresh the graph"}


def _card(entry: w.StoredEntry, source: str) -> Dict[str, Any]:
    lines = entry.text.strip().splitlines()
    first = lines[0].strip().lstrip("# ").strip() if lines else ""
    return {"id": f"memory:{source}:{entry.id}", "source": source, "timestamp": None,
            "title": (first[:80] + "…") if len(first) > 80 else first, "body": entry.text[:1200],
            "entry_id": entry.id, "scope": format_scope(entry.origin_scope)}


def curated_graph_memory(raw_config: Any, context: Optional[AdminContext], *,
                         backend_factory: Any = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """``(cards, curated_memory status)`` for the journey graph; an unavailable provider shows no memory."""
    context = context or AdminContext()
    snapshots: Dict[str, str] = {}
    status: Dict[str, Any] = {"state": "ok", "context": context.as_dict(), "snapshots": snapshots}
    cards: List[Dict[str, Any]] = []
    try:
        with admin_service(raw_config, context=context, backend_factory=backend_factory) as service:
            for target in ("memory", "user"):
                if not service.target_enabled(target):
                    continue
                snapshot = service.load_curated(target)
                snapshots[target] = snapshot.status
                cards.extend(_card(entry, _SOURCE[target]) for entry in snapshot.mutation_entries)
    except ADMIN_FAILURES as exc:
        payload = unavailable_payload(exc)
        logger.warning("journey memory unavailable (%s)", payload["code"])
        state = "configuration_error" if payload["code"] == "configuration_error" else "unavailable"
        return [], {**status, "state": state, "code": payload["code"]}
    return cards, status


# -- detail, edit and delete by stable entry ID (rulings R42-1, R42-6, R42-7) ------------------------------


def _parse(node_id: str) -> Tuple[str, str]:
    """``memory:<memory|profile>:<entry-id>`` -> (target, entry id); the native id error otherwise."""
    parts = node_id.split(":", 2) if isinstance(node_id, str) else []
    if len(parts) != 3 or parts[0] != "memory" or parts[1] not in _TARGET or not parts[2]:
        raise ValueError(f"bad memory node id: {node_id!r}")
    return _TARGET[parts[1]], parts[2]


def _find(snapshot: w.CuratedSnapshot, entry_id: str) -> Optional[w.StoredEntry]:
    return next((entry for entry in snapshot.mutation_entries if entry.id == entry_id), None)


def _mutate(raw_config: Any, context: Optional[AdminContext], target: str, planner: Planner,
            backend_factory: Any) -> Dict[str, Any]:
    """C5 under the explicit administrative identity. No ``extra_approval`` and no ``approval=``: mandatory
    approvals fail closed with a typed ``approval_unavailable`` until the R42 follow-up (R42-7, X6b-4)."""
    try:
        with admin_service(raw_config, context=context or AdminContext(), backend_factory=backend_factory) as service:
            outcome = run_curated_mutation(service, target, planner, actor_kind="human",
                                           initiating_surface=JOURNEY_SURFACE)
    except ADMIN_FAILURES as exc:
        return unavailable_payload(exc)
    return outcome_payload(outcome)


def curated_detail(raw_config: Any, node_id: str, context: Optional[AdminContext], *,
                   backend_factory: Any = None) -> Dict[str, Any]:
    """The entry's current text from a fresh snapshot, for an edit prefill; a positional or gone id is stale."""
    try:
        target, entry_id = _parse(node_id)
    except ValueError as exc:
        return {"ok": False, "message": str(exc)}
    try:
        with admin_service(raw_config, context=context or AdminContext(), backend_factory=backend_factory) as service:
            entry = _find(service.load_curated(target), entry_id)
    except ADMIN_FAILURES as exc:
        return unavailable_payload(exc)
    if entry is None:
        return dict(_STALE)
    body = entry.text.strip()
    return {"ok": True, "kind": "memory", "id": node_id, "label": body.splitlines()[0][:80] if body else "",
            "content": body}


def curated_delete(raw_config: Any, node_id: str, context: Optional[AdminContext], *,
                   backend_factory: Any = None) -> Dict[str, Any]:
    """Retire exactly the addressed entry (a ``remove`` intent with one ``retire``)."""
    try:
        target, entry_id = _parse(node_id)
    except ValueError as exc:
        return {"ok": False, "message": str(exc)}

    def planner(snapshot: w.CuratedSnapshot):
        entry = _find(snapshot, entry_id)
        if entry is None:
            return PlanShortCircuit(dict(_STALE))
        return PlannedMutation(
            intent=w.MutationIntent(kind="remove", matched_entry_id=entry.id),
            mutation_delta=(w.MutationDeltaItem(action="retire", record_id=entry.id),), candidate_entries=(),
            requested_write_scopes=(entry.origin_scope,),
            projected_texts=tuple(e.text for e in snapshot.mutation_entries if e.id != entry.id),
            message=f"deleted memory ({format_scope(entry.origin_scope)})")

    return _mutate(raw_config, context, target, planner, backend_factory)


def curated_edit(raw_config: Any, node_id: str, content: str, context: Optional[AdminContext], *,
                 backend_factory: Any = None) -> Dict[str, Any]:
    """Supersede exactly the addressed entry, with the memory tool's replace validation (ruling R42-7).

    Normalization, the non-empty check and the strict threat scan run before any provider call (§9.4
    L1517); the per-target char quota runs over the snapshot's ``mutation_entries`` (X-3).
    """
    try:
        target, entry_id = _parse(node_id)
    except ValueError as exc:  # id errors win over the empty-body message, as natively
        return {"ok": False, "message": str(exc)}
    body = MemoryStore.normalize_entry(content or "")
    if not body:
        return {"ok": False, "message": "empty memory — use delete to remove it"}
    threat = _scan_memory_content(body)
    if threat:
        return {"ok": False, "message": threat}

    def planner(snapshot: w.CuratedSnapshot):
        entry = _find(snapshot, entry_id)
        if entry is None:
            return PlanShortCircuit(dict(_STALE))
        if MemoryStore.normalize_entry(entry.text) == body:
            return PlanShortCircuit({"ok": True, "message": "no changes"})
        projected = [body if e.id == entry.id else e.text for e in snapshot.mutation_entries]
        limit = snapshot.limits.user_chars if target == "user" else snapshot.limits.memory_chars
        total = len(ENTRY_DELIMITER.join(projected))
        if total > limit:
            return PlanShortCircuit({"ok": False, "message": (
                f"Replacement would put memory at {total:,}/{limit:,} chars. Shorten the new content, or "
                "delete other stale or less important entries to make room, then retry.")})
        return PlannedMutation(
            intent=w.MutationIntent(kind="replace", matched_entry_id=entry.id),
            mutation_delta=(w.MutationDeltaItem(action="supersede", old_record_id=entry.id,
                                                replacement_client_ref="c1"),),
            candidate_entries=(w.CandidateEntry(client_ref="c1", text=body, destination_scope=entry.origin_scope,
                                                target=target, proposed_policy_key=None,
                                                import_source_identity=None),),
            requested_write_scopes=(entry.origin_scope,), projected_texts=tuple(projected),
            message="updated memory")

    return _mutate(raw_config, context, target, planner, backend_factory)
