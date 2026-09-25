"""Host-side curated mutation flow over :class:`MemoryService` (§9.4, §9.5).

A caller supplies a *planner*: a pure function from one complete
:class:`~agent.memory_service.wire.CuratedSnapshot` to an explicit intent and
ID-based delta (:class:`PlannedMutation`), or to a host response that needs no
provider mutation (:class:`PlanShortCircuit`: a no-op or a semantic refusal).
This module owns everything that is not semantics and knows no provider and
no tool: the fresh load before every plan (§9.4 step 3, §9.6 L1564); approval
prediction, refusing before any stage while no approval channel exists (R38-1);
stage then commit; ``version_conflict`` replay with a fresh load and a NEW
request ID (§9.5 L1554, R38-7); one exact retry of an unknown stage/commit
outcome (§9.5 L1560, R38-8); the commit/stage admissions check carried from
R36-C (R38-10); and the reload of every enabled target after a commit
(§6.3 L547, R38-9).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Tuple, Union

from agent.memory_service import wire as w

#: Ruling R38-7: planning attempts per call (the first plus two conflict replays).
MAX_ATTEMPTS = 3
_INTENT_REQUIREMENTS = frozenset({"bulk_edit", "reset", "import"})


@dataclass(frozen=True)
class PlannedMutation:
    intent: w.MutationIntent
    mutation_delta: Tuple[w.MutationDeltaItem, ...]
    candidate_entries: Tuple[w.CandidateEntry, ...]
    requested_write_scopes: Tuple[w.ScopeRef, ...]
    projected_texts: Tuple[str, ...]
    message: str
    threat_decision_id: Optional[str] = None


@dataclass(frozen=True)
class PlanShortCircuit:
    """The planner answered without a provider mutation (a no-op or a semantic refusal)."""

    response: Mapping[str, Any]


Planner = Callable[[w.CuratedSnapshot], Union[PlannedMutation, PlanShortCircuit]]


class MutationStatus(str, Enum):
    COMMITTED = "committed"
    SHORT_CIRCUIT = "short_circuit"
    APPROVAL_UNAVAILABLE = "approval_unavailable"
    CONFLICT_EXHAUSTED = "conflict_exhausted"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"
    OUTCOME_UNKNOWN = "outcome_unknown"


@dataclass(frozen=True)
class MutationOutcome:
    status: MutationStatus
    target: str
    attempts: int
    plan: Optional[PlannedMutation] = None
    short_circuit: Optional[PlanShortCircuit] = None
    approval_requirements: Tuple[str, ...] = ()
    stage: Optional[w.StageResult] = None
    commit: Optional[w.CommitResult] = None
    error_code: Optional[str] = None
    error_detail: Optional[str] = None  # content-free only: a store_blocked reason, detector code or limit name
    reloaded: Mapping[str, w.CuratedSnapshot] = field(default_factory=dict)
    reload_failed: bool = False


def predict_approval_requirements(snapshot: w.CuratedSnapshot, plan: PlannedMutation) -> Tuple[str, ...]:
    """§9.3 L1336, in its order: the requirements ``stage_curated`` will derive for ``plan``."""
    requirements = []
    if snapshot.target == "user":
        requirements.append("target_user")
    if any(scope != snapshot.default_write_scope for scope in plan.requested_write_scopes):
        requirements.append("non_default_scope")
    if plan.intent.kind in _INTENT_REQUIREMENTS:
        requirements.append(plan.intent.kind)
    if plan.threat_decision_id is not None:
        requirements.append("threat")
    return tuple(requirements)
