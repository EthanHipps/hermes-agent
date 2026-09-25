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

import dataclasses
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

from agent.memory_service import wire as w
from agent.memory_service.errors import MemoryBlockedError, MemoryServiceError, ProviderError, ProviderTransportError
from agent.memory_service.service import CommitIntent, MemoryService, MutationRequest

logger = logging.getLogger(__name__)

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


_DETAIL_KEY = {"store_blocked": "reason", "secret_rejected": "detector_code", "limit_exceeded": "limit"}


class _OutcomeUnknown(Exception):
    """A stage/commit may or may not have executed, and the one exact retry did not settle it."""


def _new_request_id() -> str:
    return uuid.uuid4().hex


def _error_code(exc: BaseException) -> str:
    return getattr(exc, "code", None) or {"StatelessSessionError": "stateless",
                                           "TargetDisabledError": "target_disabled"}.get(type(exc).__name__, "blocked")


def _detail(exc: ProviderError) -> Optional[str]:
    key = _DETAIL_KEY.get(exc.code)
    value = exc.details.get(key) if key and isinstance(exc.details, dict) else None
    return str(value) if value else None


def _publication_possible(exc: ProviderError) -> bool:
    """Correction R-5 (reconciliation-fork §6), required by ruling X-4.

    ygg answers an unknown mutation outcome itself on several internal paths
    (`approval/mod.rs` L448, L481, L529-535, L562, L576), and a
    ``stage_not_found`` whose ``details.state`` is ``"committed"`` *proves* an
    acknowledgement exists (§9.5 L1558). Neither may be reported as a rejection,
    whose model-facing text says "Nothing was saved".
    """
    details = exc.details if isinstance(exc.details, dict) else {}
    return getattr(exc, "outcome", None) == "unknown" or (
        exc.code == "stage_not_found" and details.get("state") == "committed")


def _with_exact_retry(service: MemoryService, target: str, call: Callable[[], Any]) -> Any:
    """R38-8: one identical retry after a fresh load clears the fail-closed latch (§9.5 L1560)."""
    try:
        return call()
    except ProviderTransportError:
        logger.warning("memory %s outcome unknown; retrying the identical request once", target)
    try:
        service.load_curated(target)
        return call()
    except (ProviderTransportError, MemoryBlockedError) as exc:
        raise _OutcomeUnknown() from exc


def _mutation_request(service: MemoryService, snapshot: w.CuratedSnapshot, plan: PlannedMutation,
                      request_id: str, actor_kind: str, initiating_surface: str) -> MutationRequest:
    identity = service.identity
    provenance = w.MutationProvenance(actor_kind=actor_kind, principal_id=identity.principal_id,
                                      logical_session_id=identity.logical_session_id,
                                      initiating_surface=initiating_surface, source_entry_ids=(),
                                      source_commit=None, threat_decision_id=plan.threat_decision_id)
    return MutationRequest(target=snapshot.target, request_id=request_id, expected_revision=snapshot.revision,
                           hidden_preservation_state=snapshot.hidden_preservation_state,
                           requested_write_scopes=plan.requested_write_scopes, intent=plan.intent,
                           mutation_delta=plan.mutation_delta, candidate_entries=plan.candidate_entries,
                           provenance=provenance)


_CONFLICT = object()


def _stage_then_commit(service: MemoryService, request: MutationRequest):
    """``(stage, commit)``, ``_CONFLICT``, or a terminal :class:`MutationOutcome` (attempts filled by the caller)."""
    target = request.target
    try:
        stage = _with_exact_retry(service, target, lambda: service.stage_curated(request))
        if stage.approval_requirements:  # R38-1: the provider asked for more than Hermes predicted
            return MutationOutcome(MutationStatus.APPROVAL_UNAVAILABLE, target, 0, stage=stage,
                                   approval_requirements=tuple(stage.approval_requirements))
        intent = CommitIntent(target=target, request_id=request.request_id,
                              stage_handle_b64url=stage.stage_handle_b64url,
                              approval_binding_sha256=stage.approval_binding_sha256,
                              authorized_write_scopes=stage.requested_write_scopes,
                              authorization=w.ApprovalAuthorization(kind="not_required"))
        commit = _with_exact_retry(service, target, lambda: service.commit_curated(intent))
    except ProviderError as exc:
        if exc.code == "version_conflict":  # §9.5 L1554: publishes nothing; X-4 (a) makes a re-plan safe
            return _CONFLICT
        if _publication_possible(exc):  # correction R-5: never "Nothing was saved"
            return MutationOutcome(MutationStatus.OUTCOME_UNKNOWN, target, 0, error_code="outcome_unknown")
        return MutationOutcome(MutationStatus.REJECTED, target, 0, error_code=exc.code, error_detail=_detail(exc))
    except _OutcomeUnknown:
        return MutationOutcome(MutationStatus.OUTCOME_UNKNOWN, target, 0, error_code="outcome_unknown")
    except MemoryServiceError as exc:
        return MutationOutcome(MutationStatus.UNAVAILABLE, target, 0, error_code=_error_code(exc))
    except w.WireError:
        logger.error("curated mutation request failed wire validation before transmission", exc_info=True)
        return MutationOutcome(MutationStatus.REJECTED, target, 0, error_code="invalid_request")
    return stage, commit


def _reload_enabled_targets(service: MemoryService) -> Tuple[Dict[str, w.CuratedSnapshot], bool]:
    """§6.3 L547 / R38-9: after a commit, reload every enabled target before anything stages again."""
    reloaded: Dict[str, w.CuratedSnapshot] = {}
    for target in ("memory", "user"):
        if not service.target_enabled(target):
            continue
        try:
            reloaded[target] = service.load_curated(target)
        except MemoryServiceError:
            logger.warning("post-commit reload failed (target=%s)", target)
            return reloaded, True
    return reloaded, False


def run_curated_mutation(
    service: MemoryService,
    target: str,
    planner: Planner,
    *,
    actor_kind: str = "hermes",
    initiating_surface: str = "memory_tool",
    extra_approval: Optional[str] = None,
    max_attempts: int = MAX_ATTEMPTS,
    new_request_id: Callable[[], str] = _new_request_id,
) -> MutationOutcome:
    """Load, plan, stage and commit one curated mutation (§9.4 steps 3–6 and 9–10)."""
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        try:
            snapshot = service.load_curated(target)
        except MemoryServiceError as exc:
            return MutationOutcome(MutationStatus.UNAVAILABLE, target, attempts, error_code=_error_code(exc))
        plan = planner(snapshot)
        if isinstance(plan, PlanShortCircuit):
            return MutationOutcome(MutationStatus.SHORT_CIRCUIT, target, attempts, short_circuit=plan)
        required = predict_approval_requirements(snapshot, plan) + ((extra_approval,) if extra_approval else ())
        if required:
            return MutationOutcome(MutationStatus.APPROVAL_UNAVAILABLE, target, attempts, plan=plan,
                                   approval_requirements=required)
        request = _mutation_request(service, snapshot, plan, new_request_id(), actor_kind, initiating_surface)
        step = _stage_then_commit(service, request)
        if step is _CONFLICT:
            continue
        if isinstance(step, MutationOutcome):
            return dataclasses.replace(step, attempts=attempts, plan=plan)
        stage, commit = step
        if commit.admissions != stage.admissions:  # R38-10 (carried R36-C)
            logger.warning("memory commit reply does not match its stage; outcome unknown")
            return MutationOutcome(MutationStatus.OUTCOME_UNKNOWN, target, attempts, plan=plan, stage=stage,
                                   error_code="commit_mismatch")
        reloaded, reload_failed = _reload_enabled_targets(service)
        return MutationOutcome(MutationStatus.COMMITTED, target, attempts, plan=plan, stage=stage, commit=commit,
                               reloaded=reloaded, reload_failed=reload_failed)
    return MutationOutcome(MutationStatus.CONFLICT_EXHAUSTED, target, attempts)
