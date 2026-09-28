"""Host-side curated mutation flow over :class:`MemoryService` (§9.4, §9.5).

A caller supplies a *planner*: a pure function from one complete
:class:`~agent.memory_service.wire.CuratedSnapshot` to an explicit intent and
ID-based delta (:class:`PlannedMutation`), or to a host response that needs no
provider mutation (:class:`PlanShortCircuit`: a no-op or a semantic refusal).
This module owns everything that is not semantics and knows no provider and
no tool: the fresh load before every plan (§9.4 step 3, §9.6 L1564); approval
prediction; with no usable approval channel, refusal before any stage (R38-1);
otherwise stage, inspect, approve or defer, and a write-ahead approved record
(R39; contract C6b-2, ruling X6b-2); stage then commit; ``version_conflict`` replay with a fresh load and a NEW
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
from agent.memory_service.approval import ApprovalChannel, ApprovalPrompt, build_authorization, render_stage_inspection, rfc3339
from agent.memory_service.approval_store import (
    ApprovalStoreError, PendingApproval, drop_pending_approval, new_pending_id, save_pending_approval)
from agent.memory_service.errors import (
    BindingInvalidError, MemoryBlockedError, MemoryServiceError, ProviderError, ProviderTransportError)
from agent.memory_service.service import CommitIntent, InspectRequest, MemoryService, MutationRequest

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
    DENIED = "denied"                        # R39 (C6b-2): a host denial; nothing persisted (§9.5 L1542)
    PENDING_APPROVAL = "pending_approval"    # R39 (C6b-2): staged and waiting for /memory pending


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
    pending_id: Optional[str] = None  # a handle-only approval record: waiting, or kept for an identical replay


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


def _forget(record: Optional[PendingApproval]) -> None:
    """A typed reply settled the request (X-4: not_committed is per request ID), so its record goes."""
    if record is None:
        return
    try:
        drop_pending_approval(record.pending_id)
    except ApprovalStoreError:
        logger.warning("could not remove a settled memory approval record; /memory pending reconciles it")


def _keep_unknown(record: Optional[PendingApproval]) -> Optional[str]:
    """§9.5 L1562: an unknown outcome keeps the approved request for an identical replay."""
    if record is None:
        return None
    try:
        save_pending_approval(dataclasses.replace(record, outcome="unknown"))
    except ApprovalStoreError:
        logger.warning("could not mark a memory approval record unknown")
    return record.pending_id


def _deferrable(approval: Optional[ApprovalChannel], service: MemoryService) -> bool:
    """Ruling X6b-2 (§4.1 L250): persist an approval record only when replay can resume the staging identity."""
    if approval is None or not approval.session_id:
        return False
    from agent.memory_service.host_state import load_host_state
    try:
        record = load_host_state(approval.session_id)
    except BindingInvalidError:
        return False
    state = getattr(service, "session_state", None)
    return record is not None and record.state is not None and state is not None and record.state == state


def _approval_record(service: MemoryService, request: MutationRequest, stage: w.StageResult,
                     required: Tuple[str, ...], approval: ApprovalChannel, *, decision: str,
                     authorization: Optional[w.ApprovalAuthorization] = None) -> PendingApproval:
    return PendingApproval(
        pending_id=new_pending_id(), session_id=approval.session_id,
        logical_session_id=service.identity.logical_session_id,
        provider_epoch=stage.expected_revision.provider_epoch, target=request.target,
        intent_kind=request.intent.kind, request_id=request.request_id,
        stage_handle_b64url=stage.stage_handle_b64url, approval_binding_sha256=stage.approval_binding_sha256,
        expected_revision=stage.expected_revision, requested_write_scopes=tuple(stage.requested_write_scopes),
        approval_requirements=tuple(required), expires_at=stage.expires_at, decision=decision,
        authorization=authorization, origin=approval.origin, created_at=rfc3339(approval.clock()))


def _store_or_fail(record: PendingApproval, stage: w.StageResult, required: Tuple[str, ...]) -> Optional[MutationOutcome]:
    try:
        save_pending_approval(record)
    except ApprovalStoreError:
        logger.warning("memory approval store unavailable; failing closed")
        return MutationOutcome(MutationStatus.APPROVAL_UNAVAILABLE, record.target, 0, stage=stage,
                               approval_requirements=required, error_code="approval_store_unavailable")
    return None


def _obtain_approval(service: MemoryService, request: MutationRequest, stage: w.StageResult,
                     required: Tuple[str, ...], approval: ApprovalChannel, deferrable: bool):
    """§9.4 steps 7–8: ``(authorization, record-or-None)`` to commit, or a terminal outcome.

    ``deferrable`` (ruling X6b-2) gates both the deferral and the write-ahead record.
    """
    target = request.target
    if approval.prompt is not None:
        inspection = service.inspect_staged(InspectRequest(target=target, request_id=request.request_id,
                                                           stage_handle_b64url=stage.stage_handle_b64url))
        if inspection.summary != stage:  # the human must see exactly what the binding covers
            logger.warning("memory stage inspection does not match its stage; approval not requested")
            return MutationOutcome(MutationStatus.UNAVAILABLE, target, 0, stage=stage, error_code="inspection_mismatch")
        text = render_stage_inspection(inspection, intent_kind=request.intent.kind, requirements=required)
        answer = approval.prompt(ApprovalPrompt(target=target, intent_kind=request.intent.kind,
                                                requirements=required, stage=stage, inspection=inspection, text=text))
        if answer is False:  # §9.5 L1542: a host denial persists nothing and drops the handle
            return MutationOutcome(MutationStatus.DENIED, target, 0, stage=stage, approval_requirements=required)
        if answer is True:
            authorization = build_authorization(stage, principal_id=service.identity.principal_id,
                                                now=approval.clock())
            if not deferrable:
                return authorization, None
            record = _approval_record(service, request, stage, required, approval, decision="approved",
                                      authorization=authorization)
            failed = _store_or_fail(record, stage, required)  # ruling R39-7: written before the first send
            return failed if failed is not None else (authorization, record)
    if not deferrable:  # nobody answered and nothing may wait: fail closed (§9.5 L1538)
        return MutationOutcome(MutationStatus.APPROVAL_UNAVAILABLE, target, 0, stage=stage,
                               approval_requirements=required)
    record = _approval_record(service, request, stage, required, approval, decision="pending")
    failed = _store_or_fail(record, stage, required)
    return failed if failed is not None else MutationOutcome(
        MutationStatus.PENDING_APPROVAL, target, 0, stage=stage, approval_requirements=required,
        pending_id=record.pending_id)


def _stage_then_commit(service: MemoryService, request: MutationRequest, *, required: Tuple[str, ...] = (),
                       approval: Optional[ApprovalChannel] = None, deferrable: bool = False):
    """``(stage, commit)``, ``_CONFLICT``, or a terminal :class:`MutationOutcome` (attempts filled by the caller)."""
    target = request.target
    record: Optional[PendingApproval] = None
    try:
        stage = _with_exact_retry(service, target, lambda: service.stage_curated(request))
        if not set(stage.approval_requirements) <= set(required):  # R38-1 re-check (K-5); ruling R39-10
            return MutationOutcome(MutationStatus.APPROVAL_UNAVAILABLE, target, 0, stage=stage,
                                   approval_requirements=tuple(stage.approval_requirements))
        authorization = w.ApprovalAuthorization(kind="not_required")
        if required:
            granted = _obtain_approval(service, request, stage, required, approval, deferrable)
            if isinstance(granted, MutationOutcome):
                return granted
            authorization, record = granted
        intent = CommitIntent(target=target, request_id=request.request_id,
                              stage_handle_b64url=stage.stage_handle_b64url,
                              approval_binding_sha256=stage.approval_binding_sha256,
                              authorized_write_scopes=stage.requested_write_scopes, authorization=authorization)
        commit = _with_exact_retry(service, target, lambda: service.commit_curated(intent))
    except ProviderError as exc:
        if exc.code == "version_conflict":  # §9.5 L1554: publishes nothing; X-4 (a) makes a re-plan safe
            _forget(record)
            return _CONFLICT
        if _publication_possible(exc):  # correction R-5: never "Nothing was saved"
            return MutationOutcome(MutationStatus.OUTCOME_UNKNOWN, target, 0, error_code="outcome_unknown",
                                   pending_id=_keep_unknown(record))
        _forget(record)
        return MutationOutcome(MutationStatus.REJECTED, target, 0, error_code=exc.code, error_detail=_detail(exc))
    except _OutcomeUnknown:
        return MutationOutcome(MutationStatus.OUTCOME_UNKNOWN, target, 0, error_code="outcome_unknown",
                               pending_id=_keep_unknown(record))
    except MemoryServiceError as exc:
        _forget(record)
        return MutationOutcome(MutationStatus.UNAVAILABLE, target, 0, error_code=_error_code(exc))
    except w.WireError:
        _forget(record)
        logger.error("curated mutation request failed wire validation before transmission", exc_info=True)
        return MutationOutcome(MutationStatus.REJECTED, target, 0, error_code="invalid_request")
    _forget(record)
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
    approval: Optional[ApprovalChannel] = None,
) -> MutationOutcome:
    """Load, plan, stage and commit one curated mutation (§9.4 steps 3–10; contract C6b-2).

    ``approval=None`` (or a channel with no prompt and no deferrable session) refuses an
    approval-requiring mutation before staging, exactly as R38-1. Deferral and write-ahead
    need the channel's session to hold a C2 record equal to this service's
    ``session_state`` (ruling X6b-2).
    """
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
        deferrable = bool(required) and _deferrable(approval, service)
        if required and not ((approval is not None and approval.prompt is not None) or deferrable):  # R38-1
            return MutationOutcome(MutationStatus.APPROVAL_UNAVAILABLE, target, attempts, plan=plan,
                                   approval_requirements=required)
        request = _mutation_request(service, snapshot, plan, new_request_id(), actor_kind, initiating_surface)
        step = _stage_then_commit(service, request, required=required, approval=approval, deferrable=deferrable)
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
