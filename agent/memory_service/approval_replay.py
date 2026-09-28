"""Deferred approvals: review, approve and reject (§9.4 steps 7–9; §9.5 L1542, L1562; R39; contract C6b-4).

A replay never borrows a live agent's service (D-R39-5). It resumes the staging
session's persisted identity on its own transport, using the C2 host-state record
that the approval record names (§4.1 L250, L262; ruling R39-4). The transport is
``bootstrap.open_session_view``: negotiate plus validate, no bind, as R40's
continuity capture uses it (K-4). The commit is byte-free, and its authorization
is written to the record before the first send (ruling R39-7). An unknown outcome
is therefore retried only with the identical request.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple

from agent.memory_service import wire as w
from agent.memory_service.approval import build_authorization, render_stage_inspection, utcnow
from agent.memory_service.approval_store import (
    ApprovalStoreError, PendingApproval, drop_pending_approval, list_pending_approvals, load_pending_approval,
    save_pending_approval, valid_pending_id)
from agent.memory_service.errors import (
    BindingInvalidError, MemoryBlockedError, MemoryServiceError, ProviderError, ProviderTransportError,
    TargetDisabledError)
from agent.memory_service.mutation import _OutcomeUnknown, _with_exact_retry
from agent.memory_service.service import CommitIntent, InspectRequest

logger = logging.getLogger(__name__)


class ReviewState(str, Enum):
    AWAITING = "awaiting"
    UNKNOWN = "unknown"
    EXPIRED = "expired"
    COMMITTED = "committed"
    VOID = "void"
    VOID_UNCONFIRMED = "void_unconfirmed"  # an approved record: its commit may have landed (R39-7; correction R-5)
    UNAVAILABLE = "unavailable"
    UNREADABLE = "unreadable"


class ReplayStatus(str, Enum):
    COMMITTED = "committed"
    NOT_FOUND = "not_found"
    EXPIRED = "expired"
    CONFLICT = "conflict"
    REJECTED = "rejected"
    VOID = "void"
    VOID_UNCONFIRMED = "void_unconfirmed"  # an approved record: its commit may have landed (R39-7; correction R-5)
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"


class RejectStatus(str, Enum):
    DROPPED = "dropped"
    NOT_FOUND = "not_found"
    REFUSED = "refused"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class PendingReview:
    pending_id: str
    state: ReviewState
    record: Optional[PendingApproval] = None
    text: Optional[str] = None  # the live inspection, for the immediate UI only (§9.3 L1357)


@dataclass(frozen=True)
class ReplayReport:
    pending_id: str
    status: ReplayStatus
    code: Optional[str] = None  # content-free: a protocol code, a commit outcome, or a store_blocked reason
    audit_pending: bool = False


_VOIDING = frozenset({"provider_epoch_changed", "binding_invalid", "binding_revoked"})


def _voided(exc: MemoryBlockedError) -> bool:
    """An epoch change or a lost binding voids stages and approvals (§9.5 L1562; D-R39-8).

    A transport failure reaches here as a MemoryBlockedError chained from a
    ProviderTransportError (``ProviderAuthoritativeMemoryService._fail_transport``).
    Every other block during resume or a call is an epoch or binding verdict,
    including ``resume``'s uncoded epoch-change and identity-mismatch refusals.
    """
    return exc.code in _VOIDING or (exc.code is None and not isinstance(exc.__cause__, ProviderTransportError))


def _drop(pending_id: str) -> None:
    try:
        drop_pending_approval(pending_id)
    except ApprovalStoreError:
        logger.warning("could not remove a settled memory approval record")


def _open_view(record: PendingApproval, raw_config: Any, backend_factory) -> Tuple[Any, Optional[ReplayStatus], Optional[str]]:
    from agent.memory_service.bootstrap import open_session_view
    from agent.memory_service.config import MemoryConfigurationError, MemoryMode, resolve_memory_service_config
    from agent.memory_service.host_state import load_host_state

    try:
        config = resolve_memory_service_config(raw_config)
    except MemoryConfigurationError:
        return None, ReplayStatus.UNAVAILABLE, "configuration"
    if config.provider_mode is not MemoryMode.AUTHORITATIVE:
        return None, ReplayStatus.UNAVAILABLE, "not_authoritative"
    if not config.target_enabled(record.target):  # D-R39-9 (§9.1 L945)
        return None, ReplayStatus.VOID, "target_disabled"
    try:
        host = load_host_state(record.session_id)
    except BindingInvalidError:
        host = None
    state = host.state if host is not None else None
    if (state is None or state.provider_epoch != record.provider_epoch
            or state.identity.logical_session_id != record.logical_session_id):
        return None, ReplayStatus.VOID, "binding_invalid"  # §4.1 L250
    try:
        return open_session_view(config, state, backend_factory=backend_factory), None, None
    except MemoryBlockedError as exc:
        if _voided(exc):
            return None, ReplayStatus.VOID, exc.code or "provider_epoch_changed"
        return None, ReplayStatus.UNAVAILABLE, "unavailable"
    except (MemoryConfigurationError, MemoryServiceError, w.WireError):
        return None, ReplayStatus.UNAVAILABLE, "unavailable"


def _inspect(view, record: PendingApproval) -> w.StageInspection:
    return view.inspect_staged(InspectRequest(target=record.target, request_id=record.request_id,
                                              stage_handle_b64url=record.stage_handle_b64url))


def _settled_review(exc: ProviderError) -> Optional[ReviewState]:
    """``inspect_staged`` is the host's only stage probe (§9.3 L1359; D-R39-2, D-R39-7)."""
    details = exc.details if isinstance(exc.details, dict) else {}
    if exc.code == "stage_expired":
        return ReviewState.EXPIRED
    if exc.code == "stage_not_found":
        return ReviewState.COMMITTED if details.get("state") == "committed" else ReviewState.VOID
    if exc.code == "invalid_request":
        return ReviewState.VOID
    return None


def _review_one(record: PendingApproval, raw_config: Any, backend_factory) -> PendingReview:
    pid = record.pending_id
    void = ReviewState.VOID_UNCONFIRMED if record.decision == "approved" else ReviewState.VOID  # §9.5 L1562
    view, status, _ = _open_view(record, raw_config, backend_factory)
    if view is None:
        if status is ReplayStatus.VOID:
            _drop(pid)
            return PendingReview(pid, void, record)
        return PendingReview(pid, ReviewState.UNAVAILABLE, record)
    try:
        inspection = _inspect(view, record)
    except ProviderError as exc:
        state = _settled_review(exc)
        if state is None:
            return PendingReview(pid, ReviewState.UNAVAILABLE, record)
        _drop(pid)
        return PendingReview(pid, void if state is ReviewState.VOID else state, record)
    except MemoryBlockedError as exc:
        if _voided(exc):
            _drop(pid)
            return PendingReview(pid, void, record)
        return PendingReview(pid, ReviewState.UNAVAILABLE, record)
    except (MemoryServiceError, w.WireError):
        return PendingReview(pid, ReviewState.UNAVAILABLE, record)
    finally:
        view.shutdown()
    if inspection.summary.approval_binding_sha256 != record.approval_binding_sha256:
        _drop(pid)
        return PendingReview(pid, void, record)
    text = render_stage_inspection(inspection, intent_kind=record.intent_kind, requirements=record.approval_requirements)
    return PendingReview(pid, ReviewState.AWAITING if record.decision == "pending" else ReviewState.UNKNOWN,
                         record, text)


def review_pending_approvals(*, raw_config: Any, backend_factory=None) -> List[PendingReview]:
    """Ruling R39-14: one live inspection per record; terminal proofs are settled here."""
    return [PendingReview(pid, ReviewState.UNREADABLE) if record is None else _review_one(record, raw_config, backend_factory)
            for pid, record in list_pending_approvals()]


def _keep_unknown(record: PendingApproval) -> None:
    try:
        save_pending_approval(dataclasses.replace(record, outcome="unknown"))
    except ApprovalStoreError:
        logger.warning("could not mark a memory approval record unknown")


def _settle(record: PendingApproval, exc: ProviderError) -> ReplayReport:
    pid = record.pending_id
    details = exc.details if isinstance(exc.details, dict) else {}
    already = (exc.code == "stage_not_found" and details.get("state") == "committed") or (
        exc.code == "idempotency_mismatch" and exc.operation == "commit_curated")  # D-R39-7
    if already:
        _drop(pid)
        return ReplayReport(pid, ReplayStatus.COMMITTED, "already_committed")
    if getattr(exc, "outcome", None) == "unknown":  # K-7
        _keep_unknown(record)
        return ReplayReport(pid, ReplayStatus.UNKNOWN, "outcome_unknown")
    if exc.code == "store_blocked":  # not committed and transient: the approved request stays replayable
        reason = details.get("reason")
        return ReplayReport(pid, ReplayStatus.UNAVAILABLE, f"store_blocked: {reason}" if reason else "store_blocked")
    _drop(pid)
    status = {"stage_expired": ReplayStatus.EXPIRED, "version_conflict": ReplayStatus.CONFLICT}.get(
        exc.code, ReplayStatus.REJECTED)  # D-R39-1: a conflict cannot be re-planned without candidate text
    return ReplayReport(pid, status, exc.code)


def _replay_on(view, record: PendingApproval, clock: Callable[[], datetime]) -> ReplayReport:
    pid, stage_admissions = record.pending_id, None
    # As loaded: a pending record approved below and then refused by a typed block was not saved (§9.5 L1562).
    void = ReplayStatus.VOID_UNCONFIRMED if record.decision == "approved" else ReplayStatus.VOID
    try:
        if record.decision == "pending":
            inspection = _inspect(view, record)  # ruling R39-6: inspect before this decision
            if inspection.summary.approval_binding_sha256 != record.approval_binding_sha256:
                _drop(pid)
                return ReplayReport(pid, ReplayStatus.VOID, "inspection_mismatch")
            stage_admissions = inspection.summary.admissions
            record = dataclasses.replace(record, decision="approved", authorization=build_authorization(
                inspection.summary, principal_id=view.identity.principal_id, now=clock()))
            save_pending_approval(record)  # ruling R39-7: written before the first send
        intent = CommitIntent(target=record.target, request_id=record.request_id,
                              stage_handle_b64url=record.stage_handle_b64url,
                              approval_binding_sha256=record.approval_binding_sha256,
                              authorized_write_scopes=record.requested_write_scopes,
                              authorization=record.authorization)
        commit = _with_exact_retry(view, record.target, lambda: view.commit_curated(intent))
    except ApprovalStoreError:
        return ReplayReport(pid, ReplayStatus.UNAVAILABLE, "approval_store_unavailable")
    except ProviderError as exc:
        return _settle(record, exc)
    except _OutcomeUnknown:
        _keep_unknown(record)
        return ReplayReport(pid, ReplayStatus.UNKNOWN, "outcome_unknown")
    except MemoryBlockedError as exc:
        if _voided(exc):
            _drop(pid)
            return ReplayReport(pid, void, exc.code or "provider_epoch_changed")
        return ReplayReport(pid, ReplayStatus.UNAVAILABLE, "unavailable")
    except TargetDisabledError:
        _drop(pid)
        return ReplayReport(pid, void, "target_disabled")
    except (MemoryServiceError, w.WireError):
        return ReplayReport(pid, ReplayStatus.UNAVAILABLE, "unavailable")
    _drop(pid)
    if stage_admissions is not None and commit.admissions != stage_admissions:  # R38-10 / K-6
        logger.warning("memory commit reply does not match its stage; outcome unknown")
        return ReplayReport(pid, ReplayStatus.UNKNOWN, "commit_mismatch")
    return ReplayReport(pid, ReplayStatus.COMMITTED, commit.outcome,
                        audit_pending=commit.outcome == "committed_audit_pending")


def replay_pending_approval(pending_id: str, *, raw_config: Any, backend_factory=None,
                            clock: Callable[[], datetime] = utcnow) -> ReplayReport:
    """Approve one record: the byte-free ``commit_curated`` with the identical handle, binding hash,
    request ID, scopes and authorization (§9.7 L1604; §9.5 L1562)."""
    if not valid_pending_id(pending_id):
        return ReplayReport(pending_id, ReplayStatus.NOT_FOUND)
    try:
        record = load_pending_approval(pending_id)
    except ApprovalStoreError:
        return ReplayReport(pending_id, ReplayStatus.UNAVAILABLE, "unreadable")
    if record is None:
        return ReplayReport(pending_id, ReplayStatus.NOT_FOUND)
    view, status, code = _open_view(record, raw_config, backend_factory)
    if view is None:
        if status is ReplayStatus.VOID:
            _drop(pending_id)
            if record.decision == "approved":  # its commit may have landed (§9.5 L1562)
                status = ReplayStatus.VOID_UNCONFIRMED
        return ReplayReport(pending_id, status, code)
    try:
        return _replay_on(view, record, clock)
    finally:
        view.shutdown()


def reject_pending_approval(pending_id: str) -> RejectStatus:
    """§9.5 L1542: a host denial persists nothing and drops the handle. v1 has no discard, so no provider call."""
    if not valid_pending_id(pending_id):
        return RejectStatus.NOT_FOUND
    try:
        record = load_pending_approval(pending_id)
    except ApprovalStoreError:
        record = None  # unreadable: removing it is the only cleanup
    else:
        if record is None:
            return RejectStatus.NOT_FOUND
        if record.decision == "approved":  # its commit may have landed; /memory pending settles it
            return RejectStatus.REFUSED
    try:
        return RejectStatus.DROPPED if drop_pending_approval(pending_id) else RejectStatus.NOT_FOUND
    except ApprovalStoreError:
        return RejectStatus.UNAVAILABLE
