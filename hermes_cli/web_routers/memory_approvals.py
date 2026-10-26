"""Pending provider-managed memory approvals: list, approve, deny (contract C6b-9; §9.4 steps 7-9).

The dashboard and desktop counterpart of ``/memory pending|approve|reject``. A change that needs approval
(a reset, or a journey edit or delete of the user profile or of a non-default scope) is staged and waits
here under the identity that staged it. Approving replays the identical byte-free commit under that
identity; denying drops the handle and persists nothing (§9.5 L1542, L1562). This is an owner surface: it
lists every record and acts as the staging identity's principal. ``profile`` selects the home, as on the
journey routes that stage these changes. Inspection text holds entry bodies for the immediate UI only;
records never do (§9.3 L1357).
"""

from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, HTTPException

from hermes_cli.web_deps import late
from hermes_cli.web_models import MemoryApprovalDecision
from hermes_cli.web_routers._common import scoped_to_thread

router = APIRouter()

# Late-bound so a test's monkeypatch on the owning module wins at call time.
load_config = late("load_config", "hermes_cli.config")

_NOT_PROVIDER_MANAGED = ("Pending memory approvals apply to provider-managed memory only "
                         "(memory.provider_mode: authoritative).")

#: ReplayStatus value -> (HTTP status, message). A result that may have been saved (unknown, unavailable,
#: void_unconfirmed) is never a client error, which would read as "refused, nothing saved".
_APPROVE: Dict[str, Tuple[int, str]] = {
    "committed": (200, "Approved and saved."),
    "not_found": (404, "No pending memory approval has this id."),
    "expired": (409, "The staged change expired before it was approved; nothing was saved."),
    "conflict": (409, "Memory changed after this was staged, so nothing was saved; make the change again."),
    "rejected": (409, "The memory provider refused the approved change ({code}); nothing was saved."),
    "void": (409, "This change can no longer be approved ({code}); nothing was saved."),
    "void_unconfirmed": (503, "This approved change can no longer be settled ({code}); it may have been saved."),
    "unknown": (503, "The memory provider did not confirm the result; approving again retries the identical "
                     "request."),
    "unavailable": (503, "The approval was not applied ({code}); the request is kept."),
}
#: RejectStatus value -> (HTTP status, message).
_DENY: Dict[str, Tuple[int, str]] = {
    "dropped": (200, "Denied; nothing was saved."),
    "not_found": (404, "No pending memory approval has this id."),
    "refused": (409, "This change was already approved and its result is unconfirmed; approve it again to "
                     "settle it."),
    "unavailable": (503, "This approval could not be removed."),
}


def _row(review: Any) -> Dict[str, Any]:
    from agent.memory_service.admin import format_scope

    row: Dict[str, Any] = {"pending_id": review.pending_id, "state": review.state.value}
    record = review.record
    if record is not None:
        row.update(target=record.target, intent=record.intent_kind, decision=record.decision,
                   scopes=[format_scope(s) for s in record.requested_write_scopes],
                   approval_required=list(record.approval_requirements), expires_at=record.expires_at,
                   origin=record.origin, created_at=record.created_at)
    if review.text:
        row["inspection"] = review.text
    return row


def _provider_managed_config() -> Any:
    from agent.memory_service.bootstrap import requests_authoritative_mode

    cfg = load_config()
    return cfg if requests_authoritative_mode(cfg) else None


def _list() -> Dict[str, Any]:
    from agent.memory_service.approval_replay import review_pending_approvals

    cfg = _provider_managed_config()
    if cfg is None:
        return {"provider_managed": False, "approvals": []}
    return {"provider_managed": True, "approvals": [_row(r) for r in review_pending_approvals(raw_config=cfg)]}


def _decide(pending_id: str, decision: str) -> Tuple[int, Any]:
    from agent.memory_service.approval_replay import reject_pending_approval, replay_pending_approval

    cfg = _provider_managed_config()
    if cfg is None:
        return 409, _NOT_PROVIDER_MANAGED
    warning = None
    if decision == "approve":
        report = replay_pending_approval(pending_id, raw_config=cfg)
        status, code, table = report.status.value, report.code, _APPROVE
        if report.audit_pending:
            warning = "Saved and published; the memory provider's audit trail is pending."
    else:
        status, code, table = reject_pending_approval(pending_id).value, None, _DENY
    http_status, message = table[status]
    payload = {"ok": http_status == 200, "pending_id": pending_id, "status": status, "code": code,
               "message": message.format(code=code or "unknown")}
    if warning:
        payload["warning"] = warning
    return http_status, payload


@router.get("/api/memory/approvals")
async def list_memory_approvals(profile: Optional[str] = None):
    """Every pending record with its live inspection; settled ones (saved, expired, void) are reported once."""
    return await scoped_to_thread(profile, _list)


@router.post("/api/memory/approvals/{pending_id}")
async def decide_memory_approval(pending_id: str, body: MemoryApprovalDecision):
    http_status, payload = await scoped_to_thread(body.profile, lambda: _decide(pending_id, body.decision))
    if http_status != 200:
        raise HTTPException(status_code=http_status, detail=payload)
    return payload
