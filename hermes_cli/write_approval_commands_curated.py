"""/memory write-approval subcommands in a provider-managed home (R39; §9.4 steps 7–9; §9.5; contract C6b-4).

Reached from ``write_approval_commands.handle_pending_subcommand`` when the home's
config requests authoritative mode (ruling R39-13). Only handle-only approval
records are listed, approved or rejected. Native ``pending/memory/`` records stay
dormant (§9.1 L950). Nothing here prints a provider token (§9.3 L987; D-R39-3).
``approve`` acts as the staging identity's principal on every surface that reaches
this handler (ruling R39-15); gating gateway users (I3) is R41's.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from agent.memory_service.approval_replay import (
    RejectStatus, ReplayStatus, ReviewState, reject_pending_approval, replay_pending_approval,
    review_pending_approvals)
from agent.memory_service.approval_store import list_pending_approvals

_LABEL = {"memory": "memory", "user": "user profile"}
_FOOTER = "Apply: /memory approve <id>   Reject: /memory reject <id>"
_REVIEW_LINE = {
    ReviewState.AWAITING: "awaiting approval",
    ReviewState.UNKNOWN: "approved; the provider has not confirmed the result — /memory approve {id} retries the identical request",
    ReviewState.EXPIRED: "expired before approval; nothing was saved (removed)",
    ReviewState.COMMITTED: "saved (removed from the list)",
    ReviewState.VOID: "can no longer be approved (its memory binding or target is gone); nothing was saved (removed)",
    ReviewState.VOID_UNCONFIRMED: ("approved, but the result was never confirmed and can no longer be settled (its memory "
                                   "binding or target is gone); it may have been saved (removed)"),
    ReviewState.UNAVAILABLE: "details unavailable: the memory provider could not be reached",
    ReviewState.UNREADABLE: "unreadable approval record — /memory reject {id} removes it",
}
_REPLAY_TEXT = {
    ReplayStatus.COMMITTED: "{id}: approved and saved.",
    ReplayStatus.NOT_FOUND: "No pending memory write with id '{id}'.",
    ReplayStatus.EXPIRED: "{id}: the staged change expired before approval; nothing was saved.",
    ReplayStatus.CONFLICT: "{id}: memory changed after this was staged, so nothing was saved; ask for the change again.",
    ReplayStatus.REJECTED: "{id}: the memory provider refused the approved change ({code}); nothing was saved.",
    ReplayStatus.VOID: "{id}: this change can no longer be approved ({code}); nothing was saved.",
    ReplayStatus.VOID_UNCONFIRMED: ("{id}: this approved change can no longer be settled ({code}); its result is "
                                    "unconfirmed and it may have been saved."),
    ReplayStatus.UNKNOWN: "{id}: the memory provider did not confirm the result; /memory approve {id} retries the identical request.",
    ReplayStatus.UNAVAILABLE: "{id}: the approval was not applied ({code}); the request is kept.",
}
_REJECT_TEXT = {
    RejectStatus.DROPPED: "Rejected pending memory write '{id}'. Nothing was saved.",
    RejectStatus.NOT_FOUND: "No pending memory write with id '{id}'.",
    RejectStatus.REFUSED: "'{id}' was already approved and its result is unconfirmed; run /memory pending to settle it.",
    RejectStatus.UNAVAILABLE: "Could not remove '{id}' from the approval store.",
}


def _state_line() -> str:
    from tools import write_approval as wa
    on = wa.write_approval_enabled(wa.MEMORY, fail_closed=True)
    return (f"memory.write_approval = {'on' if on else 'off'} (provider-managed memory: user-profile, "
            "non-default-scope, bulk, reset and import changes always need approval)")


def _describe(record) -> str:
    scopes = ", ".join(f"{s.kind}:{s.id}" for s in record.requested_write_scopes) or "none"
    return (f"{_LABEL[record.target]} {record.intent_kind} · scopes {scopes} · needs "
            f"{', '.join(record.approval_requirements)} · expires {record.expires_at}")


def _review(raw: Any) -> str:
    reviews = review_pending_approvals(raw_config=raw)
    if not reviews:
        return "No pending memory writes."
    lines = [f"Pending memory writes ({len(reviews)}):"]
    for review in reviews:
        record = review.record
        tag = " [auto]" if record is not None and record.origin == "background_review" else ""
        lines.append(f"  {review.pending_id}{tag}" + (f" · {_describe(record)}" if record is not None else ""))
        lines.append("    " + _REVIEW_LINE[review.state].format(id=review.pending_id))
        lines.extend("    " + line for line in (review.text or "").splitlines())
    return "\n".join(lines + ["", _FOOTER])


def _ids(rest: List[str]) -> Optional[List[str]]:
    if not rest:
        return None
    return [pid for pid, _ in list_pending_approvals()] if rest[0].lower() == "all" else [rest[0]]


def _pending(rest: List[str], raw: Any, set_mode_fn) -> str:
    return _review(raw)


def _approve(rest: List[str], raw: Any, set_mode_fn) -> str:
    from hermes_cli.write_approval_commands import _usage
    ids = _ids(rest)
    if ids is None:
        return _usage("memory")
    lines = []
    for pid in ids:
        report = replay_pending_approval(pid, raw_config=raw)
        text = _REPLAY_TEXT[report.status].format(id=pid, code=report.code or "unknown")
        lines.append(text + (" The memory provider's audit trail is pending." if report.audit_pending else ""))
    return "\n".join(lines) or "No pending memory writes."


def _reject(rest: List[str], raw: Any, set_mode_fn) -> str:
    from hermes_cli.write_approval_commands import _usage
    ids = _ids(rest)
    if ids is None:
        return _usage("memory")
    return "\n".join(_REJECT_TEXT[reject_pending_approval(pid)].format(id=pid) for pid in ids) or "No pending memory writes."


def _approval(rest: List[str], raw: Any, set_mode_fn) -> str:
    from hermes_cli.write_approval_commands import _set_approval
    from tools import write_approval as wa
    return _set_approval(wa.MEMORY, rest, set_mode_fn)


_HANDLERS: Dict[str, Callable[[List[str], Any, Any], str]] = {
    "pending": _pending, "approve": _approve, "apply": _approve, "reject": _reject, "deny": _reject,
    "drop": _reject, "approval": _approval, "mode": _approval}


def handle_curated_memory_subcommand(args: List[str], *, set_mode_fn=None) -> Optional[str]:
    """``None`` for an unknown subcommand, so the caller prints its own usage (as the native path does)."""
    from hermes_cli.config import load_config
    raw = load_config()
    if not args:
        return f"{_state_line()}\n\n{_review(raw)}"
    handler = _HANDLERS.get(args[0].lower())
    return None if handler is None else handler(args[1:], raw, set_mode_fn)
