"""Host approval of a staged curated mutation (§9.4 steps 7–8, §9.5; R39; contract C6b-1).

Provider-neutral. An :class:`ApprovalChannel` says how a human is asked (an inline
prompt function) and whether an unanswered request may wait for ``/memory pending``:
the Hermes session whose host-state record holds the frozen identity (ruling R39-4).
A deferral also needs that record to equal the staging service's identity (ruling
X6b-2; ``mutation._deferrable``). Hermes owns the approval UI (§9.4 L1500). The
rendered text is for the immediate UI only (§9.3 L1357): it is never persisted,
logged or returned to a model.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional, Sequence, Tuple

from agent.memory_service import wire as w


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def rfc3339(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class ApprovalPrompt:
    target: str
    intent_kind: str
    requirements: Tuple[str, ...]      # Hermes's full list: the provider's plus host-only ("memory.write_approval")
    stage: w.StageResult
    inspection: w.StageInspection      # bodies: immediate UI only; never persist or log
    text: str                          # render_stage_inspection(...)


PromptFn = Callable[[ApprovalPrompt], Optional[bool]]  # True approve, False deny, None: no answer. Must not raise.


@dataclass(frozen=True)
class ApprovalChannel:
    prompt: Optional[PromptFn] = None
    session_id: Optional[str] = None   # Hermes session id; deferral + write-ahead only when its C2 record matches (X6b-2)
    origin: str = "foreground"
    clock: Callable[[], datetime] = utcnow

    @property
    def available(self) -> bool:
        return self.prompt is not None or bool(self.session_id)


def build_authorization(stage: w.StageResult, *, principal_id: str, now: datetime) -> w.ApprovalAuthorization:
    """Ruling R39-8: bound to the stage's hash and expiring with it (§9.3 L1397: never later).

    ``approval_id`` is lowercase hex, which passes ygg's structural token check and
    its container secret scan (``transaction::token``).
    """
    return w.ApprovalAuthorization(kind="approved", approval_id=uuid.uuid4().hex,
                                   approved_by_principal_id=principal_id, approved_at=rfc3339(now),
                                   expires_at=stage.expires_at,
                                   approval_binding_sha256=stage.approval_binding_sha256)


_LABEL = {"memory": "memory", "user": "user profile"}


def _scope(scope: w.ScopeRef) -> str:
    return f"{scope.kind}:{scope.id}"


def render_stage_inspection(inspection: w.StageInspection, *, intent_kind: str, requirements: Sequence[str]) -> str:
    """§9.5 L1546's approval UI; never a provider token (§9.3 L987, D-R39-3)."""
    summary = inspection.summary
    before = {entry.id: entry for entry in inspection.visible_before}
    after_ids = {entry.id for entry in inspection.visible_after}
    candidates = {cand.client_ref: cand for cand in inspection.canonical_candidates}
    superseded = {a.superseded_id for a in summary.admissions if a.superseded_id}
    lines = [f"Memory change awaiting approval ({_LABEL[summary.target]})",
             f"  operation: {intent_kind}",
             f"  scopes: {', '.join(_scope(s) for s in summary.requested_write_scopes) or 'none'}",
             f"  approval needed for: {', '.join(requirements)}",
             f"  expires: {summary.expires_at}",
             "  changes:"]
    header = len(lines)
    lines += [f"    - {entry.text}" for entry in inspection.visible_before
              if entry.id not in after_ids and entry.id not in superseded]
    for admission in summary.admissions:
        candidate = candidates.get(admission.client_ref)
        text = candidate.text if candidate is not None else ""
        tag = (f"[{admission.disposition}, {admission.publication_effect}, {_scope(admission.origin_scope)}, "
               f"key {admission.policy_key or 'none'}]")
        old = before.get(admission.superseded_id) if admission.superseded_id else None
        lines += [f"    ~ {old.text}", f"      -> {text} {tag}"] if old is not None else [f"    + {text} {tag}"]
    lines += [f"    hidden: {h.action} {h.count} {h.lane} record(s) in {_scope(h.scope)} ({h.record_channel})"
              for h in summary.hidden_effects]
    if len(lines) == header:
        lines.append("    (no visible change)")
    return "\n".join(lines)
