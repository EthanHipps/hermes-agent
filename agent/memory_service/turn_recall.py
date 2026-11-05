"""Fit one target's recall result into the host's per-turn recall budget.

Pure: no I/O, no provider call, no re-ranking. Only ``scoped_evidence`` is rendered; the
trusted-instruction budget for a recall event is zero. The result is the longest prefix of
whole records, in result order, whose rendered text plus its exact omission note fits both
the scalar budget (counted after escaping and fencing) and the entry cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from agent.memory_service import wire as w
from agent.memory_service.render import OMISSION_NOTE, evidence_record_lines

RECALL_EVIDENCE_SCALARS = 3000
RECALL_ENTRIES = 20


@dataclass(frozen=True)
class TargetRecallRender:
    text: str
    delivered_ids: Tuple[str, ...]
    omitted: int
    scalars: int


def render_target_recall(recall: w.TypedRecall, *, evidence_scalars: int, entries: int) -> TargetRecallRender:
    records = recall.scoped_evidence
    candidates = ["\n".join(evidence_record_lines(entry)) for entry in records]
    total = len(records)
    for taken in range(min(total, max(entries, 0)), -1, -1):
        text = "\n".join(candidates[:taken])
        omitted = total - taken
        if omitted:
            note = OMISSION_NOTE.format(count=omitted, label=recall.target, limit=evidence_scalars)
            text = f"{text}\n{note}" if taken else note
        if len(text) <= evidence_scalars:
            return TargetRecallRender(text=text, delivered_ids=tuple(e.id for e in records[:taken]),
                                      omitted=omitted, scalars=len(text))
    return TargetRecallRender(text="", delivered_ids=(), omitted=total, scalars=0)
