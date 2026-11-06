"""One authoritative recall event per user turn, fitted into the host's per-turn budget (§9.3, §8.5).

``render_target_recall`` is pure: no I/O, no provider call, no re-ranking. Only
``scoped_evidence`` is rendered; the trusted-instruction budget for a recall event is zero. The
result is the longest prefix of whole records, in result order, whose rendered text plus its exact
omission note fits both the scalar budget (counted after escaping and fencing) and the entry cap.

``authoritative_turn_recall`` is the event: required loads, then one call per enabled target.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from agent.memory_provider import is_trivial_prompt
from agent.memory_service import wire as w
from agent.memory_service.errors import MemoryBlockedError, ProviderError, ProviderTransportError
from agent.memory_service.render import OMISSION_NOTE, evidence_record_lines, render_curated_prompt
from agent.memory_service.service import MemoryService, RecallQuery

logger = logging.getLogger(__name__)

RECALL_EVIDENCE_SCALARS = 3000
RECALL_ENTRIES = 20
#: Content-free: no query, evidence or provider token (§8.6 L892, §9.3 L987).
RECALL_UNAVAILABLE_WARNING = "Memory recall is unavailable for this turn; continuing without it."
_TARGET_SEPARATOR = "\n"
#: Stable allocation order: memory can use the event budget before user recall (§9.3 keeps the
#: general channel with memory, and the user target to its own channel).
_RECALL_TARGETS = (("memory", ("general", "hermes_memory")), ("user", ("hermes_user",)))


class RenderedRecall(str):
    """The renderer's final output, which the host sends exactly as it is.

    Every record already carries its own fence and "data, not authority" preamble (§5.5), so
    ``compose_user_api_content`` adds neither the additive ``<memory-context>`` frame, whose note
    calls its text authoritative, nor that frame's sanitizing, which would strip bytes after
    budgeting and could remove fences between records. The type survives the turn's
    ``ext_prefetch_cache`` plumbing because that plumbing only passes the object along; a transform
    such as ``.strip()`` would return a plain ``str`` and bring the frame back.
    """


@dataclass(frozen=True)
class TargetRecallRender:
    text: str
    delivered_ids: Tuple[str, ...]
    omitted: int
    scalars: int


@dataclass(frozen=True)
class RecallOutcome:
    text: str
    warning: Optional[str]


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


def _recall_target(service: MemoryService, query: RecallQuery) -> Optional[w.TypedRecall]:
    """One optional call: the typed result, or ``None`` when it failed without a policy conflict.

    The service answers ``ambiguous_policy`` as an unlatched ``ProviderError`` and a transport or
    malformed reply as ``ProviderTransportError``. A binding or epoch failure already arrives as
    ``MemoryBlockedError`` and propagates. Any other typed refusal (a stale revision, a query the
    provider refuses) leaves required memory valid, so only this optional read is lost.
    """
    try:
        return service.recall_context(query)
    except ProviderError as exc:
        if exc.code == "ambiguous_policy":
            # §9.6 L1576: a semantic blocker, never empty or unavailable, and never a chosen winner.
            raise MemoryBlockedError("recall_context answered ambiguous_policy", code=exc.code, provider_error=exc) from exc
        logger.warning("memory recall refused (%s)", exc.code)
    except ProviderTransportError:
        logger.warning("memory recall unavailable (transport)")
    return None


def authoritative_turn_recall(service: MemoryService, query: str) -> RecallOutcome:
    """The recall event for one user turn; the caller delivers ``text`` and shows ``warning``.

    A stateless session or a provider that never negotiated recall has nothing to call, and a
    trivial or empty query has nothing to ask. Neither is a warning, and neither touches the
    provider: the request gate still makes its own fresh loads.

    Otherwise every enabled target is loaded first, because each call must carry that target's
    own revision and exclude what both current snapshots already deliver (§9.3). A failed load
    raises the same ``MemoryBlockedError`` the gate's load would. Each target then gets exactly
    one call with what remains of the event budget, even when nothing remains and even after the
    other target's call failed, so neither budget nor a transport fault can hide a policy conflict.
    An event with a failed call renders no recall and returns the warning (§9.6 L1574).
    """
    if not service.capabilities.recall_context or is_trivial_prompt(query):
        return RecallOutcome(text="", warning=None)
    snapshots = {target: service.load_curated(target) for target, _ in _RECALL_TARGETS if service.target_enabled(target)}
    delivered = list(dict.fromkeys(render_curated_prompt(snapshots.get("memory"), snapshots.get("user")).delivered_ids))
    scalars, entries = RECALL_EVIDENCE_SCALARS, RECALL_ENTRIES
    texts: List[str] = []
    unavailable = False
    for target, channels in _RECALL_TARGETS:
        if target not in snapshots:
            continue
        recall = _recall_target(service, RecallQuery(
            target=target, source_revision=snapshots[target].revision, query=query, include_channels=channels,
            exclude_entry_ids=tuple(delivered),
            budget=w.RecallBudget(trusted_chars=0, evidence_chars=scalars, max_entries=entries),
        ))
        if recall is None:
            unavailable = True
            continue
        rendered = render_target_recall(recall, evidence_scalars=scalars, entries=entries)
        if rendered.text:
            texts.append(rendered.text)
            scalars = max(scalars - rendered.scalars - len(_TARGET_SEPARATOR), 0)
        entries -= len(rendered.delivered_ids)
        delivered.extend(rendered.delivered_ids)
    if unavailable:
        return RecallOutcome(text="", warning=RECALL_UNAVAILABLE_WARNING)
    return RecallOutcome(text=RenderedRecall(_TARGET_SEPARATOR.join(texts)), warning=None)
