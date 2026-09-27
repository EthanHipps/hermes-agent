"""Explicit memory dependencies for background and scheduled surfaces (§9.7 L1610, L1613, L1625; R43; C6b-7).

Background review, its /btw sibling, cron and any future surface receive MemoryService as an
explicit dependency (``AIAgent(memory_service=...)``). Nothing here discovers memory by
filename, provider module, working directory or global singleton, and an injected service is
never re-resolved and never given a host-state record (ruling R43-10).
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from agent.memory_service import wire as w
from agent.memory_service.config import MemoryConfigurationError, MemoryServiceConfig
from agent.memory_service.errors import CapabilityUnavailableError, MemoryBlockedError, MemoryServiceError
from agent.memory_service.identity import FrozenMemoryIdentity, HostSessionState
from agent.memory_service.service import (
    CommitIntent, ContinuityCapture, InspectRequest, MemoryDisposition, MemoryService, MutationRequest,
    RecallQuery, ServiceCapabilities, StatelessMemoryService, is_provider_managed,
)

logger = logging.getLogger(__name__)

#: Ruling R43-8: platforms that never bind for themselves (§9.3 L1229: directory binds are human-initiated).
UNBOUND_PLATFORMS = frozenset({"cron", "subagent"})
#: Ruling R43-3: only the background-review fork may mutate; every other fork surface is read-only.
_MUTATING_SURFACES = frozenset({"background_review"})
_INJECTED_ATTR = "_memory_service_injected"
_NO_OPTIONAL_CAPABILITIES = ServiceCapabilities(recall_context=False, capture_continuity=False)


class UnboundMemoryService(StatelessMemoryService):
    """A background or scheduled surface with no explicit frozen identity (§9.6 L1585).

    Rejects every read and write and never contacts the provider. It reuses the stateless
    disposition so every existing reader (C1, the renderer, the request gate, C7) treats it as
    "no curated memory, no mutation"; only its reason and warning differ.
    """

    def __init__(self, config: MemoryServiceConfig, *, surface: str) -> None:
        super().__init__(config, reason=f"{surface} has no explicit memory identity")
        self._surface = surface

    @property
    def surface(self) -> str:
        return self._surface

    def degraded_warning(self) -> Optional[str]:
        return ("Curated memory is off for this run: this background surface has no explicit memory "
                "identity, so nothing is loaded and nothing can be saved.")


class LimitedMemoryView(MemoryService):
    """A capability-limited view over a parent's persisted frozen identity (ruling R43-3).

    Its own transport (``bootstrap.open_session_view``: negotiate plus validate, never bind), so
    the parent's unlocked service instance is never shared across threads. No recall, no
    continuity, never an approved commit, mutation only for the background-review surface, and
    every stage carries the surface as its ``initiating_surface``.
    """

    def __init__(self, inner: MemoryService, *, surface: str, mutations: bool) -> None:
        super().__init__(inner.config)
        self._inner, self._surface, self._mutations, self._closed = inner, surface, mutations, False

    @property
    def surface(self) -> str:
        return self._surface

    @property
    def mutations(self) -> bool:
        return self._mutations

    @property
    def disposition(self) -> MemoryDisposition:
        return self._inner.disposition

    @property
    def identity(self) -> Optional[FrozenMemoryIdentity]:
        return self._inner.identity

    @property
    def session_state(self) -> Optional[HostSessionState]:
        return getattr(self._inner, "session_state", None)

    @property
    def capabilities(self) -> ServiceCapabilities:
        return _NO_OPTIONAL_CAPABILITIES

    def target_enabled(self, target: str) -> bool:
        return self._inner.target_enabled(target)

    def load_curated(self, target: str) -> w.CuratedSnapshot:
        return self._inner.load_curated(target)

    def prompt_block(self, target: str) -> Optional[str]:
        return self._inner.prompt_block(target)

    def _require_mutation(self, what: str) -> None:
        if not self._mutations:
            raise CapabilityUnavailableError(f"{what} is unavailable: the {self._surface} surface is read-only")

    def stage_curated(self, request: MutationRequest) -> w.StageResult:
        self._require_mutation("stage_curated")
        provenance = dataclasses.replace(request.provenance, initiating_surface=self._surface) \
            if request.provenance is not None else None
        return self._inner.stage_curated(dataclasses.replace(request, provenance=provenance))

    def inspect_staged(self, request: InspectRequest) -> w.StageInspection:
        self._require_mutation("inspect_staged")
        return self._inner.inspect_staged(request)

    def commit_curated(self, intent: CommitIntent) -> w.CommitResult:
        self._require_mutation("commit_curated")
        if intent.authorization.kind != "not_required":   # approval is a human surface (§9.5; R39)
            raise CapabilityUnavailableError("a background surface never commits an approved stage")
        return self._inner.commit_curated(intent)

    def recall_context(self, query: RecallQuery) -> w.TypedRecall:
        raise CapabilityUnavailableError(f"recall is unavailable on the {self._surface} surface")

    def capture_continuity(self, capture: ContinuityCapture) -> w.ContinuityResult:
        raise CapabilityUnavailableError(f"continuity capture is unavailable on the {self._surface} surface")

    def degraded_warning(self) -> Optional[str]:
        return self._inner.degraded_warning()

    def shutdown(self) -> None:
        if not self._closed:
            self._closed = True
            self._inner.shutdown()


@dataclass(frozen=True)
class ParentMemory:
    """The reviewed conversation's memory, frozen when the review was spawned (ruling R43-4)."""

    config: MemoryServiceConfig
    disposition: str
    state: Optional[HostSessionState]


def capture_parent_memory(agent: Any) -> Optional[ParentMemory]:
    """``None`` for an absent or additive service (C1), so the additive path never changes."""
    service = getattr(agent, "_memory_service", None)
    if not is_provider_managed(service):
        return None
    disposition = getattr(service.disposition, "value", service.disposition)
    state = getattr(service, "session_state", None) if disposition == MemoryDisposition.AUTHORITATIVE.value else None
    return ParentMemory(service.config, disposition, state)


def open_fork_view(parent: ParentMemory, *, surface: str,
                   backend_factory: Optional[Callable[[MemoryServiceConfig], Any]] = None) -> MemoryService:
    """The explicit dependency a fork of ``parent`` receives (rulings R43-3, R43-4)."""
    if parent.disposition == MemoryDisposition.STATELESS.value:
        return StatelessMemoryService(parent.config, reason="the parent session is stateless")
    if parent.state is None:
        raise MemoryBlockedError("the parent session has no frozen memory identity", code="binding_invalid")
    from agent.memory_service.bootstrap import open_session_view

    try:
        inner = open_session_view(parent.config, parent.state, backend_factory=backend_factory)
    except (MemoryServiceError, MemoryConfigurationError, w.WireError) as exc:
        # §9.3 L987: the provider's message may carry epoch tokens; report the class only.
        logger.warning("background memory view unavailable (%s)", type(exc).__name__)
        raise MemoryBlockedError("could not open a memory view for this background surface",
                                 code=getattr(exc, "code", None) or "view_unavailable") from None
    return LimitedMemoryView(inner, surface=surface, mutations=surface in _MUTATING_SURFACES)
