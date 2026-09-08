"""The host-owned generic memory service (§9.1, §3.1).

One interface, three dispositions:

* :class:`BuiltinMemoryService` — additive mode; native ``MemoryStore`` is the authority (Task 5).
* :class:`ProviderAuthoritativeMemoryService` — the configured provider is the sole authority (Task 6).
* :class:`StatelessMemoryService` — explicit degraded disposition: no curated memory, no mutations.

The service owns identity propagation: host-level request types below carry
no provider epoch or frozen identity; the service injects both.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Tuple

from agent.memory_service import wire as w
from agent.memory_service.config import (
    FailurePolicy,
    MemoryConfigurationError,
    MemoryMode,
    MemoryServiceConfig,
    resolve_memory_service_config,
)
from agent.memory_service.errors import (
    MemoryBlockedError,
    MemoryServiceError,
    ProviderError,
    ProviderTransportError,
    StatelessSessionError,
)
from agent.memory_service.identity import FrozenMemoryIdentity, HostSessionState

logger = logging.getLogger(__name__)


class MemoryDisposition(str, Enum):
    """I1: a session is builtin, provider-authoritative, or explicit stateless."""

    BUILTIN = "builtin"
    AUTHORITATIVE = "provider_authoritative"
    STATELESS = "stateless"


@dataclass(frozen=True)
class ServiceCapabilities:
    recall_context: bool
    capture_continuity: bool


@dataclass(frozen=True)
class MutationRequest:
    """A stage request minus epoch and identity (§9.3 StageRequest)."""

    target: str
    request_id: str
    expected_revision: Optional[w.CompositeRevision]
    hidden_preservation_state: Optional[w.HiddenPreservationState]
    requested_write_scopes: Tuple[w.ScopeRef, ...]
    intent: Optional[w.MutationIntent]
    mutation_delta: Tuple[w.MutationDeltaItem, ...]
    candidate_entries: Tuple[w.CandidateEntry, ...]
    provenance: Optional[w.MutationProvenance]


@dataclass(frozen=True)
class InspectRequest:
    target: str
    request_id: str
    stage_handle_b64url: str


@dataclass(frozen=True)
class CommitIntent:
    target: str
    request_id: str
    stage_handle_b64url: str
    approval_binding_sha256: str
    authorized_write_scopes: Tuple[w.ScopeRef, ...]
    authorization: w.ApprovalAuthorization


@dataclass(frozen=True)
class RecallQuery:
    target: str
    source_revision: w.CompositeRevision
    query: str
    include_channels: Tuple[str, ...]
    exclude_entry_ids: Tuple[str, ...]
    budget: w.RecallBudget


@dataclass(frozen=True)
class ContinuityCapture:
    request_id: str
    kind: str
    text: str
    initiating_surface: str


class MemoryService(ABC):
    """Every curated-memory read, mutation, lifecycle event, and admin operation goes through here."""

    def __init__(self, config: MemoryServiceConfig) -> None:
        self._config = config

    @property
    def config(self) -> MemoryServiceConfig:
        return self._config

    @property
    @abstractmethod
    def disposition(self) -> MemoryDisposition: ...

    @property
    @abstractmethod
    def identity(self) -> Optional[FrozenMemoryIdentity]: ...

    @property
    @abstractmethod
    def capabilities(self) -> ServiceCapabilities: ...

    def target_enabled(self, target: str) -> bool:
        return self._config.target_enabled(target)

    @abstractmethod
    def load_curated(self, target: str) -> w.CuratedSnapshot: ...

    @abstractmethod
    def prompt_block(self, target: str) -> Optional[str]: ...

    @abstractmethod
    def stage_curated(self, request: MutationRequest) -> w.StageResult: ...

    @abstractmethod
    def inspect_staged(self, request: InspectRequest) -> w.StageInspection: ...

    @abstractmethod
    def commit_curated(self, intent: CommitIntent) -> w.CommitResult: ...

    @abstractmethod
    def recall_context(self, query: RecallQuery) -> w.TypedRecall: ...

    @abstractmethod
    def capture_continuity(self, capture: ContinuityCapture) -> w.ContinuityResult: ...

    def degraded_warning(self) -> Optional[str]:
        return None

    def shutdown(self) -> None:
        return None


class StatelessMemoryService(MemoryService):
    """§9.1: no curated-memory prompt block, no mutation capability, durable warning."""

    def __init__(self, config: MemoryServiceConfig, *, reason: str) -> None:
        super().__init__(config)
        self._reason = reason

    @property
    def disposition(self) -> MemoryDisposition:
        return MemoryDisposition.STATELESS

    @property
    def identity(self) -> Optional[FrozenMemoryIdentity]:
        return None

    @property
    def capabilities(self) -> ServiceCapabilities:
        return ServiceCapabilities(recall_context=False, capture_continuity=False)

    def target_enabled(self, target: str) -> bool:
        self._config.target_enabled(target)  # still validates the target name
        return False

    def _refuse(self, what: str):
        raise StatelessSessionError(f"{what} unavailable: session is stateless ({self._reason})")

    def load_curated(self, target: str) -> w.CuratedSnapshot:
        self._refuse("load_curated")

    def prompt_block(self, target: str) -> Optional[str]:
        return None

    def stage_curated(self, request: MutationRequest) -> w.StageResult:
        self._refuse("stage_curated")

    def inspect_staged(self, request: InspectRequest) -> w.StageInspection:
        self._refuse("inspect_staged")

    def commit_curated(self, intent: CommitIntent) -> w.CommitResult:
        self._refuse("commit_curated")

    def recall_context(self, query: RecallQuery) -> w.TypedRecall:
        self._refuse("recall_context")

    def capture_continuity(self, capture: ContinuityCapture) -> w.ContinuityResult:
        self._refuse("capture_continuity")

    def degraded_warning(self) -> Optional[str]:
        return (
            "Curated memory is STATELESS for this session: the authoritative memory provider "
            f"was unavailable at startup ({self._reason}). No memory is loaded and no memory "
            "can be saved until a new session starts with the provider available."
        )


def _default_backend_factory(config: MemoryServiceConfig) -> Callable[[MemoryServiceConfig], Any]:
    from plugins.memory import load_authoritative_backend_factory

    factory = load_authoritative_backend_factory(config.provider)
    if factory is None:
        raise MemoryConfigurationError(
            f"memory.provider {config.provider!r} does not export create_authoritative_backend; "
            "it cannot serve provider_mode: authoritative"
        )
    return factory


def select_memory_service(
    config: Any,
    *,
    store_factory: Callable[[], Any],
    requested_context: Optional[w.RequestedContext] = None,
    session_state: Optional[HostSessionState] = None,
    backend_factory: Optional[Callable[[MemoryServiceConfig], Any]] = None,
) -> MemoryService:
    """Decide the disposition from configuration, then build the service.

    ``store_factory`` constructs (and loads) the native ``MemoryStore``; it is
    invoked only after additive mode is selected, never in authoritative or
    stateless dispositions (§9.1). A configuration error propagates under
    every policy (and still shuts the backend down). A provider failure at
    negotiate, bind, validate, or the startup first load raises
    :class:`MemoryBlockedError` under ``fail_closed`` and yields a
    :class:`StatelessMemoryService` under ``stateless``.

    §9.6 fixes the disposition *before the first model request*, so after a
    successful bind or validate this function probes ``load_curated`` once for
    each enabled target and discards the snapshot. That probe is what gives a
    first-load failure a stateless outcome instead of a
    :class:`MemoryBlockedError` raised later under either policy. It does not
    weaken the per-request rule: every model request still performs its own
    fresh load, and no snapshot from the probe is ever cached or reused.
    """
    cfg = resolve_memory_service_config(config)
    if cfg.provider_mode is MemoryMode.ADDITIVE:
        from agent.memory_service.builtin import BuiltinMemoryService

        return BuiltinMemoryService(cfg, store_factory())

    factory = backend_factory or _default_backend_factory(cfg)
    backend = factory(cfg)
    from agent.memory_service.authoritative import ProviderAuthoritativeMemoryService

    service = ProviderAuthoritativeMemoryService(cfg, backend)
    try:
        if session_state is not None:
            service.resume(session_state)
        else:
            if requested_context is None:
                raise MemoryConfigurationError("authoritative mode needs a requested_context or a session_state")
            service.start(requested_context)
        for target in ("memory", "user"):
            if cfg.target_enabled(target):
                service.load_curated(target)
    except MemoryConfigurationError:
        service.shutdown()
        raise
    except (ProviderError, ProviderTransportError, w.WireError, MemoryServiceError) as exc:
        service.shutdown()
        if cfg.failure_policy is FailurePolicy.STATELESS:
            logger.warning("memory session starting stateless: authoritative provider unavailable before the first model request")
            return StatelessMemoryService(cfg, reason=str(exc))
        raise MemoryBlockedError(f"authoritative memory provider failed before the first model request: {exc}") from exc
    return service
