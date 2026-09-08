"""Errors raised by the host-owned memory service.

Typed provider failures (:class:`ProviderError`) mirror §9.3 ``WireFailure``;
transport failures (:class:`ProviderTransportError`) are adapter-synthesized
(``unavailable`` for reads, ``unknown`` outcome for a stage/commit whose fate
is unknowable). Host-level errors describe the §9.6 dispositions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from agent.memory_service.wire import ERROR_CODES

logger = logging.getLogger(__name__)


class MemoryServiceError(Exception):
    """Base class for every memory-service error."""


class MemoryBlockedError(MemoryServiceError):
    """Authoritative fail-closed: the model request or mutation must not proceed."""

    def __init__(self, message: str, *, code: Optional[str] = None, provider_error: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.code = code
        self.provider_error = provider_error


class TargetDisabledError(MemoryServiceError):
    """The target is disabled by configuration and MUST NOT be loaded, rendered, or mutated."""


class StatelessSessionError(MemoryServiceError):
    """The session runs stateless: no curated memory, no mutation capability."""


class CapabilityUnavailableError(MemoryServiceError):
    """The negotiated provider does not offer this optional operation."""


class BindingInvalidError(MemoryServiceError):
    """Persisted host session state is missing a member or the binding is invalid."""


class ProviderError(MemoryServiceError):
    """A typed §9.3 failure envelope from the provider."""

    def __init__(self, *, code: str, outcome: str, details: Optional[Dict[str, Any]], operation: str) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"unknown provider error code {code!r}")
        if outcome not in ("not_applicable", "not_committed", "unknown"):
            raise ValueError(f"unknown provider error outcome {outcome!r}")
        super().__init__(f"{operation}: provider returned {code} (outcome {outcome})")
        self.code = code
        self.outcome = outcome
        self.details = details
        self.operation = operation


class ProviderTransportError(MemoryServiceError):
    """The provider could not be reached or answered malformed output."""

    def __init__(self, *, reason: str, operation: str, mutation_outcome_unknown: bool = False) -> None:
        super().__init__(f"{operation}: provider transport failure ({reason})")
        self.reason = reason
        self.operation = operation
        self.mutation_outcome_unknown = mutation_outcome_unknown
