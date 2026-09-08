"""Host-owned generic memory service (spec §9.1–§9.3, §9.6).

Hermes routes every curated-memory read, mutation, lifecycle event, and
administrative operation through :class:`MemoryService`. Provider-specific
code lives behind :class:`AuthoritativeBackend`; nothing in this package
knows any particular provider.
"""

import logging

from agent.memory_service import wire
from agent.memory_service.authoritative import ProviderAuthoritativeMemoryService
from agent.memory_service.builtin import BuiltinMemoryService
from agent.memory_service.config import (
    PROVIDER_API_VERSION,
    REQUIRED_OPERATIONS,
    FailurePolicy,
    MemoryConfigurationError,
    MemoryMode,
    MemoryServiceConfig,
    resolve_memory_service_config,
)
from agent.memory_service.errors import (
    BindingInvalidError,
    CapabilityUnavailableError,
    MemoryBlockedError,
    MemoryServiceError,
    ProviderError,
    ProviderTransportError,
    StatelessSessionError,
    TargetDisabledError,
)
from agent.memory_service.identity import FrozenMemoryIdentity, HostSessionState
from agent.memory_service.service import (
    CommitIntent,
    ContinuityCapture,
    InspectRequest,
    MemoryDisposition,
    MemoryService,
    MutationRequest,
    RecallQuery,
    ServiceCapabilities,
    StatelessMemoryService,
    select_memory_service,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PROVIDER_API_VERSION",
    "REQUIRED_OPERATIONS",
    "FailurePolicy",
    "MemoryConfigurationError",
    "MemoryMode",
    "MemoryServiceConfig",
    "resolve_memory_service_config",
    "wire",
    "BindingInvalidError",
    "CapabilityUnavailableError",
    "MemoryBlockedError",
    "MemoryServiceError",
    "ProviderError",
    "ProviderTransportError",
    "StatelessSessionError",
    "TargetDisabledError",
    "FrozenMemoryIdentity",
    "HostSessionState",
    "BuiltinMemoryService",
    "CommitIntent",
    "ContinuityCapture",
    "InspectRequest",
    "MemoryDisposition",
    "MemoryService",
    "MutationRequest",
    "ProviderAuthoritativeMemoryService",
    "RecallQuery",
    "ServiceCapabilities",
    "StatelessMemoryService",
    "select_memory_service",
]
