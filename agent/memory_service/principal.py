"""Explicit gateway principal mapping (§4.1 L252; invariant I3, §13.1 L2191; ruling R41-5; contract C6b-12).

A gateway session's principal is never ``memory.principal_id`` by default. Each gateway user maps
explicitly, as ``"<platform>:<user_id>": <principal_id>`` under ``memory.gateway_principals``. An
unknown user, or one whose ``user_id`` and ``user_id_alt`` map to two different principals, gets no
memory: a stateless disposition that never contacts the provider, so it also never receives a
``degraded_global_only`` result. Local surfaces (CLI, TUI, desktop, ACP) carry no gateway user id and
keep ``memory.principal_id``. A session's principal is ``service.identity.principal_id``, never
``service.config.principal_id``. Provider-generic (D5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from agent.memory_service.config import MemoryConfigurationError, MemoryServiceConfig
from agent.memory_service.service import StatelessMemoryService

GATEWAY_PRINCIPALS_KEY = "gateway_principals"


@dataclass(frozen=True)
class GatewayIdentity:
    platform: str
    user_id: Optional[str]
    user_id_alt: Optional[str] = None

    def keys(self) -> Tuple[str, ...]:
        return tuple(f"{self.platform}:{uid}" for uid in (self.user_id, self.user_id_alt) if uid)


def _identity(platform: Any, user_id: Any, user_id_alt: Any) -> Optional[GatewayIdentity]:
    if not (user_id or user_id_alt):
        return None
    return GatewayIdentity(platform=str(getattr(platform, "value", platform) or ""),
                           user_id=str(user_id) if user_id else None,
                           user_id_alt=str(user_id_alt) if user_id_alt else None)


def gateway_identity_of(agent: Any) -> Optional[GatewayIdentity]:
    """The gateway user an agent serves (``agent._user_id``/``_user_id_alt``), or ``None``."""
    return _identity(getattr(agent, "platform", ""), getattr(agent, "_user_id", None),
                     getattr(agent, "_user_id_alt", None))


def gateway_identity_from_source(source: Any) -> Optional[GatewayIdentity]:
    """The same identity from a gateway ``SessionSource`` (``platform`` may be an enum)."""
    if source is None:
        return None
    return _identity(getattr(source, "platform", ""), getattr(source, "user_id", None),
                     getattr(source, "user_id_alt", None))


def _valid_key(key: Any) -> bool:
    platform, sep, user = key.partition(":") if isinstance(key, str) else ("", "", "")
    return bool(sep and platform.strip() and user.strip())


def gateway_principals(raw_config: Any) -> Dict[str, str]:
    """``memory.gateway_principals`` validated; ``MemoryConfigurationError`` when malformed (§9.1 L946)."""
    section = raw_config.get("memory") if isinstance(raw_config, Mapping) else None
    table = section.get(GATEWAY_PRINCIPALS_KEY) if isinstance(section, Mapping) else None
    if table is None:
        return {}
    if not isinstance(table, Mapping) or not all(
            _valid_key(k) and isinstance(v, str) and v.strip() for k, v in table.items()):
        raise MemoryConfigurationError(
            "memory.gateway_principals must map '<platform>:<user_id>' to a non-empty principal id")
    return {k.strip(): v.strip() for k, v in table.items()}


def map_gateway_principal(raw_config: Any, identity: GatewayIdentity) -> Optional[str]:
    """The one principal ``identity`` maps to, or ``None`` when unknown or ambiguous."""
    table = gateway_principals(raw_config)
    found = {table[key] for key in identity.keys() if key in table}
    return found.pop() if len(found) == 1 else None


class UnmappedPrincipalMemoryService(StatelessMemoryService):
    """No memory for an unknown or ambiguous gateway user (I3). Never contacts the provider."""

    def __init__(self, config: MemoryServiceConfig) -> None:
        super().__init__(config, reason="gateway user is not mapped to a memory principal")

    def degraded_warning(self) -> Optional[str]:
        return "Curated memory is not available in this chat."
