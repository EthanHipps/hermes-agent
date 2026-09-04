"""§9.1 configuration contract for the host-owned memory service.

    memory:
      provider: <plugin name>
      provider_mode: additive | authoritative        # absent = additive
      authoritative_failure_policy: fail_closed | stateless
      provider_executable: <absolute path>            # authoritative only
      memory_enabled: bool
      user_profile_enabled: bool

Every violation raises :class:`MemoryConfigurationError`; Hermes never
reinterprets a bad authoritative configuration as additive.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional

from utils import is_truthy_value

PROVIDER_API_VERSION = 1

#: Operations an authoritative provider must implement (§9.3).
REQUIRED_OPERATIONS = (
    "bind_session",
    "validate_session",
    "load_curated",
    "stage_curated",
    "inspect_staged",
    "commit_curated",
)

#: Config keys that would smuggle caller-supplied arguments to the provider
#: executable; the argv suffix is fixed by §9.3, so any of these is an error.
_EXTRA_ARGUMENT_KEYS = ("provider_args", "provider_arguments", "provider_executable_args")


class MemoryMode(str, Enum):
    """Configured authority mode. Stateless is a runtime disposition, not a mode."""

    ADDITIVE = "additive"
    AUTHORITATIVE = "authoritative"


class FailurePolicy(str, Enum):
    FAIL_CLOSED = "fail_closed"
    STATELESS = "stateless"


class MemoryConfigurationError(ValueError):
    """A §9.1 configuration error. Never downgraded to additive behavior."""


@dataclass(frozen=True)
class MemoryServiceConfig:
    provider: str
    provider_mode: MemoryMode
    failure_policy: FailurePolicy
    provider_executable: Optional[str]
    memory_enabled: bool
    user_profile_enabled: bool

    def target_enabled(self, target: str) -> bool:
        if target == "memory":
            return self.memory_enabled
        if target == "user":
            return self.user_profile_enabled
        raise ValueError(f"unknown memory target {target!r}")


def _memory_section(config: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    section = config.get("memory")
    return section if isinstance(section, Mapping) else {}


def _parse_enum(enum_cls, raw: Any, key: str, default):
    if raw is None or raw == "":
        return default
    if isinstance(raw, str):
        try:
            return enum_cls(raw)
        except ValueError:
            pass
    allowed = ", ".join(m.value for m in enum_cls)
    raise MemoryConfigurationError(f"memory.{key} {raw!r} is not one of: {allowed}")


def resolve_memory_service_config(config: Optional[Mapping[str, Any]]) -> MemoryServiceConfig:
    """Resolve the ``memory:`` section into a validated :class:`MemoryServiceConfig`."""
    section = _memory_section(config)
    provider = section.get("provider")
    provider = provider.strip() if isinstance(provider, str) else ""
    mode = _parse_enum(MemoryMode, section.get("provider_mode"), "provider_mode", MemoryMode.ADDITIVE)
    policy = _parse_enum(
        FailurePolicy,
        section.get("authoritative_failure_policy"),
        "authoritative_failure_policy",
        FailurePolicy.FAIL_CLOSED,
    )
    if policy is FailurePolicy.STATELESS and mode is not MemoryMode.AUTHORITATIVE:
        raise MemoryConfigurationError(
            "memory.authoritative_failure_policy: stateless is valid only with provider_mode: authoritative"
        )

    executable_raw = section.get("provider_executable")
    executable = executable_raw.strip() if isinstance(executable_raw, str) else None
    if mode is MemoryMode.AUTHORITATIVE:
        if not provider:
            raise MemoryConfigurationError("memory.provider_mode: authoritative requires memory.provider")
        if not executable:
            raise MemoryConfigurationError(
                "memory.provider_mode: authoritative requires memory.provider_executable"
            )
        if not os.path.isabs(executable):
            raise MemoryConfigurationError(
                f"memory.provider_executable {executable!r} must be an absolute path; PATH lookup is not allowed"
            )
        if not os.path.isfile(executable):
            raise MemoryConfigurationError(
                f"memory.provider_executable {executable!r} must be an existing file"
            )
        for key in _EXTRA_ARGUMENT_KEYS:
            if section.get(key):
                raise MemoryConfigurationError(
                    f"memory.{key}: caller-supplied provider arguments are not allowed; the argv suffix is fixed"
                )
    elif not executable:
        executable = None

    return MemoryServiceConfig(
        provider=provider,
        provider_mode=mode,
        failure_policy=policy,
        provider_executable=executable,
        memory_enabled=is_truthy_value(section.get("memory_enabled"), default=True),
        user_profile_enabled=is_truthy_value(section.get("user_profile_enabled"), default=True),
    )
