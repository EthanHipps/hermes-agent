"""Explicit identity and scope for scheduled (non-human) binds (§9.2 L968; §9.3 L1229; rulings R43-6, R43-12, X6b-3).

Contract C6b-8. A scheduled surface never infers scope from an incidental directory: the
operator names registry IDs, and the bind asserts them with ``resolution_source: explicit_ids``
(``canonical_directory: null``). An empty mapping is an explicit principal-global scope.
Administrative surfaces use ``agent/memory_service/admin.py`` (C6b-5), which never goes stateless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from agent.memory_service import wire as w
from agent.memory_service.config import (
    MemoryConfigurationError, MemoryMode, MemoryServiceConfig, resolve_memory_service_config,
)

_SCOPE_KEYS = ("org_id", "project_id", "repo_id")


@dataclass(frozen=True)
class ExplicitScope:
    org_id: Optional[str] = None
    project_id: Optional[str] = None
    repo_id: Optional[str] = None


def explicit_scope_from_config(raw_config: Any, key: str) -> Optional[ExplicitScope]:
    """``memory.<key>``: ``None`` when absent or null; a configuration error when malformed."""
    section = raw_config.get("memory") if isinstance(raw_config, Mapping) else None
    raw = section.get(key) if isinstance(section, Mapping) else None
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise MemoryConfigurationError(f"memory.{key} must be a mapping of {', '.join(_SCOPE_KEYS)}")
    unknown = sorted(str(k) for k in raw if k not in _SCOPE_KEYS)
    if unknown:
        raise MemoryConfigurationError(f"memory.{key} has unknown key(s): {', '.join(unknown)}")
    values = {}
    for name in _SCOPE_KEYS:
        value = raw.get(name)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise MemoryConfigurationError(f"memory.{key}.{name} must be a non-empty string")
        values[name] = value.strip()
    return ExplicitScope(**values)


def _no_native_store():
    raise AssertionError("an explicit authoritative bind never builds the native store (§9.1 L950)")


def bind_explicit_service(raw_config: Any, *, scope: ExplicitScope, logical_session_id: str, platform: str,
                          profile_id: Optional[str] = None,
                          backend_factory: Optional[Callable[[MemoryServiceConfig], Any]] = None):
    """Bind a new logical session from explicit IDs under R35's failure policy (§9.6).

    Persists no host-state record (ruling R43-7): the caller owns the returned service's lifecycle.
    """
    from agent.memory_service.service import select_memory_service

    config = resolve_memory_service_config(raw_config)
    if config.provider_mode is not MemoryMode.AUTHORITATIVE:
        raise ValueError("an explicit memory identity needs provider_mode: authoritative")
    if profile_id is None:
        from hermes_cli.profiles import get_active_profile_name

        profile_id = get_active_profile_name()
    context = w.RequestedContext(
        principal_id=config.principal_id, profile_id=profile_id, logical_session_id=logical_session_id,
        platform=platform, org_id=scope.org_id, project_id=scope.project_id, repo_id=scope.repo_id,
        workspace_id=None, resolution_source="explicit_ids", canonical_directory=None,
    )
    return select_memory_service(raw_config, store_factory=_no_native_store, requested_context=context,
                                 backend_factory=backend_factory)
