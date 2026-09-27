"""/memory for provider-managed curated memory: one implementation for the CLI, the TUI (live
session and slash worker) and the gateway (§9.7 L1605, L1606; the goal_command.py precedent).

``provider_memory_command`` returns ``None`` when the session is additive and the home does not
request authoritative mode: each surface then runs its pre-R41 code unchanged (§9.10 L1674).
Otherwise nothing here builds, stats or reads a native store (§9.1 L950). The live session's service
answers (ruling R41-4), or the command says there is no live session. A gateway caller must also name
a mapped principal (§4.1 L252; I3; ruling R41-5), and a gateway that maps users to more than one
principal never reviews pending memory writes (ruling R41-20). Status is content-free:
``binding_revision`` is the only revision shown (§9.3 L987, L1167; ruling R41-3; contract C6b-13).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

UNKNOWN_SUBCOMMAND = "Unknown /memory subcommand. Use: status, pending, approve <id>, reject <id>, approval <on|off>."
NOT_AVAILABLE = "Curated memory is not available in this chat."
GATEWAY_REVIEW_UNAVAILABLE = ("Pending memory writes are reviewed from the CLI while this gateway maps its users "
                              "to more than one memory principal.")
_SERVICE_SUBCOMMANDS = frozenset({"", "status", "approve", "apply"})
_REVIEW_SUBCOMMANDS = frozenset({"", "pending", "approve", "apply", "reject", "deny", "drop"})


def _memory_section(home: Optional[Path]) -> Mapping[str, Any]:
    """The requested ``memory:`` section through the one never-raising pipeline (ruling R41-10; C6b-10)."""
    from hermes_cli.backup_memory import home_memory_section
    if home is None:
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
    return home_memory_section(home)


def requests_provider_memory(home: Optional[Path] = None) -> bool:
    """File-only check: does this home request authoritative mode? (R41-22's cheap pre-check)"""
    from agent.memory_service.bootstrap import requests_authoritative_mode
    return requests_authoritative_mode({"memory": dict(_memory_section(home))})


def _pending_subcommand(args: Sequence[str], set_mode_fn) -> Optional[str]:
    """The single call into the shared write-approval handler. `memory_store=None`: nothing here applies a
    native-shaped record (§9.1 L950). In a home that requests authoritative mode, R39's branch of
    `handle_pending_subcommand` answers from handle-only approval records (contract C6b-4); before R39 the
    base handler answers and applies nothing."""
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    return handle_pending_subcommand(wa.MEMORY, list(args), memory_store=None, set_mode_fn=set_mode_fn)


def _gateway_principal_matches(section: Mapping[str, Any], identity: Any, service: Any) -> bool:
    """I3 for /memory: the caller maps to exactly one principal, and it is the live session's (R41-5)."""
    from agent.memory_service.config import MemoryConfigurationError
    from agent.memory_service.principal import UnmappedPrincipalMemoryService, map_gateway_principal
    if identity is None or isinstance(service, UnmappedPrincipalMemoryService):
        return False
    try:
        principal = map_gateway_principal({"memory": dict(section)}, identity)
    except MemoryConfigurationError:
        logger.warning("memory.gateway_principals is invalid; /memory answers no gateway user")
        return False
    live = getattr(service, "identity", None)
    return principal is not None and (live is None or live.principal_id == principal)


def _serves_several_principals(section: Mapping[str, Any]) -> bool:
    """Ruling R41-20: one home has one approval store; a gateway reviews it only for a single principal."""
    from agent.memory_service.principal import gateway_principals
    configured = str(section.get("principal_id") or "").strip()
    named = set(gateway_principals({"memory": dict(section)}).values()) | ({configured} if configured else set())
    return len(named) > 1


def _no_session_status(section: Mapping[str, Any]) -> str:
    from agent.memory_service.config import (PROVIDER_API_VERSION, MemoryConfigurationError,
                                             resolve_memory_service_config)
    try:
        cfg = resolve_memory_service_config({"memory": dict(section)})
    except MemoryConfigurationError as exc:
        return f"Curated memory: authoritative, but the configuration is invalid: {exc}"
    return (f"Curated memory: authoritative (provider: {cfg.provider}, API v{PROVIDER_API_VERSION}, "
            f"failure policy: {cfg.failure_policy.value})\n"
            "No live memory session in this process yet: send a message first, "
            "or run 'hermes memory status --session <id>'.")


def render_memory_status(status: Any) -> str:
    """A ``status.MemoryStatus`` as text (C6b-13): mode, identity, scopes, binding revision, degraded state."""
    head = {"provider_authoritative": "authoritative", "stateless": "STATELESS for this session",
            "builtin": "additive"}.get(status.disposition, status.disposition)
    lines = [f"Curated memory: {head} (provider: {status.provider or '(none)'}, API v{status.api_version}, "
             f"failure policy: {status.failure_policy})"]
    if status.identity:
        i = status.identity
        lines.append(f"Identity: principal {i['principal_id']} · profile {i['profile_id']} · platform {i['platform']} · "
                     f"session {i['logical_session_id']}")
        lines.append("Context: " + " · ".join(f"{label} {i[key] or '—'}" for label, key in (
            ("organization", "org_id"), ("project", "project_id"), ("repository", "repo_id"),
            ("workspace", "workspace_id"))))
        lines.append(f"Binding revision: {i['binding_revision']}")
    for t in status.targets:
        if not t.enabled:
            lines.append(f"{t.target}: disabled by config")
        elif t.error:
            lines.append(f"{t.target}: unavailable ({t.error})")
        else:
            lines.append(f"{t.target}: {t.status} · visible: {', '.join(t.visible_scopes) or '—'} · "
                         f"default write: {t.default_write_scope or '— (writes disabled)'} · "
                         f"eligible: {', '.join(t.eligible_write_scopes) or '—'} · entries: {t.entries}")
    lines.append(f"Degraded: {status.degraded or 'none'}")
    return "\n".join(lines)


def provider_memory_command(args: Sequence[str], *, agent: Any, set_mode_fn: Optional[Callable[[bool], Any]],
                            busy: bool = False, home: Optional[Path] = None, gateway_identity: Any = None,
                            require_gateway_principal: bool = False) -> Optional[str]:
    """``/memory`` text for provider-managed memory, or ``None`` for "additive: use the existing path"."""
    from agent.memory_service.bootstrap import requests_authoritative_mode
    from agent.memory_service.service import is_provider_managed

    service = getattr(agent, "_memory_service", None)
    managed = is_provider_managed(service)
    section = _memory_section(home)
    if not managed and not requests_authoritative_mode({"memory": dict(section)}):
        return None
    args = list(args)
    sub = args[0].lower() if args else ""
    if require_gateway_principal and not _gateway_principal_matches(section, gateway_identity,
                                                                    service if managed else None):
        return NOT_AVAILABLE
    review_blocked = (require_gateway_principal and sub in _REVIEW_SUBCOMMANDS
                      and _serves_several_principals(section))

    def pending() -> str:
        return GATEWAY_REVIEW_UNAVAILABLE if review_blocked else (_pending_subcommand(args, set_mode_fn) or "")

    if not managed:
        if sub == "status":
            return _no_session_status(section)
        if not args:
            return _no_session_status(section) + "\n\n" + pending()
        return pending() if review_blocked else (_pending_subcommand(args, set_mode_fn) or UNKNOWN_SUBCOMMAND)
    if busy and sub in _SERVICE_SUBCOMMANDS:
        return f"A turn is running; /memory {sub or 'status'} is available between turns."
    if sub == "status" or not args:
        from agent.memory_service.status import collect_memory_status
        text = render_memory_status(collect_memory_status(service))
        return text if sub else text + "\n\n" + pending()
    return pending() if review_blocked else (_pending_subcommand(args, set_mode_fn) or UNKNOWN_SUBCOMMAND)
