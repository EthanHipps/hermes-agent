"""``hermes memory status|setup|off|reset`` when authoritative mode is requested (§9.7 L1618, L1620).

A sibling of ``main_agent_cmds.py`` and ``memory_setup.py``: each of their commands gains a short
branch into here and keeps its additive body byte-for-byte (§9.10 L1674). Keyed on the REQUESTED mode
(R44-3), read through the never-raising, last-known-good-aware archive/doctor pipeline (rulings R41-9,
R41-10): an invalid authoritative config is reported, never treated as additive, and never reaches a
native file. Nothing here writes config.yaml; switching authority is ``hermes memory migrate`` (R45) or an
explicit operator edit (§9.9 L1664–L1668; ruling R41-7).
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Tuple

from agent.memory_service.config import PROVIDER_API_VERSION, MemoryConfigurationError, resolve_memory_service_config

_RULE = "─" * 40

PROVIDER_SWITCH_REFUSED = ("memory.provider_mode is authoritative: the memory provider changes only by an explicit "
                           "edit of memory.provider in config.yaml (§9.9). Nothing was changed.")


def requested_authoritative_config() -> Optional[dict]:
    """``{"memory": section}`` when the active home requests authoritative mode, else None. Never raises: the
    archive/doctor pipeline, last-known-good aware (rulings R41-9, R41-10; hermes_cli/AGENTS.md L133)."""
    from agent.memory_service.bootstrap import requests_authoritative_mode
    from hermes_cli.backup_memory import home_memory_section
    from hermes_constants import get_hermes_home
    config = {"memory": home_memory_section(get_hermes_home())}
    return config if requests_authoritative_mode(config) else None


def _session_block(cfg, session_id: str, backend_factory) -> Tuple[List[str], bool]:
    """A persisted session's frozen identity and scopes: validate (negotiate + validate_session), never bind."""
    from agent.memory_service.bootstrap import open_session_view
    from agent.memory_service.errors import BindingInvalidError, MemoryServiceError
    from agent.memory_service.host_state import load_host_state
    from agent.memory_service.status import collect_memory_status
    from hermes_cli.memory_command import render_memory_status
    try:
        record = load_host_state(session_id)
    except BindingInvalidError:
        return [f"  Session {session_id}: its memory record is unreadable (binding_invalid)"], False
    if record is None:
        return [f"  Session {session_id}: no memory record (never bound, or deleted)"], False
    if record.disposition == "stateless":
        return [f"  Session {session_id}: STATELESS — no curated memory for this session"], True
    try:
        view = open_session_view(cfg, record.state, backend_factory=backend_factory)   # validate, never bind
    except (MemoryServiceError, MemoryConfigurationError) as exc:
        return [f"  Session {session_id}: the provider refused its identity "
                f"({getattr(exc, 'code', None) or type(exc).__name__})"], False
    try:
        text = render_memory_status(collect_memory_status(view))
    finally:
        view.shutdown()
    return ["", *(f"  {line}" for line in text.splitlines())], True


def status_command(config: Mapping[str, Any], args: Any = None, *, backend_factory=None) -> int:
    """Show and validate the authoritative configuration and probe provider health (negotiate only; R41-6)."""
    from agent.memory_service.archive import archive_disposition
    from agent.memory_service.principal import gateway_principals
    from agent.memory_service.status import probe_provider
    try:
        cfg = resolve_memory_service_config(config)
    except MemoryConfigurationError as exc:
        print(f"\nMemory status\n{_RULE}\n  Invalid authoritative memory configuration: {exc}\n")
        return 1
    ok = True
    lines = ["", "Memory status", _RULE, "  Mode:             authoritative",
             f"  Provider:         {cfg.provider}", f"  Failure policy:   {cfg.failure_policy.value}",
             f"  API version:      v{PROVIDER_API_VERSION} required", f"  Principal:        {cfg.principal_id}",
             f"  Targets:          memory {'on' if cfg.memory_enabled else 'off'} · "
             f"user {'on' if cfg.user_profile_enabled else 'off'}"]
    try:
        lines.append(f"  Gateway users:    {len(gateway_principals(config))} mapped (unmapped users get no memory)")
    except MemoryConfigurationError as exc:
        ok = False
        lines.append(f"  Gateway users:    invalid mapping: {exc}")
    disposition = archive_disposition(config)
    lines.append(f"  Archives:         {disposition.disposition}, not included "
                 f"(restore_action: {disposition.restore_action})")
    probe = probe_provider(cfg, backend_factory=backend_factory)
    if probe.ok:
        lines.append(f"  Provider health:  answered negotiate (API v{probe.api_version}); recall "
                     f"{'yes' if probe.recall_context else 'no'}, continuity {'yes' if probe.capture_continuity else 'no'}")
    else:
        ok = False
        lines.append(f"  Provider health:  UNAVAILABLE — {probe.error}")
    session_id = getattr(args, "session", None)
    if session_id:
        block, session_ok = _session_block(cfg, session_id, backend_factory)
        lines += block
        ok = ok and session_ok
    else:
        lines.append("  Identity/scopes:  per session — /memory status in a session, or --session <id>")
    print("\n".join(lines) + "\n")
    return 0 if ok else 1


def setup_command(config: Mapping[str, Any], args: Any = None, *, backend_factory=None) -> int:
    """Ruling R41-7: setup shows and validates the authoritative configuration and writes nothing."""
    print("\n  Authoritative memory is configured explicitly (memory.provider_mode: authoritative).\n"
          "  Setup shows and validates it and changes nothing; edit memory.provider or provider_mode yourself.")
    return status_command(config, args, backend_factory=backend_factory)


def off_command(config: Mapping[str, Any]) -> int:
    """Ruling R41-7: ``off`` never switches authority and writes nothing (exit 2)."""
    print("\n  memory.provider_mode is authoritative: 'hermes memory off' would leave an invalid authoritative\n"
          "  configuration, and it never switches authority. Return to additive memory with\n"
          "  'hermes memory migrate rollback', which also warns that the dormant MEMORY.md/USER.md are stale.\n"
          "  Nothing was changed.\n")
    return 2


def authoritative_only_flags(args: Any) -> Tuple[str, ...]:
    """Flags that mean something only to authoritative memory; additive refuses them (ruling R41-8)."""
    return ("--scope",) if getattr(args, "scope", None) else ()


def reset_command(config: Mapping[str, Any], args: Any) -> int:
    """§9.7 L1620 in wave 6b: an explicit target and scope, then the mandatory approval (§9.5 L1535), which this
    version cannot request, so the reset fails closed before any provider contact (rulings R41-1, R41-8; §9.5 L1538).
    Never a native delete (§9.1 L950). The approved scoped commit is the R41 follow-up's (C6b-5, C6b-2)."""
    from agent.memory_service.status import format_scope_ref, parse_scope_ref
    target = getattr(args, "target", "all")
    if target not in ("memory", "user"):
        print("\n  Authoritative reset names one target: --target memory or --target user. Nothing was reset.\n")
        return 2
    if getattr(args, "yes", False):
        print("\n  --yes cannot waive the mandatory reset approval (§9.5). Nothing was reset.\n")
        return 2
    try:
        scopes = tuple(parse_scope_ref(s) for s in (getattr(args, "scope", None) or ()))
    except ValueError as exc:
        print(f"\n  {exc}. Nothing was reset.\n")
        return 2
    if not scopes or len(set(scopes)) != len(scopes):
        print("\n  Name each scope to reset once with --scope, e.g. --scope repository:<id>. Nothing was reset.\n")
        return 2
    try:
        resolve_memory_service_config(config)
    except MemoryConfigurationError as exc:
        print(f"\n  Invalid authoritative memory configuration: {exc}. Nothing was reset.\n")
        return 1
    needs = (("target_user",) if target == "user" else ()) + ("reset",)   # §9.3 L1336 order; §9.5 L1532, L1535
    print(f"\n  Resetting {target} in {', '.join(format_scope_ref(s) for s in scopes)} needs approval "
          f"({', '.join(needs)}), which this version of Hermes cannot request yet; nothing was staged or changed.\n")
    return 1
