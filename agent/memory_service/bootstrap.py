"""Host bootstrap for the memory service (§9.1 L925, §9.2, §9.7 rows 1-2).

Hermes selects the router *before* it constructs, initializes, stats, or reads
its native ``MemoryStore``. This module owns the seam between host facts
(session, platform, profile, working directory) and
:func:`agent.memory_service.service.select_memory_service`.

Hermes does not resolve ``org_id``/``project_id``/``repo_id``: §4.2 makes the
provider the deterministic resolver (registered repo paths, workspace markers,
activated registry revision), and path ties, overlapping registrations and
marker mismatches are hard errors there. Hermes names the canonical directory
and reads the resolved scopes back off the frozen identity the bind returns.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from agent.memory_service import wire as w
from agent.memory_service.config import MemoryServiceConfig

logger = logging.getLogger(__name__)


def build_requested_context(
    config: MemoryServiceConfig,
    *,
    logical_session_id: str,
    platform: str,
    profile_id: Optional[str] = None,
    working_directory: Optional[str] = None,
) -> w.RequestedContext:
    """Build the §9.3 ``RequestedContext`` for a new logical session."""
    if profile_id is None:
        from hermes_cli.profiles import get_active_profile_name

        profile_id = get_active_profile_name()
    from agent.runtime_cwd import resolve_agent_cwd

    directory = working_directory if working_directory is not None else resolve_agent_cwd()
    return w.RequestedContext(
        principal_id=config.principal_id,
        profile_id=profile_id,
        logical_session_id=logical_session_id,
        platform=platform,
        org_id=None,
        project_id=None,
        repo_id=None,
        workspace_id=None,
        resolution_source="directory",
        canonical_directory=os.path.realpath(directory),
    )


def requests_authoritative_mode(raw_config: object) -> bool:
    """Did the operator ask for authoritative mode? Must never raise.

    This decides the ERROR REGIME, not the mode: the mode itself is decided by
    the validated config. A malformed ``memory:`` section in additive mode must
    keep degrading the way it does today (§9.10 first bullet), while a bad
    authoritative config must surface (§9.1 L944). Public: ``_init_memory``
    (agent_init.py) reuses this same predicate to decide whether a failure
    *anywhere* in the additive path (not just config resolution) still
    degrades silently, or must propagate. Home initialization and doctor also
    use this predicate to keep native storage dormant even when validation fails.
    """
    try:
        section = raw_config.get("memory")  # type: ignore[union-attr]
        mode = section.get("provider_mode")  # type: ignore[union-attr]
        return isinstance(mode, str) and mode.strip() == "authoritative"
    except Exception:
        return False


@dataclass(frozen=True)
class SessionBinding:
    """How this Hermes session gets its frozen identity (contract C4)."""

    kind: str                                   # resume | inherit | new | stateless | invalid
    record: Optional["HostStateRecord"]         # noqa: F821 - agent.memory_service.host_state


def _row(session_db: Any, session_id: str) -> Optional[dict]:
    getter = getattr(session_db, "get_session", None)
    if not callable(getter):
        return None
    try:
        return getter(session_id)
    except Exception:
        logger.warning("session lookup failed while resolving the memory binding")
        return None


def _model_config(row: dict) -> dict:
    raw = row.get("model_config")
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw) if isinstance(raw, str) and raw else {}
    except ValueError:
        return {}


def resolve_session_binding(session_id: str, *, session_db: Any = None, hermes_home: Optional[Path] = None) -> SessionBinding:
    """Decide how this Hermes session gets its frozen identity (§4.1 L250, §9.2, §9.3 L1233).

    A record always wins. A branch or compression child inherits its parent's
    record (never a ``_delegate_from`` child: R43). A session that already has
    messages but no record is a continuation without identity: ``invalid``
    (D-R40-1). Anything else is a new logical session.
    """
    from agent.memory_service.host_state import inherit_host_state, load_host_state

    record = load_host_state(session_id, hermes_home=hermes_home)
    if record is not None:
        return SessionBinding("stateless" if record.disposition == "stateless" else "resume", record)
    row = _row(session_db, session_id)
    parent_id = (row or {}).get("parent_session_id")
    if parent_id:
        branched = bool(_model_config(row).get("_branched_from"))
        parent = _row(session_db, parent_id) or {}
        rotated = parent.get("end_reason") == "compression"
        if branched or rotated:
            inherited = inherit_host_state(parent_id, session_id, hermes_home=hermes_home)
            if inherited is not None:
                return SessionBinding("stateless" if inherited.disposition == "stateless" else "inherit", inherited)
    if row is not None and int(row.get("message_count") or 0) > 0:
        return SessionBinding("invalid", None)
    return SessionBinding("new", None)


def open_session_view(config: MemoryServiceConfig, state: "HostSessionState", *, backend_factory=None):  # noqa: F821
    """A second provider transport bound to the same persisted identity.

    Negotiate plus validate: no bind, and no probe load. Used off the foreground
    turn (continuity, ruling R40-6) and later by R43's capability-limited views.
    """
    from agent.memory_service.authoritative import ProviderAuthoritativeMemoryService
    from agent.memory_service.service import _default_backend_factory

    factory = backend_factory or _default_backend_factory(config)
    view = ProviderAuthoritativeMemoryService(config, factory(config))
    try:
        view.resume(state)
    except BaseException:
        view.shutdown()
        raise
    return view


def init_memory_service(
    raw_config,
    *,
    logical_session_id: str,
    platform: str,
    store_factory,
    profile_id: Optional[str] = None,
    working_directory: Optional[str] = None,
    backend_factory=None,
    session_db: Any = None,
    gateway_identity: Any = None,
):
    """Select the memory service before any native store is constructed.

    Returns ``(service, swallowed_error)``. ``service`` is ``None`` only when an
    additive configuration was too malformed to resolve, in which case the error
    is returned rather than raised so the caller can degrade exactly as it does
    today. In authoritative mode nothing is swallowed: a configuration error, a
    fail-closed provider failure, and a blocked session all propagate, because a
    session never switches mode because of failure (I1).

    In authoritative mode the session's frozen identity comes from
    :func:`resolve_session_binding`: a persisted record is resumed or inherited
    and never re-bound (§9.3 L1233), a continuation with no record fails
    ``binding_invalid`` (D-R40-1), and only a genuinely new logical session
    binds. A session with no id at all is unmanaged and binds exactly as it did
    before this resolver existed.
    """
    from agent.memory_service.config import MemoryConfigurationError, resolve_memory_service_config
    from agent.memory_service.service import select_memory_service

    authoritative_requested = requests_authoritative_mode(raw_config)
    try:
        config = resolve_memory_service_config(raw_config)
    except MemoryConfigurationError as exc:
        if authoritative_requested:
            raise
        logger.warning("memory configuration is invalid; continuing with built-in memory: %s", exc)
        return None, exc

    if config.provider_mode.value != "authoritative":
        return select_memory_service(raw_config, store_factory=store_factory, backend_factory=backend_factory), None

    from agent.memory_service.errors import MemoryBlockedError
    from agent.memory_service.host_state import HostStateRecord, save_host_state
    from agent.memory_service.service import FailurePolicy, StatelessMemoryService

    binding = (
        resolve_session_binding(logical_session_id, session_db=session_db)
        if logical_session_id else SessionBinding("new", None)
    )

    if gateway_identity is not None:
        # Ruling R41-5 (I3; §4.1 L252, §13.1 L2191; contract C6b-12): a gateway user's principal comes only
        # from an explicit mapping. Unknown, ambiguous, or different from the principal a resumed record froze:
        # no memory and no provider contact. A malformed table is a configuration error (§9.1 L946).
        # Runs after R43's cron/subagent guard, which precedes the binding statement.
        from dataclasses import replace as _replace

        from agent.memory_service.principal import UnmappedPrincipalMemoryService, map_gateway_principal

        principal = map_gateway_principal(raw_config, gateway_identity)
        frozen = binding.record.state.identity.principal_id if binding.kind in ("resume", "inherit") else None
        if principal is None or (frozen is not None and frozen != principal):
            if logical_session_id and binding.record is None:
                # Agents built without the user id (gateway hygiene, manual /compress) then resume a
                # stateless record instead of binding or failing binding_invalid (D-R40-1).
                save_host_state(HostStateRecord(logical_session_id, "stateless", None))
            return UnmappedPrincipalMemoryService(config), None     # an existing record is kept
        config = _replace(config, principal_id=principal)

    if binding.kind == "stateless":
        # §9.6 L1582 and I1: a session that started stateless stays stateless,
        # across a process restart included. The provider is not contacted.
        return StatelessMemoryService(config, reason="this session started stateless"), None

    if binding.kind == "invalid":
        if config.failure_policy is FailurePolicy.STATELESS:
            save_host_state(HostStateRecord(logical_session_id, "stateless", None))
            return StatelessMemoryService(config, reason="binding_invalid: no persisted host session state"), None
        raise MemoryBlockedError(
            "binding_invalid: this session has no persisted memory identity; start a new session",
            code="binding_invalid",
        )

    if binding.kind in ("resume", "inherit"):
        service = select_memory_service(
            raw_config, store_factory=store_factory, session_state=binding.record.state,
            backend_factory=backend_factory,
        )
    else:
        service = select_memory_service(
            raw_config,
            store_factory=store_factory,
            requested_context=build_requested_context(
                config, logical_session_id=logical_session_id, platform=platform,
                profile_id=profile_id, working_directory=working_directory,
            ),
            backend_factory=backend_factory,
        )

    if logical_session_id:
        _persist_binding(logical_session_id, service, prior=binding.record)
    return service, None


def _persist_binding(logical_session_id: str, service, *, prior) -> None:
    """Record the disposition this session settled on (§4.1 L250; ruling R40-4a).

    ``prompt_sha256`` carries forward only while the disposition AND the frozen
    identity are unchanged, which is what makes ruling R40-4d's exact-match reuse
    survive a resume; anything else leaves it unset so the prompt is rebuilt once.
    """
    from agent.memory_service.host_state import HostStateRecord, save_host_state
    from agent.memory_service.service import MemoryDisposition

    if service.disposition is MemoryDisposition.AUTHORITATIVE:
        state = service.session_state
        digest = (prior.prompt_sha256 if prior is not None
                  and prior.disposition == "provider_authoritative" and prior.state == state else None)
        record = HostStateRecord(logical_session_id, "provider_authoritative", state, prompt_sha256=digest)
    elif service.disposition is MemoryDisposition.STATELESS:
        record = HostStateRecord(logical_session_id, "stateless", None)
    else:
        return
    if prior is not None and prior == record:
        return
    save_host_state(record)
