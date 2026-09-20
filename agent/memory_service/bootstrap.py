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

import logging
import os
from typing import Optional

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
    directory = working_directory if working_directory is not None else os.getcwd()
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


def _requests_authoritative(raw_config: object) -> bool:
    """Did the operator ask for authoritative mode? Must never raise.

    This decides the ERROR REGIME, not the mode: the mode itself is decided by
    the validated config. A malformed ``memory:`` section in additive mode must
    keep degrading the way it does today (§9.10 first bullet), while a bad
    authoritative config must surface (§9.1 L944).
    """
    try:
        section = raw_config.get("memory")  # type: ignore[union-attr]
        mode = section.get("provider_mode")  # type: ignore[union-attr]
        return isinstance(mode, str) and mode.strip() == "authoritative"
    except Exception:
        return False


def init_memory_service(
    raw_config,
    *,
    logical_session_id: str,
    platform: str,
    store_factory,
    profile_id: Optional[str] = None,
    working_directory: Optional[str] = None,
    backend_factory=None,
):
    """Select the memory service before any native store is constructed.

    Returns ``(service, swallowed_error)``. ``service`` is ``None`` only when an
    additive configuration was too malformed to resolve, in which case the error
    is returned rather than raised so the caller can degrade exactly as it does
    today. In authoritative mode nothing is swallowed: a configuration error, a
    fail-closed provider failure, and a blocked session all propagate, because a
    session never switches mode because of failure (I1).
    """
    from agent.memory_service.config import resolve_memory_service_config
    from agent.memory_service.service import select_memory_service

    authoritative_requested = _requests_authoritative(raw_config)
    try:
        config = resolve_memory_service_config(raw_config)
    except Exception as exc:
        if authoritative_requested:
            raise
        logger.warning("memory configuration is invalid; continuing with built-in memory: %s", exc)
        return None, exc

    context = None
    if config.provider_mode.value == "authoritative":
        context = build_requested_context(
            config,
            logical_session_id=logical_session_id,
            platform=platform,
            profile_id=profile_id,
            working_directory=working_directory,
        )
    return select_memory_service(
        raw_config,
        store_factory=store_factory,
        requested_context=context,
        backend_factory=backend_factory,
    ), None
