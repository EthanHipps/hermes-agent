"""Cron's curated memory in authoritative mode (§9.7 L1613, L1625; rulings R43-6, R43-7; C6b-8).

A cron run is scheduled, not human-initiated, so it never binds from a directory (§9.3 L1229).
The operator names its scope in ``memory.cron_scope`` (registry IDs; ``{}`` = principal-global);
each run binds one new logical session from those IDs and is handed to the agent as an explicit
dependency. With no scope configured, the run's memory is rejected before any read or write.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional

logger = logging.getLogger(__name__)

CRON_SCOPE_KEY = "cron_scope"
_CRON_PLATFORM = "cron"


def resolve_cron_memory_service(*, session_id: str, raw_config: Optional[Mapping[str, Any]] = None,
                                backend_factory: Optional[Callable[[Any], Any]] = None):
    """``None`` in additive mode (the base path, byte-unchanged); otherwise an explicit service."""
    if raw_config is None:
        from hermes_cli.config import load_config_readonly   # the same source agent init reads

        raw_config = load_config_readonly()
    from agent.memory_service.bootstrap import requests_authoritative_mode

    if not requests_authoritative_mode(raw_config):
        return None
    from agent.memory_service.config import resolve_memory_service_config
    from agent.memory_service.explicit import bind_explicit_service, explicit_scope_from_config
    from agent.memory_service.view import UnboundMemoryService

    config = resolve_memory_service_config(raw_config)          # §9.1 L944: a bad authoritative config raises
    scope = explicit_scope_from_config(raw_config, CRON_SCOPE_KEY)
    if scope is None:
        logger.warning("cron run has no explicit memory identity (memory.%s is unset); curated memory is off",
                       CRON_SCOPE_KEY)
        return UnboundMemoryService(config, surface=_CRON_PLATFORM)
    return bind_explicit_service(raw_config, scope=scope, logical_session_id=session_id, platform=_CRON_PLATFORM,
                                 backend_factory=backend_factory)
