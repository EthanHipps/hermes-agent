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
