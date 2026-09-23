"""Agent-side session lifecycle for the host-owned MemoryService (R40).

Every function is a no-op for an absent or built-in (additive) service, so the
additive path stays byte-unchanged (§9.10 L1668).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, List, Optional, Tuple

from agent.memory_service.errors import MemoryBlockedError

logger = logging.getLogger(__name__)
_SERVICE_RENDERED = ("provider_authoritative", "stateless")


def _disposition(agent: Any) -> Optional[str]:
    disposition = getattr(getattr(agent, "_memory_service", None), "disposition", None)
    return getattr(disposition, "value", None)


def curated_prompt_parts(agent: Any) -> Optional[List[str]]:
    """The curated region for the system prompt; ``None`` means "use the native path"."""
    disposition = _disposition(agent)
    if disposition not in _SERVICE_RENDERED:
        return None
    if disposition == "stateless":
        return []
    from agent.memory_service.render import render_service_prompt

    service = agent._memory_service
    try:
        render = render_service_prompt(service)
    except MemoryBlockedError:
        prior = getattr(agent, "_curated_prompt_render", None)
        if prior is None or prior[0] != service.identity:
            raise
        render = prior[1]                        # ruling R40-7 (a): reuse the last validated render
    else:
        agent._curated_prompt_render = (service.identity, render)
        agent._curated_fresh_for_next_request = True   # ruling R40-1 (c)
    return [render.text] if render.text else []


def record_curated_prompt(agent: Any, prompt: str) -> None:
    """Remember which prompt this disposition built (ruling R40-4d)."""
    disposition = _disposition(agent)
    if disposition not in _SERVICE_RENDERED or not getattr(agent, "session_id", None):
        return
    from dataclasses import replace

    from agent.memory_service.host_state import load_host_state, save_host_state

    record = load_host_state(agent.session_id)
    if record is not None and record.disposition == disposition:
        save_host_state(replace(record, prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest()))


def curated_prompt_reusable(agent: Any, stored_prompt: str) -> bool:
    """Reuse a stored prompt only if this disposition built exactly those bytes (ruling R40-4d).

    A stored prompt from an additive-era session carries native ``MEMORY.md`` text;
    reusing it in an authoritative or stateless session would inject native data
    (§9.1 L948). A whole-prompt digest needs no heuristic, and the guard is a no-op
    for an additive or absent service.
    """
    disposition = _disposition(agent)
    if disposition not in _SERVICE_RENDERED:
        return True
    from agent.memory_service.errors import BindingInvalidError
    from agent.memory_service.host_state import load_host_state

    try:
        record = load_host_state(agent.session_id)
    except BindingInvalidError:
        return False
    return (record is not None and record.disposition == disposition
            and record.prompt_sha256 == hashlib.sha256(stored_prompt.encode("utf-8")).hexdigest())


def memory_guidance_flags(agent: Any) -> Optional[Tuple[bool, bool]]:
    """The stable-tier memory-tool guidance flags (contract C7; ruling X-2 (c)).

    ``None`` means "additive or absent service": ``_tool_guidance_block`` then reads
    ``agent._memory_enabled``/``agent._user_profile_enabled`` exactly as at
    ``5c583156f7``, so additive bytes are unchanged (§9.10 L1668). A stateless session
    returns ``(False, False)``, for which ``build_memory_guidance`` returns ``""``,
    because its tool refuses (§9.1 L946). The agent flags are never written.
    """
    disposition = _disposition(agent)
    if disposition not in _SERVICE_RENDERED:
        return None
    if disposition == "stateless":
        return (False, False)
    service = agent._memory_service
    return (bool(service.target_enabled("memory")), bool(service.target_enabled("user")))
