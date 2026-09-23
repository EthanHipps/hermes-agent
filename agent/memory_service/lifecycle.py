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

#: Ruling R40-4c (a), decided at Checkpoint A: /new binds a fresh logical session, so the prior
#: handle stays live and the old session stays resumable with its memory. The live-handle cost is
#: recorded as cross-repo obligation K-2 (ygg has no end operation in v1 and caps handles at 4,096).
NEW_SESSION_BIND_INTENT = "new_session"


def _disposition(agent: Any) -> Optional[str]:
    disposition = getattr(getattr(agent, "_memory_service", None), "disposition", None)
    return getattr(disposition, "value", None)


def _no_native_store():
    raise AssertionError("a non-additive session never builds the native store (§9.1 L925)")


def ensure_session_binding(agent: Any, conversation_history: Optional[list] = None) -> None:
    """Re-resolve the frozen identity when ``agent.session_id`` moved (ruling R40-4b).

    Same session: no-op. Same identity (branch/compression child): keep the service,
    record already inherited. Otherwise resume, bind, stay stateless, or raise
    ``MemoryBlockedError(code="binding_invalid")`` (D-R40-1).

    ``conversation_history`` is accepted but unused: the resolver decides
    "continuation" from the SessionDB ``message_count`` alone, identically at agent
    init (where no history exists yet) and at turn start. A REST caller that holds
    its own history with no SessionDB row is R42's surface.
    """
    if _disposition(agent) not in _SERVICE_RENDERED:
        return
    session_id = getattr(agent, "session_id", None)
    if not session_id or getattr(agent, "_memory_session_key", None) == session_id:
        return
    from agent.memory_service.bootstrap import init_memory_service, resolve_session_binding

    current = agent._memory_service
    binding = resolve_session_binding(session_id, session_db=getattr(agent, "_session_db", None))
    if binding.kind in ("resume", "inherit") and binding.record.state == getattr(current, "session_state", None):
        agent._memory_session_key = session_id
        return
    if binding.kind == "invalid" and current.config.failure_policy.value != "stateless":
        raise MemoryBlockedError("binding_invalid: this session has no persisted memory identity; start a new session",
                                 code="binding_invalid")
    replacement, _ = init_memory_service(
        agent._memory_boot_config, logical_session_id=session_id, platform=getattr(agent, "platform", None) or "cli",
        store_factory=_no_native_store, session_db=getattr(agent, "_session_db", None),
    )
    agent._memory_service, agent._memory_session_key = replacement, session_id
    agent._curated_prompt_render = None
    current.shutdown()


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


def curated_request_gate(agent: Any) -> Optional[str]:
    """A fresh successful load per enabled target before this model request (§9.6 L1564).

    Returns ``None`` to proceed or a content-free block message. Ruling R40-2 (a):
    the load gates; it never re-renders the byte-stable prompt. Ruling R40-1 (c):
    a load made while rendering the prompt for this same request satisfies it once.
    """
    if _disposition(agent) != "provider_authoritative":
        return None
    if getattr(agent, "_curated_fresh_for_next_request", False):
        agent._curated_fresh_for_next_request = False          # ruling R40-1 (c)
        return None
    service = agent._memory_service
    try:
        for target in ("memory", "user"):
            if service.target_enabled(target):
                service.load_curated(target)
    except MemoryBlockedError:
        logger.warning("model request blocked: curated memory unavailable")
        return ("Curated memory is unavailable, so this request was not sent to the model. "
                + (service.degraded_warning() or ""))
    return None


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
