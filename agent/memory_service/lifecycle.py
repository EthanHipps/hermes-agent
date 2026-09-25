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


def _no_native_store():
    raise AssertionError("a non-additive session never builds the native store (§9.1 L925)")


def ensure_session_binding(agent: Any, conversation_history: Optional[list] = None) -> None:
    """Turn start: scope the fresh-load flag to this turn, then ``follow_session_binding``.

    ``conversation_history`` is accepted but unused: the resolver decides
    "continuation" from the SessionDB ``message_count`` alone, identically at agent
    init (where no history exists yet) and at turn start. A REST caller that holds
    its own history with no SessionDB row is R42's surface.
    """
    if _disposition(agent) not in _SERVICE_RENDERED:
        return
    # Ruling R40-1 (c): only a load made while rendering the prompt FOR a request satisfies
    # it. A render outside this turn (/context, a TUI prompt persist, manual /compress, a turn
    # that ended after a preflight compression) is not; this turn's own prompt build or
    # compaction re-arms the flag (§9.6 L1564, L1573).
    agent._curated_fresh_for_next_request = False
    follow_session_binding(agent)


def follow_session_binding(agent: Any) -> None:
    """Re-resolve the frozen identity when ``agent.session_id`` moved (ruling R40-4b).

    Called lazily before anything uses the service for a session: at turn start, before
    every prompt render (``build_system_prompt_parts``) and before any compression state
    exists (``prepare_compression``), so an out-of-turn render or ``/compress`` right after
    ``/resume`` or ``/branch`` never serves the new session from the previous one's service.

    Same session: no-op. Same identity (branch/compression child): keep the service,
    record already inherited. Otherwise resume, bind, stay stateless, or raise
    ``MemoryBlockedError(code="binding_invalid")`` (D-R40-1).

    Ruling R40-4c (a): a genuinely new session binds with ``bind_intent="new_session"``
    — ``ProviderAuthoritativeMemoryService.start``'s default — so the prior handle
    stays live and the old session stays resumable with its memory. The live-handle
    cost is cross-repo obligation K-2: v1 has no host end operation (D-R10-2) and ygg
    caps handles at 4,096, so R28 budgets or closes them.
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
        agent._curated_fresh_for_next_request = False  # ... and the gate blocks the next request
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


def configure_compaction_text(compressor: Any, service: Any) -> None:
    """Ruling R40-5 (a): only the built-in compressor, only non-additive sessions.

    An external context engine owns its own compaction text, and an additive session
    keeps SUMMARY_PREFIX and _COMPRESSION_NOTE byte-for-byte (§9.10 L1668).
    """
    from agent.context_compressor import (
        CURATED_MEMORY_COMPRESSION_NOTE, CURATED_MEMORY_SUMMARY_PREFIX, ContextCompressor,
    )

    disposition = getattr(getattr(service, "disposition", None), "value", None)
    if isinstance(compressor, ContextCompressor) and disposition in _SERVICE_RENDERED:
        compressor.summary_prefix = CURATED_MEMORY_SUMMARY_PREFIX
        compressor.compression_note = CURATED_MEMORY_COMPRESSION_NOTE


def prepare_compression(agent: Any) -> bool:
    """Before any compression state exists: follow a session switch, then hold a validated render.

    Ruling R40-4b: a compression outside a turn (manual ``/compress`` right after ``/resume``
    or ``/branch``) must use the session's own service, so the switch is detected here —
    the boundary hook would otherwise carry the previous session's service onto this one.

    The boundary rebuild runs inside the commit fence, where a raise would leave compression
    half-done, so ruling R40-7 (a) reuses the last validated render when that refresh fails.
    A session whose stored prompt was reused (R40-4d) has rendered nothing in this process —
    ``reconstruct_static_prefix`` renders only on prompt-caching routes — so render once now.
    If either step fails, ``False`` tells the caller not to begin: nothing is committed, and
    the next request's gate blocks (§9.6 L1573).
    """
    if _disposition(agent) not in _SERVICE_RENDERED:
        return True
    try:
        follow_session_binding(agent)
        if _disposition(agent) != "provider_authoritative":
            return True
        service = agent._memory_service
        prior = getattr(agent, "_curated_prompt_render", None)
        if prior is not None and prior[0] == service.identity:
            return True
        from agent.memory_service.render import render_service_prompt

        agent._curated_prompt_render = (service.identity, render_service_prompt(service))
    except MemoryBlockedError:
        logger.warning("compression not started: curated memory unavailable")
        return False
    return True


def on_compression_boundary(agent: Any, *, compressed: list, old_session_id: Optional[str],
                            session_commit_succeeded: bool) -> None:
    """D-R40-2 / D-R40-3: carry the identity to a rotated id; then continuity.

    Compression never binds and never reads the working directory: the record is
    copied to the child session id with the frozen identity unchanged (§9.3 L1233).
    An aborted commit carries nothing.
    """
    if _disposition(agent) not in _SERVICE_RENDERED or not session_commit_succeeded:
        return
    from agent.memory_service.host_state import inherit_host_state

    if old_session_id and old_session_id != agent.session_id:
        inherit_host_state(old_session_id, agent.session_id)
    agent._memory_session_key = agent.session_id

    service = agent._memory_service
    if _disposition(agent) == "provider_authoritative" and service.capabilities.capture_continuity:
        text = compression_summary_text(compressed)
        if text:
            agent._memory_continuity_thread = submit_continuity_async(service.config, service.session_state, text)


def compression_summary_text(compressed: list) -> Optional[str]:
    """The committed handoff body, prefix stripped; ``None`` when the engine wrote none."""
    from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY, ContextCompressor

    for message in compressed:
        if isinstance(message, dict) and message.get(COMPRESSED_SUMMARY_METADATA_KEY):
            content = message.get("content")
            text = content if isinstance(content, str) else ""
            body = ContextCompressor._strip_summary_prefix(text)
            return body or None
    return None


def submit_continuity_async(config: Any, state: Any, text: str, *, backend_factory=None) -> "threading.Thread":
    """Ruling R40-6 (a): a second transport from the persisted state; content-free and non-blocking.

    The foreground service instance is documented as not thread-safe, so this never
    touches it: it opens its own transport (negotiate plus validate, no bind, no probe
    load) and sends exactly one capture_continuity. No retry and no pre-truncation —
    the provider answers ``limit_exceeded`` — and every failure is logged without the
    summary body or any provider token (§9.3 L1446, §8.6 L892).
    """
    import threading
    import uuid

    from agent.memory_service.service import ContinuityCapture

    def _capture() -> None:
        from agent.memory_service.bootstrap import open_session_view

        try:
            view = open_session_view(config, state, backend_factory=backend_factory)
            try:
                view.capture_continuity(ContinuityCapture(request_id=uuid.uuid4().hex, kind="compression_snapshot",
                                                          text=text, initiating_surface="compression"))
            finally:
                view.shutdown()
        except Exception as exc:  # §9.3 L1446: every failure is content-free and non-blocking
            logger.warning("compression continuity capture failed (%s%s)", type(exc).__name__,
                           f": {exc.code}" if getattr(exc, "code", None) else "")

    thread = threading.Thread(target=_capture, name="memory-continuity", daemon=True)
    thread.start()
    return thread


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
