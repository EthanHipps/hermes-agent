"""The memory tool's approval channel in a provider-managed session (R39; §9.4 step 8, §9.5; contract C6b-1).

Inline through the terminal approval callback when one is registered on this
thread. The native gate uses the same callback and the same mapping
(``write_approval._prompt_inline_memory_approval``): ``once``/``session`` approve
this one stage, ``deny`` denies, and anything else is no answer. Otherwise the
approval waits for ``/memory pending`` under the session's persisted identity
(ruling X6b-2 decides in ``mutation._deferrable`` whether it may wait).
"""

from __future__ import annotations

from typing import Optional

from agent.memory_service.approval import ApprovalChannel, ApprovalPrompt

_LABEL = {"memory": "memory", "user": "user profile"}


def _inline_prompt(prompt: ApprovalPrompt) -> Optional[bool]:
    from tools.write_approval import _prompt_inline_memory_approval

    header = f"approve {prompt.intent_kind} on {_LABEL[prompt.target]} ({', '.join(prompt.requirements)})"
    return _prompt_inline_memory_approval(header, prompt.text)


def memory_tool_approval_channel(session_id: Optional[str]) -> ApprovalChannel:
    from tools.skill_provenance import is_background_review
    from tools.terminal_tool import _get_approval_callback
    from tools.write_approval import current_origin

    # Ruling X6b-1: a background-review fork never prompts inline (native evaluate_gate parity);
    # its thread callback is the auto-deny guard, so a prompt there would deny, not defer.
    inline = _get_approval_callback() is not None and not is_background_review()
    return ApprovalChannel(prompt=_inline_prompt if inline else None,
                           session_id=session_id or None, origin=current_origin())
