"""S6 identity: a fresh successful load per model request per enabled target.

§9.6 L1564 (each model request), L1570/L1573 (block before the request, stay
blocked until a fresh load succeeds), L1575 (ambiguous policy delivers no partial
memory), I1 (recovery never switches mode). Rulings R40-1 (c), R40-2 (a),
R40-9 (a); D-R40-9.

The scripted tool round deliberately uses ``web_search`` rather than ``memory``,
so these tests stay independent of R38's dispatch when this branch merges ``main``
after F-R38 (reconciliation §4, after-merge step 3).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.memory_service.errors import MemoryBlockedError
from tests.agent.memory_service.test_s6_identity_lifecycle import (  # noqa: F401 - fixtures
    PG_SCOPE, REPO_SCOPE, agent_env, agent_env_factory, native_dir,
)

LOAD = "load"
MODEL = "model"


def _response(content="Done", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=None, reasoning=None)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], model="test/model")
    response.usage = None
    return response


def _tool_call():
    return SimpleNamespace(id="tc1", type="function",
                           function=SimpleNamespace(name="web_search", arguments='{"query":"test"}'))


def _record_events(agent):
    """Script a two-request turn and interleave loads and model calls in one list.

    Returns ``(events, load_patch)``. ``load_patch`` wraps the real
    ``load_curated``, so the provider still answers exactly as it would in
    production; only the ordering is observed.
    """
    events = []
    real_load = type(agent._memory_service).load_curated

    def _load(service, target):
        events.append((LOAD, target))
        return real_load(service, target)

    responses = [_response(content=None, tool_calls=[_tool_call()]), _response("Done")]

    def _create(*args, **kwargs):
        events.append((MODEL, sum(1 for kind, _ in events if kind == MODEL)))
        return responses.pop(0)

    agent.client.chat.completions.create.side_effect = _create
    return events, _load


def _two_request_turn(agent, load_patch, message="hello"):
    with (
        patch.object(type(agent._memory_service), "load_curated", load_patch),
        patch("model_tools.handle_function_call", return_value="search result"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation(message)


def test_every_model_request_is_preceded_by_a_fresh_load_per_enabled_target(agent_env):
    """§9.6 L1564: between consecutive model calls there is >= 1 load per enabled target."""
    agent = agent_env.agent
    agent_env.store.seed_record(REPO_SCOPE, "memory", "MEMORY-FACT")
    events, load_patch = _record_events(agent)
    result = _two_request_turn(agent, load_patch)
    assert result.get("completed") is True

    model_positions = [i for i, (kind, _) in enumerate(events) if kind == MODEL]
    assert len(model_positions) == 2, events
    # Ruling R40-1 (c): the load that rendered the prompt satisfies the FIRST request,
    # so the contract is stated between consecutive requests, not as an absolute count.
    between = {target for kind, target in events[model_positions[0] + 1:model_positions[1]] if kind == LOAD}
    assert between == {"memory", "user"}, events
    # And a load precedes the first request too (the render-time load).
    assert any(kind == LOAD for kind, _ in events[:model_positions[0]]), events


def test_a_render_time_load_satisfies_only_one_request(agent_env):
    """Ruling R40-1 (c): the dedupe is one-shot, never a standing exemption."""
    from agent.memory_service.lifecycle import curated_request_gate

    agent = agent_env.agent
    agent._build_system_prompt(None)
    assert agent._curated_fresh_for_next_request is True
    loads = agent_env.count("load_curated")
    assert curated_request_gate(agent) is None
    assert agent_env.count("load_curated") == loads          # satisfied by the render
    assert curated_request_gate(agent) is None
    assert agent_env.count("load_curated") == loads + 2      # both targets, fresh


def test_provider_failure_mid_turn_blocks_the_next_request(agent_env):
    """§9.6 L1573: the second model call never happens; the turn fails curated_memory_blocked."""
    agent = agent_env.agent
    agent._build_system_prompt(None)
    events, load_patch = _record_events(agent)

    armed = {"done": False}
    real_dispatch = "search result"

    def _arm(*args, **kwargs):
        if not armed["done"]:
            agent_env.store.fail_transport("load_curated", times=2)
            armed["done"] = True
        return real_dispatch

    with (
        patch.object(type(agent._memory_service), "load_curated", load_patch),
        patch("model_tools.handle_function_call", side_effect=_arm),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")

    assert result["failed"] and result.get("turn_exit_reason") == "curated_memory_blocked"
    assert len([e for e in events if e[0] == MODEL]) == 1


def test_recovery_needs_only_a_fresh_load_and_never_switches_mode(agent_env):
    """I1: blocked, then the fault clears, then the next turn proceeds on the same identity."""
    from agent.memory_service.lifecycle import curated_request_gate
    from agent.memory_service.service import MemoryDisposition

    agent = agent_env.agent
    identity = agent._memory_service.identity
    agent._build_system_prompt(None)
    agent._curated_fresh_for_next_request = False
    # One fault is enough: the gate loads "memory" first, and that failure blocks the
    # request before "user" is reached, so exactly one fault is consumed.
    agent_env.store.fail_transport("load_curated", times=1)
    assert curated_request_gate(agent) is not None
    assert agent._memory_service.disposition is MemoryDisposition.AUTHORITATIVE
    assert agent._memory_service.blocked is True
    # The fault queue is drained; the next gate load succeeds and clears the latch.
    assert curated_request_gate(agent) is None
    assert agent._memory_service.blocked is False
    assert agent._memory_service.identity == identity


def test_ambiguous_policy_blocks_with_no_partial_memory(agent_env):
    """D-R40-9 / §9.6 L1575: an ambiguous policy answer delivers nothing."""
    from agent.memory_service.lifecycle import curated_request_gate

    agent = agent_env.agent
    agent_env.store.seed_record(REPO_SCOPE, "memory", "SECRET-MEMORY-FACT")
    agent._build_system_prompt(None)
    agent._curated_fresh_for_next_request = False
    agent_env.store.add_policy_conflict("k1", 2)
    message = curated_request_gate(agent)
    assert message is not None
    assert "SECRET-MEMORY-FACT" not in message
    assert agent._memory_service.blocked is True


def test_disabled_target_is_never_gated(agent_env_factory):
    """D-R40-4 / §9.1 L943: a disabled target is never loaded by the gate."""
    from agent.memory_service.lifecycle import curated_request_gate

    env = agent_env_factory(user_profile_enabled=False)
    agent = env.agent
    agent._curated_fresh_for_next_request = False
    loads = env.count("load_curated")
    assert curated_request_gate(agent) is None
    assert env.count("load_curated") == loads + 1


def test_stateless_session_runs_no_gate_load(agent_env_factory):
    """D-R40-8: a stateless session runs no gate load at all."""
    from agent.memory_service.host_state import HostStateRecord, save_host_state
    from agent.memory_service.lifecycle import curated_request_gate

    env = agent_env_factory()
    save_host_state(HostStateRecord("stateless-gate", "stateless", None))
    agent = env.build_agent("stateless-gate")
    loads = env.count("load_curated")
    assert curated_request_gate(agent) is None
    assert env.count("load_curated") == loads


def test_additive_session_runs_no_gate_load(tmp_path):
    """§9.10 L1668: the gate is a no-op for an additive or absent service."""
    from agent.memory_service.lifecycle import curated_request_gate

    assert curated_request_gate(SimpleNamespace(_memory_service=None)) is None


def test_gate_block_text_is_content_free(agent_env):
    """§8.6 L892 / §9.3 L987: no seeded text and no provider token in the block message."""
    from agent.memory_service.lifecycle import curated_request_gate

    agent = agent_env.agent
    agent_env.store.seed_record(REPO_SCOPE, "memory", "SECRET-MEMORY-FACT")
    state = agent._memory_service.session_state
    agent._build_system_prompt(None)
    agent._curated_fresh_for_next_request = False
    agent_env.store.fail_transport("load_curated", times=2)
    message = curated_request_gate(agent)
    assert message is not None
    assert "SECRET-MEMORY-FACT" not in message
    assert state.provider_epoch not in message
    assert state.identity.opaque_binding_b64url not in message
    assert state.identity.binding_revision not in message


def test_the_gate_never_rerenders_the_prompt(agent_env):
    """Ruling R40-2 (a): the per-request load validates; it never changes prompt bytes."""
    from agent.memory_service.lifecycle import curated_request_gate

    agent = agent_env.agent
    agent_env.store.seed_record(REPO_SCOPE, "memory", "FIRST-FACT")
    prompt = agent._build_system_prompt(None)
    agent._cached_system_prompt = prompt
    agent_env.store.seed_record(REPO_SCOPE, "memory", "SECOND-FACT")
    agent._curated_fresh_for_next_request = False
    assert curated_request_gate(agent) is None
    assert agent._cached_system_prompt == prompt
    assert "SECOND-FACT" not in agent._cached_system_prompt


def test_an_out_of_turn_render_never_satisfies_a_later_request(agent_env):
    """66-V2 (b) / R40-1 (c): a render between turns is not a load made for the next request.

    CLI ``/context`` and the TUI context panel render the live agent's prompt parts; the
    binding is then revoked. The next turn reuses its cached prompt, so nothing renders for
    its first request, which must therefore run its own load and be blocked (§9.6 L1573,
    L1580), never sent on the stale render.
    """
    from agent.context_breakdown import compute_session_context_breakdown

    agent = agent_env.agent
    agent.client.chat.completions.create.return_value = _response("first answer")
    first = agent.run_conversation("first question")
    assert first.get("completed") is True and agent._cached_system_prompt
    compute_session_context_breakdown(agent, first["messages"])        # /context, out of turn
    agent_env.store.revoke(agent._memory_service.session_state.identity.opaque_binding_b64url)
    agent.client.chat.completions.create.reset_mock()

    result = agent.run_conversation("second question", conversation_history=first["messages"])

    assert result["failed"] and result.get("turn_exit_reason") == "curated_memory_blocked"
    agent.client.chat.completions.create.assert_not_called()


def test_a_blocked_gate_raises_nothing_at_the_service_boundary(agent_env):
    """The gate answers with a message; MemoryBlockedError never escapes it."""
    from agent.memory_service.lifecycle import curated_request_gate

    agent = agent_env.agent
    agent._curated_fresh_for_next_request = False
    agent_env.store.fail_transport("load_curated", times=4)
    try:
        assert curated_request_gate(agent) is not None
    except MemoryBlockedError:  # pragma: no cover - the contract is that this does not happen
        pytest.fail("curated_request_gate must not raise MemoryBlockedError")
