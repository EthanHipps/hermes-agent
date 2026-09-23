"""S6 identity: exactly one MemoryService snapshot in the system prompt.

§9.7 L1596 (one structured snapshot, never concatenated with native data), §9.1
L948, D-R40-4/5/8, rulings R40-9 (a) and X-2 (c).
"""

import hashlib

import pytest

from agent.memory_service.host_state import load_host_state
from agent.memory_service.render import REGION_HEADING
from tests.agent.memory_service.native_sentinel import native_memory_sentinel
from tests.agent.memory_service.test_s6_identity_lifecycle import (  # noqa: F401 - fixtures
    PG_SCOPE, REPO_SCOPE, agent_env, agent_env_factory, native_dir,
)


def test_authoritative_prompt_carries_exactly_one_structured_region(agent_env):
    """§9.7 L1596: one MemoryService snapshot; the general packet once (D-R40-5)."""
    agent_env.store.seed_general(REPO_SCOPE, "GENERAL-POLICY", policy_key="k1")
    agent_env.store.seed_record(REPO_SCOPE, "memory", "MEMORY-FACT")
    agent_env.store.seed_record(PG_SCOPE, "user", "USER-FACT", lane="trusted_instruction")
    prompt = agent_env.agent._build_system_prompt(None)
    assert prompt.count(REGION_HEADING) == 1 and prompt.count("GENERAL-POLICY") == 1
    assert prompt.index("GENERAL-POLICY") < prompt.index("MEMORY-FACT") < prompt.index("USER-FACT")


def test_dormant_native_files_never_reach_the_prompt(agent_env, native_dir):
    """§9.1 L948: never concatenated with native data; proven with the sentinel."""
    native_dir.mkdir(parents=True, exist_ok=True)
    (native_dir / "MEMORY.md").write_text("NATIVE-SECRET-FACT\n", encoding="utf-8")
    agent_env.store.seed_record(REPO_SCOPE, "memory", "MEMORY-FACT")
    with native_memory_sentinel(native_dir) as sentinel:
        prompt = agent_env.agent._build_system_prompt(None)
    sentinel.assert_untouched()
    assert "NATIVE-SECRET-FACT" not in prompt and "MEMORY-FACT" in prompt


def test_disabled_user_target_is_never_loaded_or_rendered(agent_env_factory):
    """D-R40-4 / §9.1 L943: a disabled target is never loaded, rendered or gated."""
    env = agent_env_factory(user_profile_enabled=False)
    env.store.seed_record(REPO_SCOPE, "memory", "MEMORY-FACT")
    env.store.seed_record(PG_SCOPE, "user", "USER-FACT")
    loads_before = env.count("load_curated")
    prompt = env.agent._build_system_prompt(None)
    assert "MEMORY-FACT" in prompt and "USER-FACT" not in prompt
    assert env.count("load_curated") - loads_before == 1        # memory only


def test_stateless_session_renders_no_region_and_loads_nothing(agent_env_factory):
    """D-R40-8: a stateless session renders no curated block and runs no load."""
    from agent.memory_service.host_state import HostStateRecord, save_host_state
    from agent.memory_service.service import MemoryDisposition

    env = agent_env_factory(session_id="never-built")
    save_host_state(HostStateRecord("stateless-1", "stateless", None))
    agent = env.build_agent("stateless-1")
    assert agent._memory_service.disposition is MemoryDisposition.STATELESS
    loads_before = env.count("load_curated")
    prompt = agent._build_system_prompt(None)
    assert REGION_HEADING not in prompt
    assert env.count("load_curated") == loads_before


def test_additive_prompt_path_is_unchanged(tmp_path, monkeypatch):
    """§9.10 L1668: with no provider_mode, _memory_parts is the native store's blocks."""
    from types import SimpleNamespace

    from agent.system_prompt import _memory_parts

    class _Store:
        def format_for_system_prompt(self, kind):
            return f"NATIVE-{kind}-BLOCK"

    agent = SimpleNamespace(_memory_store=_Store(), _memory_enabled=True, _user_profile_enabled=True,
                            _memory_manager=None, _memory_service=None)
    assert _memory_parts(agent) == ["NATIVE-memory-BLOCK", "NATIVE-user-BLOCK"]


def test_first_build_blocked_ends_the_turn_before_any_model_call(agent_env):
    """§9.6 L1570 / ruling R40-9: typed result, client never called, no user row persisted."""
    agent_env.store.fail_transport("load_curated", times=2)
    result = agent_env.agent.run_conversation("hello")
    assert result["failed"] and result.get("turn_exit_reason") == "curated_memory_blocked"
    agent_env.agent.client.chat.completions.create.assert_not_called()
    rows = agent_env.db.get_messages(agent_env.agent.session_id)
    assert not [r for r in rows if r.get("role") == "user"]


def test_blocked_turn_text_is_content_free(agent_env):
    """§8.6 L892 / §9.3 L987: no seeded text and no provider token in the result."""
    agent_env.store.seed_record(REPO_SCOPE, "memory", "SECRET-MEMORY-FACT")
    state = agent_env.agent._memory_service.session_state
    agent_env.store.fail_transport("load_curated", times=2)
    result = agent_env.agent.run_conversation("hello")
    assert result.get("turn_exit_reason") == "curated_memory_blocked"
    text = result["final_response"]
    assert "SECRET-MEMORY-FACT" not in text
    assert state.provider_epoch not in text
    assert state.identity.opaque_binding_b64url not in text
    assert state.identity.binding_revision not in text


def test_prompt_build_records_the_prompt_digest(agent_env):
    prompt = agent_env.agent._build_system_prompt(None)
    assert load_host_state(agent_env.agent.session_id).prompt_sha256 == hashlib.sha256(
        prompt.encode("utf-8")).hexdigest()


def test_additive_prompt_build_writes_no_host_state(tmp_path, monkeypatch):
    """§9.10 L1668: record_curated_prompt is a no-op without a provider-managed service."""
    from types import SimpleNamespace

    from agent.memory_service.host_state import host_state_dir
    from agent.memory_service.lifecycle import record_curated_prompt

    record_curated_prompt(SimpleNamespace(_memory_service=None, session_id="s-1"), "prompt")
    assert not host_state_dir().exists()


# ---- ruling X-2 (c): the memory-tool guidance in the stable tier ------------------------------


def test_authoritative_prompt_keeps_the_memory_tool_guidance(agent_env):
    """X-2 (c): the guidance follows service.target_enabled, not the agent flags, which stay False."""
    from agent.memory_service.lifecycle import memory_guidance_flags

    agent = agent_env.agent
    assert agent._memory_enabled is False and agent._user_profile_enabled is False
    assert memory_guidance_flags(agent) == (True, True)
    assert "memory" in agent._build_system_prompt(None).lower()   # assert the relationship, not the copy


def test_disabled_user_target_narrows_the_guidance(agent_env_factory):
    """user_profile_enabled: false in the memory config -> (True, False)."""
    from agent.memory_service.lifecycle import memory_guidance_flags

    env = agent_env_factory(user_profile_enabled=False)
    assert memory_guidance_flags(env.agent) == (True, False)


def test_disabled_memory_target_narrows_the_guidance(agent_env_factory):
    from agent.memory_service.lifecycle import memory_guidance_flags

    env = agent_env_factory(memory_enabled=False)
    assert memory_guidance_flags(env.agent) == (False, True)


def test_stateless_session_gets_no_memory_tool_guidance(agent_env_factory):
    """(False, False); build_memory_guidance returns "" and the tool refuses (§9.1 L946)."""
    from agent.memory_service.host_state import HostStateRecord, save_host_state
    from agent.memory_service.lifecycle import memory_guidance_flags
    from agent.prompt_builder import build_memory_guidance

    env = agent_env_factory(session_id="never-built")
    save_host_state(HostStateRecord("stateless-2", "stateless", None))
    agent = env.build_agent("stateless-2")
    assert memory_guidance_flags(agent) == (False, False)
    assert build_memory_guidance(False, False) == ""


def test_additive_tool_guidance_is_unchanged(tmp_path):
    """§9.10 L1668: memory_guidance_flags returns None, so _tool_guidance_block reads the agent flags."""
    from types import SimpleNamespace

    from agent.memory_service.lifecycle import memory_guidance_flags
    from agent.prompt_builder import build_memory_guidance
    from agent.system_prompt import _tool_guidance_block

    for memory_on, user_on in ((True, True), (True, False), (False, True), (False, False)):
        agent = SimpleNamespace(_memory_service=None, valid_tool_names={"memory"},
                                _memory_enabled=memory_on, _user_profile_enabled=user_on,
                                _kanban_worker_guidance=None)
        assert memory_guidance_flags(agent) is None
        expected = build_memory_guidance(memory_on, user_on, skill_manage_available=False) or None
        assert _tool_guidance_block(agent) == expected


def test_guidance_is_absent_when_the_memory_tool_is_not_loaded(agent_env_factory):
    """The guidance stays gated on the tool being present, exactly as at 5c583156f7."""
    from agent.system_prompt import _tool_guidance_block

    env = agent_env_factory(tools=("web_search",))
    assert "memory" not in env.agent.valid_tool_names
    block = _tool_guidance_block(env.agent)
    assert block is None or "persistent memory" not in block
