"""S6 identity: compression text, retention and identity across rotation.

§9.7 L1609 (compression prompt and retention without MEMORY.md/USER.md wording),
§9.7 L1597 (reuse or refresh with the same identity), §9.7 L1610 (notify without
changing identity). Rulings R40-5 (a) and R40-7 (a); D-R40-2, D-R40-3.

The four frozen historical recognizers that carry the native memory clause are
never edited: they are recognizers, not emitted text, and editing them would break
resume stripping. The curated prefix is a LIVE variant registered beside
SUMMARY_PREFIX.
"""

import pytest

from agent.context_compressor import (
    CURATED_MEMORY_COMPRESSION_NOTE, CURATED_MEMORY_SUMMARY_PREFIX, SUMMARY_PREFIX,
    _HISTORICAL_SUMMARY_PREFIXES, ContextCompressor,
)
from tests.agent.memory_service.native_sentinel import native_memory_sentinel
from tests.agent.memory_service.test_s6_identity_lifecycle import (  # noqa: F401 - fixtures
    PG_SCOPE, REPO_SCOPE, _session, agent_env, agent_env_factory, native_dir,
)

NATIVE_CLAUSE = "Your persistent memory (MEMORY.md, USER.md)"
CURATED_CLAUSE = "Your curated memory"


def test_curated_prefix_differs_from_the_live_prefix_only_in_the_memory_clause():
    assert CURATED_MEMORY_SUMMARY_PREFIX.replace(CURATED_CLAUSE, NATIVE_CLAUSE) == SUMMARY_PREFIX
    assert "MEMORY.md" not in CURATED_MEMORY_SUMMARY_PREFIX and "USER.md" not in CURATED_MEMORY_SUMMARY_PREFIX
    assert "MEMORY.md" not in CURATED_MEMORY_COMPRESSION_NOTE and "USER.md" not in CURATED_MEMORY_COMPRESSION_NOTE


def test_curated_note_differs_from_the_live_note_only_in_the_memory_clause():
    assert CURATED_MEMORY_COMPRESSION_NOTE.replace(CURATED_CLAUSE, NATIVE_CLAUSE) == ContextCompressor._COMPRESSION_NOTE


@pytest.mark.parametrize("prefix", [SUMMARY_PREFIX, CURATED_MEMORY_SUMMARY_PREFIX])
def test_both_live_prefixes_keep_every_handoff_invariant(prefix):
    lower = prefix.lower()
    assert "topic overlap" in lower and "if no user message appears after this summary" in lower
    assert "must never become the active turn" in lower and "tools remain fully active" in lower


def test_curated_prefix_is_recognized_and_stripped_but_never_frozen():
    content = CURATED_MEMORY_SUMMARY_PREFIX + "\nBODY"
    assert ContextCompressor._is_context_summary_content(content)
    assert ContextCompressor._strip_summary_prefix(content) == "BODY"
    assert CURATED_MEMORY_SUMMARY_PREFIX not in _HISTORICAL_SUMMARY_PREFIXES


def test_the_frozen_historical_recognizers_still_carry_the_native_clause():
    """§9.10 L1668: the four frozen entries are byte-identical; R40 adds, never edits."""
    assert sum(1 for prefix in _HISTORICAL_SUMMARY_PREFIXES if NATIVE_CLAUSE in prefix) == 4
    for prefix in _HISTORICAL_SUMMARY_PREFIXES:
        assert ContextCompressor._starts_with_summary_prefix(prefix + "\nbody")


def test_authoritative_compressor_emits_the_curated_texts(agent_env):
    compressor = agent_env.agent.context_compressor
    assert compressor.summary_prefix == CURATED_MEMORY_SUMMARY_PREFIX
    assert compressor.compression_note == CURATED_MEMORY_COMPRESSION_NOTE


def test_additive_compressor_text_is_unchanged(tmp_path):
    """§9.10 L1668: a relationship, never a snapshot."""
    compressor = ContextCompressor(model="test/model")
    assert compressor.summary_prefix is SUMMARY_PREFIX
    assert compressor.compression_note is ContextCompressor._COMPRESSION_NOTE


def test_configure_compaction_text_is_a_no_op_for_an_additive_service(tmp_path):
    from types import SimpleNamespace

    from agent.memory_service.lifecycle import configure_compaction_text

    compressor = ContextCompressor(model="test/model")
    configure_compaction_text(compressor, None)
    configure_compaction_text(compressor, SimpleNamespace(disposition=None))
    assert compressor.summary_prefix is SUMMARY_PREFIX
    assert compressor.compression_note is ContextCompressor._COMPRESSION_NOTE


def test_generated_summary_and_micro_marker_use_the_configured_prefix(agent_env):
    """Both emitters read the compressor's configured prefix, not the module constant."""
    compressor = agent_env.agent.context_compressor
    rendered = compressor._with_summary_prefix("BODY", prefix=compressor.summary_prefix)
    assert rendered.startswith(CURATED_MEMORY_SUMMARY_PREFIX)
    marker = ContextCompressor._render_micro_marker_content("Live handoff body",
                                                            prefix=compressor.summary_prefix)
    assert marker.startswith(CURATED_MEMORY_SUMMARY_PREFIX)
    # The default call form the existing suites use is untouched.
    assert ContextCompressor._render_micro_marker_content("Live handoff body").startswith(SUMMARY_PREFIX)


def test_the_head_note_follows_the_configured_compression_note(agent_env):
    compressor = agent_env.agent.context_compressor
    head = compressor._assemble_head([{"role": "system", "content": "SYSTEM"},
                                      {"role": "user", "content": "u"}], 2)
    assert CURATED_MEMORY_COMPRESSION_NOTE in head[0]["content"]
    assert "MEMORY.md" not in head[0]["content"]


def test_rotation_keeps_the_identity_and_rerenders_the_region(agent_env):
    """§9.7 L1597 and L1610: the child keeps the record, binds nothing, and re-renders."""
    from agent.memory_service.host_state import load_host_state
    from agent.memory_service.lifecycle import on_compression_boundary
    from agent.memory_service.render import REGION_HEADING

    agent = agent_env.agent
    agent_env.store.seed_record(REPO_SCOPE, "memory", "MEMORY-FACT")
    old_id = agent.session_id
    identity = agent._memory_service.identity
    agent._build_system_prompt(None)
    binds = agent_env.count("bind_session")

    agent.session_id = "rotated-1"                       # the rotation the compressor performs
    on_compression_boundary(agent, compressed=[], old_session_id=old_id, session_commit_succeeded=True)

    assert agent._memory_service.identity == identity
    assert agent_env.count("bind_session") == binds
    assert load_host_state("rotated-1").state == load_host_state(old_id).state
    assert agent._memory_session_key == "rotated-1"
    agent._invalidate_system_prompt()
    rebuilt = agent._build_system_prompt(None)
    assert REGION_HEADING in rebuilt and "MEMORY-FACT" in rebuilt


def test_an_aborted_commit_carries_nothing(agent_env):
    """A boundary that did not commit must not move the record."""
    from agent.memory_service.host_state import load_host_state
    from agent.memory_service.lifecycle import on_compression_boundary

    agent = agent_env.agent
    old_id = agent.session_id
    agent.session_id = "rotated-2"
    on_compression_boundary(agent, compressed=[], old_session_id=old_id, session_commit_succeeded=False)
    assert load_host_state("rotated-2") is None


def test_in_place_compaction_keeps_the_same_record(agent_env):
    """In-place compaction keeps one session id; the record is already correct."""
    from agent.memory_service.host_state import load_host_state
    from agent.memory_service.lifecycle import on_compression_boundary

    agent = agent_env.agent
    before = load_host_state(agent.session_id)
    on_compression_boundary(agent, compressed=[], old_session_id=None, session_commit_succeeded=True)
    assert load_host_state(agent.session_id) == before


def test_boundary_refresh_failure_reuses_the_last_render_and_the_gate_blocks(agent_env):
    """Ruling R40-7 (a): reuse the last validated render; the gate blocks the next request."""
    from agent.memory_service.lifecycle import curated_request_gate

    agent = agent_env.agent
    agent_env.store.seed_record(REPO_SCOPE, "memory", "MEMORY-FACT")
    first = agent._build_system_prompt(None)
    assert "MEMORY-FACT" in first

    agent_env.store.fail_transport("load_curated", times=1)
    agent._invalidate_system_prompt()
    rebuilt = agent._build_system_prompt(None)                 # the boundary rebuild
    assert "MEMORY-FACT" in rebuilt                            # the last validated render, reused
    assert agent._memory_service.blocked is True

    agent._curated_fresh_for_next_request = False
    agent_env.store.fail_transport("load_curated", times=1)
    assert curated_request_gate(agent) is not None             # and the gate still blocks


def test_a_reused_render_from_another_identity_is_refused(agent_env):
    """R40-7 reuse is scoped to the SAME frozen identity, never any prior render."""
    from agent.memory_service.errors import MemoryBlockedError
    from agent.memory_service.lifecycle import curated_prompt_parts
    from agent.memory_service.render import CuratedPromptRender

    agent = agent_env.agent
    other = agent_env.build_agent("s-other")
    foreign_identity = other._memory_service.identity
    assert foreign_identity != agent._memory_service.identity
    agent._curated_prompt_render = (foreign_identity, CuratedPromptRender("STALE", (), ()))
    agent_env.store.fail_transport("load_curated", times=1)
    with pytest.raises(MemoryBlockedError):
        curated_prompt_parts(agent)


def test_compression_never_touches_the_native_directory(agent_env, native_dir):
    """§9.10 L1685: proven with the sentinel."""
    from agent.memory_service.lifecycle import on_compression_boundary

    native_dir.mkdir(parents=True, exist_ok=True)
    (native_dir / "MEMORY.md").write_text("NATIVE-SECRET-FACT\n", encoding="utf-8")
    agent = agent_env.agent
    old_id = agent.session_id
    agent._build_system_prompt(None)
    with native_memory_sentinel(native_dir) as sentinel:
        agent.session_id = "rotated-3"
        on_compression_boundary(agent, compressed=[], old_session_id=old_id, session_commit_succeeded=True)
        agent._invalidate_system_prompt()
        prompt = agent._build_system_prompt(None)
    sentinel.assert_untouched()
    assert "NATIVE-SECRET-FACT" not in prompt


def test_a_stateless_session_carries_its_record_across_rotation(agent_env):
    """I1: a rotated stateless session is still stateless on the child id."""
    from agent.memory_service.host_state import HostStateRecord, load_host_state, save_host_state
    from agent.memory_service.lifecycle import on_compression_boundary

    env = agent_env
    save_host_state(HostStateRecord("stateless-c", "stateless", None))
    agent = env.build_agent("stateless-c")
    agent.session_id = "stateless-c-rotated"
    on_compression_boundary(agent, compressed=[], old_session_id="stateless-c", session_commit_succeeded=True)
    assert load_host_state("stateless-c-rotated").disposition == "stateless"


def test_an_additive_boundary_is_a_no_op(tmp_path):
    from types import SimpleNamespace

    from agent.memory_service.host_state import host_state_dir
    from agent.memory_service.lifecycle import on_compression_boundary

    agent = SimpleNamespace(_memory_service=None, session_id="child")
    on_compression_boundary(agent, compressed=[], old_session_id="parent", session_commit_succeeded=True)
    assert not host_state_dir().exists()
