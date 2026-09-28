"""R39: memory.write_approval in a provider-managed session (Checkpoint B carry-forward; §9.5 L1538),
and the memory tool's approval channel (Task 6)."""

from types import SimpleNamespace

import pytest

from tools import write_approval as wa
from tools.terminal_tool import set_approval_callback


def _config(monkeypatch, value=None, *, fail=False):
    def load_config():
        if fail:
            raise OSError("config unreadable")
        return {"memory": {} if value is None else {"write_approval": value}}
    monkeypatch.setattr("hermes_cli.config.load_config", load_config)  # write_approval late-imports it


@pytest.mark.parametrize("value, lenient, strict", [
    (None, False, False), (False, False, False), ("off", False, False), (" Disabled ", False, False),
    (True, True, True), ("on", True, True), ("maybe", False, True), (7, False, True)])
def test_only_an_absent_value_or_an_explicit_off_is_off_when_failing_closed(monkeypatch, value, lenient, strict):
    _config(monkeypatch, value)
    assert wa.write_approval_enabled(wa.MEMORY) is lenient            # additive reading: unchanged
    assert wa.write_approval_enabled(wa.MEMORY, fail_closed=True) is strict


def test_a_failed_read_counts_as_on_only_when_failing_closed(monkeypatch):
    _config(monkeypatch, fail=True)
    assert wa.write_approval_enabled(wa.MEMORY) is False
    assert wa.write_approval_enabled(wa.MEMORY, fail_closed=True) is True


def _prompt():
    return SimpleNamespace(target="user", intent_kind="add", requirements=("target_user",), text="rendered diff")


@pytest.fixture
def callback():
    def install(choice, seen):
        set_approval_callback(lambda command, description, **kw: seen.append((command, description)) or choice)
    yield install
    set_approval_callback(None)


def test_without_a_callback_the_channel_can_only_defer():
    from tools.write_approval_curated import memory_tool_approval_channel
    assert memory_tool_approval_channel(None).prompt is None and not memory_tool_approval_channel(None).available
    assert memory_tool_approval_channel("sess-1").available and memory_tool_approval_channel("sess-1").prompt is None


@pytest.mark.parametrize("choice, answer", [("once", True), ("session", True), ("deny", False), ("timeout", None)])
def test_the_inline_prompt_maps_the_native_choices(callback, choice, answer):
    from tools.write_approval_curated import memory_tool_approval_channel
    seen = []
    callback(choice, seen)
    assert memory_tool_approval_channel(None).prompt(_prompt()) is answer
    assert seen[-1][0] == "rendered diff" and "target_user" in seen[-1][1]


def test_a_background_review_never_prompts_inline(callback):
    """Ruling X6b-1: native evaluate_gate never prompts for the background_review origin."""
    from tools.skill_provenance import reset_current_write_origin, set_current_write_origin
    from tools.write_approval_curated import memory_tool_approval_channel
    seen = []
    callback("once", seen)
    token = set_current_write_origin("background_review")
    try:
        channel = memory_tool_approval_channel("sess-1")
    finally:
        reset_current_write_origin(token)
    assert channel.prompt is None and channel.session_id == "sess-1" and channel.origin == "background_review"
    assert seen == []
