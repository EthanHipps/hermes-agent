"""R39: memory.write_approval in a provider-managed session (Checkpoint B carry-forward; §9.5 L1538),
and the memory tool's approval channel (Task 6)."""

import pytest

from tools import write_approval as wa


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
