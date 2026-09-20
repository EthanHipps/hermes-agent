"""The sentinel must FIRE on every §9.1 L948 verb. A guard that cannot fail proves nothing."""

import os
from pathlib import Path

import pytest

from tests.agent.memory_service.native_sentinel import NativeMemoryTouched, native_memory_sentinel


@pytest.fixture
def native_dir(tmp_path):
    d = tmp_path / "memories"
    d.mkdir()
    (d / "MEMORY.md").write_text("existing user memory\n", encoding="utf-8")
    (d / "USER.md").write_text("existing profile\n", encoding="utf-8")
    return d


def test_silent_when_nothing_touches_the_directory(native_dir, tmp_path):
    with native_memory_sentinel(native_dir) as sentinel:
        (tmp_path / "elsewhere.txt").write_text("unrelated", encoding="utf-8")
    assert sentinel.accesses == []
    sentinel.assert_untouched()


@pytest.mark.parametrize("label,action", [
    ("read",     lambda d: (d / "MEMORY.md").read_text(encoding="utf-8")),
    ("write",    lambda d: (d / "MEMORY.md").write_text("mutated", encoding="utf-8")),
    ("create",   lambda d: (d / "NEW.md").write_text("created", encoding="utf-8")),
    ("mkdir",    lambda d: (d / "sub").mkdir()),
    ("listdir",  lambda d: os.listdir(d)),
    ("iterdir",  lambda d: list(d.iterdir())),
    ("remove",   lambda d: (d / "USER.md").unlink()),
    ("stat",     lambda d: (d / "MEMORY.md").stat()),
    ("exists",   lambda d: (d / "MEMORY.md").exists()),
    ("is_file",  lambda d: (d / "MEMORY.md").is_file()),
    ("getsize",  lambda d: os.path.getsize(d / "MEMORY.md")),
    ("os_stat",  lambda d: os.stat(d / "MEMORY.md")),
])
def test_sentinel_records_every_forbidden_verb(native_dir, label, action):
    with native_memory_sentinel(native_dir) as sentinel:
        action(native_dir)
    assert sentinel.accesses, f"sentinel missed {label}"
    with pytest.raises(AssertionError, match="native memory"):
        sentinel.assert_untouched()


def test_deny_mode_raises_at_the_point_of_access(native_dir):
    with pytest.raises(NativeMemoryTouched):
        with native_memory_sentinel(native_dir, deny=True):
            (native_dir / "MEMORY.md").read_text(encoding="utf-8")


def test_disarms_on_exit_even_when_the_body_raises(native_dir):
    with pytest.raises(ValueError):
        with native_memory_sentinel(native_dir):
            raise ValueError("boom")
    (native_dir / "MEMORY.md").read_text(encoding="utf-8")  # must not raise or record


def test_defaults_to_the_configured_native_memory_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.memory_tool import get_memory_dir
    target = get_memory_dir()
    target.mkdir(parents=True, exist_ok=True)
    with native_memory_sentinel() as sentinel:
        (target / "MEMORY.md").write_text("x", encoding="utf-8")
    assert sentinel.accesses
