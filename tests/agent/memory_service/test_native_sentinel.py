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
    ("access",   lambda d: os.access(d / "MEMORY.md", os.F_OK)),
    ("chmod",    lambda d: os.chmod(d / "MEMORY.md", 0o644)),
    ("utime",    lambda d: os.utime(d / "MEMORY.md", None)),
    ("truncate", lambda d: os.truncate(d / "MEMORY.md", 0)),
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


def test_nested_sentinels_do_not_blind_the_outer_guard(native_dir, tmp_path):
    """An inner context exiting must not strip the outer context's stat
    interception -- 5 of the 12 watched verbs are caught only by that patch."""
    other = tmp_path / "other"
    other.mkdir()
    with native_memory_sentinel(native_dir) as outer:
        with native_memory_sentinel(other) as inner:
            (other / "x.md").write_text("y", encoding="utf-8")
        assert inner.accesses
        (native_dir / "MEMORY.md").exists()   # stat route, outer must still see it
    assert outer.accesses, "outer sentinel went stat-blind after the inner context exited"


def test_rename_into_the_native_directory_is_caught(native_dir, tmp_path):
    """os.rename's audit args are (src, dst, ...): restoring a legacy file INTO
    the dormant directory is a forbidden 'restore'/'mirror' and must not be
    invisible just because the native path is args[1].

    Unlike POSIX, Windows' os.rename refuses to overwrite an existing
    destination (WinError 183), so the fixture's pre-existing MEMORY.md is
    moved out of the way *before* the sentinel is armed -- that unlink is not
    part of what this test is exercising and must not itself pad
    ``sentinel.accesses``. The destination path is still inside native_dir
    either way.
    """
    stale = tmp_path / "stale.md"
    stale.write_text("legacy native memory", encoding="utf-8")
    (native_dir / "MEMORY.md").unlink()
    with native_memory_sentinel(native_dir) as sentinel:
        os.rename(str(stale), str(native_dir / "MEMORY.md"))
    assert sentinel.accesses, "a restore into the native directory went unrecorded"


@pytest.mark.windows_only
def test_realpath_normalizes_a_junction_alias_of_the_watched_directory(tmp_path):
    """HERMES_HOME may be a junction/symlink alias of the platform default and
    only the spelling is preserved (hermes_cli/profiles.py:1694-1701, #82581
    junction follow-up) -- physically the same directory reached through a
    second spelling must not slip the guard.
    """
    import subprocess

    real_dir = tmp_path / "real_memories"
    real_dir.mkdir()
    (real_dir / "MEMORY.md").write_text("existing user memory\n", encoding="utf-8")
    link_dir = tmp_path / "linked_memories"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link_dir), str(real_dir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"junction creation failed: {result.stderr}"

    with native_memory_sentinel(real_dir) as sentinel:
        (link_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert sentinel.accesses, "access via a junction alias of the watched directory went unrecorded"
