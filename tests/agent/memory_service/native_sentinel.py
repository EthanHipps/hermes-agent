"""Filesystem sentinel around the native memory directory (§9.10 L1685).

§9.10 requires "not used" to be *proven* rather than inferred from output, and
§9.1 L948 forbids initialize, create, stat, read, inject, write, mirror,
import, restore and fallback on MEMORY.md / USER.md in authoritative mode.

Two mechanisms, because neither alone covers the verb list:

* ``sys.addaudithook`` sees ``open``, ``os.mkdir``, ``os.listdir``,
  ``os.scandir``, ``os.remove``, ``os.rename``, ``os.rmdir``, ``os.chmod``,
  ``os.utime``, ``os.truncate``, ``os.link`` and ``os.symlink`` from ANY code
  path -- ``pathlib``, ``os``, ``io``, ``shutil``, and C-level callers alike.
  This is what makes the guard a proof rather than a monkeypatch that a new
  call site could route around. Multi-path events carry more than one path
  (``os.rename``'s audit args are ``(src, dst, src_dir_fd, dst_dir_fd)``), so
  the hook inspects *every* argument, not just the first -- a "restore" that
  renames a legacy file INTO the native directory is forbidden by L948 and
  would otherwise be invisible because the watched path sits in ``args[1]``,
  not ``args[0]``. Non-path members (mode/flag ints, dir_fds) are rejected by
  ``_covers()``'s ``isinstance`` check, so scanning every argument does not
  risk false positives.
* CPython raises **no audit event for stat, and none for access either**.
  Measured on 3.11.16: ``os.stat``, ``os.lstat``, ``Path.exists``,
  ``Path.stat``, ``Path.is_file``, ``os.path.exists``, ``os.path.getsize``
  and ``os.access`` all produce zero audit events -- and ``stat`` is named in
  L948 (doctor calls ``.exists()`` today). So ``os.stat`` and ``os.lstat`` are
  both intercepted while armed; the five ``Path``/``os.path`` routes above
  were measured to funnel through ``os.stat``. ``os.access`` shares the
  "no audit event" problem but does NOT route through ``os.stat`` either
  (measured separately), so it gets its own guard, installed and restored
  alongside ``os.stat``/``os.lstat`` rather than folded into either.

``_covers()`` resolves both the watched directory and every candidate path
with ``os.path.realpath`` rather than ``os.path.abspath``. HERMES_HOME may be
a junction/symlink alias of the platform default and only the *spelling* is
preserved (``hermes_cli/profiles.py:1694-1701``, #82581 junction follow-up) --
``abspath`` would treat two spellings of the same physical directory as
unrelated paths and let an access through the other spelling slip the guard.

The audit hook is installed once per process and cannot be removed, so it is
gated on ``_ARMED``: disarmed, its first operation is a bool check.
``scripts/run_tests_parallel.py`` isolates per file, confining the cost.

Default mode RECORDS. It does not raise, because ``agent_init._init_memory``
wraps its native path in ``suppress(Exception)`` -- a raising guard there would
be swallowed and the test would pass for the wrong reason. ``deny=True`` opts
into raising where a hard refusal is itself under test.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple

_AUDIT_EVENTS = frozenset({
    "open", "os.mkdir", "os.listdir", "os.scandir", "os.remove", "os.rename", "os.rmdir",
    "os.chmod", "os.utime", "os.truncate", "os.link", "os.symlink",
})

_ARMED: Optional["NativeMemorySentinel"] = None
_HOOK_INSTALLED = False
_REAL_STAT = os.stat
_REAL_LSTAT = os.lstat
_REAL_ACCESS = os.access


class NativeMemoryTouched(AssertionError):
    """Raised in ``deny=True`` mode at the moment the native directory is touched."""


class NativeMemorySentinel:
    def __init__(self, directory: Path, deny: bool) -> None:
        self.directory = directory
        self.deny = deny
        self.accesses: List[Tuple[str, str]] = []
        self._prefix = os.path.normcase(os.path.realpath(str(directory)))

    def _covers(self, raw) -> bool:
        if not isinstance(raw, (str, bytes, os.PathLike)):
            return False  # an int fd, or something we cannot resolve to a path
        try:
            candidate = os.path.normcase(os.path.realpath(os.fsdecode(raw)))
        except Exception:
            return False
        return candidate == self._prefix or candidate.startswith(self._prefix + os.sep)

    def record(self, event: str, raw) -> None:
        if not self._covers(raw):
            return
        entry = (event, os.fsdecode(raw) if not isinstance(raw, str) else raw)
        self.accesses.append(entry)
        if self.deny:
            raise NativeMemoryTouched(f"{event} on native memory path {entry[1]!r}")

    def assert_untouched(self) -> None:
        if self.accesses:
            detail = "\n  ".join(f"{event}: {path}" for event, path in self.accesses)
            raise AssertionError(
                f"native memory directory was touched {len(self.accesses)} time(s) "
                f"in authoritative mode (§9.1 forbids this):\n  {detail}"
            )


def _audit_hook(event: str, args) -> None:
    sentinel = _ARMED
    if sentinel is None or event not in _AUDIT_EVENTS or not args:
        return
    # Multi-path events carry (src, dst, ...): os.rename INTO the native
    # directory is a forbidden "restore"/"mirror" and lives in args[1].
    # Non-path members (dir_fd ints) are rejected by _covers().
    for arg in args:
        sentinel.record(event, arg)


def _guarded_stat(path, *a, **k):
    sentinel = _ARMED
    if sentinel is not None:
        sentinel.record("os.stat", path)
    return _REAL_STAT(path, *a, **k)


def _guarded_lstat(path, *a, **k):
    sentinel = _ARMED
    if sentinel is not None:
        sentinel.record("os.lstat", path)
    return _REAL_LSTAT(path, *a, **k)


def _guarded_access(path, *a, **k):
    sentinel = _ARMED
    if sentinel is not None:
        sentinel.record("os.access", path)
    return _REAL_ACCESS(path, *a, **k)


@contextmanager
def native_memory_sentinel(directory=None, *, deny: bool = False):
    """Arm the sentinel around ``directory`` (default: the configured native memory dir)."""
    global _ARMED, _HOOK_INSTALLED
    if directory is None:
        from tools.memory_tool import get_memory_dir
        directory = get_memory_dir()
    if not _HOOK_INSTALLED:
        sys.addaudithook(_audit_hook)
        _HOOK_INSTALLED = True
    sentinel = NativeMemorySentinel(Path(directory), deny)
    previous = _ARMED
    previous_stat, previous_lstat, previous_access = os.stat, os.lstat, os.access
    _ARMED = sentinel
    os.stat, os.lstat, os.access = _guarded_stat, _guarded_lstat, _guarded_access
    try:
        yield sentinel
    finally:
        _ARMED = previous
        os.stat, os.lstat, os.access = previous_stat, previous_lstat, previous_access
