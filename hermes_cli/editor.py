"""Editor selection and launch shared by file-editing CLI commands."""

import os
import shlex
import shutil
import subprocess

from hermes_cli._subprocess_compat import split_windows_command_line


def preferred_editor(default: str | None = None) -> str | None:
    """Honor VISUAL before EDITOR, then the caller's existing default."""
    return os.environ.get("VISUAL") or os.environ.get("EDITOR") or default


def run_editor(command: str, path: str | os.PathLike, *, shell_fallback: bool = False) -> int:
    """Wait for the editor; only /prompt retains its historical POSIX shell fallback."""
    try:
        argv = split_windows_command_line(command) if os.name == "nt" else shlex.split(command)
        if not argv:
            raise ValueError("Editor command is empty")
        # Resolve Windows .cmd shims such as code --wait before CreateProcess.
        return subprocess.call([shutil.which(argv[0]) or argv[0], *argv[1:], str(path)])
    except (OSError, ValueError):
        if os.name == "nt" or not shell_fallback:
            raise
        # /bin/sh expands editor values containing ~ or $VARS. cmd.exe instead invokes file
        # associations for unsupported scripts, so Windows must report a launch failure.
        return subprocess.call(f"{command} {shlex.quote(str(path))}", shell=True)
