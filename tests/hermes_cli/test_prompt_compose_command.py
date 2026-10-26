"""Tests for the CLI `/prompt` editor-compose command.

`/prompt` opens `$VISUAL`/`$EDITOR` on a temp markdown file so the user can
hand-edit a multi-line prompt, then queues the saved buffer as the next
agent turn via the one-shot `_pending_agent_seed` (same path `/blueprint`
uses). These drive a fake editor subprocess to verify read-back, header
stripping, seeding, and the empty-buffer cancel path.
"""

import sys

import pytest

from hermes_cli.cli_commands_mixin import CLICommandsMixin
from hermes_cli.commands import resolve_command


class _Stub(CLICommandsMixin):
    def __init__(self):
        self._pending_agent_seed = None


def _fake_editor(tmp_path, body: str, mode: str = "append") -> str:
    """An EDITOR value naming a tiny Python 'editor' that mutates the file it is handed.

    It runs through this interpreter so the argv call starts it on every OS; a bare
    script would be opened by its file association on Windows (an "Open With" dialog).
    """
    script = tmp_path / "fake_editor.py"
    file_mode = "a" if mode == "append" else "w"  # "w" clears the buffer
    script.write_text(
        "import sys\n"
        f"with open(sys.argv[1], {file_mode!r}, encoding='utf-8') as fh:\n"
        f"    fh.write({body!r})\n",
        encoding="utf-8",
    )
    return f'"{sys.executable}" "{script}"'


@pytest.fixture(autouse=True)
def _no_visual(monkeypatch):
    monkeypatch.delenv("VISUAL", raising=False)


def test_command_registered():
    cd = resolve_command("prompt")
    assert cd and cd.name == "prompt"
    assert resolve_command("compose").name == "prompt"


def test_compose_reads_and_strips_header(monkeypatch, tmp_path):
    monkeypatch.setenv("EDITOR", _fake_editor(tmp_path, "Refactor the auth module.\nUse pytest.\n"))
    out = _Stub()._compose_in_editor("")
    assert "Refactor the auth module." in out
    assert "Use pytest." in out
    assert "#!" not in out  # the instructional header is stripped


def test_empty_buffer_does_not_seed(monkeypatch, tmp_path):
    monkeypatch.setenv("EDITOR", _fake_editor(tmp_path, "", mode="clear"))
    s = _Stub()
    s._handle_prompt_compose_command("/prompt")
    assert s._pending_agent_seed is None


@pytest.mark.windows_only
def test_unlaunchable_editor_is_reported_not_handed_to_cmd(monkeypatch, tmp_path):
    """cmd.exe would open a non-executable editor through its file association, so the
    argv failure must surface (``/prompt`` prints "Could not open editor") instead."""
    monkeypatch.setenv("EDITOR", f'"{tmp_path / "no-such-editor.exe"}"')
    with pytest.raises(OSError):
        _Stub()._compose_in_editor("")
