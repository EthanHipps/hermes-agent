"""Every CLI editor surface preserves the user's command and chosen editor."""

import json
import os
import sys

import pytest

from hermes_cli.cli_commands_mixin import CLICommandsMixin
from hermes_cli.journey import _open_in_editor


def _editor_script(tmp_path):
    script = tmp_path / "editor with spaces.py"
    script.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "Path(sys.argv[-1]).write_text(json.dumps(sys.argv[1:-1]), encoding='utf-8')\n",
        encoding="utf-8",
    )
    return script


def _edit(surface):
    if surface == "prompt":
        return CLICommandsMixin()._compose_in_editor("draft")
    if surface == "journey":
        return _open_in_editor("draft", suffix=".txt")
    from hermes_cli.config import edit_config, get_config_path

    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("# draft\n", encoding="utf-8")
    edit_config()
    return config_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("surface", ["config", "journey", "prompt"])
@pytest.mark.parametrize("preferred", ["EDITOR", "VISUAL"])
def test_editor_surfaces_preserve_arguments_and_visual_precedence(tmp_path, monkeypatch, surface, preferred):
    script = _editor_script(tmp_path)
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.setenv("EDITOR", str(tmp_path / "unused-editor"))
    monkeypatch.setenv(preferred, f'"{sys.executable}" "{script}" --profile="Work Profile"')

    assert json.loads(_edit(surface)) == ["--profile=Work Profile"]


@pytest.mark.windows_only
@pytest.mark.parametrize("surface", ["config", "journey", "prompt"])
@pytest.mark.parametrize("shape", ["unquoted_path", "cmd_shim"])
def test_windows_editor_surfaces_resolve_native_commands(tmp_path, monkeypatch, surface, shape):
    script = _editor_script(tmp_path)
    if shape == "unquoted_path":
        if any(char.isspace() for char in sys.executable):
            pytest.skip("An unquoted executable requires a space-free interpreter path")
        command = f'{sys.executable} "{script}"'
    else:
        (tmp_path / "record-editor.cmd").write_text(
            f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8"
        )
        monkeypatch.setenv("PATH", f"{tmp_path};{os.environ['PATH']}")
        command = "record-editor"
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.setenv("EDITOR", command + ' --wait')

    assert json.loads(_edit(surface)) == ["--wait"]
