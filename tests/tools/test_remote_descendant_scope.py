"""Descendant scope travels in commands, independently of SSH/cloud environment forwarding."""

import os
import shlex
import subprocess
from types import SimpleNamespace

import pytest

from tools.environments.base import BaseEnvironment
from tools.environments.local import _bash_safe_path, _find_bash
from tools.environments.managed_modal import ManagedModalEnvironment
from tools.environments.ssh import SSHEnvironment


@pytest.mark.parametrize("backend", ["ssh", "managed_modal"])
@pytest.mark.parametrize("marker", ["HERMES_KANBAN_TASK", "HERMES_DELEGATED_CHILD_CONTEXT"])
def test_remote_commands_preserve_descendant_scope_without_tainting_later_commands(
    tmp_path, monkeypatch, backend, marker,
):
    bash = _find_bash()
    shell_env = {key: value for key, value in os.environ.items()
                 if not key.startswith("HERMES_KANBAN_") and key != "HERMES_DELEGATED_CHILD_CONTEXT"}
    cwd = _bash_safe_path(str(tmp_path))
    probe = "printf 'scope=%s\\n' \"${HERMES_DELEGATED_CHILD_CONTEXT:-human}\""
    command = "bash -c " + shlex.quote(probe)
    if backend == "ssh":
        env = object.__new__(SSHEnvironment)
        BaseEnvironment.__init__(env, cwd=cwd, timeout=30)
        env.host, env.user, env.port, env.key_path = "test", "test", 22, ""
        env.control_socket = tmp_path / "ssh.sock"
        env._sync_manager = None
        snapshot = tmp_path / "snapshot.sh"
        snapshot.write_text("", encoding="utf-8")
        env._snapshot_path, env._snapshot_ready = _bash_safe_path(str(snapshot)), True

        def launch(argv, stdin_data=None, **kwargs):
            # Run the actual remote script with no inherited host markers or SendEnv support.
            assert not any(arg.startswith("SendEnv=") for arg in argv)
            script = shlex.split(argv[-1])[0]
            return subprocess.Popen([bash, "-c", script], env=shell_env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8")

        monkeypatch.setattr("tools.environments.ssh._popen_bash", launch)
        monkeypatch.setattr("tools.env_passthrough.get_all_passthrough", lambda: frozenset())
    else:
        env = object.__new__(ManagedModalEnvironment)
        BaseEnvironment.__init__(env, cwd=cwd, timeout=30)
        env._sandbox_id = "test"
        env._persistent = True

        def request(method, url, *, json, timeout):
            # Model the server's persistent shell: a later command in it must not inherit the restriction.
            script = json["command"] + "\nprintf 'after=%s\\n' \"${HERMES_DELEGATED_CHILD_CONTEXT:-human}\""
            result = subprocess.run([bash, "-c", script], env=shell_env, capture_output=True,
                                    text=True, encoding="utf-8", timeout=30)
            assert "after=human" in result.stdout
            return SimpleNamespace(status_code=200, json=lambda: {
                "status": "completed", "output": result.stdout, "returncode": result.returncode})

        monkeypatch.setattr(env, "_request", request)

    monkeypatch.setenv(marker, "1")
    restricted = env.execute(command)
    assert restricted["returncode"] == 0 and "scope=1" in restricted["output"], restricted
    if backend == "ssh":
        # A backend with environment forwarding must exclude its inherited marker from the snapshot too.
        shell_env["HERMES_DELEGATED_CHILD_CONTEXT"] = "1"
        assert "scope=1" in env.execute(command)["output"]
        shell_env.pop("HERMES_DELEGATED_CHILD_CONTEXT")
    monkeypatch.delenv(marker)
    human = env.execute(command)
    assert human["returncode"] == 0 and "scope=human" in human["output"], human
    if backend == "managed_modal":
        env._sandbox_id = None
