"""Authority isolation includes config loading and imports in a fresh interpreter."""

import os
import subprocess
import sys

import pytest
import yaml


@pytest.mark.parametrize("entrypoint", ["doctor", "agent"])
@pytest.mark.parametrize("mode", ["authoritative", "invalid", "additive", "managed_scope", "managed_install"])
def test_cold_startup_preserves_memory_authority(tmp_path, monkeypatch, entrypoint, mode):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    section = {"provider": "example", "provider_mode": "authoritative",
               "provider_executable": sys.executable, "principal_id": "ethan"}
    if mode == "invalid":
        del section["principal_id"]
    elif mode == "additive":
        section = {}
    config_home = home
    if mode == "managed_scope":
        config_home = tmp_path / "managed"
        config_home.mkdir()
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(config_home))
    if mode == "managed_install":
        monkeypatch.setenv("HERMES_MANAGED", "1")
        for subdir in ("cron", "sessions", "logs"):
            (home / subdir).mkdir()
    (config_home / "config.yaml").write_text(yaml.safe_dump({"memory": section}), encoding="utf-8")

    probe = """
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from tests.agent.memory_service.native_sentinel import native_memory_sentinel

entrypoint, mode = sys.argv[1:]
native = Path(os.environ['HERMES_HOME']) / 'memories'
with native_memory_sentinel(native) as sentinel:
    if entrypoint == 'doctor':
        import hermes_cli.doctor
        from hermes_cli.doctor_state import _check_directory_structure, _check_memory_provider
        finding = _check_memory_provider(True)
        _check_directory_structure(True)
        assert bool(finding.issues) == (mode == 'invalid'), finding.issues
    else:
        from agent.agent_init import _init_memory
        from agent.memory_service.config import MemoryConfigurationError
        from agent.memory_service.service import MemoryDisposition
        from hermes_cli.config import load_config_readonly
        from tests.agent.memory_service.fake_backend import FakeAuthoritativeBackend, FakeProviderStore, FakeRegistry
        import plugins.memory
        registry = FakeRegistry(directories={os.path.realpath(os.getcwd()): ('repo-1', 'proj-1', None)})
        backend = FakeAuthoritativeBackend(FakeProviderStore(registry=registry), provider='example')
        plugins.memory.load_authoritative_backend_factory = lambda name: lambda cfg: backend
        agent = SimpleNamespace(enabled_toolsets=None, disabled_toolsets=None, session_id='cold', tools=None)
        try:
            _init_memory(agent, load_config_readonly(), False, None)
        except MemoryConfigurationError:
            assert mode == 'invalid'
        else:
            assert mode != 'invalid'
            expected = MemoryDisposition.BUILTIN if mode == 'additive' else MemoryDisposition.AUTHORITATIVE
            assert agent._memory_service.disposition is expected
if mode == 'additive':
    assert sentinel.accesses
    assert native.is_dir()
else:
    sentinel.assert_untouched()
    assert not native.exists()
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, entrypoint, mode],
        cwd=tmp_path, env={**os.environ, "PYTHONPATH": os.getcwd()},
        capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
