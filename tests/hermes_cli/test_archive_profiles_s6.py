"""S6 archive suite D: profile export/import/clone under the native sentinel (§9.8, §9.10 L1669).

Helpers are copied rather than imported: the shared test support files stay frozen
and each suite keeps its own helpers (D-R44-f, reconciliation §3 rule 10).
"""

import importlib
import json
import tarfile
from pathlib import Path

import pytest
import yaml

from agent.memory_service.archive import DISPOSITION_RECORD_NAME
from tests.agent.memory_service.native_sentinel import native_memory_sentinel

SPEC_BLOCK = {"authority": "provider", "provider": "ygg", "provider_api": 1, "included": False,
              "disposition": "provider-managed", "restore_action": "reconnect-provider"}


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    """Path.home() and HERMES_HOME both redirected (root AGENTS.md, Testing)."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_path


@pytest.fixture()
def profiles():
    """Resolve the live profiles module at call time (test_profile_export_default_path.py L13-22)."""
    return importlib.import_module("hermes_cli.profiles")


def _authoritative_yaml(exe: Path, provider: str, policy: str = "") -> str:
    extra = f"  authoritative_failure_policy: {policy}\n" if policy else ""
    return (f"memory:\n  provider: {provider}\n  provider_mode: authoritative\n"
            f"  provider_executable: '{exe}'\n  principal_id: ethan\n{extra}")


def _profile_like(home: Path, *, provider=None) -> Path:
    """Write a config.yaml into an arbitrary directory (a real home or a staging dir)."""
    home.mkdir(parents=True, exist_ok=True)
    if provider:
        exe = home.parent / f"{home.name}-provider.exe"
        exe.write_bytes(b"MZ")
        (home / "config.yaml").write_text(_authoritative_yaml(exe, provider), encoding="utf-8")
    else:
        (home / "config.yaml").write_text("model: test\n", encoding="utf-8")
    return home


def _profile(root: Path, name: str, *, provider=None) -> Path:
    home = root / ".hermes" if name == "default" else root / ".hermes" / "profiles" / name
    return _profile_like(home, provider=provider)


def _dormant(home: Path) -> Path:
    native = home / "memories"
    native.mkdir(parents=True, exist_ok=True)
    (native / "MEMORY.md").write_text("dormant native memory\n", encoding="utf-8")
    (native / "USER.md").write_text("dormant native profile\n", encoding="utf-8")
    return native


def _migration(home: Path, state: str) -> None:
    path = home / "migrations" / "example" / "ep-1" / "run-1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": "ygg.hermes-migration/v1", "state": state}), encoding="utf-8")


def _host_state(home: Path) -> Path:
    record = home / "memory_service" / "sessions" / ("aa" * 32 + ".json")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text('{"schema": "hermes.memory-host-session/v1"}\n', encoding="utf-8")
    return record


def _tar_names(path: Path) -> set:
    with tarfile.open(path) as tf:
        return set(tf.getnames())


# --- Export ---

@pytest.mark.parametrize("name", ["default", "coder"])
def test_authoritative_export_excludes_native_and_includes_the_record(profile_env, profiles, name):
    home = _profile(profile_env, name, provider="example")
    native = _dormant(home)
    with native_memory_sentinel(native) as sentinel:
        archive = profiles.export_profile(name, str(profile_env / f"{name}.tar.gz"))
    sentinel.assert_untouched()
    names = _tar_names(archive)
    assert not [n for n in names if f"{name}/memories" in n]
    with tarfile.open(archive) as tf:
        record = yaml.safe_load(tf.extractfile(f"{name}/{DISPOSITION_RECORD_NAME}").read())
    assert record == {"curated_memory": {**SPEC_BLOCK, "provider": "example"}}


@pytest.mark.parametrize("name", ["default", "coder"])
def test_additive_export_is_unchanged(profile_env, profiles, name):
    home = _profile(profile_env, name)
    _dormant(home)
    names = _tar_names(profiles.export_profile(name, str(profile_env / f"{name}.tar.gz")))
    assert f"{name}/memories/MEMORY.md" in names
    assert f"{name}/{DISPOSITION_RECORD_NAME}" not in names


def test_active_migration_refuses_export_and_writes_no_archive(profile_env, profiles):
    home = _profile(profile_env, "coder")
    _migration(home, "active")
    from hermes_cli.backup_memory import MigrationInProgressError
    with pytest.raises(MigrationInProgressError):
        profiles.export_profile("coder", str(profile_env / "coder.tar.gz"))
    assert not (profile_env / "coder.tar.gz").exists()


@pytest.mark.parametrize("provider", [None, "example"])  # ruling X-1 (a): every mode
def test_export_and_import_never_carry_host_session_state(profile_env, profiles, provider):
    """X-1 (a): the name is pruned on export and withheld on import, authoritative or additive."""
    home = _profile(profile_env, "coder", provider=provider)
    _host_state(home)
    archive = profiles.export_profile("coder", str(profile_env / "coder.tar.gz"))
    assert not [n for n in _tar_names(archive) if "memory_service" in n]


def test_a_stale_export_record_is_never_carried(profile_env, profiles):  # ruling R44-5
    home = _profile(profile_env, "coder")
    (home / DISPOSITION_RECORD_NAME).write_text("curated_memory: {stale: true}\n", encoding="utf-8")
    archive = profiles.export_profile("coder", str(profile_env / "coder.tar.gz"))
    assert f"coder/{DISPOSITION_RECORD_NAME}" not in _tar_names(archive)


# --- Profile import ---

def test_profile_import_withholds_legacy_native_memory(profile_env, profiles):  # R44-6
    staging = profile_env / "stage" / "legacy"
    _profile_like(staging, provider="example")
    _dormant(staging)
    archive = profile_env / "legacy.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(staging, arcname="legacy")
    target_native = profile_env / ".hermes" / "profiles" / "legacy" / "memories"
    with native_memory_sentinel(target_native) as sentinel:
        pdir = profiles.import_profile(str(archive))
    sentinel.assert_untouched()
    assert not (pdir / "memories").exists()
    assert (pdir / "config.yaml").exists()


def test_additive_profile_import_still_restores_native_memory(profile_env, profiles):
    staging = profile_env / "stage" / "plain"
    _profile_like(staging)
    _dormant(staging)
    archive = profile_env / "plain.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(staging, arcname="plain")
    pdir = profiles.import_profile(str(archive))
    assert (pdir / "memories" / "MEMORY.md").read_text(encoding="utf-8") == "dormant native memory\n"


@pytest.mark.parametrize("provider", [None, "example"])  # ruling X-1 (a): every mode
def test_profile_import_withholds_host_session_state(profile_env, profiles, provider):
    staging = profile_env / "stage" / "stateful"
    _profile_like(staging, provider=provider)
    _host_state(staging)
    archive = profile_env / "stateful.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(staging, arcname="stateful")
    pdir = profiles.import_profile(str(archive))
    assert not (pdir / "memory_service").exists()
    assert (pdir / "config.yaml").exists()
