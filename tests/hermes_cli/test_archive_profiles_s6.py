"""S6 archive suite D: profile export/import/clone under the native sentinel (§9.8, §9.10 L1669).

Helpers are copied rather than imported: the shared test support files stay frozen
and each suite keeps its own helpers (D-R44-f, reconciliation §3 rule 10).
"""

import importlib
import json
import tarfile
from argparse import Namespace
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


# --- Task 8: clone (--clone, --clone-all, --clone-from) ---

CLONES = [{"clone_config": True}, {"clone_all": True}, {"clone_from": "default"}]


@pytest.mark.parametrize("kw", CLONES)
def test_clone_of_an_authoritative_source_never_touches_native(profile_env, profiles, kw):
    source = _profile(profile_env, "default", provider="example")
    native = _dormant(source)
    with native_memory_sentinel(native) as sentinel:
        pdir = profiles.create_profile("coder", no_alias=True, **kw)
    sentinel.assert_untouched()
    assert not (pdir / "memories").exists()                    # no native dir initialized (R37 precedent)
    record = yaml.safe_load((pdir / DISPOSITION_RECORD_NAME).read_text(encoding="utf-8"))
    assert record == {"curated_memory": {**SPEC_BLOCK, "provider": "example"}}


@pytest.mark.parametrize("kw", CLONES)
def test_additive_clone_still_copies_native_memory(profile_env, profiles, kw):
    _dormant(_profile(profile_env, "default"))
    pdir = profiles.create_profile("coder", no_alias=True, **kw)
    assert (pdir / "memories" / "MEMORY.md").read_text(encoding="utf-8") == "dormant native memory\n"
    assert not (pdir / DISPOSITION_RECORD_NAME).exists()


@pytest.mark.parametrize("kw", CLONES)
def test_active_migration_refuses_clone_and_creates_nothing(profile_env, profiles, kw):
    _migration(_profile(profile_env, "default"), "active")
    from hermes_cli.backup_memory import MigrationInProgressError
    with pytest.raises(MigrationInProgressError):
        profiles.create_profile("coder", no_alias=True, **kw)
    assert not (profile_env / ".hermes" / "profiles" / "coder").exists()


def test_fresh_profile_creation_is_unchanged(profile_env, profiles):
    _profile(profile_env, "default", provider="example")          # the source mode must not leak into a fresh profile
    pdir = profiles.create_profile("fresh", no_alias=True)
    assert (pdir / "memories").is_dir() and not (pdir / DISPOSITION_RECORD_NAME).exists()


@pytest.mark.parametrize("kw", CLONES)
@pytest.mark.parametrize("provider", [None, "example"])  # ruling X-1 (a): every mode
def test_clone_never_copies_host_session_state(profile_env, profiles, kw, provider):
    """X-1 (a): `memory_service/` is a mode-independent exclusion, so no clone form carries it."""
    source = _profile(profile_env, "default", provider=provider)
    _host_state(source)
    pdir = profiles.create_profile("coder", no_alias=True, **kw)
    assert not (pdir / "memory_service").exists()


# --- Task 9: terminal messaging at CLI and slash entry points (R44-11 (b)) ---

SPEC_SENTENCE = "Hermes archive complete; authoritative ygg memory is provider-managed and not included"


def _profile_cmd():
    return importlib.import_module("hermes_cli.profile_cmd")


def _create_args(name, **over):
    ns = Namespace(profile_action="create", profile_name=name, clone=False, clone_all=False,
                   no_alias=True, no_skills=False, clone_from=None, clone_channels=False,
                   description=None)
    for key, value in over.items():
        setattr(ns, key, value)
    return ns


def _mixin():
    from hermes_cli.cli_commands_mixin import CLICommandsMixin
    return CLICommandsMixin()


def test_cli_export_prints_the_disposition(profile_env, profiles, capsys):
    _profile(profile_env, "coder", provider="ygg")
    _profile_cmd().cmd_profile(Namespace(profile_action="export", profile_name="coder",
                                         output=str(profile_env / "coder.tar.gz")))
    assert SPEC_SENTENCE in capsys.readouterr().out


def test_cli_import_prints_the_restore_disposition(profile_env, profiles, capsys):
    staging = profile_env / "stage" / "shared"
    _profile_like(staging, provider="ygg")
    archive = profile_env / "shared.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(staging, arcname="shared")
    _profile_cmd().cmd_profile(Namespace(profile_action="import", archive=str(archive), import_name=None))
    out = capsys.readouterr().out
    assert "was not restored" in out and "reconnect-provider" in out


def test_cli_clone_prints_the_disposition(profile_env, profiles, capsys):
    _profile(profile_env, "default", provider="ygg")
    _profile_cmd().cmd_profile(_create_args("coder", clone=True))
    assert "authoritative ygg memory is provider-managed and not included" in capsys.readouterr().out


def test_slash_export_and_import_print_the_disposition(profile_env, profiles, capsys):
    _profile(profile_env, "coder", provider="ygg")
    mixin = _mixin()
    mixin._handle_export_command(f"/export coder -o {profile_env / 'coder.tar.gz'}")
    assert SPEC_SENTENCE in capsys.readouterr().out
    mixin._handle_import_command(f"/import {profile_env / 'coder.tar.gz'} --name copy")
    out = capsys.readouterr().out
    assert "was not restored" in out and "reconnect-provider" in out


def test_slash_snapshot_restore_prints_the_restore_disposition(profile_env, profiles, capsys, monkeypatch):
    home = _profile(profile_env, "default", provider="ygg")
    from hermes_cli import backup as backup_mod
    snap = backup_mod.create_quick_snapshot(hermes_home=home)
    assert snap is not None
    capsys.readouterr()
    _mixin()._snapshot_restore(["/snapshot", "restore", snap])
    out = capsys.readouterr().out
    assert "was not restored" in out and "reconnect-provider" in out


def test_slash_snapshot_restore_that_switches_to_additive_warns_native_is_stale(profile_env, profiles, capsys):
    """R44-1 lists /snapshot restore; R44-7 and §9.9 L1662 require the stale-native warning on the flip."""
    home = _profile(profile_env, "default")                       # additive when the snapshot is taken
    from hermes_cli import backup as backup_mod
    snap = backup_mod.create_quick_snapshot(hermes_home=home)
    assert snap is not None
    _profile(profile_env, "default", provider="ygg")              # the operator later goes authoritative
    capsys.readouterr()
    _mixin()._snapshot_restore(["/snapshot", "restore", snap])
    out = capsys.readouterr().out
    assert "Restored state from" in out
    assert "authoritative to additive" in out and "stale" in out


def test_additive_cli_and_slash_output_carries_no_disposition(profile_env, profiles, capsys):
    _profile(profile_env, "coder")
    _profile_cmd().cmd_profile(Namespace(profile_action="export", profile_name="coder",
                                         output=str(profile_env / "coder.tar.gz")))
    assert "provider-managed" not in capsys.readouterr().out
