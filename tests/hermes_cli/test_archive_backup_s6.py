"""S6 archive suite C: backup/snapshot/import archives under the native sentinel (§9.8, §9.10 L1669)."""

import importlib
import json
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from agent.memory_service.archive import DISPOSITION_RECORD_NAME
from tests.agent.memory_service.native_sentinel import native_memory_sentinel

SPEC_BLOCK = {"authority": "provider", "provider": "ygg", "provider_api": 1, "included": False,
              "disposition": "provider-managed", "restore_action": "reconnect-provider"}


@pytest.fixture(autouse=True)
def _no_real_gateway_service(monkeypatch):
    import hermes_cli.gateway as gateway_mod
    monkeypatch.setattr(gateway_mod, "ensure_gateway_service", lambda **kw: False)
    monkeypatch.setattr(gateway_mod, "_is_service_running", lambda: False)


def _backup():
    return importlib.import_module("hermes_cli.backup")


def _authoritative_yaml(exe: Path, provider: str, policy: str = "") -> str:
    extra = f"  authoritative_failure_policy: {policy}\n" if policy else ""
    return (f"memory:\n  provider: {provider}\n  provider_mode: authoritative\n"
            f"  provider_executable: '{exe}'\n  principal_id: ethan\n{extra}")


def _home(path: Path, *, provider=None, policy="") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if provider is None:
        (path / "config.yaml").write_text("model:\n  provider: openrouter\n", encoding="utf-8")
    else:
        exe = path.parent / f"{path.name}-provider.exe"
        exe.write_bytes(b"MZ")
        (path / "config.yaml").write_text(_authoritative_yaml(exe, provider, policy), encoding="utf-8")
    (path / "skills").mkdir(exist_ok=True)
    (path / "skills" / "note.md").write_text("skill\n", encoding="utf-8")
    return path


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


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    def use(home: Path) -> Path:
        monkeypatch.setenv("HERMES_HOME", str(home))
        return home

    return use


def test_authoritative_backup_touches_no_native_file_and_carries_the_record(tmp_path, env, capsys):
    home = env(_home(tmp_path / ".hermes", provider="ygg"))
    native = _dormant(home)
    out_zip = tmp_path / "out.zip"
    with native_memory_sentinel(native) as sentinel:
        _backup().run_backup(Namespace(output=str(out_zip)))
    sentinel.assert_untouched()
    with zipfile.ZipFile(out_zip) as zf:
        names = zf.namelist()
        record = yaml.safe_load(zf.read(DISPOSITION_RECORD_NAME))
    assert not [n for n in names if n == "memories" or n.startswith("memories/")]
    assert record == {"curated_memory": SPEC_BLOCK}
    out = capsys.readouterr().out
    assert "Hermes archive complete; authoritative ygg memory is provider-managed and not included" in out
    assert "memory-provider file" not in out


def test_authoritative_backup_never_discovers_provider_paths(tmp_path, env, monkeypatch):
    env(_home(tmp_path / ".hermes", provider="example"))
    calls = []
    monkeypatch.setattr(_backup(), "_collect_memory_provider_external_paths", lambda: calls.append(1) or [])
    _backup().run_backup(Namespace(output=str(tmp_path / "out.zip")))
    assert calls == []


def test_additive_backup_is_unchanged(tmp_path, env):
    home = env(_home(tmp_path / ".hermes"))
    _dormant(home)
    _backup().run_backup(Namespace(output=str(tmp_path / "out.zip")))
    with zipfile.ZipFile(tmp_path / "out.zip") as zf:
        names = set(zf.namelist())
    assert {"memories/MEMORY.md", "memories/USER.md"} <= names
    assert DISPOSITION_RECORD_NAME not in names


def test_root_backup_applies_each_homes_own_mode(tmp_path, env):  # ruling R44-2
    root = env(_home(tmp_path / ".hermes"))
    _dormant(root)
    profile = _home(root / "profiles" / "coder", provider="example")
    profile_native = _dormant(profile)
    with native_memory_sentinel(profile_native) as sentinel:
        _backup().run_backup(Namespace(output=str(tmp_path / "out.zip")))
    sentinel.assert_untouched()
    with zipfile.ZipFile(tmp_path / "out.zip") as zf:
        names = set(zf.namelist())
    assert "memories/MEMORY.md" in names and DISPOSITION_RECORD_NAME not in names
    assert not [n for n in names if n.startswith("profiles/coder/memories")]
    assert f"profiles/coder/{DISPOSITION_RECORD_NAME}" in names


def test_a_stale_record_on_disk_is_never_copied(tmp_path, env):  # ruling R44-5
    home = env(_home(tmp_path / ".hermes"))
    (home / DISPOSITION_RECORD_NAME).write_text("curated_memory: {stale: true}\n", encoding="utf-8")
    _backup().run_backup(Namespace(output=str(tmp_path / "out.zip")))
    with zipfile.ZipFile(tmp_path / "out.zip") as zf:
        assert DISPOSITION_RECORD_NAME not in zf.namelist()


@pytest.mark.parametrize("provider", [None, "example"])  # additive and authoritative: R44-8
def test_active_migration_refuses_backup_before_any_archive_exists(tmp_path, env, capsys, provider):
    home = env(_home(tmp_path / ".hermes", provider=provider))
    _migration(home, "active")
    out_zip = tmp_path / "out.zip"
    with pytest.raises(SystemExit) as exc:
        _backup().run_backup(Namespace(output=str(out_zip)))
    assert exc.value.code == 2
    assert not out_zip.exists() and not list(tmp_path.glob(".out.zip.*.partial"))
    assert "MIGRATION_IN_PROGRESS" in capsys.readouterr().out


@pytest.mark.parametrize("state", ["completed", "rolled_back"])
def test_compacted_receipt_does_not_block_and_never_enters_the_archive(tmp_path, env, state):
    home = env(_home(tmp_path / ".hermes", provider="example"))
    _migration(home, state)
    _backup().run_backup(Namespace(output=str(tmp_path / "out.zip")))
    with zipfile.ZipFile(tmp_path / "out.zip") as zf:
        assert not [n for n in zf.namelist() if n.startswith("migrations")]


@pytest.mark.parametrize("provider", [None, "example"])  # ruling X-1 (a): every mode
def test_host_session_state_never_enters_a_backup(tmp_path, env, provider):
    """X-1 (a): R40 persists HostSessionState under <home>/memory_service/sessions/ (§9.8 L1639)."""
    home = env(_home(tmp_path / ".hermes", provider=provider))
    _host_state(home)
    _backup().run_backup(Namespace(output=str(tmp_path / "out.zip")))
    with zipfile.ZipFile(tmp_path / "out.zip") as zf:
        assert not [n for n in zf.namelist() if n == "memory_service" or n.startswith("memory_service/")]


# --- Task 5: automatic full zips and quick snapshots ---

def test_pre_update_zip_is_dormant_and_carries_the_record(tmp_path, env):
    home = env(_home(tmp_path / ".hermes", provider="example"))
    native = _dormant(home)
    with native_memory_sentinel(native) as sentinel:
        out = _backup().create_pre_update_backup(hermes_home=home)
    sentinel.assert_untouched()
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert yaml.safe_load(zf.read(DISPOSITION_RECORD_NAME)) == {
            "curated_memory": {**SPEC_BLOCK, "provider": "example"}}
    assert not [n for n in names if n.startswith("memories")]


@pytest.mark.parametrize("helper", ["create_pre_update_backup", "create_pre_migration_backup"])
def test_automatic_zips_are_skipped_during_an_active_migration(tmp_path, env, capsys, helper):  # R44-8(a)
    home = env(_home(tmp_path / ".hermes"))
    _migration(home, "active")
    assert getattr(_backup(), helper)(hermes_home=home) is None
    assert not list((home / "backups").glob("*.zip"))
    assert "MIGRATION_IN_PROGRESS" in capsys.readouterr().out


def test_quick_snapshot_manifest_carries_the_disposition(tmp_path, env, capsys):
    home = env(_home(tmp_path / ".hermes", provider="example"))
    snap = _backup().create_quick_snapshot(hermes_home=home)
    manifest = json.loads((home / "state-snapshots" / snap / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["curated_memory"] == {**SPEC_BLOCK, "provider": "example"}
    assert "provider-managed and not included" in capsys.readouterr().out


def test_additive_quick_snapshot_manifest_has_no_disposition(tmp_path, env):
    home = env(_home(tmp_path / ".hermes"))
    snap = _backup().create_quick_snapshot(hermes_home=home)
    manifest = json.loads((home / "state-snapshots" / snap / "manifest.json").read_text(encoding="utf-8"))
    assert "curated_memory" not in manifest


def test_quick_snapshot_is_not_refused_by_an_active_migration(tmp_path, env):  # R44-8(a)
    home = env(_home(tmp_path / ".hermes", provider="example"))
    _migration(home, "active")
    assert _backup().create_quick_snapshot(hermes_home=home) is not None


def test_archives_never_construct_a_provider_backend(tmp_path, env, monkeypatch):  # R44-10, §9.6 L1584
    home = env(_home(tmp_path / ".hermes", provider="example"))

    def refuse(name):
        raise AssertionError("an archive operation contacted the provider")

    monkeypatch.setattr("plugins.memory.load_authoritative_backend_factory", refuse)
    _backup().run_backup(Namespace(output=str(tmp_path / "out.zip")))
    assert _backup().create_quick_snapshot(hermes_home=home) is not None
    assert _backup().create_pre_update_backup(hermes_home=home) is not None
