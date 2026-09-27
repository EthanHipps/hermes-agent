"""Checkpoint B (ledger L569): an unparseable config.yaml must not revive an authoritative home's native
memory. Doctor, the home skeleton and archive creation read the requested mode through one pipeline
(D-R44-e; ruling R41-10) that falls back to the last-known-good copy load_config() left (ruling R41-9)."""

import importlib
import zipfile
from pathlib import Path

import pytest
import yaml

from agent.memory_service.archive import DISPOSITION_RECORD_NAME
from tests.agent.memory_service.native_sentinel import native_memory_sentinel

BROKEN = "memory: [unclosed\n"


@pytest.fixture(autouse=True)
def _no_real_gateway_service(monkeypatch):
    import hermes_cli.gateway as gateway_mod
    monkeypatch.setattr(gateway_mod, "ensure_gateway_service", lambda **kw: False)
    monkeypatch.setattr(gateway_mod, "_is_service_running", lambda: False)


def _authoritative(home: Path) -> Path:
    exe = home.parent / "p.exe"
    exe.write_bytes(b"MZ")
    path = home / "config.yaml"
    path.write_text(yaml.safe_dump({"memory": {"provider": "example", "provider_mode": "authoritative",
                                               "provider_executable": str(exe), "principal_id": "ethan"}}),
                    encoding="utf-8")
    return path


def _dormant(home: Path) -> Path:
    native = home / "memories"
    native.mkdir(parents=True, exist_ok=True)
    (native / "MEMORY.md").write_text("dormant\n", encoding="utf-8")
    return native


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _break_after_good(home):
    from hermes_cli.config_backups import backup_config
    path = _authoritative(home)
    assert backup_config(path, "good") is not None
    path.write_text(BROKEN, encoding="utf-8")


def test_unparseable_config_reads_the_last_known_good_mode(home):
    from hermes_cli.backup_memory import archive_prune_names, home_memory_section
    _break_after_good(home)
    assert home_memory_section(home)["provider_mode"] == "authoritative"
    assert "memories" in archive_prune_names(home)


def test_without_a_last_known_good_copy_the_home_stays_additive(home):
    from hermes_cli.backup_memory import archive_prune_names, home_memory_section
    (home / "config.yaml").write_text(BROKEN, encoding="utf-8")
    assert home_memory_section(home) == {} and "memories" not in archive_prune_names(home)


def test_home_skeleton_does_not_create_memories_from_a_broken_authoritative_config(home):
    from hermes_cli.config_home import initialize_home
    _break_after_good(home)
    initialize_home(home, ("cron", "sessions", "logs", "memories"), set())
    assert not (home / "memories").exists()


def test_doctor_does_not_touch_memories_under_a_broken_authoritative_config(home, monkeypatch):
    import hermes_cli.doctor as doctor
    monkeypatch.setattr(doctor, "HERMES_HOME", home, raising=False)
    _break_after_good(home)
    native = _dormant(home)
    from hermes_cli.doctor_state import _check_directory_structure
    with native_memory_sentinel(native) as sentinel:
        _check_directory_structure(True)
    sentinel.assert_untouched()


def test_pre_update_zip_withholds_native_memory_under_a_broken_authoritative_config(home):
    _break_after_good(home)
    native = _dormant(home)
    with native_memory_sentinel(native) as sentinel:
        out = importlib.import_module("hermes_cli.backup").create_pre_update_backup(hermes_home=home)
    sentinel.assert_untouched()
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert yaml.safe_load(zf.read(DISPOSITION_RECORD_NAME))["curated_memory"]["provider"] == "example"
    assert not [n for n in names if n.startswith("memories")]


def test_a_failing_managed_overlay_keeps_the_homes_own_section(home, monkeypatch):
    """R41-10: doctor's tolerant overlay, now shared."""
    from hermes_cli.backup_memory import home_memory_section
    _authoritative(home)
    monkeypatch.setattr("hermes_cli.managed_scope.apply_managed_overlay", lambda config: 1 / 0)
    assert home_memory_section(home)["provider_mode"] == "authoritative"
