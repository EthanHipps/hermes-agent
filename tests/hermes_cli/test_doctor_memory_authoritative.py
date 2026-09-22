"""§9.7 Doctor row: in authoritative mode, probe the provider contract without
touching the native memory directory."""

import pytest
import yaml

from tests.agent.memory_service.native_sentinel import native_memory_sentinel


@pytest.fixture
def doctor_home(tmp_path, monkeypatch):
    # These tests isolate the checks; test_memory_cold_startup covers imports
    # and skeleton creation with the sentinel armed in a fresh interpreter.
    import hermes_cli.doctor as doctor
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(doctor, "HERMES_HOME", home, raising=False)
    return home


def _write_config(home, section):
    (home / "config.yaml").write_text(yaml.safe_dump({"memory": section}), encoding="utf-8")


def test_authoritative_doctor_never_touches_the_native_directory(doctor_home, tmp_path):
    exe = tmp_path / "p.exe"
    exe.write_bytes(b"MZ")
    _write_config(doctor_home, {"provider": "example", "provider_mode": "authoritative",
                                "provider_executable": str(exe), "principal_id": "ethan"})
    memories = doctor_home / "memories"
    memories.mkdir()
    (memories / "MEMORY.md").write_text("stale\n", encoding="utf-8")
    from hermes_cli.doctor_state import _check_directory_structure

    # @doctor_check() turns fn(should_fix, f) into check(should_fix) -> Finding
    # (doctor_report.py:68-84), so the decorated callable takes ONE argument.
    with native_memory_sentinel(memories) as sentinel:
        _check_directory_structure(True)  # should_fix=True: the creating path
    sentinel.assert_untouched()


def test_authoritative_doctor_does_not_create_the_native_directory(doctor_home, tmp_path):
    """--fix must not resurrect a dormant directory in authoritative mode."""
    exe = tmp_path / "p.exe"
    exe.write_bytes(b"MZ")
    _write_config(doctor_home, {"provider": "example", "provider_mode": "authoritative",
                                "provider_executable": str(exe), "principal_id": "ethan"})
    from hermes_cli.doctor_state import _check_directory_structure

    _check_directory_structure(True)
    assert not (doctor_home / "memories").exists()


def test_additive_doctor_still_checks_the_native_directory(doctor_home):
    _write_config(doctor_home, {"memory_enabled": True, "user_profile_enabled": True})
    from hermes_cli.doctor_state import _check_directory_structure

    _check_directory_structure(True)
    assert (doctor_home / "memories").exists(), "additive doctor must still ensure memories/"


def test_authoritative_provider_probe_never_touches_the_native_directory(doctor_home, tmp_path):
    """The probe's whole job is to check the provider contract WITHOUT touching
    native memory. Prove it with the sentinel, not by reading the code."""
    exe = tmp_path / "p.exe"
    exe.write_bytes(b"MZ")
    _write_config(doctor_home, {"provider": "example", "provider_mode": "authoritative",
                                "provider_executable": str(exe), "principal_id": "ethan"})
    memories = doctor_home / "memories"
    memories.mkdir()
    (memories / "MEMORY.md").write_text("stale\n", encoding="utf-8")
    (memories / "USER.md").write_text("stale\n", encoding="utf-8")
    from hermes_cli.doctor_state import _check_memory_provider

    with native_memory_sentinel(memories) as sentinel:
        finding = _check_memory_provider(True)
    sentinel.assert_untouched()
    assert not finding.issues, "a valid authoritative config should raise no doctor issues"


@pytest.mark.parametrize("invalid", ["principal_id", "provider_executable"])
@pytest.mark.parametrize("existing", [False, True])
def test_invalid_authoritative_config_is_reported_without_native_access(doctor_home, invalid, existing):
    import sys

    section = {"provider": "example", "provider_mode": "authoritative",
               "provider_executable": sys.executable, "principal_id": "ethan"}
    section[invalid] = "" if invalid == "principal_id" else str(doctor_home / "missing.exe")
    _write_config(doctor_home, section)
    memories = doctor_home / "memories"
    if existing:
        memories.mkdir()
        (memories / "MEMORY.md").write_text("dormant memory", encoding="utf-8")
        (memories / "USER.md").write_text("dormant profile", encoding="utf-8")
    from hermes_cli.doctor_state import _check_directory_structure, _check_memory_provider

    with native_memory_sentinel(memories) as sentinel:
        finding = _check_memory_provider(True)
        _check_directory_structure(True)
    sentinel.assert_untouched()
    assert any(invalid in issue for issue in finding.issues)
    assert memories.exists() is existing
