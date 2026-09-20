"""§9.7 Doctor row: in authoritative mode, probe the provider contract without
touching the native memory directory."""

import pytest
import yaml

from tests.agent.memory_service.native_sentinel import native_memory_sentinel


@pytest.fixture
def doctor_home(tmp_path, monkeypatch):
    # Import (and thus warm any load_config()-triggered ensure_hermes_home() skeleton
    # creation, e.g. via env_loader's terminal-config bridge) BEFORE HERMES_HOME points
    # at this test's fresh directory -- otherwise the first-ever import of
    # hermes_cli.doctor in this process creates memories/ (and SOUL.md) under our own
    # tmp_path as a side effect of module import, which the tests below must not see.
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
