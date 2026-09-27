"""§9.7 Doctor row: in authoritative mode, probe the provider contract without
touching the native memory directory."""

from types import SimpleNamespace

import pytest
import yaml

from tests.agent.memory_service.native_sentinel import native_memory_sentinel


@pytest.fixture
def fake_provider(monkeypatch):
    """Doctor probes the provider with negotiate only (ruling R41-6); the R36 fake answers it."""
    from tests.agent.memory_service.fake_backend import FakeAuthoritativeBackend, FakeProviderStore
    store, backends = FakeProviderStore(epoch="EPOCHVALUEAAAA"), []   # distinctive, so its absence is meaningful

    def factory(cfg):
        backends.append(FakeAuthoritativeBackend(store, provider="example"))
        return backends[-1]

    monkeypatch.setattr("plugins.memory.load_authoritative_backend_factory", lambda name: factory)
    return SimpleNamespace(store=store, backends=backends)


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


def test_authoritative_provider_probe_never_touches_the_native_directory(doctor_home, tmp_path, fake_provider):
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


def _authoritative_config(home, tmp_path):
    exe = tmp_path / "p.exe"
    exe.write_bytes(b"MZ")
    _write_config(home, {"provider": "example", "provider_mode": "authoritative",
                         "provider_executable": str(exe), "principal_id": "ethan"})


def test_doctor_probe_negotiates_once_and_never_binds(doctor_home, tmp_path, fake_provider, capsys):
    _authoritative_config(doctor_home, tmp_path)
    from hermes_cli.doctor_state import _check_memory_provider
    finding = _check_memory_provider(True)
    backend = fake_provider.backends[-1]
    assert not finding.issues and [op for op, _ in backend.calls] == ["negotiate"] and backend.shutdown_calls == 1
    assert fake_provider.store.epoch not in capsys.readouterr().out


def test_doctor_reports_an_unreachable_provider_as_an_issue(doctor_home, tmp_path, fake_provider):
    _authoritative_config(doctor_home, tmp_path)
    fake_provider.store.fail_transport("negotiate")
    from hermes_cli.doctor_state import _check_memory_provider
    assert any("probe" in issue.lower() for issue in _check_memory_provider(True).issues)


def test_doctor_reports_an_invalid_gateway_mapping(doctor_home, tmp_path, fake_provider):
    _authoritative_config(doctor_home, tmp_path)
    section = yaml.safe_load((doctor_home / "config.yaml").read_text(encoding="utf-8"))["memory"]
    _write_config(doctor_home, {**section, "gateway_principals": ["telegram:1"]})
    from hermes_cli.doctor_state import _check_memory_provider
    assert any("gateway_principals" in issue for issue in _check_memory_provider(True).issues)
