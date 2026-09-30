"""R45 start: §9.9 steps 1-3 and the pre-clean crash boundaries of §13.1 L2182 (D-R45-1)."""

import json

import pytest

from agent.memory_service import migration_manifest as mm
from agent.memory_service.migration import MigrationStatus
from hermes_cli.backup_memory import active_migration_manifests
from tests.agent.memory_service.migration_support import (
    MARKER, Answers, Crash, file_bytes_under, make_env, run_files, start, write_native)


@pytest.fixture
def env(tmp_path):
    e = make_env(tmp_path)
    write_native(e.home, memory=[f"uses postgres {MARKER}", "prefers tabs"], user=["name is Ethan"])
    return e


def _no_host_state(env):
    assert not (env.home / "migrations").exists()
    assert not (env.home / "memory_service" / "approvals").exists()


def _stages(env):
    return [key for key in env.store.stage_by_request]


def test_local_secret_rejection_sends_nothing_and_persists_nothing(env):
    write_native(env.home, memory=["api_key = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ012345'"])
    report = start(env, Answers())
    assert (report.status, report.code) == (MigrationStatus.REFUSED, "source_threat")
    assert env.calls("stage_curated") == []
    assert _stages(env) == [] and env.store.receipts == {}
    _no_host_state(env)


def test_provider_secret_rejection_persists_nothing(env):
    env.store.secret_detector = lambda text: "fixture_secret" if MARKER in text else None
    report = start(env, Answers())
    assert (report.status, report.code) == (MigrationStatus.REJECTED, "secret_rejected")
    assert _stages(env) == [] and report.run_id is None
    _no_host_state(env)


def test_stage_transport_failure_before_publish_persists_nothing(env):
    env.store.fail_transport("stage_curated", phase="before", times=2)
    report = start(env, Answers())
    assert (report.status, report.code) == (MigrationStatus.STOPPED, "outcome_unknown") and report.run_id is None
    assert _stages(env) == []
    _no_host_state(env)


def test_an_unknown_stage_outcome_persists_nothing_and_leaves_only_an_uncommitted_stage(env):
    # The first send persists the stage and loses the reply; the identical retry fails before reaching the store.
    env.store.fail_transport("stage_curated", phase="during")
    env.store.fail_transport("stage_curated", phase="before")
    report = start(env, Answers())
    assert (report.status, report.code, report.run_id) == (MigrationStatus.STOPPED, "outcome_unknown", None)
    _no_host_state(env)
    [(_, request_id)] = _stages(env)
    assert env.store.stage_state(request_id) == "live" and env.store.receipts == {}   # it expires on its own (§9.9 L1663)


def test_a_crash_after_the_clean_stage_before_the_manifest_leaves_no_host_state(env, monkeypatch):
    def crash(*a, **k):
        raise Crash()
    monkeypatch.setattr(mm, "write_document", crash)
    with pytest.raises(Crash):
        start(env, Answers())
    # The lock file may exist (it follows a clean stage); no run file, no C9 manifest, no approval record.
    assert run_files(env.home) == [] and active_migration_manifests(env.home) == []
    assert not (env.home / "memory_service" / "approvals").exists()
    assert len(_stages(env)) == 1 and env.store.receipts == {}
    ids = [key[1] for key in _stages(env)]
    assert all(i.encode() not in file_bytes_under(env.home) for i in ids)


def test_a_run_that_became_active_meanwhile_is_not_joined(env, monkeypatch):     # ruling R45-22
    from agent.memory_service import migration
    real_scan = migration.scan_items
    seeded = mm.manifest_path(env.home, "example", "ep-1", "e" * 32)

    def scan_and_seed(items):
        seeded.parent.mkdir(parents=True, exist_ok=True)
        seeded.write_text('{"state": "active"}', encoding="utf-8")
        return real_scan(items)
    monkeypatch.setattr(migration, "scan_items", scan_and_seed)
    report = start(env, Answers(True, True))
    assert (report.status, report.code) == (MigrationStatus.STOPPED, "MIGRATION_IN_PROGRESS")
    assert run_files(env.home) == [seeded]
    assert env.store.receipts == {}


@pytest.mark.parametrize("change,code", [
    (lambda e: e.section.update(provider_mode="authoritative"), "native_dormant"),
    (lambda e: e.section.update(principal_id=""), "configuration_error"),
    (lambda e: e.section.update(provider="Bad/Name"), "configuration_error"),
])
def test_preconditions_refuse_before_any_provider_contact(env, change, code):
    from tests.agent.memory_service.migration_support import write_memory_section
    change(env)
    write_memory_section(env.home, env.section)
    report = start(env, Answers())
    assert (report.status, report.code) == (MigrationStatus.REFUSED, code)
    assert env.store.handles == {}
    _no_host_state(env)


def test_an_active_run_refuses_a_second_start(env):
    path = mm.manifest_path(env.home, "example", "ep-1", "f" * 32)
    path.parent.mkdir(parents=True)
    path.write_text('{"state": "active"}', encoding="utf-8")
    report = start(env, Answers())
    assert (report.status, report.code) == (MigrationStatus.REFUSED, "MIGRATION_IN_PROGRESS")   # ruling R45-17
    assert env.backends == []


def test_two_items_for_one_target_refuse(env, tmp_path):     # one item per target per run (Checkpoint A, R45-6)
    report = start(env, Answers(), kind="legacy_archive", items=("memories/MEMORY.md", "profiles/w/memories/MEMORY.md"),
                   archive=tmp_path / "a.zip")
    assert (report.status, report.code) == (MigrationStatus.REFUSED, "source_invalid") and env.backends == []


def test_a_destination_outside_the_identity_refuses_before_staging(env):
    from agent.memory_service import wire as w
    report = start(env, Answers(), scope=w.ScopeRef(kind="repository", id="elsewhere"))
    assert (report.status, report.code) == (MigrationStatus.REFUSED, "unauthorized_scope") and _stages(env) == []


def test_an_empty_source_refuses(env):
    write_native(env.home, memory=[], user=[])
    report = start(env, Answers())
    assert (report.status, report.code) == (MigrationStatus.REFUSED, "nothing_to_migrate")
    assert json.dumps(report.message) and _stages(env) == []
