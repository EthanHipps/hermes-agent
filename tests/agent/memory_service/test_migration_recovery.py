"""R45 recovery: explicit rollback, reconciliation and status (§9.9 L1655, L1670; §9.5 L1546; R45-15, R45-16,
R45-18, R45-23), including §13.1 L2182's rollback-compaction boundary."""

from datetime import datetime, timezone

import pytest

from agent.memory_service import migration_manifest as mm
from agent.memory_service.migration import MigrationStatus
from agent.memory_service.migration_recovery import migration_summaries, reconcile_migration, rollback_migrations
from hermes_cli.backup_memory import active_migration_manifests
from tests.agent.memory_service.migration_support import (
    MARKER, Answers, Crash, make_env, run_files, start, write_memory_section, write_native)


@pytest.fixture
def env(tmp_path):
    e = make_env(tmp_path)
    write_native(e.home, memory=[f"uses postgres {MARKER}", "prefers tabs"], user=["name is Ethan"])
    return e


def _crash_after_manifest(env, monkeypatch):
    real = mm.write_document

    def write(path, doc):
        real(path, doc)
        raise Crash()
    monkeypatch.setattr(mm, "write_document", write)
    with pytest.raises(Crash):
        start(env, Answers(True, True))
    monkeypatch.undo()
    [path] = run_files(env.home)
    return path


def _unknown_commit(env):
    """The commit published and lost its reply; the identical retry failed before the store (correction 4)."""
    env.store.fail_transport("commit_curated", phase="during")
    env.store.fail_transport("commit_curated", phase="before")
    report = start(env, Answers(True, True))
    assert (report.status, report.code) == (MigrationStatus.STOPPED, "outcome_unknown")
    assert not env.store._transport_faults.get(("commit_curated", "during"))
    assert not env.store._transport_faults.get(("commit_curated", "before"))
    [path] = run_files(env.home)
    return path


def _doc(path):
    return mm.read_document(path, provider=path.parent.parent.name, provider_epoch=path.parent.name,
                            import_run_id=path.stem)


def _rollback(env, **kwargs):
    return rollback_migrations(env.raw(), home=env.home, backend_factory=env.factory, **kwargs)


def _reconcile(env, run_id, **kwargs):
    return reconcile_migration(env.raw(), run_id, home=env.home, backend_factory=env.factory, **kwargs)


def test_rollback_of_an_unapproved_run_needs_no_provider_call(env, monkeypatch):
    path = _crash_after_manifest(env, monkeypatch)
    backends = len(env.backends)
    report = _rollback(env)
    assert report.status is MigrationStatus.ROLLED_BACK
    doc = _doc(path)
    assert (doc.state, doc.outcome) == ("rolled_back", "operator_rollback")
    assert [b.outcome for b in doc.batches] == ["not_committed", "not_committed"]
    assert len(env.backends) == backends and active_migration_manifests(env.home) == []


def test_rollback_probes_an_approved_batch_and_records_a_commit(env):
    path = _unknown_commit(env)
    memory = next(b for b in _doc(path).batches if b.target == "memory")
    tx = env.store.receipts[("ep-1", memory.request_id)].tx_id
    report = _rollback(env)
    assert report.status is MigrationStatus.ROLLED_BACK
    doc = _doc(path)
    assert (doc.state, doc.outcome) == ("rolled_back", "operator_rollback")
    assert [(b.target, b.outcome, b.tx_id) for b in doc.batches] == [("memory", "committed", tx),
                                                                     ("user", "not_committed", None)]
    assert [a.assigned_id for a in doc.batches[0].assigned] == [a.assigned_id for a in memory.admissions]


def test_rollback_refuses_whole_when_a_batch_is_unprovable(env):
    path = _unknown_commit(env)
    before = path.read_bytes()
    env.store.fail_transport("inspect_staged")
    report = _rollback(env)
    assert (report.status, report.code) == (MigrationStatus.REFUSED, "unprovable")
    assert path.read_bytes() == before and active_migration_manifests(env.home) == [path]
    assert not env.store._transport_faults.get(("inspect_staged", "before"))


def test_rollback_never_touches_native_memory(env, monkeypatch):
    from tests.agent.memory_service.native_sentinel import native_memory_sentinel
    _crash_after_manifest(env, monkeypatch)
    with native_memory_sentinel(env.home / "memories", deny=True) as sentinel:
        report = _rollback(env)
    assert report.status is MigrationStatus.ROLLED_BACK and sentinel.accesses == []


def test_reconcile_compacts_a_provable_run_as_reconciled(env, monkeypatch):
    path = _crash_after_manifest(env, monkeypatch)
    env.store.stages.clear()
    env.store.stage_by_request.clear()                        # an R14 discard with no provable request ID
    report = _reconcile(env, path.stem)
    assert report.status is MigrationStatus.RECONCILED
    doc = _doc(path)
    assert (doc.state, doc.outcome) == ("rolled_back", "reconciled")
    assert [b.outcome for b in doc.batches] == ["not_committed", "not_committed"]


def test_reconcile_after_an_epoch_change_needs_discard(env):
    path = _unknown_commit(env)
    env.store.set_epoch("ep-2")
    report = _reconcile(env, path.stem)
    assert (report.status, report.code) == (MigrationStatus.REFUSED, "unprovable")
    assert active_migration_manifests(env.home) == [path]
    report = _reconcile(env, path.stem, discard=True)
    assert report.status is MigrationStatus.RECONCILED
    doc = _doc(path)
    assert (doc.state, doc.outcome) == ("rolled_back", "discarded")
    assert [b.outcome for b in doc.batches] == ["unknown", "not_committed"]


def test_reconcile_discard_deletes_an_unreadable_file_only_with_discard(env):
    path = mm.manifest_path(env.home, "example", "ep-1", "a" * 32)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"{\"state\": \"act")
    report = _reconcile(env, "a" * 32)
    assert (report.status, report.code) == (MigrationStatus.REFUSED, "reconcile_needs_discard") and path.exists()
    report = _reconcile(env, "a" * 32, discard=True)
    assert report.status is MigrationStatus.RECONCILED
    assert not path.exists() and active_migration_manifests(env.home) == []


def test_summaries_are_content_free(env):
    assert start(env, Answers(True, True)).status is MigrationStatus.COMPLETED
    write_memory_section(env.home, env.section)
    assert start(env, Answers(False)).status is MigrationStatus.DENIED
    summaries = migration_summaries(env.home, "example")
    assert sorted((s.state, s.outcome) for s in summaries) == [("completed", "completed"),
                                                              ("rolled_back", "operator_denied")]
    assert all(not s.blocks_archives and s.batches for s in summaries)
    text = repr(summaries)
    tokens = [staged.result.stage_handle_b64url for staged in env.store.stage_by_request.values()]
    tokens += [staged.result.approval_binding_sha256 for staged in env.store.stage_by_request.values()]
    for token in ["ep-1", MARKER, *tokens]:
        assert token not in text


def test_a_failed_rollback_compaction_stays_active_and_changes_no_mode(env, monkeypatch):
    path = _crash_after_manifest(env, monkeypatch)
    before = path.read_bytes()
    real = mm.write_document

    def write(p, doc):
        if doc["state"] == "rolled_back":
            raise OSError("disk full")
        real(p, doc)
    monkeypatch.setattr(mm, "write_document", write)
    report = _rollback(env)
    assert (report.status, report.code) == (MigrationStatus.STOPPED, "rollback_not_complete")
    assert path.stem in report.message
    assert active_migration_manifests(env.home) == [path] and path.read_bytes() == before


def test_reconcile_settles_a_run_under_another_provider_by_local_proof(env, monkeypatch):
    import json
    path = _crash_after_manifest(env, monkeypatch)
    doc = json.loads(path.read_bytes())
    doc.update(schema="other.hermes-migration/v1", import_run_id="e" * 32)
    foreign = mm.manifest_path(env.home, "other", "ep-1", "e" * 32)
    mm.write_document(foreign, doc)
    path.unlink()
    backends = len(env.backends)
    report = _reconcile(env, "e" * 32)
    assert report.status is MigrationStatus.RECONCILED and len(env.backends) == backends
    assert (_doc(foreign).state, _doc(foreign).outcome) == ("rolled_back", "reconciled")
    assert active_migration_manifests(env.home) == []


def test_recovery_without_state_creates_no_lock(env):
    assert _rollback(env).code == "nothing_to_roll_back"
    assert _reconcile(env, "0" * 32).code == "not_found"
    assert not (env.home / "migrations").exists()


def test_compaction_stamps_updated_at_from_the_clock(env, monkeypatch):
    path = _crash_after_manifest(env, monkeypatch)
    created = _doc(path).created_at
    later = datetime(2026, 9, 30, 13, 30, tzinfo=timezone.utc)
    assert _rollback(env, clock=lambda: later).status is MigrationStatus.ROLLED_BACK
    doc = _doc(path)
    assert (doc.created_at, doc.updated_at) == (created, "2026-09-30T13:30:00Z")
