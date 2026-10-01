"""R45 resume: every post-manifest boundary of §13.1 L2182 replays the exact recorded request (D-R45-2).

Each "crash" wraps ``migration_manifest.write_document`` (the engine's one writer) so it raises ``Crash`` once,
before or after the write; ``resume`` then opens a fresh administrative service over the same provider store and
home, as a restarted process would. Every test ends with one provider record per native entry.
"""

import shutil

import pytest

from agent.memory_service import migration_manifest as mm
from agent.memory_service import wire as w
from agent.memory_service.migration import RESTART_NOTICE, MigrationStatus
from hermes_cli.backup_memory import active_migration_manifests
from tests.agent.memory_service.fake_backend import DEFAULT_LIMITS
from tests.agent.memory_service.migration_support import (
    MARKER, REPO, Answers, Crash, make_env, resume, run_files, start, write_memory_section, write_native)

ENTRIES = {"memory": 2, "user": 1}


@pytest.fixture
def env(tmp_path):
    e = make_env(tmp_path)
    write_native(e.home, memory=[f"uses postgres {MARKER}", "prefers tabs"], user=["name is Ethan"])
    return e


def _crash_on(monkeypatch, predicate, *, after_write: bool):
    """Raise Crash once, on the first document ``predicate`` accepts; before or after writing it."""
    real = mm.write_document
    fired = []

    def write(path, doc):
        if not fired and predicate(doc):
            fired.append(True)
            if after_write:
                real(path, doc)
            raise Crash()
        real(path, doc)
    monkeypatch.setattr(mm, "write_document", write)


def _first_write(doc):
    return True


def _status(target, status):
    return lambda doc: any(b.get("target") == target and b.get("status") == status for b in doc.get("batches", ()))


def _manifest(env):
    [path] = run_files(env.home)
    return mm.read_document(path, provider="example", provider_epoch=path.parent.name, import_run_id=path.stem)


def _batch(doc, target):
    return next(b for b in doc.batches if b.target == target)


def _imported(env, target):
    return [r for r in env.store.records.values() if r.target == target and r.provenance.actor_kind == "import"]


def _one_record_per_entry(env):
    for target, count in ENTRIES.items():
        assert len(_imported(env, target)) == count, target


def _calls_since(env, first_backend, operation):
    return [request for b in env.backends[first_backend:] for op, request in b.calls if op == operation]


def _crash_after_manifest(env, monkeypatch):
    _crash_on(monkeypatch, _first_write, after_write=True)
    with pytest.raises(Crash):
        start(env, Answers(True, True))
    monkeypatch.undo()
    return _manifest(env)


def test_crash_after_the_manifest_fsync_resumes_with_the_exact_request(env, monkeypatch):
    manifest = _crash_after_manifest(env, monkeypatch)
    assert active_migration_manifests(env.home) == run_files(env.home)
    memory = _batch(manifest, "memory")
    assert memory.status == "staged"
    report = resume(env, Answers(True, True))
    assert report.status is MigrationStatus.COMPLETED
    commits = env.calls("commit_curated")
    assert commits[0].request_id == memory.request_id and commits[0].stage_handle_b64url == memory.stage.stage_handle_b64url
    assert len([c for c in env.calls("stage_curated") if c.target == "memory"]) == 1
    _one_record_per_entry(env)


def test_crash_while_prompting_re_prompts_on_resume(env):
    def crash(approval):
        raise Crash()
    with pytest.raises(Crash):
        start(env, crash)
    answers = Answers(True, True)
    report = resume(env, answers)
    assert report.status is MigrationStatus.COMPLETED and [p.target for p in answers.prompts] == ["memory", "user"]
    assert len([c for c in env.calls("inspect_staged") if c.target == "memory"]) == 2
    _one_record_per_entry(env)


def test_crash_after_the_approval_write_ahead_replays_the_identical_commit(env, monkeypatch):
    _crash_on(monkeypatch, _status("memory", "approved"), after_write=True)
    with pytest.raises(Crash):
        start(env, Answers(True, True))
    monkeypatch.undo()
    memory = _batch(_manifest(env), "memory")
    assert memory.status == "approved" and env.store.receipts == {}
    report = resume(env, Answers(True))
    assert report.status is MigrationStatus.COMPLETED
    assert [c.request_id for c in env.calls("commit_curated")].count(memory.request_id) == 1
    assert env.store.receipts[("ep-1", memory.request_id)]
    _one_record_per_entry(env)


def test_a_lost_commit_reply_is_an_idempotent_replay(env):
    env.store.fail_transport("commit_curated", phase="during")
    report = start(env, Answers(True, True))
    assert report.status is MigrationStatus.COMPLETED
    receipt = _manifest(env)
    memory_receipt = _batch(receipt, "memory")
    assert memory_receipt.tx_id == env.store.receipts[("ep-1", memory_receipt.request_ids[-1])].tx_id
    _one_record_per_entry(env)


def test_an_unknown_commit_stops_and_resume_retries_identically(env):
    # The first send publishes and loses its reply; the exact retry fails before reaching the store.
    env.store.fail_transport("commit_curated", phase="during")
    env.store.fail_transport("commit_curated", phase="before")
    report = start(env, Answers(True, True))
    assert (report.status, report.code) == (MigrationStatus.STOPPED, "outcome_unknown")
    memory = _batch(_manifest(env), "memory")
    assert memory.status == "approved" and active_migration_manifests(env.home)
    original_tx = env.store.receipts[("ep-1", memory.request_id)].tx_id
    first_backend = len(env.backends)
    report = resume(env, Answers(True))
    assert report.status is MigrationStatus.COMPLETED
    replays = [c for c in _calls_since(env, first_backend, "commit_curated") if c.request_id == memory.request_id]
    sent = [c for c in env.calls("commit_curated") if c.request_id == memory.request_id]
    assert len(replays) == 1 and len({w.canonical_json(c.to_wire()) for c in sent}) == 1
    assert _batch(_manifest(env), "memory").tx_id == original_tx
    assert not env.store._transport_faults.get(("commit_curated", "during"))
    assert not env.store._transport_faults.get(("commit_curated", "before"))
    _one_record_per_entry(env)


def test_crash_after_the_commit_response_before_the_result_fsync_records_the_original_tx(env, monkeypatch):
    _crash_on(monkeypatch, _status("memory", "committed"), after_write=False)
    with pytest.raises(Crash):
        start(env, Answers(True, True))
    monkeypatch.undo()
    memory = _batch(_manifest(env), "memory")
    assert memory.status == "approved"
    original_tx = env.store.receipts[("ep-1", memory.request_id)].tx_id
    report = resume(env, Answers(True))
    assert report.status is MigrationStatus.COMPLETED
    assert _batch(_manifest(env), "memory").tx_id == original_tx
    _one_record_per_entry(env)


def test_crash_after_the_result_fsync_finishes_on_resume(env, monkeypatch):
    _crash_on(monkeypatch, _status("user", "committed"), after_write=True)
    with pytest.raises(Crash):
        start(env, Answers(True, True))
    monkeypatch.undo()
    first_backend = len(env.backends)
    report = resume(env, Answers())
    assert report.status is MigrationStatus.COMPLETED and report.switched
    assert _calls_since(env, first_backend, "stage_curated") == []
    assert _calls_since(env, first_backend, "commit_curated") == []
    assert _manifest(env).state == "completed"
    _one_record_per_entry(env)


def test_crash_after_the_switch_before_completion_compacts_on_resume(env):
    real_switch = env.switch

    def crash_switch():
        real_switch()
        raise Crash()
    env.switch = crash_switch
    with pytest.raises(Crash):
        start(env, Answers(True, True))
    env.switch = real_switch
    assert _manifest(env).state == "active" and env.raw()["memory"]["provider_mode"] == "authoritative"
    report = resume(env, Answers())
    assert report.status is MigrationStatus.COMPLETED and not report.switched
    assert RESTART_NOTICE in report.message                       # running gateways still use the dormant files
    assert env.switches == ["authoritative"]
    assert _manifest(env).state == "completed"
    _one_record_per_entry(env)


def test_a_failed_completion_write_after_the_switch_still_names_the_restart(env, monkeypatch):
    real = mm.write_document
    fired = []

    def write(path, doc):
        if not fired and doc.get("state") == "completed":
            fired.append(True)
            raise OSError("disk full")
        real(path, doc)
    monkeypatch.setattr(mm, "write_document", write)
    report = start(env, Answers(True, True))
    assert (report.status, report.code) == (MigrationStatus.STOPPED, "completion_not_recorded")
    assert RESTART_NOTICE in report.message and env.switches == ["authoritative"]
    monkeypatch.undo()
    report = resume(env, Answers())
    assert report.status is MigrationStatus.COMPLETED and not report.switched
    assert RESTART_NOTICE in report.message
    assert _manifest(env).state == "completed"


def test_crash_after_completion_is_terminal(env):
    assert start(env, Answers(True, True)).status is MigrationStatus.COMPLETED
    report = resume(env, Answers())
    assert (report.status, report.code) == (MigrationStatus.REFUSED, "no_active_migration")


def test_an_expired_stage_is_restaged_only_after_stage_expired(env, monkeypatch):
    old = _batch(_crash_after_manifest(env, monkeypatch), "memory")
    env.store.clock.advance(DEFAULT_LIMITS.stage_ttl_seconds + 1)
    answers = Answers(True, True)
    report = resume(env, answers)
    assert report.status is MigrationStatus.COMPLETED
    assert [p.target for p in answers.prompts] == ["memory", "user"]
    assert env.store.stage_state(old.request_id) == "expired"
    assert ("ep-1", old.request_id) not in env.store.receipts
    [inspect_old] = [c for c in env.calls("inspect_staged") if c.request_id == old.request_id]
    assert inspect_old.stage_handle_b64url == old.stage.stage_handle_b64url
    memory = _batch(_manifest(env), "memory")
    assert memory.request_ids[0] == old.request_id and len(memory.request_ids) == 2
    _one_record_per_entry(env)


def test_the_restage_is_recorded_before_it_is_approved(env, monkeypatch):
    old = _batch(_crash_after_manifest(env, monkeypatch), "memory")
    env.store.clock.advance(DEFAULT_LIMITS.stage_ttl_seconds + 1)
    seen = []

    def prompt(approval):
        seen.append(_batch(_manifest(env), approval.target))
        return False
    resume(env, prompt)
    assert seen[0].prior_requests == (mm.Prior(old.request_id, "stage_expired"),)
    assert seen[0].request_id != old.request_id and seen[0].status == "staged"


def test_version_conflict_at_commit_restages_and_reapproves(env):
    prompts = []

    def prompt(approval):
        prompts.append(approval.target)
        if len(prompts) == 1:
            env.store.external_write(REPO, "memory", "concurrent")
        return True
    report = start(env, prompt)
    assert report.status is MigrationStatus.COMPLETED
    assert prompts == ["memory", "memory", "user"]
    memory = _batch(_manifest(env), "memory")
    assert len(memory.request_ids) == 2 and memory.outcome == "committed"
    _one_record_per_entry(env)


def test_a_plain_stage_not_found_stops_for_reconciliation(env, monkeypatch):
    _crash_after_manifest(env, monkeypatch)
    [path] = run_files(env.home)
    before = path.read_bytes()
    env.store.stages.clear()
    env.store.stage_by_request.clear()
    report = resume(env, Answers(True, True))
    assert (report.status, report.code) == (MigrationStatus.STOPPED, "reconcile_required")
    assert path.read_bytes() == before


def test_an_epoch_change_stops_for_reconciliation(env, monkeypatch):
    _crash_after_manifest(env, monkeypatch)
    [path] = run_files(env.home)
    before = path.read_bytes()
    env.store.set_epoch("ep-2")
    first_backend = len(env.backends)
    report = resume(env, Answers(True, True))
    assert (report.status, report.code) == (MigrationStatus.STOPPED, "reconcile_required")
    assert path.read_bytes() == before
    assert _calls_since(env, first_backend, "stage_curated") == []
    assert _calls_since(env, first_backend, "commit_curated") == []


def test_a_changed_native_source_stops_a_restage(env, monkeypatch):
    _crash_after_manifest(env, monkeypatch)
    [path] = run_files(env.home)
    before = path.read_bytes()
    write_native(env.home, memory=["uses sqlite now"])
    env.store.clock.advance(DEFAULT_LIMITS.stage_ttl_seconds + 1)
    first_backend = len(env.backends)
    report = resume(env, Answers(True, True))
    assert (report.status, report.code) == (MigrationStatus.STOPPED, "source_changed")
    assert path.read_bytes() == before
    assert _calls_since(env, first_backend, "stage_curated") == []


def test_source_tuple_dedupes_across_a_new_run_id(env):
    assert start(env, Answers(True, True)).status is MigrationStatus.COMPLETED
    [first] = mm.list_runs(env.home)
    first_ids = {a.assigned_id for b in first.document.batches for a in b.assigned}
    records = len(env.store.records)
    write_memory_section(env.home, env.section)                 # back to additive by hand
    report = start(env, Answers(True, True))
    assert report.status is MigrationStatus.COMPLETED
    second = next(r for r in mm.list_runs(env.home) if r.import_run_id != first.import_run_id)
    assigned = [a for b in second.document.batches for a in b.assigned]
    assert {a.publication_effect for a in assigned} == {"reuse_existing_import"}
    assert {a.assigned_id for a in assigned} == first_ids
    assert len(env.store.records) == records


def test_source_tuple_dedupes_after_hermes_archive_state_loss(env):
    assert start(env, Answers(True, True)).status is MigrationStatus.COMPLETED
    handles = len(env.store.handles)
    records = len(env.store.records)
    shutil.rmtree(env.home / "migrations")
    shutil.rmtree(env.home / "memory_service")                  # what a restore into a fresh home loses (X-1)
    write_memory_section(env.home, env.section)
    report = start(env, Answers(True, True))
    assert report.status is MigrationStatus.COMPLETED
    assert len(env.store.handles) == handles + 1                 # a second bind: a new administrative identity
    [second] = mm.list_runs(env.home)
    assert {a.publication_effect for b in second.document.batches for a in b.assigned} == {"reuse_existing_import"}
    assert len(env.store.records) == records


def test_resume_reads_no_native_file_before_finish(env, monkeypatch):
    from tests.agent.memory_service.native_sentinel import native_memory_sentinel
    _crash_after_manifest(env, monkeypatch)
    with native_memory_sentinel(env.home / "memories") as sentinel:
        report = resume(env, Answers(False))
    assert report.status is MigrationStatus.DENIED
    assert sentinel.accesses == []
