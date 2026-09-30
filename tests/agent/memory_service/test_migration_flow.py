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


# -- Task 6: inspection, approval, denial and commit (§9.9 steps 3-4; §9.5 L1546, L1562, L1564) --------------

from hermes_cli.backup_memory import refuse_if_migration_in_progress  # noqa: E402


def _only_run(env):
    [path] = run_files(env.home)
    return path


def _read(env):
    path = _only_run(env)
    return mm.read_document(path, provider="example", provider_epoch="ep-1", import_run_id=path.stem)


def test_a_lost_stage_reply_is_replayed_exactly_and_proceeds(env):
    env.store.fail_transport("stage_curated", phase="during", times=1)
    report = start(env, Answers(True, True))
    assert report.status is MigrationStatus.COMPLETED
    assert len(env.store.receipts) == 2                     # one commit per batch, no duplicate stage committed
    assert len({call.request_id for call in env.calls("stage_curated")}) == 2      # the replay reused its request ID


def test_the_first_clean_stage_writes_the_active_manifest_before_the_inspection(env):
    seen = []

    def prompt(approval):
        [path] = run_files(env.home)                        # already fsynced when the mapping is shown (§9.9 L1663)
        seen.append((json.loads(path.read_bytes())["state"], len(env.calls("inspect_staged"))))
        return False
    start(env, prompt)
    assert seen == [("active", 1)]


def test_denial_compacts_to_operator_denied_and_unblocks_archives(env):
    report = start(env, Answers(False))
    assert report.status is MigrationStatus.DENIED
    doc = _read(env)
    assert (doc.state, doc.outcome) == ("rolled_back", "operator_denied")
    assert active_migration_manifests(env.home) == []
    refuse_if_migration_in_progress([env.home])
    assert env.store.receipts == {}                           # no commit was sent
    assert MARKER.encode() not in _only_run(env).read_bytes()


@pytest.mark.parametrize("answer", [None, "raise"])
def test_cancellation_or_a_failing_prompt_is_a_denial(env, answer):
    def prompt(approval):
        if answer == "raise":
            raise RuntimeError("terminal went away")
        return None
    report = start(env, prompt)
    assert report.status is MigrationStatus.DENIED


def test_a_failed_denial_compaction_stays_active_and_blocks_archives(env, monkeypatch):
    real = mm.write_document

    def write(path, doc):
        if doc["state"] == "rolled_back":
            raise OSError("disk full")
        real(path, doc)
    monkeypatch.setattr(mm, "write_document", write)
    report = start(env, Answers(False))
    assert (report.status, report.code) == (MigrationStatus.STOPPED, "denial_not_complete")
    assert active_migration_manifests(env.home) == [_only_run(env)]
    with pytest.raises(Exception):
        refuse_if_migration_in_progress([env.home])
    assert env.store.receipts == {}


def test_approval_writes_the_authorization_ahead_of_the_commit(env, monkeypatch):
    order = []
    real_write = mm.write_document

    def write(path, doc):
        order.append(("write", tuple(b.get("status") for b in doc["batches"]), len(env.store.receipts)))
        real_write(path, doc)
    monkeypatch.setattr(mm, "write_document", write)
    start(env, Answers(True, True))
    first_approved = next(i for i, (_, statuses, _) in enumerate(order) if statuses[0] == "approved")
    assert order[first_approved][2] == 0                     # written before any commit reached the store (R39-7)


def test_two_batches_commit_sequentially_with_two_prompts(env):
    answers = Answers(True, True)
    report = start(env, answers)
    assert report.status is MigrationStatus.COMPLETED
    assert [p.target for p in answers.prompts] == ["memory", "user"]
    assert [p.requirements for p in answers.prompts] == [("import",), ("target_user", "import")]
    assert len(env.store.receipts) == 2


def test_a_second_batch_denied_after_the_first_committed_records_both(env):
    start(env, Answers(True, False))
    doc = _read(env)
    assert (doc.state, doc.outcome) == ("rolled_back", "operator_denied")
    assert [b.outcome for b in doc.batches] == ["committed", "not_committed"]
    assert doc.batches[0].tx_id and doc.batches[0].assigned and not doc.batches[1].assigned


def test_a_second_batch_rejected_by_the_provider_rolls_back_as_provider_rejected(env):
    env.store.secret_detector = lambda text: "fixture_secret" if "Ethan" in text else None
    report = start(env, Answers(True))
    assert (report.status, report.code) == (MigrationStatus.REJECTED, "secret_rejected")
    doc = _read(env)
    assert (doc.state, doc.outcome) == ("rolled_back", "provider_rejected")
    assert [b.outcome for b in doc.batches] == ["committed", "not_committed"]
    assert doc.batches[1].request_ids == ()                   # its stage was never clean: nothing correlating it


def test_the_prompt_shows_the_create_reuse_mapping_and_no_provider_token(env):
    answers = Answers(False)
    start(env, answers)
    [approval] = answers.prompts
    assert "create_record" in approval.text and MARKER in approval.text
    for token in (approval.stage.stage_handle_b64url, approval.stage.approval_binding_sha256, "ep-1"):
        assert token not in approval.text


# -- Task 7: conformance, the mode switch and completion (§9.9 steps 5-6; R45-12, R45-13, R45-14) ------------

def test_completion_switches_an_additive_home_and_compacts(env):
    from agent.memory_service.migration import RESTART_NOTICE
    report = start(env, Answers(True, True))
    assert report.status is MigrationStatus.COMPLETED and report.switched and env.switches == ["authoritative"]
    assert report.message.count(RESTART_NOTICE) == 1
    doc = _read(env)
    assert (doc.state, doc.outcome) == ("completed", "completed")
    assert active_migration_manifests(env.home) == []


def test_switch_happens_after_conformance_and_before_compaction(env, monkeypatch):
    order = []
    real = mm.write_document
    monkeypatch.setattr(mm, "write_document", lambda p, d: (order.append(d["state"]), real(p, d)))
    real_switch = env.switch
    env.switch = lambda: (order.append("switch"), real_switch())
    start(env, Answers(True, True))
    assert order[-2:] == ["switch", "completed"]


def test_withheld_raw_is_verified_absent_and_counted(env):
    env.store.admission_classifier = lambda cand: "withheld_raw" if "tabs" in cand.text else (
        "scoped_evidence" if cand.target == "memory" else "trusted_instruction")
    report = start(env, Answers(True, True))
    assert report.status is MigrationStatus.COMPLETED
    assert [(b.target, b.created, b.withheld) for b in report.batches] == [("memory", 1, 1), ("user", 1, 0)]


def test_a_conformance_failure_keeps_the_run_active_and_the_home_additive(env, monkeypatch):
    """A record retired by another writer between commit and reload (§9.9 L1665)."""
    from agent.memory_service import migration
    real_conformance = migration._Run.conformance

    def conformance(self, service):
        for record in env.store.records.values():
            if record.target == "memory":
                record.lifecycle = "retired"
        return real_conformance(self, service)
    monkeypatch.setattr(migration._Run, "conformance", conformance)
    report = start(env, Answers(True, True))
    assert (report.status, report.code) == (MigrationStatus.STOPPED, "conformance_failed")
    assert env.switches == [] and active_migration_manifests(env.home) == [_only_run(env)]


def test_a_native_edit_during_the_run_stops_before_the_switch(env):
    def prompt(approval):
        if approval.target == "user":
            write_native(env.home, memory=["edited while migrating"])
        return True
    report = start(env, prompt)
    assert (report.status, report.code) == (MigrationStatus.STOPPED, "source_changed")
    assert env.switches == []
