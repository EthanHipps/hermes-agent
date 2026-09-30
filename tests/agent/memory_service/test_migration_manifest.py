"""R45: the <provider>.hermes-migration/v1 host state (§9.9 L1655, L1659; K-9; C7F-1; rulings R45-5, R45-6, R45-12)."""

import json
from dataclasses import replace

import pytest

from agent.memory_service import migration_manifest as mm
from agent.memory_service import wire as w
from tests.agent.memory_service.migration_support import GLOBAL, REPO

EPOCH, RUN = "ep-1", "0123456789abcdef0123456789abcdef"
REVISION = w.CompositeRevision(provider_epoch=EPOCH, visibility_revision="1",
                               scope_revisions=(w.ScopeRevision(scope=REPO, revision="3"),))
AUTH = w.ApprovalAuthorization(kind="approved", approval_id="e" * 32, approved_by_principal_id="ethan",
                               approved_at="2026-09-30T12:05:00Z", expires_at="2026-09-30T13:00:00Z",
                               approval_binding_sha256="d" * 64)
CREATED, UPDATED, COMPACTED = "2026-09-30T12:00:00Z", "2026-09-30T12:05:00Z", "2026-09-30T12:10:00Z"


def _identity():
    return mm.Identity(principal_id="ethan", profile_id="default", logical_session_id="admin-" + "f" * 32,
                       platform="admin", org_id=None, project_id="proj-1", repo_id="repo-1", workspace_id=None,
                       binding_revision="rev-1")


def _approved_batch():
    return mm.Batch(batch_id="1" * 16, target="memory", destination_scope=REPO, requested_write_scopes=(REPO,),
                    items=(mm.Item("MEMORY.md", "c" * 64),), status="approved", request_id="2" * 32,
                    prior_requests=(mm.Prior("5" * 32, "stage_expired"),),
                    stage=mm.StageRef("A" * 43, "d" * 64, REVISION, "2026-09-30T13:00:00Z", ("import",)),
                    candidate_hashes=(w.CandidateHash(client_ref="a" * 16, canonical_sha256="a" * 64),),
                    admissions=(mm.Admission("a" * 16, "rec-0001", REPO, "scoped_evidence", "create_record", None),),
                    authorization=AUTH)


def _manifest(*batches):
    return mm.ActiveManifest(provider="example", provider_epoch=EPOCH, import_run_id=RUN, identity=_identity(),
                             source=mm.Source("native_memory", "hermes-native-v0.20.6", "home-default"),
                             batches=batches or (_approved_batch(), mm.Batch(batch_id="3" * 16, target="user",
                                 destination_scope=GLOBAL, requested_write_scopes=(GLOBAL,),
                                 items=(mm.Item("USER.md", None),))),
                             created_at=CREATED, updated_at=UPDATED)


def _committed(batch, tx):
    return replace(batch, status="committed", result=mm.CommitRef(tx, "committed_audit_clean", None))


def _committed_user_batch():
    return _committed(replace(_approved_batch(), batch_id="3" * 16, target="user", destination_scope=GLOBAL,
                              requested_write_scopes=(GLOBAL,), items=(mm.Item("USER.md", "f" * 64),),
                              prior_requests=(), request_id="4" * 32,
                              admissions=(mm.Admission("a" * 16, "rec-0003", GLOBAL, "trusted_instruction",
                                                       "create_record", None),)), "tx-2")


def _load(doc):
    return mm.from_document(json.loads(mm.encode(mm.to_document(doc))), provider="example",
                            provider_epoch=EPOCH, import_run_id=RUN)


def test_an_active_manifest_round_trips_through_canonical_bytes():
    manifest = _manifest()
    assert _load(manifest) == manifest
    raw = mm.encode(mm.to_document(manifest))
    assert b"\n" not in raw and w.canonical_json(json.loads(raw)) == raw


def test_the_active_member_sets_are_exactly_the_contract():
    doc = mm.to_document(_manifest())
    assert set(doc) == mm.ACTIVE_KEYS and doc["state"] == "active"
    assert doc["schema"] == "example.hermes-migration/v1" and "provider" not in doc
    assert set(doc["identity"]) == mm.IDENTITY_KEYS and "opaque_binding_b64url" not in doc["identity"]
    assert all(set(b) == mm.ACTIVE_BATCH_KEYS for b in doc["batches"])


@pytest.mark.parametrize("state,outcome", [("completed", "completed"), ("rolled_back", "operator_denied")])
def test_compaction_keeps_only_the_receipt_members(state, outcome):
    # A completed receipt needs every batch committed, so that case compacts two committed batches.
    manifest = (_manifest(_committed(_approved_batch(), "tx-1"), _committed_user_batch()) if state == "completed"
                else _manifest())
    settled = {b.batch_id: ("committed", b.result.tx_id) if b.result else ("not_committed", None)
               for b in manifest.batches}
    receipt = mm.compact(manifest, state=state, outcome=outcome, settled=settled, updated_at=COMPACTED)
    doc = mm.to_document(receipt)
    assert set(doc) == mm.RECEIPT_KEYS and all(set(b) == mm.RECEIPT_BATCH_KEYS for b in doc["batches"])
    assert "provider" not in doc                                           # the schema name carries it
    assert (doc["created_at"], doc["updated_at"]) == (CREATED, COMPACTED)  # created fixed; updated on compaction
    raw = mm.encode(doc)
    for forbidden in (b"A" * 43, b"d" * 64, b"c" * 64, b"e" * 32, b"admin-", b"rev-1"):   # handle, binding, digest, auth, identity
        assert forbidden not in raw
    assert doc["batches"][0]["request_ids"] == ["5" * 32, "2" * 32]
    assert _load(receipt) == receipt


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(state="paused"),
    lambda d: d.update(extra=1),
    lambda d: d.update(schema="other.hermes-migration/v1"),
    lambda d: d.update(import_run_id="f" * 32),                          # disagrees with the file name
    lambda d: d["batches"][0].update(status="committed"),                 # committed without a result
    lambda d: d["batches"][0].update(authorization=None),                 # approved without an authorization
    lambda d: d["batches"][1].update(request_id="9" * 32),                # pending with a request
    lambda d: d["batches"][0]["items"][0].update(item_key="../MEMORY.md"),
    lambda d: d["source"].update(parser_version="hermes-legacy-archive-v1"),  # kind/parser pair
    lambda d: d["batches"][1]["items"].append({"item_key": "USER.md", "source_sha256": None}),   # one item per batch
    lambda d: d["batches"][1].update(items=[]),
    lambda d: d.update(created_at="2026-09-30T12:00:00.5Z"),                # second precision only
    lambda d: d.update(updated_at="2026-09-30 12:05:00Z"),
    lambda d: d.update(updated_at=None),
    lambda d: d.update(provider="example"),                                 # no provider member
])
def test_a_malformed_document_is_reported_never_guessed(mutate):
    doc = mm.to_document(_manifest())
    mutate(doc)
    with pytest.raises(mm.MigrationStateError):
        mm.from_document(doc, provider="example", provider_epoch=EPOCH, import_run_id=RUN)


def test_the_writer_never_emits_a_state_outside_the_three():
    assert set(mm.STATES) == {"active", "completed", "rolled_back"}
    with pytest.raises(ValueError):
        mm.compact(_manifest(), state="active", outcome="completed", settled={}, updated_at=COMPACTED)


def test_the_timestamps_are_the_manifests_own_members():
    doc = mm.to_document(_manifest())
    assert (doc["created_at"], doc["updated_at"]) == (CREATED, UPDATED)
    assert {"created_at", "updated_at"} <= mm.ACTIVE_KEYS and {"created_at", "updated_at"} <= mm.RECEIPT_KEYS


# -- Task 2: the store, the lock and agreement with C9 (R45-4, R45-5, K-9) ----------------------------------

from hermes_cli.backup_memory import (  # noqa: E402
    MIGRATION_STATE_DIRNAME, MigrationInProgressError, active_migration_manifests, refuse_if_migration_in_progress)


def test_the_root_is_c9s_directory(tmp_path):
    assert mm.migration_root(tmp_path) == tmp_path / MIGRATION_STATE_DIRNAME


@pytest.mark.parametrize("provider,epoch", [("../x", EPOCH), ("Ex", EPOCH), ("example", "a/b"), ("example", ""),
                                            ("example", "x" * 129)])
def test_unsafe_segments_never_become_paths(tmp_path, provider, epoch):
    with pytest.raises(mm.MigrationStateError):
        mm.manifest_path(tmp_path, provider, epoch, RUN)


def test_written_files_are_canonical_and_read_back(tmp_path):
    path = mm.manifest_path(tmp_path, "example", EPOCH, RUN)
    mm.write_document(path, mm.to_document(_manifest()))
    assert path.read_bytes() == mm.encode(mm.to_document(_manifest()))
    assert mm.read_document(path, provider="example", provider_epoch=EPOCH, import_run_id=RUN) == _manifest()


def test_a_non_canonical_file_is_invalid(tmp_path):
    path = mm.manifest_path(tmp_path, "example", EPOCH, RUN)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(mm.to_document(_manifest()), indent=2), encoding="utf-8")
    with pytest.raises(mm.MigrationStateError):
        mm.read_document(path, provider="example", provider_epoch=EPOCH, import_run_id=RUN)


def _c9_agrees(home):
    """K-9: what R45 calls active (or unreadable) is exactly what C9 blocks on."""
    ours = {r.path for r in mm.list_runs(home) if r.document is None or r.document.state == "active"}
    assert ours == set(active_migration_manifests(home))


def test_writer_and_c9_reader_agree_through_every_state(tmp_path):
    path = mm.manifest_path(tmp_path, "example", EPOCH, RUN)
    mm.write_document(path, mm.to_document(_manifest()))
    _c9_agrees(tmp_path)
    with pytest.raises(MigrationInProgressError):
        refuse_if_migration_in_progress([tmp_path])
    done = _manifest(_committed(_approved_batch(), "tx-1"), _committed_user_batch())
    for manifest, state, outcome in ((_manifest(), "rolled_back", "operator_denied"), (done, "completed", "completed")):
        settled = {b.batch_id: ("committed", b.result.tx_id) if b.result else ("not_committed", None)
                   for b in manifest.batches}
        mm.write_document(path, mm.to_document(mm.compact(manifest, state=state, outcome=outcome, settled=settled,
                                                           updated_at=COMPACTED)))
        _c9_agrees(tmp_path)
        assert active_migration_manifests(tmp_path) == [] and mm.list_runs(tmp_path)[0].document.state == state


@pytest.mark.parametrize("raw", [b"{\"state\": \"comp", b"", b"[]", b"\xff\xfe"])
def test_a_torn_or_corrupt_file_blocks_both_readers(tmp_path, raw):
    path = mm.manifest_path(tmp_path, "example", EPOCH, RUN)
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    [run] = mm.list_runs(tmp_path)
    assert run.document is None and run.error
    _c9_agrees(tmp_path)


def test_an_interrupted_write_leaves_a_temp_file_both_readers_ignore(tmp_path):
    path = mm.manifest_path(tmp_path, "example", EPOCH, RUN)
    mm.write_document(path, mm.to_document(_manifest()))
    (path.parent / ".tmp_leftover.tmp").write_bytes(b"{\"state\": \"act")
    _c9_agrees(tmp_path)
    assert [r.path for r in mm.list_runs(tmp_path)] == [path]


def test_the_lock_file_is_outside_every_provider_directory_and_invisible_to_c9(tmp_path):
    with mm.migration_lock(tmp_path):
        assert (mm.migration_root(tmp_path) / mm.LOCK_NAME).exists()
        assert active_migration_manifests(tmp_path) == [] and mm.list_runs(tmp_path) == []


def test_a_second_holder_is_busy(tmp_path):
    with mm.migration_lock(tmp_path):
        with pytest.raises(mm.MigrationBusyError):
            with mm.migration_lock(tmp_path, timeout_seconds=0.1):
                pass
    with mm.migration_lock(tmp_path, timeout_seconds=0.1):     # released on exit
        pass


# -- Task 3: the cross-repository fixture (R45-6, R45-7; contract C7F-2) ------------------------------------

import hashlib  # noqa: E402
from pathlib import Path  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "hermes-migration"
EXAMPLE_EPOCH = "019244f0-6b2a-7c3d-8e4f-000000000001"


def test_the_fork_copy_matches_its_pinned_digests():
    lines = (FIXTURE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert sorted(line.split()[1] for line in lines) == ["active.json", "completed.json", "rolled_back.json",
                                                         "schema.json"]
    for line in lines:
        digest, name = line.split()
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == digest, name


@pytest.mark.parametrize("name,state", [("active", "active"), ("completed", "completed"), ("rolled_back", "rolled_back")])
def test_every_example_round_trips_byte_for_byte_through_the_writer(name, state):
    raw = (FIXTURE / f"{name}.json").read_bytes()
    doc = mm.from_document(json.loads(raw), provider="ygg", provider_epoch=EXAMPLE_EPOCH,
                           import_run_id="0123456789abcdef0123456789abcdef")
    assert doc.state == state and mm.encode(mm.to_document(doc)) == raw


def test_the_schema_file_names_exactly_the_code_member_sets():
    schema = json.loads((FIXTURE / "schema.json").read_text(encoding="utf-8"))
    defs = schema["$defs"]
    assert set(defs["active"]["properties"]) == mm.ACTIVE_KEYS
    assert set(defs["activeBatch"]["properties"]) == mm.ACTIVE_BATCH_KEYS
    assert set(defs["receipt"]["properties"]) == mm.RECEIPT_KEYS
    assert set(defs["receiptBatch"]["properties"]) == mm.RECEIPT_BATCH_KEYS
    assert defs["receipt"]["properties"]["outcome"]["enum"] == list(mm.RUN_OUTCOMES)
    assert defs["activeBatch"]["properties"]["status"]["enum"] == list(mm.BATCH_STATUSES)
    assert defs["receiptBatch"]["properties"]["outcome"]["enum"] == list(mm.BATCH_OUTCOMES)
    # one item per batch in both shapes, as from_document enforces (Checkpoint A, R45-6)
    for shape, key in (("activeBatch", "items"), ("receiptBatch", "item_keys")):
        assert (defs[shape]["properties"][key]["minItems"], defs[shape]["properties"][key]["maxItems"]) == (1, 1)
