"""R39: handle-only approval records (§9.4 step 8; §9.5 L1540, L1546; rulings R39-2, R39-3; contract C6b-3)."""

import dataclasses
import json

import pytest

from agent.memory_service import wire as w
from agent.memory_service.approval_store import (
    APPROVAL_SCHEMA, ApprovalStoreError, PendingApproval, approvals_dir, drop_pending_approval,
    list_pending_approvals, load_pending_approval, new_pending_id, save_pending_approval, valid_pending_id)
from agent.memory_service.host_state import host_state_dir
from hermes_cli.backup_memory import HOST_STATE_DIRNAME

REPO = w.ScopeRef(kind="repository", id="repo-1")
REVISION = w.CompositeRevision(provider_epoch="ep-1", visibility_revision="1",
                               scope_revisions=(w.ScopeRevision(scope=REPO, revision="3"),))
BINDING = "a" * 64
AUTH = w.ApprovalAuthorization(kind="approved", approval_id="0f" * 16, approved_by_principal_id="ethan",
                               approved_at="2026-09-17T00:10:00Z", expires_at="2026-09-17T01:00:00Z",
                               approval_binding_sha256=BINDING)


def _record(**overrides):
    base = dict(pending_id=new_pending_id(), session_id="sess-1", logical_session_id="sess-1", provider_epoch="ep-1",
                target="user", intent_kind="add", request_id="req-1", stage_handle_b64url="A" * 43,
                approval_binding_sha256=BINDING, expected_revision=REVISION, requested_write_scopes=(REPO,),
                approval_requirements=("target_user",), expires_at="2026-09-17T01:00:00Z",
                origin="foreground", created_at="2026-09-17T00:00:00Z")
    base.update(overrides)
    return PendingApproval(**base)


def test_pending_and_approved_records_round_trip(tmp_path):
    pending = _record()
    save_pending_approval(pending, hermes_home=tmp_path)
    assert load_pending_approval(pending.pending_id, hermes_home=tmp_path) == pending
    approved = dataclasses.replace(pending, decision="approved", authorization=AUTH, outcome="unknown")
    save_pending_approval(approved, hermes_home=tmp_path)
    assert load_pending_approval(pending.pending_id, hermes_home=tmp_path) == approved


def test_records_live_under_the_host_state_root_every_archive_excludes(tmp_path):
    """Ruling R39-2: X-1 (a) prunes memory_service/ from every archive and withholds it on restore."""
    assert (approvals_dir(tmp_path).relative_to(tmp_path).parts[0]
            == host_state_dir(tmp_path).relative_to(tmp_path).parts[0] == HOST_STATE_DIRNAME)


def test_a_record_holds_only_the_allowed_members(tmp_path):
    """Ruling R39-3 / §9.4 step 8: the allow-list is the contract, so no content-bearing member can appear."""
    record = _record()
    save_pending_approval(record, hermes_home=tmp_path)
    data = json.loads((approvals_dir(tmp_path) / f"{record.pending_id}.json").read_text(encoding="utf-8"))
    assert data["schema"] == APPROVAL_SCHEMA
    assert set(data) == {"schema", "pending_id", "session_id", "logical_session_id", "provider_epoch", "target",
                         "intent_kind", "request_id", "stage_handle_b64url", "approval_binding_sha256",
                         "expected_revision", "requested_write_scopes", "approval_requirements", "expires_at",
                         "decision", "authorization", "outcome", "origin", "created_at"}


@pytest.mark.parametrize("bad", ["../x", "ABCDEF012345", "123", "", "0123456789abc", "0123456789a/"])
def test_malformed_ids_never_reach_the_filesystem(tmp_path, bad):
    assert not valid_pending_id(bad)
    with pytest.raises(ApprovalStoreError):
        load_pending_approval(bad, hermes_home=tmp_path)
    assert not approvals_dir(tmp_path).exists()


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(decision="approved"),                   # approved without its authorization
    lambda d: d.update(authorization=AUTH.to_wire()),          # pending with an authorization
    lambda d: d.update(target="native"),
    lambda d: d.pop("stage_handle_b64url"),
    lambda d: d.update(schema="other/v1")])
def test_a_malformed_record_is_reported_never_guessed(tmp_path, mutate):
    record = _record()
    save_pending_approval(record, hermes_home=tmp_path)
    path = approvals_dir(tmp_path) / f"{record.pending_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ApprovalStoreError):
        load_pending_approval(record.pending_id, hermes_home=tmp_path)
    assert list_pending_approvals(hermes_home=tmp_path) == [(record.pending_id, None)]


def test_list_is_oldest_first_and_drop_removes(tmp_path):
    older, newer = _record(created_at="2026-09-17T00:00:00Z"), _record(created_at="2026-09-17T00:05:00Z")
    for record in (newer, older):
        save_pending_approval(record, hermes_home=tmp_path)
    assert [pid for pid, _ in list_pending_approvals(hermes_home=tmp_path)] == [older.pending_id, newer.pending_id]
    assert drop_pending_approval(older.pending_id, hermes_home=tmp_path) is True
    assert drop_pending_approval(older.pending_id, hermes_home=tmp_path) is False


def test_a_write_failure_is_an_approval_store_error(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr("utils.atomic_json_write", boom)     # approval_store late-imports it
    with pytest.raises(ApprovalStoreError):
        save_pending_approval(_record(), hermes_home=tmp_path)
