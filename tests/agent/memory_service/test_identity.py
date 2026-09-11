"""§4.1 FrozenMemoryIdentity and the persisted host session state."""

import base64
import dataclasses

import pytest

from agent.memory_service.errors import BindingInvalidError, ProviderError, ProviderTransportError
from agent.memory_service.identity import FrozenMemoryIdentity, HostSessionState
from agent.memory_service.wire import FrozenIdentityWire

HANDLE = base64.urlsafe_b64encode(bytes(range(32))).rstrip(b"=").decode()


def _identity(**over):
    base = dict(
        provider="example",
        provider_mode="authoritative",
        principal_id="ethan",
        profile_id="default",
        logical_session_id="sess-1",
        org_id=None,
        project_id="proj-1",
        repo_id="repo-1",
        workspace_id=None,
        platform="cli",
        binding_revision="rev-7",
        opaque_binding_b64url=HANDLE,
    )
    base.update(over)
    return FrozenMemoryIdentity(**base)


def test_field_set_is_exactly_section_4_1():
    names = [f.name for f in dataclasses.fields(FrozenMemoryIdentity)]
    assert names == [
        "provider", "provider_mode", "principal_id", "profile_id", "logical_session_id",
        "org_id", "project_id", "repo_id", "workspace_id", "platform", "binding_revision", "opaque_binding_b64url",
    ]


def test_identity_is_frozen():
    ident = _identity()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ident.repo_id = "other"  # type: ignore[misc]


def test_wire_round_trip_requires_authoritative_mode():
    ident = _identity()
    wire = ident.to_wire()
    assert isinstance(wire, FrozenIdentityWire)
    assert wire.to_wire()["opaque_binding_b64url"] == HANDLE
    assert FrozenMemoryIdentity.from_wire(wire) == ident
    # M14: a bare ValueError must never escape the seam -- resume() does not
    # catch it, so a hand-built non-authoritative identity needs the
    # package's own typed error here.
    with pytest.raises(BindingInvalidError, match="authoritative"):
        _identity(provider_mode="additive").to_wire()


def test_host_session_state_round_trip_and_missing_member():
    state = HostSessionState(provider_epoch="ep-1", identity=_identity())
    data = state.to_dict()
    assert data["provider_epoch"] == "ep-1"
    assert data["identity"]["principal_id"] == "ethan"
    assert HostSessionState.from_dict(data) == state
    with pytest.raises(BindingInvalidError):
        HostSessionState.from_dict({"identity": data["identity"]})
    with pytest.raises(BindingInvalidError):
        HostSessionState.from_dict({"provider_epoch": "ep-1"})
    with pytest.raises(BindingInvalidError):
        HostSessionState.from_dict({"provider_epoch": "", "identity": data["identity"]})
    with pytest.raises(BindingInvalidError):
        HostSessionState.from_dict({"provider_epoch": "ep-1", "identity": {**data["identity"], "extra": 1}})
    with pytest.raises(BindingInvalidError):
        HostSessionState.from_dict("not a dict")


def test_provider_errors_carry_typed_fields():
    err = ProviderError(code="store_blocked", outcome="not_committed", details={"reason": "git_dirty"}, operation="stage_curated")
    assert err.code == "store_blocked" and err.outcome == "not_committed"
    assert "store_blocked" in str(err) and "git_dirty" not in str(err)
    with pytest.raises(ValueError):
        ProviderError(code="disk_full", outcome="not_applicable", details=None, operation="load_curated")
    transport = ProviderTransportError(reason="timeout", operation="commit_curated", mutation_outcome_unknown=True)
    assert transport.mutation_outcome_unknown and "commit_curated" in str(transport)
