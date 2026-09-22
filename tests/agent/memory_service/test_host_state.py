"""§4.1 L250: one atomic host session state per Hermes session (ruling R40-4a)."""

import stat

import pytest

from agent.memory_service.errors import BindingInvalidError
from agent.memory_service.host_state import (
    HostStateRecord, host_state_dir, inherit_host_state, load_host_state, save_host_state,
)
from agent.memory_service.identity import FrozenMemoryIdentity, HostSessionState

IDENTITY = FrozenMemoryIdentity(provider="example", provider_mode="authoritative", principal_id="ethan",
                                profile_id="default", logical_session_id="s-1", org_id=None, project_id="proj-1",
                                repo_id="repo-1", workspace_id=None, platform="cli", binding_revision="rev-000001",
                                opaque_binding_b64url="A" * 43)
STATE = HostSessionState(provider_epoch="ep-1", identity=IDENTITY)


def test_authoritative_record_round_trips():
    save_host_state(HostStateRecord("s-1", "provider_authoritative", STATE, prompt_sha256="ab" * 32))
    assert load_host_state("s-1") == HostStateRecord("s-1", "provider_authoritative", STATE, prompt_sha256="ab" * 32)


def test_stateless_record_round_trips_without_state():
    save_host_state(HostStateRecord("s-2", "stateless", None))
    assert load_host_state("s-2").state is None


def test_missing_record_is_none():
    assert load_host_state("never-saved") is None


def test_corrupt_record_is_binding_invalid_and_content_free():
    save_host_state(HostStateRecord("s-3", "provider_authoritative", STATE))
    path = next(host_state_dir().glob("*.json"))
    path.write_text('{"schema": "hermes.memory-host-session/v1", "session_id": "s-3", "disposition": "provider_authoritative"}', encoding="utf-8")
    with pytest.raises(BindingInvalidError) as exc:
        load_host_state("s-3")
    assert "A" * 43 not in str(exc.value)


def test_unparseable_record_is_binding_invalid():
    save_host_state(HostStateRecord("s-4", "provider_authoritative", STATE))
    next(host_state_dir().glob("*.json")).write_text("{not json", encoding="utf-8")
    with pytest.raises(BindingInvalidError):
        load_host_state("s-4")


def test_record_from_another_session_is_refused():
    """A record whose body names a different session never satisfies this one."""
    save_host_state(HostStateRecord("s-5", "provider_authoritative", STATE))
    path = next(host_state_dir().glob("*.json"))
    path.write_text(path.read_text(encoding="utf-8").replace('"s-5"', '"someone-else"'), encoding="utf-8")
    with pytest.raises(BindingInvalidError):
        load_host_state("s-5")


def test_malformed_prompt_digest_is_binding_invalid():
    save_host_state(HostStateRecord("s-6", "stateless", None))
    path = next(host_state_dir().glob("*.json"))
    path.write_text(path.read_text(encoding="utf-8").replace('"prompt_sha256": null', '"prompt_sha256": "short"'), encoding="utf-8")
    with pytest.raises(BindingInvalidError):
        load_host_state("s-6")


def test_inherit_copies_the_parent_record_under_the_child_id():
    save_host_state(HostStateRecord("parent", "provider_authoritative", STATE, prompt_sha256="cd" * 32))
    child = inherit_host_state("parent", "child")
    assert child.session_id == "child" and child.state == STATE
    assert load_host_state("parent").session_id == "parent"
    assert inherit_host_state("unknown", "orphan") is None and load_host_state("orphan") is None


def test_inherit_persists_the_child_record():
    save_host_state(HostStateRecord("p2", "provider_authoritative", STATE, prompt_sha256="ef" * 32))
    inherit_host_state("p2", "c2")
    reloaded = load_host_state("c2")
    assert reloaded is not None and reloaded.state == STATE and reloaded.prompt_sha256 == "ef" * 32


def test_record_lives_outside_the_native_memory_directory():
    from tools.memory_tool import get_memory_dir
    assert not host_state_dir().resolve().is_relative_to(get_memory_dir().resolve())


def test_records_live_under_the_contracted_home_root_name(tmp_path):
    """Contract C2 / ruling X-1 (a): R44 prunes exactly this name from every archive."""
    from agent.memory_service.host_state import HOST_STATE_ROOT_DIRNAME
    assert host_state_dir(tmp_path).relative_to(tmp_path).parts[0] == HOST_STATE_ROOT_DIRNAME


def test_explicit_home_is_honoured_over_the_ambient_one(tmp_path):
    save_host_state(HostStateRecord("s-7", "stateless", None), hermes_home=tmp_path)
    assert load_host_state("s-7", hermes_home=tmp_path) is not None
    assert load_host_state("s-7") is None


def test_session_ids_with_path_characters_are_safe():
    save_host_state(HostStateRecord("agent:main:telegram/123", "stateless", None))
    assert load_host_state("agent:main:telegram/123") is not None
    assert all(p.parent == host_state_dir() for p in host_state_dir().iterdir())


@pytest.mark.linux_only  # POSIX permission bits; Windows ACLs do not map onto them
def test_saved_record_is_owner_only():
    """§9.3 L987: required host state, written owner-only and never exposed."""
    save_host_state(HostStateRecord("s-perm", "provider_authoritative", STATE))
    path = next(host_state_dir().glob("*.json"))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
