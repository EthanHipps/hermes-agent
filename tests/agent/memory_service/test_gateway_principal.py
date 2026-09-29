"""R41 / I3 (§4.1 L252, §13.1 L2191): gateway users map explicitly; unknown or ambiguous get no memory.

Contract C6b-12; ruling R41-5."""

import os
from types import SimpleNamespace

import pytest

from agent.memory_service.config import MemoryConfigurationError
from agent.memory_service.principal import (GatewayIdentity, UnmappedPrincipalMemoryService, gateway_identity_of,
                                            gateway_principals, map_gateway_principal)
from tests.agent.memory_service.fake_backend import FakeAuthoritativeBackend, FakeProviderStore, FakeRegistry


@pytest.fixture
def authoritative_env(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    store = FakeProviderStore(epoch="EPOCHVALUEAAAA", registry=FakeRegistry(
        directories={os.path.realpath(os.getcwd()): ("repo-1", "proj-1", None)}))
    backends = []

    def factory(cfg):
        backends.append(FakeAuthoritativeBackend(store, provider="example"))
        return backends[-1]

    monkeypatch.setattr("plugins.memory.load_authoritative_backend_factory", lambda name: factory)
    exe = tmp_path / "provider.exe"
    exe.write_bytes(b"MZ")
    memory = {"provider": "example", "provider_mode": "authoritative", "provider_executable": str(exe),
              "principal_id": "ethan"}
    return SimpleNamespace(store=store, backends=backends, memory=memory, factory=factory)


def _raw(env, table):
    return {"memory": {**env.memory, "gateway_principals": table}}


def test_identity_exists_only_for_a_gateway_user():
    assert gateway_identity_of(SimpleNamespace(platform="cli", _user_id=None, _user_id_alt=None)) is None
    ident = gateway_identity_of(SimpleNamespace(platform="telegram", _user_id=111, _user_id_alt=None))
    assert ident == GatewayIdentity("telegram", "111", None) and ident.keys() == ("telegram:111",)


@pytest.mark.parametrize("table,expected", [({"telegram:111": "ethan"}, "ethan"), ({}, None),
                                            ({"telegram:111": "ethan", "telegram:a-1": "other"}, None),
                                            ({"telegram:111": "ethan", "telegram:a-1": "ethan"}, "ethan")])
def test_mapping_is_explicit_and_ambiguity_gets_nothing(authoritative_env, table, expected):
    assert map_gateway_principal(_raw(authoritative_env, table), GatewayIdentity("telegram", "111", "a-1")) == expected


@pytest.mark.parametrize("table", [["telegram:111"], {"111": "ethan"}, {"telegram:111": ""}, {"telegram:": "ethan"}])
def test_a_malformed_table_is_a_configuration_error(authoritative_env, table):
    with pytest.raises(MemoryConfigurationError):
        gateway_principals(_raw(authoritative_env, table))


def _init(env, raw, session_id, identity, **kw):
    from agent.memory_service.bootstrap import init_memory_service
    return init_memory_service(raw, logical_session_id=session_id, platform="telegram", store_factory=lambda: 1 / 0,
                               backend_factory=env.factory, gateway_identity=identity, **kw)[0]


def test_unmapped_user_gets_no_memory_no_provider_call_and_a_stateless_record(authoritative_env):
    from agent.memory_service.host_state import load_host_state
    service = _init(authoritative_env, _raw(authoritative_env, {"telegram:111": "ethan"}), "gw-1",
                    GatewayIdentity("telegram", "999"))
    assert isinstance(service, UnmappedPrincipalMemoryService) and authoritative_env.backends == []
    assert service.disposition.value == "stateless" and load_host_state("gw-1").disposition == "stateless"
    assert "not available" in service.degraded_warning()


def test_a_keyless_gateway_turn_gets_no_memory_and_a_stateless_record(authoritative_env):
    """EDD-67-A3: a gateway source with no user id (anonymous admin, channel post) is not a local surface."""
    from agent.memory_service.host_state import load_host_state
    identity = gateway_identity_of(SimpleNamespace(platform="telegram", _user_id=None, _user_id_alt=None,
                                                   _chat_id="-100"))
    service = _init(authoritative_env, _raw(authoritative_env, {"telegram:111": "ethan"}), "gw-k", identity)
    assert isinstance(service, UnmappedPrincipalMemoryService) and authoritative_env.backends == []
    assert load_host_state("gw-k").disposition == "stateless"
    assert identity == GatewayIdentity("telegram", None) and identity.keys() == ()


def test_an_existing_stateless_record_is_kept_byte_for_byte(authoritative_env):
    """C67-6: an unmapped user whose session already has a record gets no memory; the record is untouched."""
    from agent.memory_service.host_state import HostStateRecord, host_state_dir, save_host_state
    save_host_state(HostStateRecord("gw-4", "stateless", None))
    (path,) = list(host_state_dir().iterdir())
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    service = _init(authoritative_env, _raw(authoritative_env, {"telegram:111": "ethan"}), "gw-4",
                    GatewayIdentity("telegram", "999"))
    assert isinstance(service, UnmappedPrincipalMemoryService) and authoritative_env.backends == []
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_mapped_user_binds_as_the_mapped_principal(authoritative_env):
    raw = _raw(authoritative_env, {"telegram:111": "ethan"})
    raw["memory"]["principal_id"] = "someone-else"   # the configured principal is NOT used for a gateway user
    service = _init(authoritative_env, raw, "gw-2", GatewayIdentity("telegram", "111"))
    assert service.identity.principal_id == "ethan"


def test_resume_with_a_changed_mapping_gets_no_memory_and_keeps_the_record(authoritative_env):
    from agent.memory_service.host_state import load_host_state
    _init(authoritative_env, _raw(authoritative_env, {"telegram:111": "ethan"}), "gw-3", GatewayIdentity("telegram", "111"))
    record = load_host_state("gw-3")
    later = _init(authoritative_env, _raw(authoritative_env, {"telegram:111": "other"}), "gw-3",
                  GatewayIdentity("telegram", "111"))
    assert isinstance(later, UnmappedPrincipalMemoryService) and load_host_state("gw-3") == record


def test_additive_ignores_gateway_identity(authoritative_env):
    from agent.memory_service.bootstrap import init_memory_service
    service, _ = init_memory_service({"memory": {}}, logical_session_id="a-1", platform="telegram",
                                     store_factory=lambda: SimpleNamespace(),
                                     gateway_identity=GatewayIdentity("telegram", "999"))
    assert service.disposition.value == "builtin"
