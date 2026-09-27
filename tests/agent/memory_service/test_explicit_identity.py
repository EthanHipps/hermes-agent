"""R43 explicit.py unit suite: scheduled surfaces bind from explicit registry IDs, never a directory
(§9.2 L968; §9.3 L1229; rulings R43-6, R43-7, R43-12, X6b-3; contract C6b-8)."""

import os

import pytest

from agent.memory_service.config import MemoryConfigurationError, resolve_memory_service_config
from agent.memory_service.errors import MemoryBlockedError
from agent.memory_service.explicit import ExplicitScope, bind_explicit_service, explicit_scope_from_config
from agent.memory_service.service import MemoryDisposition
from tests.agent.memory_service.fake_backend import FakeAuthoritativeBackend, FakeProviderStore, FakeRegistry


class _Env:
    def __init__(self, tmp_path, monkeypatch, **extra):
        repo = tmp_path / "repo"
        repo.mkdir()
        monkeypatch.chdir(repo)
        self.store = FakeProviderStore(registry=FakeRegistry(
            directories={os.path.realpath(os.getcwd()): ("repo-1", "proj-1", None)}))
        self.backends = []
        exe = tmp_path / "provider.exe"
        exe.write_bytes(b"MZ")
        self.raw = {"memory": {"provider": "example", "provider_mode": "authoritative",
                               "provider_executable": str(exe), "principal_id": "ethan", **extra}}
        self.config = resolve_memory_service_config(self.raw)

    def factory(self, cfg):
        self.backends.append(FakeAuthoritativeBackend(self.store, provider=cfg.provider))
        return self.backends[-1]


@pytest.fixture
def env(tmp_path, monkeypatch):
    return _Env(tmp_path, monkeypatch)


def test_scope_absent_or_null_is_none():
    assert explicit_scope_from_config({"memory": {}}, "cron_scope") is None
    assert explicit_scope_from_config({"memory": {"cron_scope": None}}, "cron_scope") is None
    assert explicit_scope_from_config({}, "cron_scope") is None


def test_empty_mapping_is_an_explicit_principal_global_scope():
    assert explicit_scope_from_config({"memory": {"cron_scope": {}}}, "cron_scope") == ExplicitScope()


def test_ids_are_parsed_and_stripped():
    raw = {"memory": {"cron_scope": {"repo_id": " repo-1 ", "project_id": "proj-1"}}}
    assert explicit_scope_from_config(raw, "cron_scope") == ExplicitScope(project_id="proj-1", repo_id="repo-1")


@pytest.mark.parametrize("bad", [["repo-1"], "repo-1", {"repo": "r"}, {"repo_id": ""}, {"repo_id": 3}])
def test_malformed_scope_is_a_configuration_error(bad):
    with pytest.raises(MemoryConfigurationError, match=r"memory\.cron_scope"):
        explicit_scope_from_config({"memory": {"cron_scope": bad}}, "cron_scope")


def test_explicit_bind_uses_explicit_ids_and_never_a_directory(env, tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)                                       # an unregistered cwd
    service = bind_explicit_service(env.raw, scope=ExplicitScope(project_id="proj-1", repo_id="repo-1"),
                                    logical_session_id="cron_j_1", platform="cron", profile_id="default",
                                    backend_factory=env.factory)
    bind = [req for op, req in env.backends[-1].calls if op == "bind_session"][-1]
    ctx = bind.requested_context
    assert (ctx.resolution_source, ctx.canonical_directory, ctx.platform, ctx.logical_session_id) == \
        ("explicit_ids", None, "cron", "cron_j_1")
    assert service.identity.repo_id == "repo-1" and service.identity.platform == "cron"
    from agent.memory_service.host_state import load_host_state
    assert load_host_state("cron_j_1") is None                         # ruling R43-7: no record


def test_empty_scope_binds_principal_global_only(env):
    service = bind_explicit_service(env.raw, scope=ExplicitScope(), logical_session_id="cron_j_2",
                                    platform="cron", profile_id="default", backend_factory=env.factory)
    snapshot = service.load_curated("memory")
    assert snapshot.status == "degraded_global_only" and snapshot.default_write_scope is None


@pytest.mark.parametrize("policy", ["fail_closed", "stateless"])
def test_outage_follows_the_failure_policy(tmp_path, monkeypatch, policy):
    env = _Env(tmp_path, monkeypatch, authoritative_failure_policy=policy)
    env.store.fail_transport("negotiate")

    def call():
        return bind_explicit_service(env.raw, scope=ExplicitScope(repo_id="repo-1", project_id="proj-1"),
                                     logical_session_id="cron_j_3", platform="cron", profile_id="default",
                                     backend_factory=env.factory)

    if policy == "fail_closed":
        with pytest.raises(MemoryBlockedError):
            call()
    else:
        assert call().disposition is MemoryDisposition.STATELESS


def test_additive_config_is_refused():
    with pytest.raises(ValueError):
        bind_explicit_service({"memory": {}}, scope=ExplicitScope(), logical_session_id="s", platform="cron",
                              profile_id="default")
