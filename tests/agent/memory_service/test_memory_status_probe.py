"""R41: agent/memory_service/status.py — negotiate-only probe and content-free status (C6b-13; §9.7, §9.3 L987)."""

import json
import os
from types import SimpleNamespace

import pytest

from agent.memory_service import wire as w
from agent.memory_service.status import collect_memory_status, format_scope_ref, parse_scope_ref, probe_provider
from tests.agent.memory_service.fake_backend import FakeAuthoritativeBackend, FakeProviderStore, FakeRegistry

EPOCH_A = "EPOCHVALUEAAAA"


@pytest.fixture
def authoritative_env(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    store = FakeProviderStore(epoch=EPOCH_A, registry=FakeRegistry(
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


def _cfg(memory):
    from agent.memory_service.config import resolve_memory_service_config
    return resolve_memory_service_config({"memory": memory})


def _context():
    return w.RequestedContext(principal_id="ethan", profile_id="default", logical_session_id="s-1", platform="cli",
                              org_id=None, project_id=None, repo_id=None, workspace_id=None,
                              resolution_source="directory", canonical_directory=os.path.realpath(os.getcwd()))


@pytest.mark.parametrize("text,kind", [("global:ethan", "principal_global"), ("organization:o-1", "organization"),
                                       ("project:p-1", "project"), ("repository:r-1", "repository")])
def test_scope_refs_round_trip_the_cli_encoding(text, kind):   # §11.2 L1837
    scope = parse_scope_ref(text)
    assert scope.kind == kind and format_scope_ref(scope) == text


@pytest.mark.parametrize("text", ["", "repo:r-1", "repository:", "r-1", "principal_global:ethan"])
def test_scope_refs_reject_anything_else(text):
    with pytest.raises(ValueError):
        parse_scope_ref(text)


def test_probe_reports_a_healthy_provider_without_binding(authoritative_env):
    probe = probe_provider(_cfg(authoritative_env.memory), backend_factory=authoritative_env.factory)
    backend = authoritative_env.backends[-1]
    assert probe.ok and probe.provider == "example" and probe.api_version == 1
    assert [op for op, _ in backend.calls] == ["negotiate"] and backend.shutdown_calls == 1


def test_probe_reports_an_unreachable_provider_content_free(authoritative_env):
    authoritative_env.store.fail_transport("negotiate")
    probe = probe_provider(_cfg(authoritative_env.memory), backend_factory=authoritative_env.factory)
    assert not probe.ok and probe.error and EPOCH_A not in probe.error
    assert authoritative_env.backends[-1].shutdown_calls == 1


def test_probe_reports_a_provider_name_mismatch(authoritative_env):
    probe = probe_provider(_cfg(authoritative_env.memory),
                           backend_factory=lambda c: FakeAuthoritativeBackend(authoritative_env.store, provider="other"))
    assert not probe.ok and "negotiated as" in probe.error


def test_probe_reports_a_missing_backend_factory(authoritative_env, monkeypatch):
    monkeypatch.setattr("plugins.memory.load_authoritative_backend_factory", lambda name: None)
    probe = probe_provider(_cfg(authoritative_env.memory))
    assert not probe.ok and "create_authoritative_backend" in probe.error


def test_status_shows_identity_scopes_and_binding_revision_but_no_opaque_value(authoritative_env):
    from agent.memory_service.authoritative import ProviderAuthoritativeMemoryService
    authoritative_env.store.seed_record(w.ScopeRef(kind="repository", id="repo-1"), "memory", "uses pnpm")
    service = ProviderAuthoritativeMemoryService(_cfg(authoritative_env.memory), authoritative_env.factory(None))
    service.start(_context())
    status = collect_memory_status(service)
    snapshot = service.load_curated("memory")
    memory = next(t for t in status.targets if t.target == "memory")
    assert status.disposition == "provider_authoritative" and status.identity["principal_id"] == "ethan"
    assert status.identity["binding_revision"] == service.identity.binding_revision
    assert memory.default_write_scope == format_scope_ref(snapshot.default_write_scope) and memory.entries == 1
    dumped = json.dumps(status.as_dict())
    assert "opaque_binding_b64url" not in dumped and service.identity.opaque_binding_b64url not in dumped
    assert EPOCH_A not in dumped   # the fake's revision tokens are small integers, so only the epoch is asserted absent


def test_status_of_a_stateless_session_reports_it_and_loads_nothing(authoritative_env):
    from agent.memory_service.service import StatelessMemoryService
    status = collect_memory_status(StatelessMemoryService(_cfg(authoritative_env.memory), reason="test"))
    assert status.disposition == "stateless" and status.identity is None and "stateless" in status.degraded.lower()
    assert all(t.error == "stateless" for t in status.targets if t.enabled)


def test_status_reports_a_blocked_target_by_code(authoritative_env):
    from agent.memory_service.authoritative import ProviderAuthoritativeMemoryService
    service = ProviderAuthoritativeMemoryService(_cfg(authoritative_env.memory), authoritative_env.factory(None))
    service.start(_context())
    authoritative_env.store.fail_transport("load_curated", times=4)
    status = collect_memory_status(service)
    assert {t.error for t in status.targets if t.enabled} == {"blocked"} and status.degraded
