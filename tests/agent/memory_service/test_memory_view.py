"""R43 view.py unit suite: explicit memory dependencies for background surfaces
(§9.7 L1610, L1625; §9.6 L1585; rulings R43-3, R43-4, R43-8; contract C6b-7)."""

import os
from types import SimpleNamespace

import pytest

from agent.memory_service import wire as w
from agent.memory_service.bootstrap import build_requested_context
from agent.memory_service.config import resolve_memory_service_config
from agent.memory_service.errors import CapabilityUnavailableError, MemoryBlockedError, StatelessSessionError
from agent.memory_service.service import (CommitIntent, MemoryDisposition, StatelessMemoryService,
                                          select_memory_service)
from agent.memory_service.view import (LimitedMemoryView, ParentMemory, UnboundMemoryService,
                                       capture_parent_memory, open_fork_view)
from tests.agent.memory_service.fake_backend import FakeAuthoritativeBackend, FakeProviderStore, FakeRegistry
from tools.memory_tool_curated import curated_memory_tool

REPO = w.ScopeRef(kind="repository", id="repo-1")


def _never():
    raise AssertionError("native store built in a non-additive session")


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

    def parent(self):
        ctx = build_requested_context(self.config, logical_session_id="parent-1", platform="cli",
                                      profile_id="default", working_directory=os.getcwd())
        return select_memory_service(self.raw, store_factory=_never, requested_context=ctx,
                                     backend_factory=self.factory)


@pytest.fixture
def env(tmp_path, monkeypatch):
    return _Env(tmp_path, monkeypatch)


def test_unbound_service_rejects_every_read_and_write_without_a_provider(env):
    unbound = UnboundMemoryService(env.config, surface="cron")
    assert unbound.disposition is MemoryDisposition.STATELESS and unbound.identity is None
    assert unbound.surface == "cron" and unbound.target_enabled("memory") is False
    with pytest.raises(StatelessSessionError):
        unbound.load_curated("memory")
    assert "no explicit memory identity" in unbound.degraded_warning()
    assert env.backends == []                                         # no provider was ever contacted


def test_view_pins_the_parent_identity_on_its_own_transport(env):
    parent = env.parent()
    view = open_fork_view(capture_parent_memory(SimpleNamespace(_memory_service=parent)),
                          surface="background_review", backend_factory=env.factory)
    assert isinstance(view, LimitedMemoryView) and view.identity == parent.identity
    own = env.backends[-1]
    assert own.count("negotiate") == 1 and own.count("validate_session") == 1 and own.count("bind_session") == 0
    view.load_curated("memory")
    assert own.count("load_curated") == 1                             # the parent's backend is not used


def test_view_stamps_its_surface_on_every_stage(env):
    parent = env.parent()
    view = open_fork_view(capture_parent_memory(SimpleNamespace(_memory_service=parent)),
                          surface="background_review", backend_factory=env.factory)
    assert '"success": true' in curated_memory_tool(view, action="add", content="uses pnpm")
    stage = [req for op, req in env.backends[-1].calls if op == "stage_curated"][-1]
    assert stage.provenance.initiating_surface == "background_review"
    assert stage.provenance.logical_session_id == parent.identity.logical_session_id
    assert [r.text for r in env.store.records_for(REPO, "memory")] == ["uses pnpm"]


def test_side_question_view_is_read_only(env):
    parent = env.parent()
    view = open_fork_view(capture_parent_memory(SimpleNamespace(_memory_service=parent)),
                          surface="side_question", backend_factory=env.factory)
    assert view.mutations is False
    with pytest.raises(CapabilityUnavailableError):
        view.stage_curated(None)
    assert env.backends[-1].count("stage_curated") == 0


def test_view_never_recalls_captures_or_commits_an_approval(env):
    parent = env.parent()
    view = open_fork_view(capture_parent_memory(SimpleNamespace(_memory_service=parent)),
                          surface="background_review", backend_factory=env.factory)
    assert view.capabilities.recall_context is False and view.capabilities.capture_continuity is False
    approved = w.ApprovalAuthorization(kind="approved", approval_id="a-1", approved_by_principal_id="ethan",
                                       approved_at="2026-09-27T00:00:00Z", expires_at="2026-09-28T00:00:00Z",
                                       approval_binding_sha256="0" * 64)
    intent = CommitIntent(target="memory", request_id="r-1", stage_handle_b64url="x", approval_binding_sha256="0" * 64,
                          authorized_write_scopes=(REPO,), authorization=approved)
    for call in (lambda: view.recall_context(None), lambda: view.capture_continuity(None),
                 lambda: view.commit_curated(intent)):
        with pytest.raises(CapabilityUnavailableError):
            call()
    own = env.backends[-1]
    assert own.count("recall_context") == own.count("capture_continuity") == own.count("commit_curated") == 0


def test_stateless_parent_yields_a_stateless_service_without_provider_contact(env):
    view = open_fork_view(ParentMemory(env.config, "stateless", None), surface="background_review",
                          backend_factory=env.factory)
    assert isinstance(view, StatelessMemoryService) and env.backends == []


def test_view_open_failure_is_content_free(env):
    parent = env.parent()
    env.store.set_epoch("ep-9")
    with pytest.raises(MemoryBlockedError) as exc:
        open_fork_view(capture_parent_memory(SimpleNamespace(_memory_service=parent)),
                       surface="background_review", backend_factory=env.factory)
    assert "ep-" not in str(exc.value)
    assert exc.value.__cause__ is None and exc.value.__suppress_context__ is True


def test_view_shutdown_is_idempotent(env):
    parent = env.parent()
    view = open_fork_view(capture_parent_memory(SimpleNamespace(_memory_service=parent)),
                          surface="background_review", backend_factory=env.factory)
    view.shutdown()
    view.shutdown()
    assert env.backends[-1].shutdown_calls == 1


def test_capture_is_none_for_an_absent_or_additive_service():
    assert capture_parent_memory(SimpleNamespace()) is None
    assert capture_parent_memory(SimpleNamespace(_memory_service=SimpleNamespace(disposition="builtin"))) is None
