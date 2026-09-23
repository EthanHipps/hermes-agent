"""S6 identity: one frozen identity per conversation, across every transition.

Shared helpers for the whole S6 suite live at the top of this file; the other
``test_s6_identity_*.py`` modules import them from here.

Part 1 (Task 3) covers the session-binding resolver at agent init: resume,
inherit, new, stateless and invalid, plus D-R40-1 (a continuation with no
persisted state never binds a new logical session into an old conversation).

The R36 fake does not model ygg's one-live-handle-per-logical-session rule
(ledger D-R10-3), so every assertion here counts binds rather than expecting a
refusal; K-1 hands the rule-enforcing run to R47/R48 conformance.
"""

import os
from typing import Any, List, Optional

import pytest

from agent.memory_service.bootstrap import init_memory_service
from agent.memory_service.errors import MemoryBlockedError
from agent.memory_service.host_state import HostStateRecord, host_state_dir, load_host_state, save_host_state
from agent.memory_service.service import MemoryDisposition
from tests.agent.memory_service.fake_backend import FakeAuthoritativeBackend, FakeProviderStore, FakeRegistry
from tests.agent.memory_service.native_sentinel import native_memory_sentinel

REPO = "repo-1"


def _never():
    """A ``store_factory`` no non-additive session may ever call (§9.1 L925)."""
    raise AssertionError("native store built in a non-additive session")


def _native_stub():
    from tools.memory_tool import MemoryStore

    return MemoryStore()


class MemoryEnv:
    """An authoritative config, a SessionDB and the R36 fake behind the real seam.

    The backend factory is patched at ``plugins.memory.load_authoritative_backend_factory``
    — where production reads it — so every code path that builds a service
    (``init_memory_service`` at init and ``ensure_session_binding`` at turn start)
    gets the fake without any of them taking a test-only argument.
    """

    def __init__(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        self.tmp_path = tmp_path
        self.registry = FakeRegistry(directories={os.path.realpath(os.getcwd()): (REPO, "proj-1", None)})
        self.store = FakeProviderStore(registry=self.registry)
        self.backends: List[FakeAuthoritativeBackend] = []
        self.factory_calls = 0
        executable = tmp_path / "p.exe"
        executable.write_bytes(b"MZ")
        self._executable = str(executable)
        self.config = self.config_with()
        from hermes_state import SessionDB

        self.db = SessionDB(db_path=tmp_path / "state.db")
        monkeypatch.setattr("plugins.memory.load_authoritative_backend_factory", lambda name: self._factory)

    def _factory(self, cfg) -> FakeAuthoritativeBackend:
        self.factory_calls += 1
        backend = FakeAuthoritativeBackend(self.store, provider=cfg.provider)
        self.backends.append(backend)
        return backend

    def config_with(self, **extra) -> dict:
        section = {"provider": "example", "provider_mode": "authoritative",
                   "provider_executable": self._executable, "principal_id": "ethan"}
        section.update(extra)
        return {"memory": section}

    def count(self, operation: str) -> int:
        return sum(backend.count(operation) for backend in self.backends)

    def init(self, session_id: str, *, config: Optional[dict] = None, **kwargs) -> Any:
        service, _ = init_memory_service(
            config if config is not None else self.config, logical_session_id=session_id,
            platform="cli", store_factory=_never, session_db=self.db, **kwargs)
        return service


@pytest.fixture
def env(tmp_path, monkeypatch) -> MemoryEnv:
    return MemoryEnv(tmp_path, monkeypatch)


@pytest.fixture
def native_dir():
    """The native memory directory under this test's isolated HERMES_HOME."""
    from tools.memory_tool import get_memory_dir

    return get_memory_dir()


def _session(db, session_id, *, messages=0, parent=None, branched=False, delegate=False,
             end_reason=None, parent_end_reason=None):
    """Create a SessionDB row through the public API, as the real surfaces do."""
    if parent is not None and db.get_session(parent) is None:
        _session(db, parent, messages=2, end_reason=parent_end_reason)
    elif parent is not None and parent_end_reason:
        db.end_session(parent, parent_end_reason)
    model_config = {}
    if branched:
        model_config["_branched_from"] = parent
    if delegate:
        model_config["_delegate_from"] = parent
    db.create_session(session_id, "cli", parent_session_id=parent, model_config=model_config or None)
    if messages:
        rows = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(messages)]
        db.append_messages_batch(session_id, rows)
    if end_reason:
        db.end_session(session_id, end_reason)
    return session_id


# ---------------------------------------------------------------------------
# Task 3: the resolver at agent init
# ---------------------------------------------------------------------------


def test_first_init_binds_once_and_persists_the_frozen_identity(env):
    service = env.init("s-1")
    record = load_host_state("s-1")
    assert record.state == service.session_state and env.count("bind_session") == 1
    assert record.disposition == "provider_authoritative"


def test_rebuilt_agent_for_the_same_session_resumes_without_binding(env):
    """Gateway eviction / --resume / hygiene agent: validate, never bind (D-R40-2)."""
    env.init("s-1")
    _session(env.db, "s-1", messages=4)
    binds = env.count("bind_session")
    resumed = env.init("s-1")
    assert env.count("bind_session") == binds and env.count("validate_session") == 1
    assert resumed.identity == load_host_state("s-1").state.identity


def test_branch_child_inherits_the_parent_identity(env):
    parent_identity = env.init("parent").identity
    binds = env.count("bind_session")
    _session(env.db, "child", parent="parent", branched=True)
    child = env.init("child")
    assert child.identity == parent_identity
    assert env.count("bind_session") == binds
    assert load_host_state("child").state == load_host_state("parent").state


def test_compression_child_inherits_the_parent_identity(env):
    parent_identity = env.init("parent").identity
    binds = env.count("bind_session")
    _session(env.db, "rotated", parent="parent", parent_end_reason="compression")
    rotated = env.init("rotated")
    assert rotated.identity == parent_identity
    assert env.count("bind_session") == binds


def test_delegate_child_never_inherits(env):
    """A ``_delegate_from`` child is a new logical session; R43 owns delegation identity."""
    parent_identity = env.init("parent").identity
    _session(env.db, "delegate", parent="parent", delegate=True)
    child = env.init("delegate")
    assert child.identity != parent_identity
    assert env.count("bind_session") == 2
    assert load_host_state("delegate").state != load_host_state("parent").state


@pytest.mark.parametrize("policy", ["fail_closed", "stateless"])
def test_continuation_without_state_is_binding_invalid(env, policy):
    """D-R40-1: never bind a new logical session into an old conversation."""
    _session(env.db, "old", messages=6)
    config = env.config_with(authoritative_failure_policy=policy)
    if policy == "fail_closed":
        with pytest.raises(MemoryBlockedError) as exc:
            env.init("old", config=config)
        assert exc.value.code == "binding_invalid"
        assert load_host_state("old") is None
    else:
        service = env.init("old", config=config)
        assert service.disposition is MemoryDisposition.STATELESS
        assert load_host_state("old").disposition == "stateless"
    assert env.count("bind_session") == 0
    assert env.factory_calls == 0


def test_stateless_record_stays_stateless_without_contacting_the_provider(env):
    """I1 and §9.6 L1582: only a new logical session may select authoritative operation."""
    save_host_state(HostStateRecord("s-9", "stateless", None))
    service = env.init("s-9")
    assert service.disposition is MemoryDisposition.STATELESS and env.factory_calls == 0


def test_a_session_with_no_row_at_all_binds_as_new(env):
    """A brand-new id with no SessionDB row is a new logical session, not a continuation."""
    service = env.init("brand-new")
    assert service.disposition is MemoryDisposition.AUTHORITATIVE
    assert env.count("bind_session") == 1


def test_an_empty_row_with_no_messages_binds_as_new(env):
    _session(env.db, "empty")
    service = env.init("empty")
    assert service.disposition is MemoryDisposition.AUTHORITATIVE
    assert env.count("bind_session") == 1


def test_corrupt_record_blocks_instead_of_rebinding(env):
    """A record that cannot be trusted never silently becomes a fresh bind."""
    env.init("s-1")
    next(host_state_dir().glob("*.json")).write_text("{not json", encoding="utf-8")
    binds = env.count("bind_session")
    with pytest.raises(Exception):
        env.init("s-1")
    assert env.count("bind_session") == binds


def test_additive_init_never_creates_host_state(tmp_path):
    init_memory_service({"memory": {}}, logical_session_id="s-1", platform="cli", store_factory=_native_stub)
    assert not host_state_dir().exists()


def test_a_session_without_an_id_is_not_managed(env):
    """An id-less agent takes the unchanged bind path and leaves no stray record.

    ``FrozenIdentityWire.validate`` requires a non-empty ``logical_session_id``, so
    an id-less authoritative bind already fails at ``5c583156f7``. What R40 adds is
    that the resolver is not consulted for it and no host-state record is written
    under a nameless key (X-5).
    """
    with pytest.raises(MemoryBlockedError):
        env.init("")
    assert env.count("bind_session") == 1
    assert not host_state_dir().exists()


def test_a_resume_keeps_the_prompt_digest_the_session_recorded(env):
    """Ruling R40-4d: the digest survives a resume of the same disposition and identity."""
    env.init("s-1")
    record = load_host_state("s-1")
    save_host_state(HostStateRecord(record.session_id, record.disposition, record.state, prompt_sha256="ab" * 32))
    _session(env.db, "s-1", messages=2)
    env.init("s-1")
    assert load_host_state("s-1").prompt_sha256 == "ab" * 32


def test_every_resolver_path_leaves_the_native_directory_untouched(env, native_dir):
    """§9.10 L1685: proven with the sentinel, never inferred from output."""
    native_dir.mkdir(parents=True, exist_ok=True)
    (native_dir / "MEMORY.md").write_text("NATIVE-SECRET-FACT\n", encoding="utf-8")
    _session(env.db, "old", messages=6)
    _session(env.db, "parent2")
    env.init("parent2")                                              # bound while still empty
    env.db.append_messages_batch("parent2", [{"role": "user", "content": "hi"}])
    _session(env.db, "child", parent="parent2", branched=True)
    save_host_state(HostStateRecord("s-stateless", "stateless", None))
    with native_memory_sentinel(native_dir) as sentinel:
        env.init("s-new")                                            # new
        env.init("s-new")                                            # resume
        env.init("child")                                            # inherit
        env.init("s-stateless")                                      # stateless
        with pytest.raises(MemoryBlockedError):
            env.init("old")                                          # invalid
    sentinel.assert_untouched()
    assert (native_dir / "MEMORY.md").read_text(encoding="utf-8") == "NATIVE-SECRET-FACT\n"
