"""S6 inventory suite B part 3: desktop/web memory status, reset and journey REST (§9.7 L1607-L1609)."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent.memory_service import wire as w
from agent.memory_service.archive import archive_disposition
from hermes_constants import get_hermes_home
from tests.agent.memory_service.fake_backend import FakeAuthoritativeBackend, FakeProviderStore, FakeRegistry
from tests.agent.memory_service.native_sentinel import native_memory_sentinel

POISON = "POISON-NATIVE-7f3a"


def _client():
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    import hermes_state
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    # Keep the state DB under the isolated HERMES_HOME for any handler that touches it.
    hermes_state.DEFAULT_DB_PATH = get_hermes_home() / "state.db"
    return client


@pytest.fixture
def env(tmp_path, monkeypatch):
    from tools.memory_tool import get_memory_dir
    store = FakeProviderStore(registry=FakeRegistry())
    backends = []

    def factory(cfg):
        backends.append(FakeAuthoritativeBackend(store, provider="example"))
        return backends[-1]
    monkeypatch.setattr("plugins.memory.load_authoritative_backend_factory", lambda name: factory)
    exe = tmp_path / "provider.exe"
    exe.write_bytes(b"MZ")
    (get_hermes_home() / "config.yaml").write_text(yaml.safe_dump({"memory": {
        "provider": "example", "provider_mode": "authoritative", "provider_executable": str(exe),
        "principal_id": "ethan"}}), encoding="utf-8")
    native = get_memory_dir()
    native.mkdir(parents=True, exist_ok=True)
    (native / "MEMORY.md").write_text(f"{POISON} memory", encoding="utf-8")
    (native / "USER.md").write_text(f"{POISON} user", encoding="utf-8")
    return SimpleNamespace(store=store, backends=backends, native=native)


def _count(env, op):
    return sum(b.count(op) for b in env.backends)


def _native_state(native):
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(native.iterdir())}


# -- status and reset (rulings R42-8, R42-9, R42-1) -----------------------------------------------------


def test_status_reports_provider_managed_authority_without_touching_native(env):
    client = _client()
    with native_memory_sentinel(env.native) as sentinel:
        data = client.get("/api/memory").json()
    sentinel.assert_untouched()
    from hermes_cli.config import load_config
    mapping = archive_disposition(load_config()).as_mapping()
    assert {k: data["curated_memory"][k] for k in mapping} == mapping
    assert data["curated_memory"]["provider_mode"] == "authoritative"
    assert data["builtin_files"] == {"memory": 0, "user": 0} and env.backends == []   # no provider probe (R42-8)


def test_additive_status_is_unchanged(tmp_path):
    from tools.memory_tool import get_memory_dir
    native = get_memory_dir()
    native.mkdir(parents=True, exist_ok=True)
    (native / "MEMORY.md").write_text("notes", encoding="utf-8")
    data = _client().get("/api/memory").json()
    assert "curated_memory" not in data and data["builtin_files"]["memory"] == len(b"notes")


@pytest.mark.parametrize("body", [{"target": "all", "scopes": ["global:ethan"]}, {"target": "memory"},
                                  {"target": "memory", "scopes": ["repo:x"]},
                                  {"target": "user", "scopes": ["global:ethan"], "repo_id": "a\nb"}])
def test_reset_input_errors_are_400_and_touch_nothing(env, body):
    before = _native_state(env.native)
    assert _client().post("/api/memory/reset", json=body).status_code == 400
    assert _native_state(env.native) == before and _count(env, "bind_session") == 0


@pytest.mark.parametrize("body,need", [
    ({"target": "memory", "scopes": ["repository:repo-1"], "repo_id": "repo-1", "project_id": "proj-1"}, ["reset"]),
    ({"target": "user", "scopes": ["global:ethan"]}, ["target_user", "reset"])])
def test_reset_fails_closed_until_approval_and_never_deletes_native(env, body, need):
    r = env.store.seed_record(w.ScopeRef(kind="repository", id="repo-1"), "memory", "keep me")
    before = _native_state(env.native)
    with native_memory_sentinel(env.native) as sentinel:
        resp = _client().post("/api/memory/reset", json=body)
    sentinel.assert_untouched()
    detail = resp.json()["detail"]
    assert resp.status_code == 409 and detail["error"] == "approval_unavailable" and detail["approval_required"] == need
    assert _count(env, "stage_curated") == 0 and env.store.records[r.id].lifecycle == "active"
    assert _native_state(env.native) == before


def test_reset_of_an_ineligible_scope_is_400(env):
    resp = _client().post("/api/memory/reset", json={"target": "memory", "scopes": ["repository:repo-1"]})
    assert resp.status_code == 400 and resp.json()["detail"]["error"] == "unauthorized_scope"


def test_reset_with_the_provider_down_is_503(env):
    env.store.fail_transport("negotiate")
    resp = _client().post("/api/memory/reset", json={"target": "user", "scopes": ["global:ethan"]})
    assert resp.status_code == 503 and resp.json()["detail"]["error"] == "memory_unavailable"


# -- journey REST routes take the explicit selector (ruling R42-12) --------------------------------------


def test_journey_rest_routes_by_stable_id(env):
    r = env.store.seed_record(w.ScopeRef(kind="repository", id="repo-1"), "memory", "uses pnpm")
    ctx, node = {"repo_id": "repo-1", "project_id": "proj-1"}, f"memory:memory:{r.id}"
    client = _client()
    with native_memory_sentinel(env.native) as sentinel:
        graph = client.get("/api/learning/graph", params=ctx).json()
        detail = client.get("/api/learning/node", params={"id": node, **ctx}).json()
        edited = client.put("/api/learning/node", json={"id": node, "content": "uses pnpm 9", **ctx})
    sentinel.assert_untouched()
    assert node in {n["id"] for n in graph["nodes"]} and detail["content"] == "uses pnpm"
    assert edited.status_code == 200 and env.store.records[r.id].lifecycle == "superseded"
    assert client.get("/api/learning/graph", params={"repo_id": "a\nb"}).status_code == 400
    new_id = next(x.id for x in env.store.records_for(w.ScopeRef(kind="repository", id="repo-1"), "memory"))
    assert client.request("DELETE", "/api/learning/node", json={"id": f"memory:memory:{new_id}", **ctx}).status_code == 200


# -- typed disposition fields on REST archive routes (ruling R42-10; ledger R44-11) -----------------------


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    """Path.home() and HERMES_HOME both redirected (root AGENTS.md, Testing)."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_path


def _profile_like(home: Path, *, provider=None) -> Path:
    """Write a config.yaml into an arbitrary home directory, plus dormant native memory files."""
    home.mkdir(parents=True, exist_ok=True)
    if provider:
        exe = home.parent / f"{home.name}-provider.exe"
        exe.write_bytes(b"MZ")
        (home / "config.yaml").write_text(yaml.safe_dump({"memory": {
            "provider": provider, "provider_mode": "authoritative", "provider_executable": str(exe),
            "principal_id": "ethan"}}), encoding="utf-8")
    else:
        (home / "config.yaml").write_text("model: test\n", encoding="utf-8")
    native = home / "memories"
    native.mkdir(exist_ok=True)
    (native / "MEMORY.md").write_text(f"{POISON} memory", encoding="utf-8")
    return home


def _mapping(home: Path) -> dict:
    return archive_disposition(yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))).as_mapping()


def test_profile_export_carries_the_typed_disposition(profile_env):
    coder = _profile_like(profile_env / ".hermes" / "profiles" / "coder", provider="example")
    _profile_like(profile_env / ".hermes" / "profiles" / "plain")
    client = _client()
    with native_memory_sentinel(coder / "memories") as sentinel:
        body = client.post("/api/profiles/coder/export", json={"output": str(profile_env / "coder.tar.gz")}).json()
    sentinel.assert_untouched()
    assert {k: body["curated_memory"][k] for k in _mapping(coder)} == _mapping(coder)
    assert "provider-managed and not included" in body["curated_memory"]["message"]
    plain = client.post("/api/profiles/plain/export", json={"output": str(profile_env / "plain.tar.gz")}).json()
    assert plain["ok"] is True and "curated_memory" not in plain


def test_profile_import_and_clone_carry_the_restore_or_clone_sentence(profile_env, monkeypatch):
    import hermes_cli.profiles as profiles_mod
    imported = _profile_like(profile_env / ".hermes" / "profiles" / "imported", provider="example")
    cloned = _profile_like(profile_env / ".hermes" / "profiles" / "cloned", provider="example")
    monkeypatch.setattr(profiles_mod, "import_profile", lambda archive, name=None: imported)
    monkeypatch.setattr(profiles_mod, "create_profile", lambda **kwargs: cloned)
    monkeypatch.setattr(profiles_mod, "check_alias_collision", lambda name: "x")
    monkeypatch.setattr(profiles_mod, "seed_profile_skills", lambda *a, **k: None)
    monkeypatch.setattr(profiles_mod, "create_wrapper_script", lambda *a, **k: None)
    client = _client()
    with native_memory_sentinel(imported / "memories") as sentinel:
        restored = client.post("/api/profiles/import", json={"archive": str(profile_env / "x.tar.gz")}).json()
    sentinel.assert_untouched()
    assert {k: restored["curated_memory"][k] for k in _mapping(imported)} == _mapping(imported)
    assert "was not restored" in restored["curated_memory"]["message"]
    with native_memory_sentinel(cloned / "memories") as sentinel:
        clone = client.post("/api/profiles", json={"name": "cloned", "clone_from": "default"}).json()
    sentinel.assert_untouched()
    assert {k: clone["curated_memory"][k] for k in _mapping(cloned)} == _mapping(cloned)
    assert "clone complete" in clone["curated_memory"]["message"]


def test_backup_lists_authoritative_homes_without_a_completion_sentence(profile_env, monkeypatch):
    monkeypatch.setattr("hermes_cli.web_server_gateway._spawn_hermes_action",
                        lambda argv, name: SimpleNamespace(pid=1))
    _profile_like(profile_env / ".hermes")
    coder = _profile_like(profile_env / ".hermes" / "profiles" / "coder", provider="example")
    client = _client()
    with native_memory_sentinel(coder / "memories") as sentinel:
        body = client.post("/api/ops/backup", json={}).json()
    sentinel.assert_untouched()
    assert body["curated_memory"] == [{"home": "profiles/coder/", **_mapping(coder)}]
    _profile_like(coder)                                                           # every home additive now
    assert "curated_memory" not in client.post("/api/ops/backup", json={}).json()


def test_debug_share_carries_the_typed_disposition(profile_env, monkeypatch):
    import hermes_cli.debug as dbg
    monkeypatch.setattr(dbg, "upload_to_pastebin", lambda c, expiry_days=7: "https://paste.rs/x")
    monkeypatch.setattr(dbg, "_schedule_auto_delete", lambda *a, **k: None)
    monkeypatch.setattr(dbg, "_best_effort_sweep_expired_pastes", lambda: None)
    monkeypatch.setattr("hermes_cli.dump.run_dump", lambda a: None)
    home = _profile_like(get_hermes_home(), provider="example")
    logs = home / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for name in ("agent.log", "errors.log", "gateway.log"):
        (logs / name).write_text("line\n", encoding="utf-8")
    with native_memory_sentinel(home / "memories") as sentinel:
        body = _client().post("/api/ops/debug-share", json={}).json()
    sentinel.assert_untouched()
    assert {k: body["curated_memory"][k] for k in _mapping(home)} == _mapping(home)
    assert "provider-managed and not included" in body["curated_memory"]["message"]
