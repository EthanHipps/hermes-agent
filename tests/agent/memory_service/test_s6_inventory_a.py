"""S6 inventory suite A for R41 (§13.1 L2211; §14.1 row 41 L2311).

The non-agent memory command path, CLI/TUI/Gateway /memory, setup/status/off, doctor and CLI reset
route through MemoryService, or fail closed before any provider contact where the approval flow is
not yet available (ruling R41-1); nothing touches the native directory; gateway principals map
explicitly (I3, §13.1 L2191). Helpers are local (reconciliation rule 10: shared test support is
frozen)."""

import asyncio
import json
import os
import threading
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agent.memory_service import wire as w
from agent.memory_service.errors import MemoryBlockedError
from agent.memory_service.service import MemoryDisposition
from gateway.run import GatewayRunner
from gateway.slash_commands import GatewaySlashCommandsMixin
from tests.agent.memory_service.fake_backend import FakeAuthoritativeBackend, FakeProviderStore, FakeRegistry
from tests.agent.memory_service.native_sentinel import native_memory_sentinel

EPOCH_A, EPOCH_B = "EPOCHVALUEAAAA", "EPOCHVALUEBBBB"
REPO = w.ScopeRef(kind="repository", id="repo-1")


def _tool_defs(*names):
    return [{"type": "function", "function": {"name": n, "description": f"{n} tool",
                                              "parameters": {"type": "object", "properties": {}}}} for n in names]


@pytest.fixture
def native_dir():
    from tools.memory_tool import get_memory_dir
    return get_memory_dir()


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


def _write_config(memory_section, home=None):
    from hermes_constants import get_hermes_home
    path = (home or get_hermes_home()) / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"memory": memory_section}), encoding="utf-8")
    return path


def _agent(memory_section, **kwargs):
    # Config first: the first ``run_agent`` import in a process runs ensure_hermes_home() (load_config at
    # import), which must already see the requested mode, as it does in production.
    _write_config(memory_section)
    from run_agent import AIAgent
    with patch("model_tools.get_tool_definitions", return_value=_tool_defs("memory", "web_search")), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=False, **kwargs)
    agent.client = MagicMock()
    return agent


def _context():
    return w.RequestedContext(principal_id="ethan", profile_id="default", logical_session_id="s-1", platform="cli",
                              org_id=None, project_id=None, repo_id=None, workspace_id=None,
                              resolution_source="directory", canonical_directory=os.path.realpath(os.getcwd()))


def _no_opaque(text, service=None):
    """§9.3 L987: neither fixture epoch nor the binding handle may appear."""
    assert EPOCH_A not in text and EPOCH_B not in text, text
    handle = getattr(getattr(service, "identity", None), "opaque_binding_b64url", None)
    assert not handle or handle not in text, text


def _cfg(memory):
    from agent.memory_service.config import resolve_memory_service_config
    return resolve_memory_service_config({"memory": memory})


def test_epoch_values_never_reach_a_blocked_message_or_warning(authoritative_env):
    """Checkpoint B carry-forward (D-R41-1): reachable from /context, /compress and the TUI persist log."""
    from agent.memory_service.authoritative import ProviderAuthoritativeMemoryService
    from agent.memory_service.service import select_memory_service
    cfg = _cfg(authoritative_env.memory)
    bound = ProviderAuthoritativeMemoryService(cfg, FakeAuthoritativeBackend(authoritative_env.store))
    bound.start(_context())
    state = bound.session_state
    authoritative_env.store.envelope_epoch_override("load_curated", EPOCH_B)
    with pytest.raises(MemoryBlockedError) as loaded:
        bound.load_curated("memory")
    authoritative_env.store.set_epoch(EPOCH_B)
    view = ProviderAuthoritativeMemoryService(cfg, FakeAuthoritativeBackend(authoritative_env.store))
    with pytest.raises(MemoryBlockedError) as resumed:
        view.resume(state)
    stateless = select_memory_service({"memory": {**authoritative_env.memory, "authoritative_failure_policy": "stateless"}},
                                      store_factory=lambda: None, session_state=state,
                                      backend_factory=lambda c: FakeAuthoritativeBackend(authoritative_env.store))
    for text in (str(loaded.value), str(resumed.value), bound.degraded_warning() or "", stateless.degraded_warning() or ""):
        _no_opaque(text)


def test_negotiate_only_negotiates_and_nothing_else(authoritative_env):
    from agent.memory_service.authoritative import ProviderAuthoritativeMemoryService
    backend = FakeAuthoritativeBackend(authoritative_env.store)
    service = ProviderAuthoritativeMemoryService(_cfg(authoritative_env.memory), backend)
    negotiation = service.negotiate_only()
    assert negotiation.provider == "example" and negotiation.selected_api_version == 1
    assert [op for op, _ in backend.calls] == ["negotiate"]
    assert service.identity is None and not service.blocked


GATEWAY = {"telegram:111": "ethan"}


def test_unmapped_gateway_user_gets_no_memory_and_no_provider_call(authoritative_env, native_dir):
    with native_memory_sentinel(native_dir) as sentinel:
        agent = _agent({**authoritative_env.memory, "gateway_principals": GATEWAY}, platform="telegram",
                       user_id="999", session_id="gw-unmapped")
    sentinel.assert_untouched()
    assert agent._memory_service.disposition is MemoryDisposition.STATELESS
    assert authoritative_env.backends == []          # no bind, no load: never a degraded_global_only result (I3)
    out = json.loads(agent._invoke_tool("memory", {"action": "add", "target": "user", "content": "x"}, "t"))
    assert out["success"] is False


def test_ambiguous_gateway_user_gets_no_memory(authoritative_env):
    agent = _agent({**authoritative_env.memory, "gateway_principals": {**GATEWAY, "telegram:alt-9": "other"}},
                   platform="telegram", user_id="111", user_id_alt="alt-9", session_id="gw-ambiguous")
    assert agent._memory_service.disposition is MemoryDisposition.STATELESS and authoritative_env.backends == []


def test_mapped_gateway_user_binds_as_its_principal(authoritative_env):
    agent = _agent({**authoritative_env.memory, "gateway_principals": GATEWAY}, platform="telegram", user_id="111",
                   session_id="gw-mapped")
    assert agent._memory_service.identity.principal_id == "ethan"
    assert authoritative_env.backends[-1].count("bind_session") == 1


def test_local_surfaces_keep_the_configured_principal(authoritative_env):
    agent = _agent(authoritative_env.memory, session_id="cli-1")
    assert agent._memory_service.identity.principal_id == "ethan"


def test_doctor_probes_negotiate_only_without_native_access(authoritative_env, native_dir, monkeypatch):
    import hermes_cli.doctor as doctor
    from hermes_constants import get_hermes_home
    monkeypatch.setattr(doctor, "HERMES_HOME", get_hermes_home(), raising=False)
    _write_config(authoritative_env.memory)
    native_dir.mkdir(parents=True, exist_ok=True)
    (native_dir / "MEMORY.md").write_text("dormant\n", encoding="utf-8")
    from hermes_cli.doctor_state import _check_directory_structure, _check_memory_provider
    with native_memory_sentinel(native_dir) as sentinel:
        finding = _check_memory_provider(True)
        _check_directory_structure(True)
    sentinel.assert_untouched()
    assert not finding.issues and [op for op, _ in authoritative_env.backends[-1].calls] == ["negotiate"]


def _cli(agent, running=False):
    from hermes_cli.cli_commands_mixin import CLICommandsMixin
    handler = CLICommandsMixin.__new__(CLICommandsMixin)
    handler.agent, handler._agent_running = agent, running
    return handler


def test_cli_memory_status_uses_the_live_service(authoritative_env, native_dir, capsys):
    from agent.memory_service.status import format_scope_ref
    agent = _agent(authoritative_env.memory, session_id="cli-status")
    loads = authoritative_env.backends[-1].count("load_curated")
    with native_memory_sentinel(native_dir) as sentinel:
        _cli(agent)._handle_memory_command("/memory status")
    sentinel.assert_untouched()
    out = capsys.readouterr().out
    snapshot = agent._memory_service.load_curated("memory")
    assert format_scope_ref(snapshot.default_write_scope) in out and agent._memory_service.identity.binding_revision in out
    assert authoritative_env.backends[-1].count("load_curated") > loads and authoritative_env.backends[-1].count("bind_session") == 1
    _no_opaque(out, agent._memory_service)


def test_cli_memory_without_a_live_agent_never_reads_native_files(authoritative_env, native_dir, capsys):
    _write_config(authoritative_env.memory)
    native_dir.mkdir(parents=True, exist_ok=True)
    (native_dir / "MEMORY.md").write_text("dormant\n", encoding="utf-8")
    with native_memory_sentinel(native_dir) as sentinel:
        _cli(None)._handle_memory_command("/memory")
        _cli(None)._handle_memory_command("/memory approve all")
    sentinel.assert_untouched()
    assert "no live memory session" in capsys.readouterr().out.lower()


def test_cli_memory_approve_never_applies_a_native_pending_record(authoritative_env, native_dir):
    """H15 / C67-9e: holds before and after R39 merges; no base handler text is asserted."""
    from tools import write_approval as wa
    agent = _agent(authoritative_env.memory, session_id="cli-approve")
    record = wa.stage_write(wa.MEMORY, {"action": "add", "target": "memory", "content": "legacy"}, summary="legacy",
                            origin="foreground")
    staged = wa._pending_path(wa.MEMORY, record["id"])
    before = (sorted(p.name for p in staged.parent.iterdir()), staged.read_bytes())
    with native_memory_sentinel(native_dir) as sentinel:
        _cli(agent)._handle_memory_command("/memory approve all")
    sentinel.assert_untouched()
    assert wa.pending_count(wa.MEMORY) == 1
    assert (sorted(p.name for p in staged.parent.iterdir()), staged.read_bytes()) == before


def test_cli_memory_refuses_service_subcommands_while_a_turn_runs(authoritative_env, capsys):
    agent = _agent(authoritative_env.memory, session_id="cli-busy")
    loads = authoritative_env.backends[-1].count("load_curated")
    _cli(agent, running=True)._handle_memory_command("/memory status")
    assert "between turns" in capsys.readouterr().out and authoritative_env.backends[-1].count("load_curated") == loads


def _tui_session(agent, running=False):
    from tui_gateway.transport import StdioTransport
    return {"agent": agent, "session_key": "tui-k", "history": [], "history_lock": threading.Lock(),
            "running": running, "transport": StdioTransport(lambda: None, threading.Lock()), "cwd": "", "source": "tui"}


def test_tui_live_memory_uses_the_session_service(authoritative_env, native_dir):
    from tui_gateway import server
    agent = _agent(authoritative_env.memory, session_id="tui-1")
    with patch.object(server, "_session_uses_compute_host", return_value=False), native_memory_sentinel(native_dir) as sentinel:
        out = server._live_slash_command_output("sid-1", _tui_session(agent), "memory", "status")
        busy = server._live_slash_command_output("sid-1", _tui_session(agent, running=True), "memory", "")
        approval = server._live_slash_command_output("sid-1", _tui_session(agent), "memory", "approval on")
    sentinel.assert_untouched()
    assert "repository:repo-1" in out and "between turns" in busy and approval is None
    _no_opaque(out, agent._memory_service)


def test_tui_live_memory_leaves_additive_sessions_to_the_worker(tmp_path, monkeypatch):
    from tui_gateway import server
    monkeypatch.chdir(tmp_path)
    agent = _agent({"memory_enabled": True})
    with patch.object(server, "_session_uses_compute_host", return_value=False):
        assert server._live_slash_command_output("sid-2", _tui_session(agent), "memory", "") is None


class _Gateway(GatewaySlashCommandsMixin):
    """The tests/gateway/test_slash_config_writes_routed_profile.py L22-30 shape, plus a warm agent cache."""

    _run_in_executor_with_context = GatewayRunner._run_in_executor_with_context
    _get_executor = GatewayRunner._get_executor

    def __init__(self, agent):
        self._agent_cache, self._agent_cache_lock = {"k": (agent, "sig")}, threading.Lock()

    def _session_key_for_source(self, _source):
        return "k"

    def _evict_cached_agent(self, _session_key):
        pass


def _event(args, user_id="111", alt=None):
    source = SimpleNamespace(platform=SimpleNamespace(value="telegram"), user_id=user_id, user_id_alt=alt)
    return SimpleNamespace(source=source, get_command_args=lambda: args)


@pytest.fixture
def gateway_home(monkeypatch):
    import gateway.run as gateway_run
    from hermes_constants import get_hermes_home
    monkeypatch.setattr(gateway_run, "_hermes_home", get_hermes_home())


def test_gateway_memory_uses_the_cached_session_service(authoritative_env, native_dir, gateway_home):
    memory = {**authoritative_env.memory, "gateway_principals": GATEWAY}
    agent = _agent(memory, platform="telegram", user_id="111", session_id="gw-status")
    with native_memory_sentinel(native_dir) as sentinel:
        out = asyncio.run(_Gateway(agent)._handle_memory_command(_event("status")))
    sentinel.assert_untouched()
    assert "repository:repo-1" in out and authoritative_env.backends[-1].count("bind_session") == 1
    _no_opaque(out, agent._memory_service)


def test_gateway_memory_refuses_an_unmapped_or_other_user(authoritative_env, gateway_home, monkeypatch):
    memory = {**authoritative_env.memory, "gateway_principals": {**GATEWAY, "telegram:222": "other"}}
    agent = _agent(memory, platform="telegram", user_id="111", session_id="gw-group")
    loads = authoritative_env.backends[-1].count("load_curated")
    calls = []
    monkeypatch.setattr("hermes_cli.write_approval_commands.handle_pending_subcommand",
                        lambda *a, **k: calls.append(a) or "listed")
    runner = _Gateway(agent)
    for event in (_event("status", user_id="999"), _event("", user_id="222"), _event("pending", user_id=None),
                  _event("approve all", user_id="999")):
        assert asyncio.run(runner._handle_memory_command(event)) == "Curated memory is not available in this chat."
    assert authoritative_env.backends[-1].count("load_curated") == loads
    assert calls == []          # an unmapped user's approve never reaches the shared handler (C67-9e)


def test_additive_sessions_keep_the_pre_r41_memory_command(tmp_path, monkeypatch):
    from hermes_cli.memory_command import provider_memory_command
    monkeypatch.chdir(tmp_path)
    agent = _agent({"memory_enabled": True})
    assert provider_memory_command(["status"], agent=agent, set_mode_fn=None) is None
    assert provider_memory_command([], agent=None, set_mode_fn=None) is None


def test_gateway_provider_memory_runs_off_the_event_loop(authoritative_env, gateway_home):
    """Ruling R41-22: provider-managed /memory is executed off-loop; additive never hops."""
    agent = _agent({**authoritative_env.memory, "gateway_principals": GATEWAY}, platform="telegram", user_id="111",
                   session_id="gw-hop")
    runner, hops = _Gateway(agent), []
    real = runner._run_in_executor_with_context

    async def spy(func, *args):
        hops.append(func)
        return await real(func, *args)

    runner._run_in_executor_with_context = spy
    asyncio.run(runner._handle_memory_command(_event("status")))
    assert len(hops) == 1


def test_additive_gateway_memory_stays_on_the_base_path(tmp_path, monkeypatch, gateway_home):
    _write_config({"memory_enabled": True})
    runner, hops = _Gateway(None), []

    async def spy(func, *args):
        hops.append(func)
        return func(*args)

    runner._run_in_executor_with_context = spy
    out = asyncio.run(runner._handle_memory_command(_event("")))
    assert hops == [] and out.startswith("memory.write_approval")


def test_a_multi_principal_gateway_never_reviews_pending_writes(authoritative_env, gateway_home, monkeypatch):
    """Ruling R41-20: R39's review renders every principal's staged diff; a multi-principal gateway never calls it."""
    from hermes_cli.memory_command import GATEWAY_REVIEW_UNAVAILABLE
    agent = _agent({**authoritative_env.memory, "gateway_principals": {**GATEWAY, "telegram:222": "other"}},
                   platform="telegram", user_id="111", session_id="gw-multi")
    calls = []
    monkeypatch.setattr("hermes_cli.write_approval_commands.handle_pending_subcommand",
                        lambda *a, **k: calls.append(a) or "listed")
    runner = _Gateway(agent)
    for args in ("", "pending", "approve all", "reject abc"):
        assert GATEWAY_REVIEW_UNAVAILABLE in asyncio.run(runner._handle_memory_command(_event(args)))
    assert calls == []


def test_a_single_principal_gateway_reaches_the_shared_handler(authoritative_env, gateway_home, monkeypatch):
    agent = _agent({**authoritative_env.memory, "gateway_principals": GATEWAY}, platform="telegram", user_id="111",
                   session_id="gw-single")
    calls = []
    monkeypatch.setattr("hermes_cli.write_approval_commands.handle_pending_subcommand",
                        lambda *a, **k: calls.append(k.get("memory_store", "absent")) or "listed")
    assert asyncio.run(_Gateway(agent)._handle_memory_command(_event("pending"))) == "listed"
    assert calls == [None]      # memory_store=None: nothing native-shaped can be applied (H15)


def test_load_on_disk_store_refuses_in_authoritative_mode(authoritative_env, native_dir):
    from tools.memory_tool import load_on_disk_store
    _write_config(authoritative_env.memory)
    with native_memory_sentinel(native_dir) as sentinel, pytest.raises(MemoryBlockedError) as refused:
        load_on_disk_store()
    sentinel.assert_untouched()
    assert refused.value.code == "native_dormant"


def test_onboarding_personalization_never_writes_user_md(authoritative_env, tmp_path, monkeypatch):
    """§9.7 L1605 (R41-19): fails closed before any provider contact; the target:user mutation is the R41 follow-up's."""
    from hermes_cli.profiles import get_profile_dir
    from tui_gateway.onboarding_personalization import remember_onboarding
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    default = get_profile_dir("default")
    _write_config(authoritative_env.memory, home=default)
    native = default / "memories"
    native.mkdir(parents=True, exist_ok=True)
    (native / "USER.md").write_text("dormant profile\n", encoding="utf-8")
    before = (native / "USER.md").stat()
    with native_memory_sentinel(native) as sentinel, pytest.raises(ValueError, match="approval"):
        remember_onboarding({"name": "Ethan"})
    sentinel.assert_untouched()
    assert (native / "USER.md").stat().st_mtime_ns == before.st_mtime_ns
    assert authoritative_env.backends == []   # C67-10b: no provider contact


def _create_clone():
    from tui_gateway import server
    return server._methods["profiles.create"]("r-1", {"name": "coder", "clone_from": "default", "no_alias": True,
                                                      "mirror_credentials": False})["result"]


def test_tui_profile_clone_reports_the_typed_disposition(authoritative_env, tmp_path, monkeypatch):
    from hermes_cli.profiles import get_profile_dir
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    _write_config({**authoritative_env.memory, "provider": "example"}, home=get_profile_dir("default"))
    assert _create_clone()["curated_memory"] == {"authority": "provider", "provider": "example", "provider_api": 1,
                                                 "included": False, "disposition": "provider-managed",
                                                 "restore_action": "reconnect-provider"}


def test_tui_profile_clone_of_an_additive_home_has_no_disposition(tmp_path, monkeypatch):
    from hermes_cli.profiles import get_profile_dir
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    _write_config({"memory_enabled": True}, home=get_profile_dir("default"))
    assert "curated_memory" not in _create_clone()


def test_hermes_memory_status_setup_off_never_touch_native_memory(authoritative_env, native_dir):
    """§9.7 L1618 (R41-7): status and setup show and validate; off refuses; none of them reaches memories/."""
    from hermes_cli.main_agent_cmds import _cmd_memory_off, cmd_memory
    _write_config(authoritative_env.memory)
    native_dir.mkdir(parents=True, exist_ok=True)
    (native_dir / "MEMORY.md").write_text("dormant\n", encoding="utf-8")
    with native_memory_sentinel(native_dir) as sentinel:
        for args in (Namespace(memory_command="status", session=None), Namespace(memory_command="setup", provider=None)):
            try:
                cmd_memory(args)
            except SystemExit:
                pass
        with pytest.raises(SystemExit):
            _cmd_memory_off()
    sentinel.assert_untouched()


def _reset_args(**over):
    base = dict(memory_command="reset", target="memory", yes=False, scope=["repository:repo-1"])
    return Namespace(**{**base, **over})


def _poison(native_dir):
    native_dir.mkdir(parents=True, exist_ok=True)
    for name in ("MEMORY.md", "USER.md"):
        (native_dir / name).write_text(f"dormant {name}\n", encoding="utf-8")
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in native_dir.iterdir()}


@pytest.mark.parametrize("over,code", [({}, 1), ({"yes": True}, 2), ({"scope": None}, 2), ({"target": "all"}, 2),
                                       ({"scope": ["repo:x"]}, 2)])
def test_cli_reset_requires_scope_and_approval_and_deletes_nothing(authoritative_env, native_dir, over, code, capsys):
    from hermes_cli.main_agent_cmds import cmd_memory
    _write_config(authoritative_env.memory)
    before = _poison(native_dir)
    with native_memory_sentinel(native_dir) as sentinel, pytest.raises(SystemExit) as exited:
        cmd_memory(_reset_args(**over))
    sentinel.assert_untouched()
    assert exited.value.code == code
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in native_dir.iterdir()} == before
    assert authoritative_env.backends == []          # no case contacts the provider (R41-1)
    if not over:
        out = capsys.readouterr().out
        assert "reset" in out and "repository:repo-1" in out


def test_invalid_authoritative_config_reset_deletes_nothing(authoritative_env, native_dir):
    from hermes_cli.main_agent_cmds import cmd_memory
    _write_config({**authoritative_env.memory, "principal_id": ""})
    before = _poison(native_dir)
    with native_memory_sentinel(native_dir) as sentinel, pytest.raises(SystemExit) as exited:
        cmd_memory(_reset_args())
    sentinel.assert_untouched()
    assert exited.value.code == 1 and {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in native_dir.iterdir()} == before


def test_additive_reset_refuses_authoritative_flags_and_still_deletes_without_them(tmp_path, native_dir):
    from hermes_cli.main_agent_cmds import cmd_memory
    _write_config({"memory_enabled": True})
    _poison(native_dir)
    with pytest.raises(SystemExit) as refused:
        cmd_memory(_reset_args(yes=True))
    assert refused.value.code == 2 and (native_dir / "MEMORY.md").exists()
    cmd_memory(Namespace(memory_command="reset", target="memory", yes=True, scope=None))
    assert not (native_dir / "MEMORY.md").exists() and (native_dir / "USER.md").exists()


def test_prompt_size_inspection_binds_no_session_and_touches_nothing(authoritative_env, native_dir):
    """Ruling R41-11 (X-5), under the native sentinel: no bind, no host record, no native access."""
    from hermes_constants import get_hermes_home
    _write_config(authoritative_env.memory)
    from hermes_cli.prompt_size import compute_prompt_breakdown
    with native_memory_sentinel(native_dir) as sentinel:
        data = compute_prompt_breakdown("cli")
    sentinel.assert_untouched()
    assert authoritative_env.backends == [] and not (get_hermes_home() / "memory_service").exists()
    assert "provider-managed" in data["curated_memory"]
