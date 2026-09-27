"""S6 inventory suite B part 4: a REST continuation that holds its own history but has no persisted memory identity never gets a new logical session (D-R40-1; §4.1 L250; ledger L552)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent.memory_service.bootstrap import claim_history_continuation, init_memory_service
from agent.memory_service.errors import MemoryBlockedError
from agent.memory_service.host_state import HostStateRecord, load_host_state, save_host_state
from agent.memory_service.service import StatelessMemoryService
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_constants import get_hermes_home
from tests.agent.memory_service.fake_backend import FakeAuthoritativeBackend, FakeProviderStore, FakeRegistry

HISTORY = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]


@pytest.fixture
def env(tmp_path):
    store = FakeProviderStore(registry=FakeRegistry())
    backends = []

    def factory(cfg):
        backends.append(FakeAuthoritativeBackend(store, provider="example"))
        return backends[-1]
    exe = tmp_path / "provider.exe"
    exe.write_bytes(b"MZ")

    def cfg(policy="fail_closed"):
        return {"memory": {"provider": "example", "provider_mode": "authoritative", "provider_executable": str(exe),
                           "principal_id": "ethan", "authoritative_failure_policy": policy}}
    return SimpleNamespace(store=store, backends=backends, factory=factory, cfg=cfg)


def _binds(env):
    return sum(b.count("bind_session") for b in env.backends)


def _boom():
    raise AssertionError("a native store was constructed")


def test_history_without_identity_is_binding_invalid_under_fail_closed(env):
    with pytest.raises(MemoryBlockedError) as exc:
        claim_history_continuation(env.cfg(), "sid-1", has_history=True)
    assert exc.value.code == "binding_invalid"
    assert _binds(env) == 0 and load_host_state("sid-1") is None


def test_history_without_identity_starts_stateless_under_stateless_policy(env):
    claim_history_continuation(env.cfg("stateless"), "sid-1", has_history=True)
    assert load_host_state("sid-1").disposition == "stateless"
    service = init_memory_service(env.cfg("stateless"), logical_session_id="sid-1", platform="api_server",
                                  store_factory=_boom, backend_factory=env.factory)[0]
    assert isinstance(service, StatelessMemoryService) and _binds(env) == 0


def test_no_history_an_existing_record_or_additive_is_a_no_op(env):
    claim_history_continuation(env.cfg(), "sid-1", has_history=False)
    assert load_host_state("sid-1") is None
    record = HostStateRecord("sid-2", "stateless", None)
    save_host_state(record)
    claim_history_continuation(env.cfg(), "sid-2", has_history=True)
    assert load_host_state("sid-2") == record
    claim_history_continuation({}, "sid-3", has_history=True)
    assert load_host_state("sid-3") is None and env.backends == []


@pytest.mark.asyncio
async def test_api_server_refuses_before_building_an_agent(env, monkeypatch):
    (get_hermes_home() / "config.yaml").write_text(yaml.safe_dump(env.cfg()), encoding="utf-8")
    monkeypatch.setattr("plugins.memory.load_authoritative_backend_factory", lambda name: env.factory)

    class _NeverBuilt:
        def __init__(self, *args, **kwargs):
            raise AssertionError("agent built")
    monkeypatch.setattr("run_agent.AIAgent", _NeverBuilt)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    with pytest.raises(MemoryBlockedError):
        await adapter._run_agent(user_message="hi", conversation_history=list(HISTORY), session_id="sid-3")
    assert _binds(env) == 0


def _runs_app(adapter: APIServerAdapter) -> web.Application:
    """/v1/runs only (copied shape of tests/gateway/test_api_server_runs.py::_create_runs_app)."""
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize("history,expected", [(HISTORY, True), (None, False)])
async def test_runs_route_declares_whether_the_client_sent_history(history, expected):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    body = {"input": "hello", **({"conversation_history": history} if history else {})}
    async with TestClient(TestServer(_runs_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent") as mock_create:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "done"}
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0
            mock_create.return_value = mock_agent
            resp = await cli.post("/v1/runs", json=body)
            assert resp.status == 202
            for _ in range(40):
                if mock_create.call_args is not None:
                    break
                await asyncio.sleep(0.05)
    assert mock_create.call_args.kwargs["has_history"] is expected
