"""Mode selection happens before any native store exists; stateless disposition."""

import pytest

from agent.memory_service.config import MemoryConfigurationError, resolve_memory_service_config
from agent.memory_service.errors import StatelessSessionError
from agent.memory_service.service import (
    MemoryDisposition,
    MemoryService,
    MutationRequest,
    StatelessMemoryService,
    select_memory_service,
)


class _Store:
    """Stand-in for the deferred native MemoryStore factory."""

    def __init__(self):
        self.constructed = 0

    def factory(self):
        self.constructed += 1
        from tools.memory_tool import MemoryStore

        return MemoryStore()


def test_default_config_selects_builtin_and_constructs_the_store_once():
    store = _Store()
    service = select_memory_service({}, store_factory=store.factory)
    assert isinstance(service, MemoryService)
    assert service.disposition is MemoryDisposition.BUILTIN
    assert store.constructed == 1
    assert service.identity is not None and service.identity.provider == "builtin"


def test_configuration_error_never_selects_a_service():
    store = _Store()
    with pytest.raises(MemoryConfigurationError):
        select_memory_service({"memory": {"provider_mode": "sidecar"}}, store_factory=store.factory)
    assert store.constructed == 0


def test_authoritative_never_constructs_the_native_store(tmp_path):
    exe = tmp_path / "p.exe"
    exe.write_bytes(b"MZ")
    store = _Store()
    config = {"memory": {"provider": "example", "provider_mode": "authoritative", "provider_executable": str(exe)}}

    def factory(cfg):
        raise AssertionError("backend factory reached; store must still not be built")

    with pytest.raises(AssertionError, match="backend factory reached"):
        select_memory_service(config, store_factory=store.factory, backend_factory=factory)
    assert store.constructed == 0


def test_authoritative_requires_a_backend_factory_or_plugin(tmp_path, monkeypatch):
    exe = tmp_path / "p.exe"
    exe.write_bytes(b"MZ")
    config = {"memory": {"provider": "no-such-provider-xyz", "provider_mode": "authoritative", "provider_executable": str(exe)}}
    with pytest.raises(MemoryConfigurationError, match="create_authoritative_backend"):
        select_memory_service(config, store_factory=_Store().factory)


def test_stateless_service_has_no_memory_and_a_durable_warning():
    cfg = resolve_memory_service_config({"memory": {"memory_enabled": True, "user_profile_enabled": False}})
    service = StatelessMemoryService(cfg, reason="bind_session: provider transport failure (timeout)")
    assert service.disposition is MemoryDisposition.STATELESS
    assert service.identity is None
    assert service.degraded_warning() and "stateless" in service.degraded_warning().lower()
    assert service.prompt_block("memory") is None
    assert not service.target_enabled("memory") and not service.target_enabled("user")
    with pytest.raises(StatelessSessionError):
        service.load_curated("memory")
    with pytest.raises(StatelessSessionError):
        service.stage_curated(MutationRequest(target="memory", request_id="r", expected_revision=None, hidden_preservation_state=None, requested_write_scopes=(), intent=None, mutation_delta=(), candidate_entries=(), provenance=None))
    assert service.capabilities.recall_context is False
    service.shutdown()


def test_plugin_discovery_reports_missing_factory(tmp_path, monkeypatch):
    import plugins.memory as pm

    plugin_dir = tmp_path / "plugins" / "demo"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text("PROVIDER_NAME = 'demo'\n", encoding="utf-8")
    monkeypatch.setattr(pm, "_get_user_plugins_dir", lambda: tmp_path / "plugins")
    monkeypatch.setattr(pm, "_is_memory_provider_dir", lambda p: True)
    assert pm.load_authoritative_backend_factory("demo") is None
    (plugin_dir / "__init__.py").write_text(
        "def create_authoritative_backend(config):\n    return ('backend-for', config.provider)\n",
        encoding="utf-8",
    )
    monkeypatch.delitem(__import__("sys").modules, pm._module_name(plugin_dir, "demo"), raising=False)
    factory = pm.load_authoritative_backend_factory("demo")
    cfg = resolve_memory_service_config({"memory": {"provider": "demo"}})
    assert factory(cfg) == ("backend-for", "demo")
    assert pm.load_authoritative_backend_factory("absent-provider") is None


def test_plugin_import_failure_is_reported_distinctly_from_missing_factory(tmp_path, monkeypatch):
    """M12: an import failure in the provider module (e.g. a syntax or
    top-level exception) must surface as the import failure, not as the
    generic 'does not export create_authoritative_backend' message -- that
    message means something different: the module imported fine but simply
    lacks the symbol."""
    import plugins.memory as pm

    plugin_dir = tmp_path / "plugins" / "broken"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text("raise ValueError('boom')\n", encoding="utf-8")
    monkeypatch.setattr(pm, "_get_user_plugins_dir", lambda: tmp_path / "plugins")
    monkeypatch.setattr(pm, "_is_memory_provider_dir", lambda p: True)

    with pytest.raises(MemoryConfigurationError, match="boom"):
        pm.load_authoritative_backend_factory("broken")
