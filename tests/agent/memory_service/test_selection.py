"""Mode selection happens before any native store exists; stateless disposition."""

import sys

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
    config = {"memory": {"provider": "example", "provider_mode": "authoritative", "provider_executable": str(exe), "principal_id": "ethan"}}

    def factory(cfg):
        raise AssertionError("backend factory reached; store must still not be built")

    with pytest.raises(AssertionError, match="backend factory reached"):
        select_memory_service(config, store_factory=store.factory, backend_factory=factory)
    assert store.constructed == 0


def test_authoritative_requires_a_backend_factory_or_plugin(tmp_path, monkeypatch):
    exe = tmp_path / "p.exe"
    exe.write_bytes(b"MZ")
    config = {"memory": {"provider": "no-such-provider-xyz", "provider_mode": "authoritative", "provider_executable": str(exe), "principal_id": "ethan"}}
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


_PLUGIN_SOURCES = ("user", "project", "package_entry_point", "module_entry_point", "attribute_entry_point")


def _install_plugin(tmp_path, monkeypatch, source, name, code):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)
    if source in ("user", "project"):
        root = tmp_path / ("home" if source == "user" else ".hermes") / "plugins"
        if source == "project":
            monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "1")
        package = root / name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(code, encoding="utf-8")
    else:
        module_name = f"{name}_{source}"
        if source == "package_entry_point":
            package = tmp_path / module_name
            package.mkdir()
            (package / "__init__.py").write_text(code, encoding="utf-8")
        else:
            (tmp_path / f"{module_name}.py").write_text(code, encoding="utf-8")
        distribution = tmp_path / f"{module_name}-1.0.dist-info"
        distribution.mkdir()
        (distribution / "METADATA").write_text(f"Name: {module_name}\nVersion: 1.0\n", encoding="utf-8")
        suffix = ":register" if source == "attribute_entry_point" else ""
        (distribution / "entry_points.txt").write_text(
            f"[hermes_agent.memory_providers]\n{name} = {module_name}{suffix}\n", encoding="utf-8"
        )
        monkeypatch.syspath_prepend(str(tmp_path))


def _authoritative_config(name):
    return {"memory": {"provider": name, "provider_mode": "authoritative", "provider_executable": sys.executable, "principal_id": "ethan"}}


@pytest.mark.parametrize("source", _PLUGIN_SOURCES)
def test_plugin_discovery_reports_missing_factory(tmp_path, monkeypatch, source):
    _install_plugin(
        tmp_path, monkeypatch, source, "legacy_demo",
        "from agent.memory_provider import MemoryProvider\n"
        "def register(ctx):\n    raise AssertionError('legacy registration must not run')\n",
    )
    store = _Store()
    with pytest.raises(MemoryConfigurationError, match="does not export create_authoritative_backend"):
        select_memory_service(_authoritative_config("legacy_demo"), store_factory=store.factory)
    assert store.constructed == 0


@pytest.mark.parametrize("source", _PLUGIN_SOURCES)
def test_plugin_factory_must_be_callable(tmp_path, monkeypatch, source):
    _install_plugin(
        tmp_path, monkeypatch, source, "invalid_factory",
        "create_authoritative_backend = 42\n"
        "def register(ctx):\n    raise AssertionError('legacy registration must not run')\n",
    )
    store = _Store()
    with pytest.raises(MemoryConfigurationError, match="create_authoritative_backend.*callable"):
        select_memory_service(_authoritative_config("invalid_factory"), store_factory=store.factory)
    assert store.constructed == 0


@pytest.mark.parametrize("source", _PLUGIN_SOURCES)
def test_plugin_import_failure_is_reported_distinctly_from_missing_factory(tmp_path, monkeypatch, source):
    """M12: an import failure in the provider module (e.g. a syntax or
    top-level exception) must surface as the import failure, not as the
    generic 'does not export create_authoritative_backend' message -- that
    message means something different: the module imported fine but simply
    lacks the symbol."""
    _install_plugin(
        tmp_path, monkeypatch, source, "broken_factory",
        "raise ValueError('boom')\n"
        "def create_authoritative_backend(config):\n    return None\n"
        "def register(ctx):\n    pass\n",
    )
    store = _Store()
    with pytest.raises(MemoryConfigurationError, match="failed to import.*boom") as exc:
        select_memory_service(_authoritative_config("broken_factory"), store_factory=store.factory)
    assert isinstance(exc.value.__cause__, ValueError)
    assert store.constructed == 0
