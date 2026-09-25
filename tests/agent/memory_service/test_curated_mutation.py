"""R38: the curated mutation flow over MemoryService (§9.4, §9.5, §6.3)."""

from types import SimpleNamespace

from agent.memory_service.config import resolve_memory_service_config
from agent.memory_service.service import MemoryDisposition, StatelessMemoryService, is_provider_managed


def test_provider_managed_means_authoritative_or_stateless():
    assert is_provider_managed(SimpleNamespace(disposition=MemoryDisposition.AUTHORITATIVE))
    assert is_provider_managed(StatelessMemoryService(resolve_memory_service_config({}), reason="test"))
    assert is_provider_managed(SimpleNamespace(disposition="provider_authoritative"))


def test_missing_or_builtin_disposition_is_additive():
    """R37's reader rule: a missing service or disposition is additive."""
    assert not is_provider_managed(SimpleNamespace(disposition=MemoryDisposition.BUILTIN))
    assert not is_provider_managed(None)
    assert not is_provider_managed(SimpleNamespace())
