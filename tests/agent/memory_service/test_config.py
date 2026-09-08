"""§9.1 configuration contract: every rule, and no reinterpretation as additive."""

import pytest

from agent.memory_service.config import (
    FailurePolicy,
    MemoryConfigurationError,
    MemoryMode,
    MemoryServiceConfig,
    resolve_memory_service_config,
)


def _cfg(**memory):
    return {"memory": memory}


def test_missing_section_is_additive_with_both_targets_enabled():
    cfg = resolve_memory_service_config({})
    assert cfg.provider_mode is MemoryMode.ADDITIVE
    assert cfg.failure_policy is FailurePolicy.FAIL_CLOSED
    assert cfg.provider == ""
    assert cfg.provider_executable is None
    assert cfg.memory_enabled and cfg.user_profile_enabled
    assert resolve_memory_service_config(None).provider_mode is MemoryMode.ADDITIVE


def test_absent_provider_mode_is_additive_even_with_a_provider():
    cfg = resolve_memory_service_config(_cfg(provider="honcho"))
    assert cfg.provider_mode is MemoryMode.ADDITIVE
    assert cfg.provider == "honcho"


def test_empty_provider_mode_string_is_additive():
    assert resolve_memory_service_config(_cfg(provider_mode="")).provider_mode is MemoryMode.ADDITIVE


@pytest.mark.parametrize("memory", ["authoritative", [1, 2]])
def test_non_mapping_memory_value_is_a_configuration_error(memory):
    """M6: a scalar or sequence `memory:` value must not be silently treated
    as an absent (and therefore additive) section -- that would let a typo
    like `memory: authoritative` downgrade authority without any error."""
    with pytest.raises(MemoryConfigurationError, match="memory"):
        resolve_memory_service_config({"memory": memory})


def test_null_or_missing_memory_key_is_still_additive():
    assert resolve_memory_service_config({"memory": None}).provider_mode is MemoryMode.ADDITIVE
    assert resolve_memory_service_config({}).provider_mode is MemoryMode.ADDITIVE


def test_authoritative_requires_provider_and_absolute_existing_executable(tmp_path):
    exe = tmp_path / "provider.exe"
    exe.write_bytes(b"MZ")
    cfg = resolve_memory_service_config(
        _cfg(provider="example", provider_mode="authoritative", provider_executable=str(exe))
    )
    assert cfg.provider_mode is MemoryMode.AUTHORITATIVE
    assert cfg.provider == "example"
    assert cfg.provider_executable == str(exe)
    assert cfg.failure_policy is FailurePolicy.FAIL_CLOSED


@pytest.mark.parametrize(
    "memory, fragment",
    [
        ({"provider_mode": "authoritative"}, "memory.provider"),
        ({"provider": "example", "provider_mode": "authoritative"}, "provider_executable"),
        ({"provider": "example", "provider_mode": "authoritative", "provider_executable": ""}, "provider_executable"),
        ({"provider": "example", "provider_mode": "authoritative", "provider_executable": "provider.exe"}, "absolute"),
        ({"provider": "example", "provider_mode": "authoritative", "provider_executable": "bin/provider.exe"}, "absolute"),
        ({"provider_mode": "sidecar"}, "provider_mode"),
        ({"provider_mode": "Authoritative"}, "provider_mode"),
        ({"provider_mode": True}, "provider_mode"),
        ({"authoritative_failure_policy": "open"}, "authoritative_failure_policy"),
        ({"authoritative_failure_policy": "stateless"}, "stateless"),
        ({"provider": "x", "authoritative_failure_policy": "stateless"}, "stateless"),
    ],
)
def test_configuration_errors_are_never_additive(memory, fragment):
    with pytest.raises(MemoryConfigurationError) as exc:
        resolve_memory_service_config(_cfg(**memory))
    assert fragment in str(exc.value)


def test_missing_executable_file_is_a_configuration_error(tmp_path):
    missing = tmp_path / "nope.exe"
    with pytest.raises(MemoryConfigurationError) as exc:
        resolve_memory_service_config(
            _cfg(provider="example", provider_mode="authoritative", provider_executable=str(missing))
        )
    assert "existing" in str(exc.value)


def test_directory_as_executable_is_a_configuration_error(tmp_path):
    with pytest.raises(MemoryConfigurationError):
        resolve_memory_service_config(
            _cfg(provider="example", provider_mode="authoritative", provider_executable=str(tmp_path))
        )


@pytest.mark.parametrize("key", ["provider_args", "provider_arguments", "provider_executable_args"])
def test_extra_arguments_are_a_configuration_error(tmp_path, key):
    exe = tmp_path / "provider.exe"
    exe.write_bytes(b"MZ")
    memory = {"provider": "example", "provider_mode": "authoritative", "provider_executable": str(exe), key: ["--fast"]}
    with pytest.raises(MemoryConfigurationError) as exc:
        resolve_memory_service_config(_cfg(**memory))
    assert "argument" in str(exc.value)


def test_stateless_policy_is_valid_only_with_authoritative(tmp_path):
    exe = tmp_path / "provider.exe"
    exe.write_bytes(b"MZ")
    cfg = resolve_memory_service_config(
        _cfg(
            provider="example",
            provider_mode="authoritative",
            provider_executable=str(exe),
            authoritative_failure_policy="stateless",
        )
    )
    assert cfg.failure_policy is FailurePolicy.STATELESS


def test_explicit_fail_closed_is_accepted_in_additive_mode():
    cfg = resolve_memory_service_config(_cfg(authoritative_failure_policy="fail_closed"))
    assert cfg.provider_mode is MemoryMode.ADDITIVE
    assert cfg.failure_policy is FailurePolicy.FAIL_CLOSED


def test_target_flags_are_independent():
    cfg = resolve_memory_service_config(_cfg(memory_enabled=False, user_profile_enabled=True))
    assert not cfg.target_enabled("memory")
    assert cfg.target_enabled("user")
    cfg = resolve_memory_service_config(_cfg(memory_enabled="false", user_profile_enabled="0"))
    assert not cfg.target_enabled("memory") and not cfg.target_enabled("user")
    with pytest.raises(ValueError):
        cfg.target_enabled("profile")


def test_config_is_frozen():
    cfg = resolve_memory_service_config({})
    with pytest.raises(Exception):
        cfg.provider = "x"  # type: ignore[misc]
    assert isinstance(cfg, MemoryServiceConfig)
