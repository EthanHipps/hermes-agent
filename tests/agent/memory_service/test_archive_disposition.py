"""S6 archive suite A: the generic curated_memory disposition (§9.8 L1627-1637, §12.1 L2044)."""

import pytest
import yaml

from agent.memory_service.archive import ArchiveDisposition, archive_disposition
from agent.memory_service.config import PROVIDER_API_VERSION

SPEC_BLOCK = {"authority": "provider", "provider": "ygg", "provider_api": 1, "included": False,
              "disposition": "provider-managed", "restore_action": "reconnect-provider"}


def _cfg(**memory):
    return {"memory": memory}


def test_record_is_the_spec_block_for_the_configured_provider():
    d = archive_disposition(_cfg(provider="ygg", provider_mode="authoritative"))
    assert yaml.safe_load(d.record_text()) == {"curated_memory": SPEC_BLOCK}


def test_provider_is_the_configured_name_not_a_constant():
    d = archive_disposition(_cfg(provider="example", provider_mode="authoritative"))
    assert d.as_mapping() == {**SPEC_BLOCK, "provider": "example"}


def test_provider_api_is_the_version_hermes_requires():
    d = archive_disposition(_cfg(provider="example", provider_mode="authoritative"))
    assert d.provider_api == PROVIDER_API_VERSION


def test_success_message_is_the_spec_sentence():
    d = archive_disposition(_cfg(provider="ygg", provider_mode="authoritative"))
    assert d.archive_message() == (
        "Hermes archive complete; authoritative ygg memory is provider-managed and not included")


def test_incomplete_archive_says_so_and_still_claims_no_coverage():
    msg = archive_disposition(_cfg(provider="ygg", provider_mode="authoritative")).archive_message(complete=False)
    assert msg.startswith("Hermes archive incomplete;") and msg.endswith("not included")


@pytest.mark.parametrize("memory", [{}, {"provider": "honcho"}, {"provider": "ygg", "provider_mode": "additive"}])
def test_no_disposition_unless_authoritative_is_requested(memory):
    assert archive_disposition({"memory": memory}) is None


@pytest.mark.parametrize("raw", [None, "text", {"memory": "broken"}, {"memory": None}, {"memory": {"provider_mode": 7}}])
def test_malformed_config_never_raises(raw):
    assert archive_disposition(raw) is None


def test_stateless_policy_declares_the_same_disposition():  # §9.6 L1584, last column
    base = _cfg(provider="ygg", provider_mode="authoritative")
    stateless = _cfg(provider="ygg", provider_mode="authoritative", authoritative_failure_policy="stateless")
    assert archive_disposition(base) == archive_disposition(stateless)


def test_requested_but_invalid_authoritative_config_still_declares_it():  # ruling R44-3
    d = archive_disposition(_cfg(provider_mode="authoritative"))
    assert d is not None and d.as_mapping()["included"] is False
    assert "authoritative memory is provider-managed" in d.archive_message()


def test_provider_names_yaml_would_coerce_stay_strings():
    d = archive_disposition(_cfg(provider="yes", provider_mode="authoritative"))
    assert yaml.safe_load(d.record_text())["curated_memory"]["provider"] == "yes"


def test_restore_message_names_the_reconnect_step_and_never_claims_restoration():
    msg = archive_disposition(_cfg(provider="ygg", provider_mode="authoritative")).restore_message()
    assert "reconnect-provider" in msg and "was not restored" in msg


def test_disposition_type_is_importable_and_frozen():
    d = ArchiveDisposition(provider="example")
    with pytest.raises(Exception):
        d.provider = "other"  # type: ignore[misc]
