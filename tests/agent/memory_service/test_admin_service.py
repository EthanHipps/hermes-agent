"""S6 inventory suite B part 1 (§13.1 L2211; §14.1 row 42 L2312): explicit administrative identity (§9.2 L968, §4.2 L271)."""

import pytest

from agent.memory_service.admin import (AdminContext, AdminIdentityError, authority_report, format_scope,
                                        outcome_payload, parse_scope_selector)
from agent.memory_service.archive import archive_disposition
from agent.memory_service.mutation import MutationOutcome, MutationStatus


def _cfg(exe, **extra):
    return {"memory": {"provider": "example", "provider_mode": "authoritative",
                       "provider_executable": str(exe), "principal_id": "ethan", **extra}}


@pytest.fixture
def exe(tmp_path):
    p = tmp_path / "provider.exe"
    p.write_bytes(b"MZ")
    return p


@pytest.mark.parametrize("text", ["global:ethan", "organization:org-1", "project:proj-1", "repository:repo-1"])
def test_scope_selectors_round_trip(text):
    assert format_scope(parse_scope_selector(text)) == text


@pytest.mark.parametrize("bad", ["", "repo:x", "repository:", "repository:a\x07b", "nocolon"])
def test_malformed_selectors_are_refused(bad):
    with pytest.raises(AdminIdentityError):
        parse_scope_selector(bad)


def test_context_fields_are_explicit_identifiers():
    assert AdminContext.from_fields(" ", None, "") == AdminContext()
    assert AdminContext.from_fields(project_id=" proj-1 ").project_id == "proj-1"
    with pytest.raises(AdminIdentityError):
        AdminContext.from_fields(repo_id="a\nb")


def test_authority_report_is_the_archive_disposition_plus_mode(exe):
    assert authority_report({}) is None and authority_report({"memory": {"provider": "x"}}) is None
    cfg = _cfg(exe, authoritative_failure_policy="stateless", user_profile_enabled=False)
    report = authority_report(cfg)
    mapping = archive_disposition(cfg).as_mapping()
    assert {k: report[k] for k in mapping} == mapping                  # relationship with C8, not a snapshot
    assert (report["provider_mode"], report["failure_policy"], report["configuration_error"]) == \
           ("authoritative", "stateless", None)
    assert report["targets"] == {"memory": True, "user": False}


def test_authority_report_names_a_configuration_error(exe):
    report = authority_report({"memory": {"provider": "example", "provider_mode": "authoritative"}})
    assert report["configuration_error"] and report["failure_policy"] is None and report["targets"] is None


def test_outcome_payloads_are_typed_and_never_claim_unknown_is_unchanged():
    unknown = outcome_payload(MutationOutcome(MutationStatus.OUTCOME_UNKNOWN, "memory", 1, error_code="outcome_unknown"))
    assert unknown["ok"] is False and unknown["code"] == "outcome_unknown"
    assert "nothing was changed" not in unknown["message"].lower()          # X-4 / correction R-5
    gated = outcome_payload(MutationOutcome(MutationStatus.APPROVAL_UNAVAILABLE, "user", 1,
                                            approval_requirements=("target_user", "reset")))
    assert gated["code"] == "approval_unavailable" and gated["approval_required"] == ["target_user", "reset"]


def test_every_mutation_status_has_a_typed_payload():
    """Relationship, not a snapshot: members added to C6b-2 later still map (R39 adds two). Correction C68-5."""
    from agent.memory_service.admin import _OUTCOME_PAYLOADS
    for status in set(MutationStatus) - set(_OUTCOME_PAYLOADS):
        payload = outcome_payload(MutationOutcome(status, "memory", 1))
        assert payload["ok"] is False and payload["code"] == status.value
