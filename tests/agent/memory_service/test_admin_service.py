"""S6 inventory suite B part 1 (§13.1 L2211; §14.1 row 42 L2312): explicit administrative identity (§9.2 L968, §4.2 L271)."""

import os
import types

import pytest

from agent.memory_service import wire as w
from agent.memory_service.admin import (ADMIN_PLATFORM, AdminContext, AdminIdentityError, admin_service,
                                        admin_state_path, authority_report, format_scope, outcome_payload,
                                        parse_scope_selector, plan_reset)
from agent.memory_service.archive import archive_disposition
from agent.memory_service.config import MemoryConfigurationError, resolve_memory_service_config
from agent.memory_service.errors import MemoryBlockedError
from agent.memory_service.mutation import (MutationOutcome, MutationStatus, PlannedMutation, PlanShortCircuit,
                                           predict_approval_requirements)
from tests.agent.memory_service.fake_backend import FakeAuthoritativeBackend, FakeProviderStore, FakeRegistry

REPO_CTX = AdminContext(repo_id="repo-1", project_id="proj-1")


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


# -- the administrative service (ruling R42-2) ----------------------------------------------------------


@pytest.fixture
def env(exe):
    store = FakeProviderStore(registry=FakeRegistry())            # C:\work\repo -> (repo-1, proj-1, None)
    backends = []

    def factory(cfg):
        backends.append(FakeAuthoritativeBackend(store, provider="example"))
        return backends[-1]
    return types.SimpleNamespace(store=store, backends=backends, factory=factory, cfg=_cfg(exe))


def _count(env, op):
    return sum(b.count(op) for b in env.backends)


def test_identity_is_explicit_and_never_the_process_directory(env, tmp_path, monkeypatch):
    here = tmp_path / "work"
    here.mkdir()
    monkeypatch.chdir(here)
    env.store.registry.directories[os.path.realpath(str(here))] = ("repo-9", "proj-9", None)
    with admin_service(env.cfg, context=AdminContext(), backend_factory=env.factory) as service:
        identity = service.identity
        assert (identity.repo_id, identity.project_id, identity.platform) == (None, None, ADMIN_PLATFORM)
        assert service.load_curated("memory").status == "degraded_global_only"
    with admin_service(env.cfg, context=REPO_CTX, backend_factory=env.factory) as service:
        assert service.identity.repo_id == "repo-1"


def test_first_open_binds_and_persists_then_resumes(env):
    key = REPO_CTX.record_key(resolve_memory_service_config(env.cfg))
    with admin_service(env.cfg, context=REPO_CTX, backend_factory=env.factory) as service:
        handle = service.identity.opaque_binding_b64url
    assert _count(env, "bind_session") == 1 and admin_state_path(key).is_file()
    with admin_service(env.cfg, context=REPO_CTX, backend_factory=env.factory) as service:
        assert service.identity.opaque_binding_b64url == handle
    assert _count(env, "bind_session") == 1 and _count(env, "validate_session") == 1


@pytest.mark.parametrize("refuse", ["revoke", "epoch"])
def test_a_refused_binding_is_rebound(env, refuse):
    with admin_service(env.cfg, context=REPO_CTX, backend_factory=env.factory) as service:
        handle = service.identity.opaque_binding_b64url
    env.store.revoke(handle) if refuse == "revoke" else env.store.set_epoch("ep-2")
    with admin_service(env.cfg, context=REPO_CTX, backend_factory=env.factory) as service:
        assert service.identity.opaque_binding_b64url != handle
    assert _count(env, "bind_session") == 2


def test_a_transport_failure_is_reported_and_keeps_the_record(env):
    key = REPO_CTX.record_key(resolve_memory_service_config(env.cfg))
    with admin_service(env.cfg, context=REPO_CTX, backend_factory=env.factory):
        pass
    before = admin_state_path(key).read_bytes()
    env.store.fail_transport("validate_session")
    with pytest.raises(MemoryBlockedError):
        with admin_service(env.cfg, context=REPO_CTX, backend_factory=env.factory):
            pass
    assert _count(env, "bind_session") == 1 and admin_state_path(key).read_bytes() == before


def test_admin_state_lives_under_the_archive_excluded_root(env):
    from hermes_cli import backup_memory
    from hermes_constants import get_hermes_home
    key = REPO_CTX.record_key(resolve_memory_service_config(env.cfg))
    parts = admin_state_path(key).relative_to(get_hermes_home()).parts
    assert parts[0] == backup_memory.HOST_STATE_DIRNAME and parts[1] == "admin"   # X-1 relationship


def test_additive_or_invalid_configuration_never_reaches_a_provider(env):
    for cfg in ({}, {"memory": {"provider": "example", "provider_mode": "authoritative"}}):
        with pytest.raises(MemoryConfigurationError):
            with admin_service(cfg, context=AdminContext(), backend_factory=env.factory):
                pass
    assert env.backends == []


# -- plan_reset (ruling R42-9) --------------------------------------------------------------------------

REPO, PROJ, USER = (w.ScopeRef(kind="repository", id="repo-1"), w.ScopeRef(kind="project", id="proj-1"),
                    w.ScopeRef(kind="principal_global", id="ethan"))


def test_reset_plans_exactly_the_requested_eligible_scopes(env):
    with admin_service(env.cfg, context=REPO_CTX, backend_factory=env.factory) as service:
        memory, user = service.load_curated("memory"), service.load_curated("user")
    plan = plan_reset(memory, [REPO, REPO])
    assert isinstance(plan, PlannedMutation) and plan.intent.kind == "reset"
    assert plan.intent.reset_scopes == (REPO,) and plan.requested_write_scopes == (REPO,)
    assert plan.mutation_delta == () and plan.candidate_entries == ()
    assert predict_approval_requirements(memory, plan) == ("reset",)
    assert predict_approval_requirements(memory, plan_reset(memory, [PROJ])) == ("non_default_scope", "reset")
    assert predict_approval_requirements(user, plan_reset(user, [USER])) == ("target_user", "reset")


def test_reset_refuses_missing_or_ineligible_scopes(env):
    with admin_service(env.cfg, context=REPO_CTX, backend_factory=env.factory) as service:
        memory = service.load_curated("memory")
    empty, foreign = plan_reset(memory, []), plan_reset(memory, [USER])
    assert isinstance(empty, PlanShortCircuit) and empty.response["code"] == "invalid_request"
    assert isinstance(foreign, PlanShortCircuit) and foreign.response["code"] == "unauthorized_scope"
    assert foreign.response["scopes"] == ["global:ethan"]
