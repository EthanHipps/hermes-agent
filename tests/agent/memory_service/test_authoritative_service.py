"""Authoritative disposition over the stub transport: §9.1 selection, §9.2 frozen identity, §9.6 fail-closed."""

import json
import logging
from dataclasses import replace

import pytest

from agent.memory_service import wire as w
from agent.memory_service.authoritative import ProviderAuthoritativeMemoryService
from agent.memory_service.config import MemoryConfigurationError, resolve_memory_service_config
from agent.memory_service.errors import BindingInvalidError, CapabilityUnavailableError, MemoryBlockedError, ProviderError, ProviderTransportError, TargetDisabledError
from agent.memory_service.identity import FrozenMemoryIdentity, HostSessionState
from agent.memory_service.service import CommitIntent, ContinuityCapture, InspectRequest, MemoryDisposition, MutationRequest, RecallQuery, ServiceCapabilities, StatelessMemoryService, select_memory_service

from tests.agent.memory_service.stub_backend import PG, REPO, StubBackend

PROVENANCE = w.MutationProvenance(actor_kind="hermes", principal_id="ethan", logical_session_id="sess-1", initiating_surface="memory_tool", source_entry_ids=(), source_commit=None, threat_decision_id=None)


def _never_built():
    raise AssertionError("native store built")


def _config(tmp_path, **memory):
    exe = tmp_path / "provider.exe"
    exe.write_bytes(b"MZ")
    base = {"provider": "example", "provider_mode": "authoritative", "provider_executable": str(exe)}
    base.update(memory)
    return {"memory": base}


def _context(session="sess-1"):
    return w.RequestedContext(principal_id="ethan", profile_id="default", logical_session_id=session, platform="cli", org_id=None, project_id=None, repo_id=None, workspace_id=None, resolution_source="directory", canonical_directory="C:\\work\\repo")


def _select(tmp_path, backend, **memory):
    return select_memory_service(_config(tmp_path, **memory), store_factory=_never_built, requested_context=_context(), backend_factory=lambda cfg: backend)


def _mutation(snapshot, request_id="r1", text="x"):
    scope = snapshot.default_write_scope
    return MutationRequest(target="memory", request_id=request_id, expected_revision=snapshot.revision, hidden_preservation_state=snapshot.hidden_preservation_state, requested_write_scopes=(scope,), intent=w.MutationIntent(kind="add"), mutation_delta=(w.MutationDeltaItem(action="add", client_ref="c1"),), candidate_entries=(w.CandidateEntry(client_ref="c1", text=text, destination_scope=scope, target="memory", proposed_policy_key=None, import_source_identity=None),), provenance=PROVENANCE)


def test_start_negotiates_binds_and_freezes_identity(tmp_path):
    backend = StubBackend()
    service = _select(tmp_path, backend)
    assert isinstance(service, ProviderAuthoritativeMemoryService)
    assert service.disposition is MemoryDisposition.AUTHORITATIVE
    # negotiate, bind, then select_memory_service's startup load probe (one per
    # enabled target) that fixes the disposition before the first model request.
    calls = [op for op, _ in backend.calls]
    assert calls[:2] == ["negotiate", "bind_session"]
    assert set(calls[2:]) == {"load_curated"}
    neg = backend.calls[0][1]
    assert neg.host == "hermes" and list(neg.supported_api_versions) == [1]
    assert list(neg.required_operations) == ["bind_session", "validate_session", "load_curated", "stage_curated", "inspect_staged", "commit_curated"]
    bind = backend.calls[1][1]
    assert bind.bind_intent == "new_session" and bind.prior_identity is None and bind.expected_provider_epoch == "ep-1"
    ident = service.identity
    assert ident.provider == "example" and ident.provider_mode == "authoritative" and ident.logical_session_id == "sess-1"
    assert service.session_state == HostSessionState(provider_epoch="ep-1", identity=ident)
    assert service.capabilities.recall_context is False
    assert service.load_curated("memory").frozen_identity == ident.to_wire()
    assert service.identity is ident  # frozen: same object after a load


def test_incompatible_api_and_missing_operation_are_configuration_errors(tmp_path):
    with pytest.raises(MemoryConfigurationError, match="API version"):
        _select(tmp_path, StubBackend(api_version=2))
    with pytest.raises(MemoryConfigurationError, match="commit_curated"):
        _select(tmp_path, StubBackend(operations=["bind_session", "validate_session", "load_curated", "stage_curated", "inspect_staged"]))
    with pytest.raises(MemoryConfigurationError, match="provider"):
        _select(tmp_path, StubBackend(provider="other"))
    # configuration errors are errors under the stateless policy too
    with pytest.raises(MemoryConfigurationError):
        _select(tmp_path, StubBackend(api_version=2), authoritative_failure_policy="stateless")


def test_capability_flag_without_its_operation_is_a_configuration_error(tmp_path):
    """M13: a provider negotiating capabilities.recall_context: true without
    listing recall_context in operations contradicts itself; ServiceCapabilities
    must not simply follow the flag -- a provider that contradicts itself is
    not a provider to trust."""
    contradictory = StubBackend(
        operations=["bind_session", "validate_session", "load_curated", "stage_curated", "inspect_staged", "commit_curated"],
        recall=True,
    )
    with pytest.raises(MemoryConfigurationError, match="recall_context"):
        _select(tmp_path, contradictory)

    consistent = StubBackend(recall=True)
    service = _select(tmp_path, consistent)
    assert service.capabilities.recall_context is True


def test_bind_failure_fails_closed_or_starts_stateless(tmp_path, caplog):
    backend = StubBackend()
    backend.fail_transport("bind_session")
    with pytest.raises(MemoryBlockedError):
        _select(tmp_path, backend)
    assert backend.shutdown_calls == 1
    backend = StubBackend()
    backend.fail_typed("bind_session", "unavailable")
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="agent.memory_service.service"):
        service = _select(tmp_path, backend, authoritative_failure_policy="stateless")
    # M8: the stateless-start transition emits exactly one record.
    assert len(caplog.records) == 1
    assert isinstance(service, StatelessMemoryService)
    assert "unavailable" in service.degraded_warning()
    assert backend.count("load_curated") == 0


def test_first_load_failure_starts_stateless_or_fails_closed(tmp_path):
    """§9.6: the disposition is fixed before the first model request, so a
    failing FIRST load is handled exactly like a bind failure -- stateless
    under the stateless policy, blocked under fail_closed -- never a
    MemoryBlockedError raised later under both policies."""
    backend = StubBackend()
    backend.fail_transport("load_curated")
    service = _select(tmp_path, backend, authoritative_failure_policy="stateless")
    assert isinstance(service, StatelessMemoryService)
    assert backend.shutdown_calls == 1
    assert backend.count("load_curated") == 1
    assert "stateless" in service.degraded_warning().lower()

    closed = StubBackend()
    closed.fail_transport("load_curated")
    with pytest.raises(MemoryBlockedError):
        _select(tmp_path, closed)
    assert closed.shutdown_calls == 1


def test_configuration_error_at_negotiation_shuts_the_backend_down(tmp_path):
    backend = StubBackend(api_version=2)
    with pytest.raises(MemoryConfigurationError, match="API version"):
        _select(tmp_path, backend)
    assert backend.shutdown_calls == 1


def test_stub_round_trips_requests_and_results_through_the_codec(tmp_path):
    """The stub is an honest transport: the service's own requests must be
    wire-legal, and every result the service sees is a strictly decoded type."""
    backend = StubBackend()
    service = _select(tmp_path, backend)
    snap = service.load_curated("memory")
    staged = service.stage_curated(_mutation(snap))
    assert type(staged) is w.StageResult  # decoded by the stub, not the same object it built
    assert staged is not backend._stages[staged.stage_handle_b64url][1]
    assert staged == backend._stages[staged.stage_handle_b64url][1]
    commits_before = backend.count("commit_curated")
    with pytest.raises(w.WireError, match="stage_handle_b64url"):
        service.commit_curated(CommitIntent("memory", "r1", "not/a valid+handle", "c" * 64, (REPO,), w.ApprovalAuthorization(kind="not_required")))
    assert backend.count("commit_curated") == commits_before  # refused locally, never sent


def test_every_load_is_fresh_and_disabled_targets_are_never_loaded(tmp_path):
    backend = StubBackend()
    service = _select(tmp_path, backend, user_profile_enabled=False)
    # select_memory_service probes the enabled target once at startup; the
    # disabled one is never probed, so counts below are measured as deltas.
    startup = backend.count("load_curated")
    assert startup == 1
    assert [r.target for op, r in backend.calls if op == "load_curated"] == ["memory"]
    service.load_curated("memory")
    service.load_curated("memory")
    assert backend.count("load_curated") - startup == 2
    with pytest.raises(TargetDisabledError):
        service.load_curated("user")
    with pytest.raises(TargetDisabledError):
        service.prompt_block("user")
    assert backend.count("load_curated") - startup == 2
    assert service.prompt_block("memory") is None  # empty store renders nothing
    backend.entries["memory"].append("Prefer uv")
    block = service.prompt_block("memory")
    assert block and "Prefer uv" in block and backend.count("load_curated") - startup == 4


def test_malformed_snapshot_fails_closed(tmp_path):
    for kind in ("identity", "target", "wire"):
        backend = StubBackend()
        service = _select(tmp_path, backend)
        backend.malformed_snapshot = kind
        with pytest.raises(MemoryBlockedError):
            service.load_curated("memory")
        assert service.blocked
        backend.malformed_snapshot = None
        assert service.load_curated("memory").target == "memory"
        assert not service.blocked


def test_blocked_load_exposes_typed_code_and_provider_error(tmp_path):
    """M5: MemoryBlockedError carries its typed code and the underlying
    ProviderError as attributes, so a caller (row 40) can distinguish
    ambiguous_policy from unavailable without parsing str(err) or __cause__."""
    backend = StubBackend()
    service = _select(tmp_path, backend)
    backend.fail_typed("load_curated", "ambiguous_policy", details={"ambiguities": [{"policy_key": "k", "tier": "dependency", "candidate_count": 2}]})
    with pytest.raises(MemoryBlockedError) as excinfo:
        service.load_curated("memory")
    err = excinfo.value
    assert err.code == "ambiguous_policy"
    assert isinstance(err.provider_error, ProviderError)
    assert err.provider_error.code == "ambiguous_policy"

    backend2 = StubBackend()
    service2 = _select(tmp_path, backend2)
    backend2.fail_typed("load_curated", "unavailable")
    with pytest.raises(MemoryBlockedError) as excinfo2:
        service2.load_curated("memory")
    assert excinfo2.value.code == "unavailable"


def test_failure_after_success_blocks_until_a_fresh_load(tmp_path, caplog):
    backend = StubBackend()
    service = _select(tmp_path, backend)
    snap = service.load_curated("memory")
    backend.fail_transport("load_curated")
    with caplog.at_level(logging.WARNING, logger="agent.memory_service.authoritative"):
        with pytest.raises(MemoryBlockedError):
            service.load_curated("memory")
    # M8: the blocked transition is exactly what an operator needs when a
    # session fails closed -- one record, no more, no payload asserted.
    assert len(caplog.records) == 1
    assert service.blocked
    with pytest.raises(MemoryBlockedError):
        service.stage_curated(_mutation(snap))
    with pytest.raises(MemoryBlockedError):
        service.commit_curated(CommitIntent("memory", "r1", "AAAA", "c" * 64, (REPO,), w.ApprovalAuthorization(kind="not_required")))
    assert backend.count("stage_curated") == 0
    service.load_curated("memory")
    assert not service.blocked
    staged = service.stage_curated(_mutation(snap))
    assert staged.request_id == "r1"
    wire_stage = backend.calls[-1][1]
    assert wire_stage.expected_provider_epoch == "ep-1" and wire_stage.frozen_identity == service.identity.to_wire()


def test_epoch_change_invalidates_state_until_rebind(tmp_path, caplog):
    backend = StubBackend()
    service = _select(tmp_path, backend)
    backend.epoch = "ep-2"
    with caplog.at_level(logging.WARNING, logger="agent.memory_service.authoritative"):
        with pytest.raises(MemoryBlockedError, match="epoch"):
            service.load_curated("memory")
    # M8: the epoch-changed transition emits exactly one record.
    assert len(caplog.records) == 1
    assert service.blocked and service.epoch_changed
    backend.epoch = "ep-1"
    with pytest.raises(MemoryBlockedError, match="rebind"):
        service.load_curated("memory")
    backend2 = StubBackend()
    service2 = _select(tmp_path, backend2)
    # scripted after selection: a typed provider_epoch_changed during the
    # startup probe is a select-time failure (covered by the first-load test).
    backend2.fail_typed("load_curated", "provider_epoch_changed", details={"expected_provider_epoch": "ep-1", "current_provider_epoch": "ep-9"})
    with pytest.raises(MemoryBlockedError, match="epoch"):
        service2.load_curated("memory")
    assert service2.epoch_changed


def test_typed_mutation_errors_propagate_and_transport_uncertainty_is_unknown(tmp_path):
    backend = StubBackend()
    service = _select(tmp_path, backend)
    snap = service.load_curated("memory")
    backend.fail_typed("stage_curated", "store_blocked", outcome="not_committed", details={"reason": "git_dirty"})
    with pytest.raises(ProviderError) as exc:
        service.stage_curated(_mutation(snap))
    assert exc.value.code == "store_blocked" and exc.value.details == {"reason": "git_dirty"}
    assert not service.blocked  # reads keep working; mutations are rejected per call
    backend.fail_typed("stage_curated", "version_conflict", outcome="not_committed", details={"current_snapshot": snap.to_wire()})
    with pytest.raises(ProviderError) as exc:
        service.stage_curated(_mutation(snap, "r2"))
    assert exc.value.code == "version_conflict"
    backend.fail_transport("commit_curated")
    with pytest.raises(ProviderTransportError) as exc:
        service.commit_curated(CommitIntent("memory", "r1", "AAAA", "c" * 64, (REPO,), w.ApprovalAuthorization(kind="not_required")))
    assert exc.value.mutation_outcome_unknown
    assert service.blocked
    service.load_curated("memory")
    committed = service.commit_curated(CommitIntent("memory", "r1", "AAAA", "c" * 64, (REPO,), w.ApprovalAuthorization(kind="not_required")))
    assert committed.outcome == "committed_audit_clean" and committed.snapshot.mutation_entries[-1].text == "committed:r1"


def test_optional_capabilities_follow_negotiation(tmp_path):
    service = _select(tmp_path, StubBackend())
    snap = service.load_curated("memory")
    with pytest.raises(CapabilityUnavailableError):
        service.recall_context(RecallQuery("memory", snap.revision, "q", ("general", "hermes_memory"), (), w.RecallBudget(0, 3000, 20)))
    with pytest.raises(CapabilityUnavailableError):
        service.capture_continuity(ContinuityCapture("cr1", "compression_snapshot", "text", "compression"))
    backend = StubBackend(recall=True, continuity=True)
    service = _select(tmp_path, backend)
    assert service.capabilities == ServiceCapabilities(recall_context=True, capture_continuity=True)
    snap = service.load_curated("memory")
    recall = service.recall_context(RecallQuery("memory", snap.revision, "q", ("general", "hermes_memory"), (), w.RecallBudget(0, 3000, 20)))
    assert recall.target == "memory" and backend.calls[-1][1].exclude_entry_ids == ()
    cont = service.capture_continuity(ContinuityCapture("cr1", "compression_snapshot", "text", "compression"))
    assert cont.outcome == "stored"


def test_resume_uses_validate_never_bind(tmp_path):
    backend = StubBackend()
    first = _select(tmp_path, backend)
    state = first.session_state
    backend2 = StubBackend()
    resumed = select_memory_service(_config(tmp_path), store_factory=_never_built, session_state=state, backend_factory=lambda cfg: backend2)
    # resume never binds; the trailing calls are the startup load probe.
    calls = [op for op, _ in backend2.calls]
    assert calls[:2] == ["negotiate", "validate_session"]
    assert set(calls[2:]) == {"load_curated"}
    assert resumed.identity == first.identity
    assert backend2.calls[1][1].expected_provider_epoch == "ep-1"
    backend3 = StubBackend(epoch="ep-2")
    with pytest.raises(MemoryBlockedError, match="epoch"):
        select_memory_service(_config(tmp_path), store_factory=_never_built, session_state=state, backend_factory=lambda cfg: backend3)
    with pytest.raises(BindingInvalidError):
        HostSessionState.from_dict({"provider_epoch": "ep-1"})


def test_resume_with_non_authoritative_identity_raises_a_typed_error(tmp_path):
    """M14: resume() must never let a bare ValueError escape from
    FrozenMemoryIdentity.to_wire() when a hand-built HostSessionState carries
    a non-authoritative identity -- select_memory_service does not catch a
    bare ValueError, so it would otherwise reach the caller unhandled."""
    backend = StubBackend()
    cfg = resolve_memory_service_config(_config(tmp_path))
    service = ProviderAuthoritativeMemoryService(cfg, backend)
    bad_identity = FrozenMemoryIdentity(
        provider="builtin", provider_mode="additive", principal_id="ethan", profile_id="default",
        logical_session_id="sess-1", org_id=None, project_id=None, repo_id=None, workspace_id=None,
        platform="cli", binding_revision="rev-1", opaque_binding_b64url="",
    )
    state = HostSessionState(provider_epoch="ep-1", identity=bad_identity)
    with pytest.raises(BindingInvalidError):
        service.resume(state)


def test_shutdown_reaches_the_backend(tmp_path):
    backend = StubBackend()
    service = _select(tmp_path, backend)
    service.shutdown()
    assert backend.shutdown_calls == 1


def test_revoked_binding_is_latched_until_rebind(tmp_path, caplog):
    backend = StubBackend()
    service = _select(tmp_path, backend)
    snap = service.load_curated("memory")
    old_identity = service.identity
    backend.fail_typed("stage_curated", "binding_revoked")
    with caplog.at_level(logging.WARNING, logger="agent.memory_service.authoritative"):
        with pytest.raises(MemoryBlockedError):
            service.stage_curated(_mutation(snap))
    # M8: the binding-lost transition emits exactly one record.
    assert len(caplog.records) == 1
    # a later successful-looking load must not re-enable mutations: the
    # binding is latched invalid until an explicit rebind, not just blocked
    # until a fresh load.
    with pytest.raises(MemoryBlockedError, match="rebind"):
        service.load_curated("memory")
    with pytest.raises(MemoryBlockedError):
        service.stage_curated(_mutation(snap))
    stage_calls_before = backend.count("stage_curated")
    service.start(_context("sess-2"), bind_intent="explicit_rebind", prior_identity=old_identity)
    assert service.identity.logical_session_id == "sess-2"
    service.stage_curated(_mutation(snap))
    assert backend.count("stage_curated") > stage_calls_before


def test_degraded_warning_after_a_lost_binding_asks_for_a_rebind(tmp_path):
    """_require_bound() refuses the very load that would clear _blocked, so the
    warning must not promise recovery on a fresh load."""
    backend = StubBackend()
    service = _select(tmp_path, backend)
    snap = service.load_curated("memory")
    backend.fail_typed("stage_curated", "binding_revoked")
    with pytest.raises(MemoryBlockedError):
        service.stage_curated(_mutation(snap))
    warning = service.degraded_warning()
    assert "binding_revoked" in warning and "rebind" in warning
    assert "fresh load" not in warning


def test_a_call_without_a_bound_epoch_is_refused(tmp_path):
    """Epoch continuity is checked on every envelope; with no expected epoch
    there is nothing to check, so the call is refused instead of skipping it."""
    backend = StubBackend()
    service = _select(tmp_path, backend)
    sent_before = backend.count("load_curated")
    service._state = None
    request = w.LoadRequest(expected_provider_epoch="ep-1", frozen_identity=backend.identity(), target="memory")
    with pytest.raises(MemoryBlockedError, match="bound"):
        service._call("load_curated", request)
    assert backend.count("load_curated") == sent_before  # refused locally, never sent


def test_explicit_rebind_recovers_after_epoch_change(tmp_path):
    backend = StubBackend()
    service = _select(tmp_path, backend)
    backend.epoch = "ep-2"
    with pytest.raises(MemoryBlockedError, match="epoch"):
        service.load_curated("memory")
    assert service.blocked and service.epoch_changed
    old_identity = service.identity
    # the backend is still at ep-2; an explicit rebind against the current
    # epoch must clear epoch_changed/blocked, not require the epoch to
    # revert.
    service.start(_context("sess-3"), bind_intent="explicit_rebind", prior_identity=old_identity)
    assert not service.epoch_changed
    assert not service.blocked
    assert service.session_state.provider_epoch == "ep-2"
    bind = backend.calls[-1][1]
    assert bind.bind_intent == "explicit_rebind" and bind.prior_identity == old_identity.to_wire()


def test_malformed_mutation_response_is_unknown_outcome_and_malformed_continuity_never_blocks(tmp_path):
    backend = StubBackend()
    service = _select(tmp_path, backend)
    snap = service.load_curated("memory")
    backend.fail("stage_curated", w.WireError("$.result", "bad"))
    with pytest.raises(ProviderTransportError) as exc:
        service.stage_curated(_mutation(snap))
    assert exc.value.mutation_outcome_unknown is True
    assert service.blocked
    service.load_curated("memory")
    assert not service.blocked

    cont_backend = StubBackend(continuity=True)
    cont_service = _select(tmp_path, cont_backend)
    cont_backend.fail("capture_continuity", w.WireError("$.result", "bad"))
    with pytest.raises(ProviderTransportError) as exc:
        cont_service.capture_continuity(ContinuityCapture("cr1", "compression_snapshot", "text", "compression"))
    assert exc.value.mutation_outcome_unknown is False
    assert not cont_service.blocked


def test_inspect_staged_returns_the_live_stage(tmp_path):
    backend = StubBackend()
    service = _select(tmp_path, backend)
    snap = service.load_curated("memory")
    staged = service.stage_curated(_mutation(snap))
    inspection = service.inspect_staged(InspectRequest("memory", "r1", staged.stage_handle_b64url))
    assert inspection.summary == staged
    assert inspection.canonical_candidates[0].text == "x"
    with pytest.raises(ProviderError) as exc:
        service.inspect_staged(InspectRequest("memory", "r-other", staged.stage_handle_b64url))
    assert exc.value.code == "stage_not_found"


def _replace_reply_field(reply, path, replacement):
    if not path:
        return replacement(reply) if callable(replacement) else replacement
    field, _, remaining = path.partition(".")
    return replace(reply, **{field: _replace_reply_field(getattr(reply, field), remaining, replacement)})


def _other_snapshot_target(snapshot):
    return replace(snapshot, target="user", mutation_entries=(), delivery_entries=(),
                   hidden_preservation_state=replace(snapshot.hidden_preservation_state, target="user"))


class _CorruptReplyBackend(StubBackend):
    corruption = None
    reply_epoch = None

    def _send(self, operation, result):
        if self.corruption and operation == self.corruption[0]:
            result = _replace_reply_field(result, *self.corruption[1:])
        # Corrupt before the real codec round trip: these are structurally
        # valid replies, and commits have already reached the provider store.
        reply = super()._send(operation, result)
        return replace(reply, provider_epoch=self.reply_epoch or reply.provider_epoch)


_CORRELATION_CASES = [
    ("load_curated", "frozen_identity.logical_session_id", "other-session"),
    ("load_curated", "", _other_snapshot_target),
    ("load_curated", "revision.provider_epoch", "ep-other"),
    ("commit_curated", "snapshot.frozen_identity.logical_session_id", "other-session"),
    ("commit_curated", "snapshot", _other_snapshot_target),
    ("commit_curated", "snapshot.revision.provider_epoch", "ep-other"),
    ("commit_curated", "request_id", "other-request"),
    ("recall_context", "frozen_identity.logical_session_id", "other-session"),
    ("recall_context", "target", "user"),
    ("recall_context", "source_revision.provider_epoch", "ep-other"),
    ("recall_context", "source_revision.visibility_revision", "other-visibility"),
    ("recall_context", "source_revision.scope_revisions", lambda scopes: scopes[::-1]),
    ("recall_context", "source_revision.scope_revisions", lambda scopes: (replace(scopes[0], revision="other-revision"), *scopes[1:])),
    ("stage_curated", "request_id", "other-request"),
    ("stage_curated", "target", "user"),
    ("stage_curated", "expected_revision.provider_epoch", "ep-other"),
    ("stage_curated", "expected_revision.visibility_revision", "other-visibility"),
    ("stage_curated", "expected_revision.scope_revisions", lambda scopes: scopes[::-1]),
    ("stage_curated", "expected_revision.scope_revisions", lambda scopes: (replace(scopes[0], revision="other-revision"), *scopes[1:])),
    ("stage_curated", "requested_write_scopes", ()),
    ("stage_curated", "requested_write_scopes", lambda scopes: scopes[::-1]),
    ("inspect_staged", "summary.request_id", "other-request"),
    ("inspect_staged", "summary.target", "user"),
    ("inspect_staged", "summary.stage_handle_b64url", "AAAA"),
    ("capture_continuity", "request_id", "other-request"),
]


def _correlation_actions(service, snapshot, staged):
    mutation = replace(_mutation(snapshot), requested_write_scopes=(REPO, PG))
    return {
        "load_curated": lambda: service.load_curated("memory"),
        "stage_curated": lambda: service.stage_curated(mutation),
        "commit_curated": lambda: service.commit_curated(CommitIntent("memory", "r1", staged.stage_handle_b64url, staged.approval_binding_sha256, (REPO,), w.ApprovalAuthorization(kind="not_required"))),
        "recall_context": lambda: service.recall_context(RecallQuery("memory", snapshot.revision, "q", ("general", "hermes_memory"), (), w.RecallBudget(0, 3000, 20))),
        "inspect_staged": lambda: service.inspect_staged(InspectRequest("memory", "r1", staged.stage_handle_b64url)),
        "capture_continuity": lambda: service.capture_continuity(ContinuityCapture("cr1", "compression_snapshot", "text", "compression")),
    }


@pytest.mark.parametrize("operation,path,replacement,conflict", [
    pytest.param(op, path, replacement, False, id=f"{op}-{path or 'target'}-{i}")
    for i, (op, path, replacement) in enumerate(_CORRELATION_CASES)
] + [
    pytest.param(op, path, replacement, True, id=f"{op}-conflict-{path or 'target'}")
    for op in ("load_curated", "stage_curated", "commit_curated", "recall_context", "inspect_staged")
    for _, path, replacement in _CORRELATION_CASES[:3]
] + [
    pytest.param("capture_continuity", path, replacement, True, id=f"capture_continuity-conflict-{path}")
    for _, path, replacement in (_CORRELATION_CASES[0], _CORRELATION_CASES[2])
])
def test_provider_replies_must_correlate_before_results_escape_or_loads_unblock(tmp_path, operation, path, replacement, conflict):
    backend = _CorruptReplyBackend(recall=True, continuity=True)
    service = _select(tmp_path, backend)
    snapshot = service.load_curated("memory")
    staged = service.stage_curated(replace(_mutation(snapshot), requested_write_scopes=(REPO, PG)))
    actions = _correlation_actions(service, snapshot, staged)
    mutation = operation in ("stage_curated", "commit_curated")

    if conflict:
        # Null details and a correlated current snapshot (whose revision may
        # have advanced) remain legitimate provider decisions.
        current = replace(snapshot, revision=replace(snapshot.revision, visibility_revision="new-visibility"))
        for details in (None, {"current_snapshot": current.to_wire()}):
            backend.fail_typed(operation, "version_conflict", outcome="not_committed" if mutation else "not_applicable", details=details)
            with pytest.raises(MemoryBlockedError if operation == "load_curated" else ProviderError) as exc:
                actions[operation]()
            assert exc.value.code == "version_conflict"
            service.load_curated("memory")
        bad_snapshot = _replace_reply_field(current, path, replacement)
        # Exercise the same strict error-details decoder as the transport.
        details = w.decode_error_details("version_conflict", json.loads(w.canonical_json({"current_snapshot": bad_snapshot.to_wire()})))
        backend.fail_typed(operation, "version_conflict", outcome="not_committed" if mutation else "not_applicable", details=details.to_wire())
    else:
        backend.corruption = (operation, path, replacement)

    expected_error = ProviderTransportError if mutation or operation == "capture_continuity" else MemoryBlockedError
    with pytest.raises(expected_error, match="malformed provider output") as exc:
        actions[operation]()
    if isinstance(exc.value, ProviderTransportError):
        assert exc.value.mutation_outcome_unknown is mutation
    assert not service.epoch_changed  # Only the envelope can change the session epoch.
    assert service.blocked is (operation != "capture_continuity")
    backend.corruption = None

    if service.blocked:
        calls_before = len(backend.calls)
        for blocked_operation in ("stage_curated", "commit_curated"):
            with pytest.raises(MemoryBlockedError):
                actions[blocked_operation]()
        assert len(backend.calls) == calls_before
        assert actions["capture_continuity"]().outcome == "stored"
        assert service.blocked  # A successful non-load cannot clear the latch.
        backend.corruption = ("load_curated", "revision.provider_epoch", "ep-other")
        with pytest.raises(MemoryBlockedError):
            service.load_curated("memory")
        assert service.blocked and not service.epoch_changed
        backend.corruption = None

    fresh = service.load_curated("memory")
    assert not service.blocked
    recovered = service.stage_curated(_mutation(fresh, request_id="recovered"))
    assert recovered.expected_revision == fresh.revision


@pytest.mark.parametrize("operation", ["load_curated", "stage_curated", "commit_curated", "recall_context", "inspect_staged", "capture_continuity"])
@pytest.mark.parametrize("changed_envelope", [False, True])
def test_correlation_failures_preserve_publication_uncertainty_and_epoch_precedence(tmp_path, operation, changed_envelope):
    backend = _CorruptReplyBackend(recall=True, continuity=True)
    service = _select(tmp_path, backend)
    snapshot = service.load_curated("memory")
    staged = service.stage_curated(_mutation(snapshot))
    actions = _correlation_actions(service, snapshot, staged)
    path = {
        "load_curated": "frozen_identity.logical_session_id",
        "stage_curated": "request_id", "commit_curated": "request_id",
        "recall_context": "frozen_identity.logical_session_id",
        "inspect_staged": "summary.request_id", "capture_continuity": "request_id",
    }[operation]
    backend.corruption = (operation, path, "other-session-or-request")
    backend.reply_epoch = "ep-other" if changed_envelope else None
    mutation = operation in ("stage_curated", "commit_curated")
    expected_error = MemoryBlockedError if changed_envelope or not (mutation or operation == "capture_continuity") else ProviderTransportError
    with pytest.raises(expected_error) as exc:
        actions[operation]()
    assert service.epoch_changed is changed_envelope
    assert service.blocked is (changed_envelope or operation != "capture_continuity")
    if isinstance(exc.value, ProviderTransportError):
        assert exc.value.mutation_outcome_unknown is mutation
    if operation == "commit_curated":
        assert backend.entries["memory"] == ["committed:r1"]  # Publication preceded the corrupt acknowledgement.

    backend.corruption = None
    backend.reply_epoch = None
    if changed_envelope:
        calls_before = len(backend.calls)
        for blocked_operation in ("load_curated", "stage_curated", "commit_curated"):
            with pytest.raises(MemoryBlockedError, match="rebind"):
                actions[blocked_operation]()
        assert len(backend.calls) == calls_before
        service.start(_context("sess-2"), bind_intent="explicit_rebind", prior_identity=service.identity)
    fresh = service.load_curated("memory")
    assert not service.blocked and not service.epoch_changed
    if operation == "commit_curated":
        assert [entry.text for entry in fresh.mutation_entries] == ["committed:r1"]
        assert fresh.revision != snapshot.revision
    # A correct post-write snapshot advances beyond the staged pre-write revision.
    next_stage = service.stage_curated(_mutation(fresh, request_id="r2"))
    committed = service.commit_curated(CommitIntent("memory", "r2", next_stage.stage_handle_b64url, next_stage.approval_binding_sha256, (REPO,), w.ApprovalAuthorization(kind="not_required")))
    assert committed.snapshot.revision != next_stage.expected_revision
