"""In-memory scripted AuthoritativeBackend: the S5 stub transport.

Records every call, answers from a tiny state, and can be told to fail a
named operation once or permanently with a typed or transport error.

The stub is an honest transport about the wire, not only about call recording:
every request is encoded with :func:`wire.canonical_json` and strictly decoded
back through ``wire.REQUEST_TYPES`` before it is handled, and every result
makes the same round trip through ``wire.RESULT_TYPES`` before it is returned.
So a request the service emits that is not wire-legal fails here exactly as it
would against a real provider, and no result reaches the service without
passing a strict decode.
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

from agent.memory_service import wire as w
from agent.memory_service.backend import ProviderResult
from agent.memory_service.errors import ProviderError, ProviderTransportError

HANDLE = base64.urlsafe_b64encode(bytes(range(32))).rstrip(b"=").decode()
PG = w.ScopeRef(kind="principal_global", id="ethan")
REPO = w.ScopeRef(kind="repository", id="repo-1")


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


class StubBackend:
    def __init__(self, *, epoch: str = "ep-1", api_version: int = 1, operations: Optional[List[str]] = None, recall: bool = False, continuity: bool = False, provider: str = "example"):
        self.epoch = epoch
        self.api_version = api_version
        self.operations = list(operations) if operations is not None else ["bind_session", "validate_session", "load_curated", "stage_curated", "inspect_staged", "commit_curated"] + (["recall_context"] if recall else []) + (["capture_continuity"] if continuity else [])
        self.recall = recall
        self.continuity = continuity
        self.provider = provider
        self.calls: List[Tuple[str, Any]] = []
        self.failures: Dict[str, List[Exception]] = {}
        self.entries: Dict[str, List[str]] = {"memory": [], "user": []}
        self.malformed_snapshot: Optional[str] = None  # "identity" | "target" | "wire"
        self.shutdown_calls = 0
        self._stages: Dict[str, Tuple[w.StageRequest, w.StageResult]] = {}

    # scripting -------------------------------------------------------------
    def fail(self, operation: str, error: Exception, *, times: int = 1) -> None:
        self.failures.setdefault(operation, []).extend([error] * times)

    def fail_typed(self, operation: str, code: str, *, outcome: str = "not_applicable", details=None, times: int = 1) -> None:
        self.fail(operation, ProviderError(code=code, outcome=outcome, details=details, operation=operation), times=times)

    def fail_transport(self, operation: str, reason: str = "timeout", *, times: int = 1) -> None:
        unknown = operation in ("stage_curated", "commit_curated")
        self.fail(operation, ProviderTransportError(reason=reason, operation=operation, mutation_outcome_unknown=unknown), times=times)

    def _maybe_fail(self, operation: str) -> None:
        queue = self.failures.get(operation)
        if queue:
            raise queue.pop(0)

    def count(self, operation: str) -> int:
        return sum(1 for op, _ in self.calls if op == operation)

    # codec ------------------------------------------------------------------
    def _receive(self, operation: str, request: Any) -> Any:
        """Decode the request as a real provider would: canonical JSON in, strict decode back."""
        decoded = w.REQUEST_TYPES[operation].from_wire(json.loads(w.canonical_json(request.to_wire())))
        self.calls.append((operation, decoded))
        return decoded

    def _send(self, operation: str, result: Any) -> ProviderResult:
        """Encode the result and strictly decode it back before the service sees it."""
        return ProviderResult(w.RESULT_TYPES[operation].from_wire(json.loads(w.canonical_json(result.to_wire()))), self.epoch)

    # identity/snapshot helpers ---------------------------------------------
    def identity(self, logical_session_id: str = "sess-1") -> w.FrozenIdentityWire:
        return w.FrozenIdentityWire(provider=self.provider, provider_mode="authoritative", principal_id="ethan", profile_id="default", logical_session_id=logical_session_id, org_id=None, project_id="proj-1", repo_id="repo-1", workspace_id=None, platform="cli", binding_revision="rev-1", opaque_binding_b64url=HANDLE)

    def _stored_for(self, target: str, texts: List[str]) -> Tuple[w.StoredEntry, ...]:
        channel = "hermes_memory" if target == "memory" else "hermes_user"
        prov = w.AcceptedProvenance(actor_kind="hermes", principal_id="ethan", logical_session_id=None, surface="stub", source_entry_ids=(), source_commit=None, transaction_id=None)
        return tuple(w.StoredEntry(id=hashlib.sha256(t.encode()).hexdigest()[:12], text=t, origin_scope=REPO, target=target, record_channel=channel, lane="scoped_evidence", lifecycle="active", policy_key=None, provenance=prov) for t in texts)

    def snapshot(self, identity: w.FrozenIdentityWire, target: str) -> w.CuratedSnapshot:
        texts = self.entries[target]
        channel = "hermes_memory" if target == "memory" else "hermes_user"
        prov = w.AcceptedProvenance(actor_kind="hermes", principal_id="ethan", logical_session_id=None, surface="stub", source_entry_ids=(), source_commit=None, transaction_id=None)
        stored = self._stored_for(target, texts)
        delivered = tuple(w.DeliveredEntry(id=e.id, text=e.text, origin_scope=REPO, target=target, record_channel=channel, lane=e.lane, delivery_tier="repository", policy_key=None, provenance=prov) for e in stored)
        rev = w.CompositeRevision(provider_epoch=self.epoch, visibility_revision="vis-1", scope_revisions=(w.ScopeRevision(scope=REPO, revision=hashlib.sha256("\n".join(texts).encode()).hexdigest()[:8]), w.ScopeRevision(scope=PG, revision="r0")))
        ident = identity
        if self.malformed_snapshot == "identity":
            ident = w.FrozenIdentityWire(**{**identity.to_wire(), "logical_session_id": "someone-else"})
        snap_target = "user" if self.malformed_snapshot == "target" and target == "memory" else target
        return w.CuratedSnapshot(api_version=1, status="ok", frozen_identity=ident, target=snap_target, visible_scopes=(REPO, PG), default_write_scope=REPO, eligible_write_scopes=(REPO,), complete_for_scopes=(REPO,), revision=rev, limits=w.CuratedLimits(memory_chars=2200, user_chars=1375, initial_general_chars=3000, max_entry_chars=2200, max_entries=100), mutation_entries=stored if snap_target == target else (), delivery_entries=delivered if snap_target == target else (), hidden_preservation_state=w.HiddenPreservationState(target=snap_target, complete_for_scopes=(REPO,), opaque_state_b64url=_b64(hashlib.sha256(w.canonical_json(rev.to_wire())).digest())))

    # protocol --------------------------------------------------------------
    def negotiate(self, request: w.NegotiateRequest) -> ProviderResult[w.Negotiation]:
        request = self._receive("negotiate", request)
        self._maybe_fail("negotiate")
        neg = w.Negotiation(provider=self.provider, selected_api_version=self.api_version, operations=tuple(self.operations), capabilities=w.Capabilities(recall_context=self.recall, capture_continuity=self.continuity), limits=w.Limits(max_request_bytes=1_000_000, max_response_bytes=4_000_000, max_stage_bytes=200_000, stage_ttl_seconds=3600, max_opaque_binding_bytes=32, max_continuity_bytes=65_536))
        return self._send("negotiate", neg)

    def bind_session(self, request: w.BindRequest) -> ProviderResult[w.BindResult]:
        request = self._receive("bind_session", request)
        self._maybe_fail("bind_session")
        ident = self.identity(request.requested_context.logical_session_id)
        return self._send("bind_session", w.BindResult(frozen_identity=ident, visible_scopes=(REPO, PG), memory_default_write_scope=REPO, memory_eligible_write_scopes=(REPO,), user_write_scope=PG))

    def validate_session(self, request: w.ValidateSessionRequest) -> ProviderResult[w.ValidateSessionResult]:
        request = self._receive("validate_session", request)
        self._maybe_fail("validate_session")
        return self._send("validate_session", w.ValidateSessionResult(valid=True, frozen_identity=request.frozen_identity, visible_scopes=(REPO, PG)))

    def load_curated(self, request: w.LoadRequest) -> ProviderResult[w.CuratedSnapshot]:
        request = self._receive("load_curated", request)
        self._maybe_fail("load_curated")
        if self.malformed_snapshot == "wire":
            raise w.WireError("$.result.status", "must be one of 'ok', 'degraded_global_only'")
        return self._send("load_curated", self.snapshot(request.frozen_identity, request.target))

    def stage_curated(self, request: w.StageRequest) -> ProviderResult[w.StageResult]:
        request = self._receive("stage_curated", request)
        self._maybe_fail("stage_curated")
        admissions = tuple(w.AdmissionDecision(client_ref=c.client_ref, assigned_id=hashlib.sha256(c.text.encode()).hexdigest()[:12], target=request.target, record_channel="hermes_memory" if request.target == "memory" else "hermes_user", origin_scope=REPO, disposition="scoped_evidence", publication_effect="create_record", superseded_id=None, policy_key=None) for c in request.candidate_entries)
        result = w.StageResult(stage_handle_b64url=_b64(hashlib.sha256(request.request_id.encode()).digest()), request_id=request.request_id, target=request.target, expected_revision=request.expected_revision, requested_write_scopes=request.requested_write_scopes, eligible_write_scopes=(REPO,), expires_at="2026-09-04T00:00:00Z", approval_binding_sha256="c" * 64, approval_requirements=(), candidate_hashes=tuple(w.CandidateHash(client_ref=c.client_ref, canonical_sha256=hashlib.sha256(c.text.encode()).hexdigest()) for c in request.candidate_entries), admissions=admissions, hidden_effects=())
        self._stages[result.stage_handle_b64url] = (request, result)
        return self._send("stage_curated", result)

    def inspect_staged(self, request: w.InspectStageRequest) -> ProviderResult[w.StageInspection]:
        request = self._receive("inspect_staged", request)
        self._maybe_fail("inspect_staged")
        staged = self._stages.get(request.stage_handle_b64url)
        if staged is None or staged[1].request_id != request.request_id:
            raise ProviderError(code="stage_not_found", outcome="not_applicable", details=None, operation="inspect_staged")
        stage_request, result = staged
        target = request.target
        before = self._stored_for(target, self.entries[target])
        after_texts = list(self.entries[target]) + [c.text for c in stage_request.candidate_entries]
        after = self._stored_for(target, after_texts)
        return self._send("inspect_staged", w.StageInspection(summary=result, canonical_candidates=stage_request.candidate_entries, visible_before=before, visible_after=after))

    def commit_curated(self, request: w.CommitRequest) -> ProviderResult[w.CommitResult]:
        request = self._receive("commit_curated", request)
        self._maybe_fail("commit_curated")
        self.entries[request.target].append(f"committed:{request.request_id}")
        self._stages.pop(request.stage_handle_b64url, None)
        return self._send("commit_curated", w.CommitResult(outcome="committed_audit_clean", request_id=request.request_id, tx_id="tx-" + request.request_id, snapshot=self.snapshot(request.frozen_identity, request.target), admissions=()))

    def recall_context(self, request: w.RecallRequest) -> ProviderResult[w.TypedRecall]:
        request = self._receive("recall_context", request)
        self._maybe_fail("recall_context")
        return self._send("recall_context", w.TypedRecall(frozen_identity=request.frozen_identity, target=request.target, source_revision=request.source_revision, trusted_instructions=(), scoped_evidence=()))

    def capture_continuity(self, request: w.ContinuityRequest) -> ProviderResult[w.ContinuityResult]:
        request = self._receive("capture_continuity", request)
        self._maybe_fail("capture_continuity")
        return self._send("capture_continuity", w.ContinuityResult(buffer_id="buf-1", request_id=request.request_id, kind=request.kind, expires_at="2026-09-04T00:00:00Z", outcome="stored"))

    def shutdown(self) -> None:
        self.shutdown_calls += 1
