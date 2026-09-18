"""Provider-authoritative memory service (§9.1 authoritative mode, §9.6).

The configured provider is the sole curated-memory authority. This class
owns: negotiation (API version and required operations), binding and the
frozen identity, the persisted host session state (epoch + identity), fresh
per-request loads, and the fail-closed state machine. It never touches the
native memory files and knows no particular provider.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from agent.memory_service import wire as w
from agent.memory_service.backend import ProviderResult
from agent.memory_service.config import PROVIDER_API_VERSION, REQUIRED_OPERATIONS, MemoryConfigurationError, MemoryServiceConfig
from agent.memory_service.errors import CapabilityUnavailableError, MemoryBlockedError, ProviderError, ProviderTransportError, TargetDisabledError
from agent.memory_service.identity import FrozenMemoryIdentity, HostSessionState
from agent.memory_service.service import CommitIntent, ContinuityCapture, InspectRequest, MemoryDisposition, MemoryService, MutationRequest, RecallQuery, ServiceCapabilities

logger = logging.getLogger(__name__)

_NATIVE_HEADER = {"memory": "Memory", "user": "User profile"}
_RESPONSE_ECHO_FIELDS = {
    "recall_context": ("frozen_identity", "target", "source_revision"),
    "stage_curated": ("request_id", "target", "expected_revision", "requested_write_scopes"),
    "inspect_staged": ("request_id", "target", "stage_handle_b64url"),
    "commit_curated": ("request_id",),
    "capture_continuity": ("request_id",),
}


def _validate_snapshot_correlation(request: Any, snapshot: w.CuratedSnapshot, path: str) -> None:
    for field in ("frozen_identity", "target"):
        # Error snapshots can also accompany calls without a target (e.g.
        # continuity); compare only identity/context fixed by that request.
        if hasattr(request, field) and getattr(snapshot, field) != getattr(request, field):
            raise w.WireError(f"{path}.{field}", "does not match the request")
    if snapshot.revision.provider_epoch != request.expected_provider_epoch:
        raise w.WireError(f"{path}.revision.provider_epoch", "does not match the bound provider epoch")


def _validate_response_correlation(operation: str, request: Any, result: Any) -> None:
    """Check request-dependent promises that a standalone wire decode cannot."""
    body = result.summary if operation == "inspect_staged" else result
    path = "$.result.summary" if operation == "inspect_staged" else "$.result"
    for field in _RESPONSE_ECHO_FIELDS.get(operation, ()):
        # Dataclass equality includes the complete revision and tuple order.
        if getattr(body, field) != getattr(request, field):
            raise w.WireError(f"{path}.{field}", "does not match the request")
    if operation == "load_curated":
        _validate_snapshot_correlation(request, result, "$.result")
    elif operation == "commit_curated":
        # A committed snapshot has a post-write revision, not the staged base.
        _validate_snapshot_correlation(request, result.snapshot, "$.result.snapshot")


class ProviderAuthoritativeMemoryService(MemoryService):
    """Provider-authoritative disposition (§9.1, §9.6).

    An instance is bound to one logical session. The fail-closed latches
    (``_blocked``, ``_epoch_changed``, ``_binding_lost``) are plain
    attributes with no lock; Hermes runs gateway sessions on threads, so
    callers MUST serialize access to a given instance rather than share it
    across concurrent calls.
    """

    def __init__(self, config: MemoryServiceConfig, backend: Any) -> None:
        super().__init__(config)
        self._backend = backend
        self._negotiation: Optional[w.Negotiation] = None
        self._state: Optional[HostSessionState] = None
        self._blocked = False
        self._epoch_changed = False
        self._binding_lost = False
        self._block_reason = ""

    # -- lifecycle -----------------------------------------------------------

    def start(self, requested_context: w.RequestedContext, *, bind_intent: str = "new_session", prior_identity: Optional[FrozenMemoryIdentity] = None) -> None:
        # start() is the single reset point: an explicit rebind or a new
        # logical session clears every latch a prior session may have set.
        self._state = None
        self._blocked = False
        self._epoch_changed = False
        self._binding_lost = False
        self._block_reason = ""
        epoch = self._negotiate()
        request = w.BindRequest(expected_provider_epoch=epoch, binding_request_id=uuid.uuid4().hex, bind_intent=bind_intent, requested_context=requested_context, prior_identity=prior_identity.to_wire() if prior_identity else None)
        request.validate()
        result = self._call("bind_session", request, expected_epoch=epoch)
        self._state = HostSessionState(provider_epoch=result.provider_epoch, identity=FrozenMemoryIdentity.from_wire(result.result.frozen_identity))

    def resume(self, session_state: HostSessionState) -> None:
        epoch = self._negotiate()
        if epoch != session_state.provider_epoch:
            self._epoch_changed = True
            logger.warning("memory provider epoch changed (operation=resume)")
            raise MemoryBlockedError(f"provider epoch changed from {session_state.provider_epoch} to {epoch}; explicit rebind or a new logical session is required")
        request = w.ValidateSessionRequest(expected_provider_epoch=session_state.provider_epoch, frozen_identity=session_state.identity.to_wire())
        result = self._call("validate_session", request, expected_epoch=session_state.provider_epoch)
        if result.result.frozen_identity != session_state.identity.to_wire():
            raise MemoryBlockedError("validate_session returned a different identity; refusing to continue")
        self._state = session_state

    def _negotiate(self) -> str:
        request = w.NegotiateRequest(host="hermes", supported_api_versions=(PROVIDER_API_VERSION,), required_operations=tuple(REQUIRED_OPERATIONS))
        result = self._backend.negotiate(request)
        neg = result.result
        if neg.selected_api_version != PROVIDER_API_VERSION:
            raise MemoryConfigurationError(f"memory provider {self._config.provider!r} selected API version {neg.selected_api_version}; Hermes requires API version {PROVIDER_API_VERSION}")
        if neg.provider != self._config.provider:
            raise MemoryConfigurationError(f"memory provider {self._config.provider!r} negotiated as {neg.provider!r}")
        missing = [op for op in REQUIRED_OPERATIONS if op not in neg.operations]
        if missing:
            raise MemoryConfigurationError(f"memory provider {self._config.provider!r} lacks required operation(s): {', '.join(missing)}")
        # A capability flag names an optional operation of the same name
        # (§9.1); a provider that sets the flag but omits the operation from
        # `operations` contradicts itself and ServiceCapabilities must not
        # just follow the flag -- fail the negotiation instead of trusting it.
        for capability in ("recall_context", "capture_continuity"):
            if getattr(neg.capabilities, capability) and capability not in neg.operations:
                raise MemoryConfigurationError(
                    f"memory provider {self._config.provider!r} negotiated capabilities.{capability}: true "
                    f"but does not list {capability!r} in operations"
                )
        self._negotiation = neg
        return result.provider_epoch

    # -- interface -----------------------------------------------------------

    @property
    def disposition(self) -> MemoryDisposition:
        return MemoryDisposition.AUTHORITATIVE

    @property
    def identity(self) -> Optional[FrozenMemoryIdentity]:
        return self._state.identity if self._state else None

    @property
    def session_state(self) -> Optional[HostSessionState]:
        return self._state

    @property
    def capabilities(self) -> ServiceCapabilities:
        caps = self._negotiation.capabilities if self._negotiation else None
        return ServiceCapabilities(recall_context=bool(caps and caps.recall_context), capture_continuity=bool(caps and caps.capture_continuity))

    @property
    def blocked(self) -> bool:
        return self._blocked

    @property
    def epoch_changed(self) -> bool:
        return self._epoch_changed

    def load_curated(self, target: str) -> w.CuratedSnapshot:
        self._require_target(target)
        self._require_bound()
        request = w.LoadRequest(expected_provider_epoch=self._state.provider_epoch, frozen_identity=self._state.identity.to_wire(), target=target)
        result = self._call("load_curated", request, fresh_load=True)
        self._blocked = False
        self._block_reason = ""
        return result.result

    def prompt_block(self, target: str) -> Optional[str]:
        snapshot = self.load_curated(target)
        texts = [e.text for e in snapshot.delivery_entries]
        if not texts:
            return None
        return f"{_NATIVE_HEADER[target]} (authoritative provider {self._config.provider})\n" + "\n".join(f"- {t}" for t in texts)

    def stage_curated(self, request: MutationRequest) -> w.StageResult:
        self._require_target(request.target)
        self._require_bound()
        self._require_unblocked()
        if request.intent is None or request.provenance is None or request.expected_revision is None or request.hidden_preservation_state is None:
            raise ProviderError(code="invalid_request", outcome="not_committed", details=None, operation="stage_curated")
        wire_request = w.StageRequest(expected_provider_epoch=self._state.provider_epoch, frozen_identity=self._state.identity.to_wire(), target=request.target, expected_revision=request.expected_revision, hidden_preservation_state=request.hidden_preservation_state, request_id=request.request_id, requested_write_scopes=request.requested_write_scopes, intent=request.intent, mutation_delta=request.mutation_delta, candidate_entries=request.candidate_entries, provenance=request.provenance)
        wire_request.validate()
        return self._call("stage_curated", wire_request, mutation=True).result

    def inspect_staged(self, request: InspectRequest) -> w.StageInspection:
        self._require_target(request.target)
        self._require_bound()
        self._require_unblocked()
        wire_request = w.InspectStageRequest(expected_provider_epoch=self._state.provider_epoch, frozen_identity=self._state.identity.to_wire(), target=request.target, request_id=request.request_id, stage_handle_b64url=request.stage_handle_b64url)
        return self._call("inspect_staged", wire_request).result

    def commit_curated(self, intent: CommitIntent) -> w.CommitResult:
        self._require_target(intent.target)
        self._require_bound()
        self._require_unblocked()
        wire_request = w.CommitRequest(expected_provider_epoch=self._state.provider_epoch, frozen_identity=self._state.identity.to_wire(), target=intent.target, request_id=intent.request_id, stage_handle_b64url=intent.stage_handle_b64url, approval_binding_sha256=intent.approval_binding_sha256, authorized_write_scopes=intent.authorized_write_scopes, authorization=intent.authorization)
        wire_request.validate()
        return self._call("commit_curated", wire_request, mutation=True).result

    def recall_context(self, query: RecallQuery) -> w.TypedRecall:
        self._require_target(query.target)
        self._require_bound()
        self._require_unblocked()
        if not self.capabilities.recall_context:
            raise CapabilityUnavailableError("provider did not negotiate recall_context")
        wire_request = w.RecallRequest(expected_provider_epoch=self._state.provider_epoch, frozen_identity=self._state.identity.to_wire(), target=query.target, source_revision=query.source_revision, query=query.query, include_channels=tuple(query.include_channels), exclude_entry_ids=tuple(query.exclude_entry_ids), budget=query.budget)
        wire_request.validate()
        return self._call("recall_context", wire_request, non_blocking=True).result

    def capture_continuity(self, capture: ContinuityCapture) -> w.ContinuityResult:
        self._require_bound()
        if not self.capabilities.capture_continuity:
            raise CapabilityUnavailableError("provider did not negotiate capture_continuity")
        wire_request = w.ContinuityRequest(expected_provider_epoch=self._state.provider_epoch, frozen_identity=self._state.identity.to_wire(), request_id=capture.request_id, kind=capture.kind, text=capture.text, initiating_surface=capture.initiating_surface)
        wire_request.validate()
        return self._call("capture_continuity", wire_request, non_blocking=True).result

    def degraded_warning(self) -> Optional[str]:
        if self._epoch_changed:
            return "Curated memory is unavailable: the memory provider's epoch changed. Start a new logical session or rebind explicitly."
        # A lost binding is checked before _blocked: _require_bound() refuses the
        # very fresh load that would clear _blocked, so "blocked until a fresh
        # load" would be false here. Only a rebind or a new session recovers.
        if self._binding_lost:
            return f"Curated memory is unavailable: the provider binding is {self._block_reason or 'invalid'}. Start a new logical session or rebind explicitly."
        if self._blocked:
            return f"Curated memory is blocked until the provider answers a fresh load: {self._block_reason}"
        return None

    def shutdown(self) -> None:
        self._backend.shutdown()

    # -- internals -----------------------------------------------------------

    def _require_target(self, target: str) -> None:
        if not self.target_enabled(target):
            raise TargetDisabledError(f"target {target} is disabled by configuration")

    def _require_bound(self) -> None:
        if self._state is None:
            raise MemoryBlockedError("memory service is not bound to a logical session")
        if self._epoch_changed:
            raise MemoryBlockedError("provider epoch changed; explicit rebind or a new logical session is required")
        if self._binding_lost:
            raise MemoryBlockedError("binding is invalid or revoked; explicit rebind or a new logical session is required")

    def _require_unblocked(self) -> None:
        if self._blocked:
            raise MemoryBlockedError(f"provider mutations are blocked until a fresh load succeeds: {self._block_reason}")

    def _fail_transport(self, exc: ProviderTransportError, *, operation: str, fresh_load: bool, mutation: bool, non_blocking: bool) -> None:
        """Shared handling for a real or WireError-synthesized transport failure."""
        if non_blocking:
            raise exc
        self._blocked = True
        self._block_reason = f"{operation}: {exc.reason}"
        logger.warning("memory service blocked (operation=%s)", operation)
        # A read (or any non-mutation call) can be reported as a plain block:
        # nothing was attempted at the provider's store. A mutation's caller
        # instead needs the raw ProviderTransportError, whose
        # mutation_outcome_unknown says the write may or may not have landed --
        # MemoryBlockedError would lose that distinction (§9.6).
        if fresh_load or not mutation:
            raise MemoryBlockedError(self._block_reason) from exc
        raise exc

    def _call(self, operation: str, request: Any, *, expected_epoch: Optional[str] = None, fresh_load: bool = False, mutation: bool = False, non_blocking: bool = False) -> ProviderResult:
        expected = expected_epoch if expected_epoch is not None else (self._state.provider_epoch if self._state else None)
        # Every operation but negotiate is epoch-continuity checked. Without a
        # caller-supplied epoch and without bound state there is nothing to
        # check against, so refuse rather than silently skip the check.
        if expected is None and operation != "negotiate":
            raise MemoryBlockedError(f"{operation} attempted before the session was bound to a provider epoch")
        try:
            try:
                result = getattr(self._backend, operation)(request)
            except ProviderError as exc:
                if exc.code == "provider_epoch_changed":
                    self._epoch_changed = True
                    self._blocked = True
                    self._block_reason = "provider epoch changed"
                    logger.warning("memory provider epoch changed (operation=%s)", operation)
                    raise MemoryBlockedError("provider epoch changed; explicit rebind or a new logical session is required", code=exc.code, provider_error=exc) from exc
                if exc.code in ("binding_invalid", "binding_revoked"):
                    self._binding_lost = True
                    self._blocked = True
                    self._block_reason = exc.code
                    logger.warning("memory provider binding lost (operation=%s)", operation)
                    raise MemoryBlockedError(f"binding is {exc.code}; explicit rebind or a new logical session is required", code=exc.code, provider_error=exc) from exc
                if exc.code == "version_conflict" and exc.details is not None:
                    details = w.decode_error_details(exc.code, exc.details)
                    _validate_snapshot_correlation(request, details.current_snapshot, "$.error.details.current_snapshot")
                if fresh_load and not non_blocking:
                    self._blocked = True
                    self._block_reason = f"{operation} failed with {exc.code}"
                    logger.warning("memory service blocked (operation=%s)", operation)
                    raise MemoryBlockedError(self._block_reason, code=exc.code, provider_error=exc) from exc
                raise
            # An envelope epoch change takes precedence over a simultaneous
            # correlation failure: only rebind/new session may recover it.
            if expected is not None and result.provider_epoch != expected:
                self._epoch_changed = True
                self._blocked = True
                self._block_reason = "provider epoch changed"
                logger.warning("memory provider epoch changed (operation=%s)", operation)
                raise MemoryBlockedError(f"provider epoch changed from {expected} to {result.provider_epoch}; explicit rebind or a new logical session is required")
            _validate_response_correlation(operation, request, result.result)
        except ProviderTransportError as exc:
            self._fail_transport(exc, operation=operation, fresh_load=fresh_load, mutation=mutation, non_blocking=non_blocking)
        except w.WireError as exc:
            # This also catches correlation errors in success results and
            # ProviderError details, preserving the same failure semantics.
            transport_exc = ProviderTransportError(reason=f"malformed provider output: {exc}", operation=operation, mutation_outcome_unknown=mutation)
            self._fail_transport(transport_exc, operation=operation, fresh_load=fresh_load, mutation=mutation, non_blocking=non_blocking)
        return result
