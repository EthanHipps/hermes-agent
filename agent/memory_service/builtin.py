"""Built-in (additive) memory service: native ``MemoryStore`` semantics
exposed through the generic interface as complete snapshots and explicit
deltas (§9.1 built-in backend, D5, D7).

Mapping (plan decision D-R35-3): one ``principal_global`` scope; ``memory``
entries are ``hermes_memory``/``scoped_evidence`` and ``user`` entries are
``hermes_user``/``trusted_instruction``; an entry's stable ID is ``"b"`` plus
the first 24 hex characters of SHA-256 of its text; revisions are content
hashes; approval requirements are empty because native approval is the host
``write_approval`` gate, not a provider requirement.

Two §9.3 intents need explicit treatment here, because neither is expressible
as an ordinary delta over the native files:

* ``reset`` means "retire every current-epoch record of this target in the
  exact approved reset scopes" (§9.3). The native store has exactly one scope,
  so a reset is accepted only when ``intent.reset_scopes`` and
  ``requested_write_scopes`` are both exactly that scope (otherwise
  ``unauthorized_scope``), and it retires every entry the staged snapshot
  listed. ``hidden_effects`` stays empty on purpose: the native store keeps no
  hidden raw records, so a reset hides nothing beyond what
  ``inspect_staged`` already shows as the empty after-state.
* ``import`` is REFUSED at stage with ``invalid_request``. §9.3 import
  semantics (the ``reuse_existing_import`` publication effect and the
  ``(provider_epoch, principal_id, source_kind, parser_version, source_id,
  item_key, target, destination_scope, canonical_sha256)`` uniqueness tuple)
  require an acknowledgement-derived import-source index. The native store has
  none, and reporting ``create_record`` for a candidate that commit would
  silently dedupe away would be a false success.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from agent.memory_service import wire as w
from agent.memory_service.config import MemoryServiceConfig
from agent.memory_service.errors import CapabilityUnavailableError, MemoryServiceError, ProviderError, TargetDisabledError
from agent.memory_service.identity import FrozenMemoryIdentity
from agent.memory_service.service import (
    CommitIntent,
    ContinuityCapture,
    InspectRequest,
    MemoryDisposition,
    MemoryService,
    MutationRequest,
    RecallQuery,
    ServiceCapabilities,
)
from tools.memory_tool_store import MemoryFileUnreadableError

logger = logging.getLogger(__name__)

_STAGE_TTL = timedelta(hours=1)
_CHANNEL = {"memory": "hermes_memory", "user": "hermes_user"}
_LANE = {"memory": "scoped_evidence", "user": "trusted_instruction"}


class BuiltinStoreError(MemoryServiceError):
    """The native store refused a write; ``response`` is its native result dict."""

    def __init__(self, response: Dict[str, Any]) -> None:
        super().__init__(str(response.get("error") or "native memory store refused the write"))
        self.response = response


def entry_id(text: str) -> str:
    return "b" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


class BuiltinMemoryService(MemoryService):
    def __init__(
        self,
        config: MemoryServiceConfig,
        store: Any,
        *,
        principal_id: str = "local",
        profile_id: str = "default",
        logical_session_id: str = "",
        platform: str = "cli",
    ) -> None:
        super().__init__(config)
        self._store = store
        self._identity = FrozenMemoryIdentity(
            provider="builtin",
            provider_mode="additive",
            principal_id=principal_id,
            profile_id=profile_id,
            logical_session_id=logical_session_id or uuid.uuid4().hex,
            org_id=None,
            project_id=None,
            repo_id=None,
            workspace_id=None,
            platform=platform,
            binding_revision="builtin",
            opaque_binding_b64url="",
        )
        self._scope = w.ScopeRef(kind="principal_global", id=principal_id)
        # A live stage keeps its request, its result, the snapshot it was
        # staged against, its expiry, and the exact entry texts commit must
        # retire (computed at stage time so reset -- which carries no delta --
        # and inspect/commit all agree on one list).
        self._stages: Dict[str, Tuple[MutationRequest, w.StageResult, w.CuratedSnapshot, datetime, Tuple[str, ...]]] = {}
        self._stage_by_request: Dict[str, Tuple[bytes, str]] = {}
        self._receipts: Dict[str, Tuple[str, w.CommitResult]] = {}
        #: Request IDs whose stage expired. §9.3: the expired state answers
        #: ``stage_expired`` on every door, so the tombstone outlives the stage.
        self._expired_requests: set = set()

    # -- interface -----------------------------------------------------------

    @property
    def disposition(self) -> MemoryDisposition:
        return MemoryDisposition.BUILTIN

    @property
    def identity(self) -> Optional[FrozenMemoryIdentity]:
        return self._identity

    @property
    def capabilities(self) -> ServiceCapabilities:
        return ServiceCapabilities(recall_context=False, capture_continuity=False)

    def _require_target(self, target: str) -> None:
        if not self.target_enabled(target):
            raise TargetDisabledError(f"target {target} is disabled by configuration")

    def load_curated(self, target: str) -> w.CuratedSnapshot:
        """Read the native store's current entries and wrap them as a snapshot.

        The returned snapshot's ``frozen_identity`` is a wire-typed artifact
        only (see ``_wire_identity``); its ``provider_mode`` MUST NOT be read
        by hosts. Raises :class:`BuiltinStoreError` instead of returning a
        false ``status="ok"`` snapshot when the on-disk file exists but could
        not be read (``tools.memory_tool_store.MemoryFileUnreadableError``) -- the
        in-memory view would otherwise be silently stale.
        """
        self._require_target(target)
        try:
            entries = self._store.read_entries(target)
        except MemoryFileUnreadableError as exc:
            raise BuiltinStoreError(exc.response) from exc
        return self._snapshot(target, entries)

    def prompt_block(self, target: str) -> Optional[str]:
        self._require_target(target)
        return self._store.format_for_system_prompt(target)

    def stage_curated(self, request: MutationRequest) -> w.StageResult:
        target = request.target
        self._require_target(target)
        self._expire_stages()
        self._require_not_expired(request.request_id, "stage_curated")
        wire_request = self._to_wire_stage_request(request)
        if wire_request.intent.kind == "import":
            # See the module docstring: no import-source index, so no honest
            # publication_effect and no reuse detection.
            raise ProviderError(code="invalid_request", outcome="not_committed", details=None, operation="stage_curated")
        fingerprint = w.canonical_json(wire_request.to_wire())
        prior = self._stage_by_request.get(request.request_id)
        if prior is not None:
            prior_fp, prior_handle = prior
            if prior_fp != fingerprint:
                raise ProviderError(code="idempotency_mismatch", outcome="not_committed", details=None, operation="stage_curated")
            if prior_handle in self._stages:
                return self._stages[prior_handle][1]
            if request.request_id in self._receipts:
                raise ProviderError(code="stage_not_found", outcome="not_committed", details={"state": "committed", "tx_id": self._receipts[request.request_id][1].tx_id}, operation="stage_curated")
            raise ProviderError(code="stage_expired", outcome="not_committed", details=None, operation="stage_curated")
        current = self.load_curated(target)
        if request.hidden_preservation_state != self._hidden_state(target, request.expected_revision):
            raise ProviderError(code="invalid_request", outcome="not_committed", details=None, operation="stage_curated")
        if request.expected_revision != current.revision:
            raise ProviderError(code="version_conflict", outcome="not_committed", details={"current_snapshot": current.to_wire()}, operation="stage_curated")
        by_id = {e.id: e for e in current.mutation_entries}
        for item in request.mutation_delta:
            old = item.record_id or item.old_record_id
            if old is not None and old not in by_id:
                raise ProviderError(code="invalid_request", outcome="not_committed", details=None, operation="stage_curated")
        for scope in request.requested_write_scopes:
            if scope != self._scope:
                raise ProviderError(code="unauthorized_scope", outcome="not_committed", details=None, operation="stage_curated")
        if wire_request.intent.kind == "reset":
            # The one scope the native store has must be the exact approved
            # reset scope AND the exact requested write scope; anything else
            # would approve a reset of something this store cannot express.
            if list(wire_request.intent.reset_scopes or ()) != [self._scope] or list(request.requested_write_scopes) != [self._scope]:
                raise ProviderError(code="unauthorized_scope", outcome="not_committed", details=None, operation="stage_curated")
            retire_texts = tuple(e.text for e in current.mutation_entries)
        else:
            retire_texts = tuple(by_id[d.record_id or d.old_record_id].text for d in request.mutation_delta if d.action in ("retire", "supersede"))
        admissions = []
        hashes = []
        superseded = {d.replacement_client_ref: d.old_record_id for d in request.mutation_delta if d.action == "supersede"}
        for cand in request.candidate_entries:
            text = cand.text.strip()
            admissions.append(
                w.AdmissionDecision(
                    client_ref=cand.client_ref,
                    assigned_id=entry_id(text),
                    target=target,
                    record_channel=_CHANNEL[target],
                    origin_scope=self._scope,
                    disposition=_LANE[target],
                    publication_effect="create_record",
                    superseded_id=superseded.get(cand.client_ref),
                    policy_key=None,
                )
            )
            hashes.append(w.CandidateHash(client_ref=cand.client_ref, canonical_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest()))
        handle = _b64url(secrets.token_bytes(32))
        expires = _now() + _STAGE_TTL
        unsigned = w.StageResult(
            stage_handle_b64url=handle,
            request_id=request.request_id,
            target=target,
            expected_revision=request.expected_revision,
            requested_write_scopes=request.requested_write_scopes,
            eligible_write_scopes=(self._scope,),
            expires_at=_timestamp(expires),
            approval_binding_sha256="0" * 64,
            approval_requirements=(),
            candidate_hashes=tuple(hashes),
            admissions=tuple(admissions),
            hidden_effects=(),
        )
        binding = self._approval_binding(unsigned, wire_request)
        result = dataclasses.replace(unsigned, approval_binding_sha256=binding)
        self._stages[handle] = (request, result, current, expires, retire_texts)
        self._stage_by_request[request.request_id] = (fingerprint, handle)
        return result

    def inspect_staged(self, request: InspectRequest) -> w.StageInspection:
        self._require_target(request.target)
        self._expire_stages()
        self._require_not_expired(request.request_id, "inspect_staged")
        staged = self._lookup_stage(request.stage_handle_b64url, request.request_id, "inspect_staged")
        mutation, result, before, _, retire_texts = staged
        after = self._apply_to_entries([e.text for e in before.mutation_entries], retire_texts, mutation.candidate_entries)
        return w.StageInspection(
            summary=result,
            canonical_candidates=tuple(w.CandidateEntry(**{**c.__dict__, "text": c.text.strip()}) for c in mutation.candidate_entries),
            visible_before=before.mutation_entries,
            visible_after=tuple(self._stored_entries(request.target, after)),
        )

    def commit_curated(self, intent: CommitIntent) -> w.CommitResult:
        self._require_target(intent.target)
        self._expire_stages()
        receipt = self._receipts.get(intent.request_id)
        if receipt is not None and receipt[0] == intent.stage_handle_b64url:
            _, original = receipt
            return w.CommitResult(outcome="idempotent_replay", request_id=intent.request_id, tx_id=original.tx_id, snapshot=self.load_curated(intent.target), admissions=original.admissions)
        self._require_not_expired(intent.request_id, "commit_curated")
        mutation, result, before, _, retire = self._lookup_stage(intent.stage_handle_b64url, intent.request_id, "commit_curated")
        if intent.approval_binding_sha256 != result.approval_binding_sha256:
            raise ProviderError(code="approval_invalid", outcome="not_committed", details=None, operation="commit_curated")
        if intent.authorization.kind == "approved" and intent.authorization.approval_binding_sha256 != result.approval_binding_sha256:
            raise ProviderError(code="approval_invalid", outcome="not_committed", details=None, operation="commit_curated")
        if list(intent.authorized_write_scopes) != list(result.requested_write_scopes):
            raise ProviderError(code="unauthorized_scope", outcome="not_committed", details=None, operation="commit_curated")
        current = self.load_curated(intent.target)
        if current.revision != before.revision:
            raise ProviderError(code="version_conflict", outcome="not_committed", details={"current_snapshot": current.to_wire()}, operation="commit_curated")
        add = [c.text.strip() for c in mutation.candidate_entries]
        response = self._store.apply_exact_delta(intent.target, retire=list(retire), add=add)
        if not response.get("success"):
            raise BuiltinStoreError(response)
        tx_id = uuid.uuid4().hex
        snapshot = self.load_curated(intent.target)
        committed = w.CommitResult(outcome="committed_audit_clean", request_id=intent.request_id, tx_id=tx_id, snapshot=snapshot, admissions=result.admissions)
        self._stages.pop(intent.stage_handle_b64url, None)
        self._receipts[intent.request_id] = (intent.stage_handle_b64url, committed)
        return committed

    def recall_context(self, query: RecallQuery) -> w.TypedRecall:
        raise CapabilityUnavailableError("the built-in service has no recall_context")

    def capture_continuity(self, capture: ContinuityCapture) -> w.ContinuityResult:
        raise CapabilityUnavailableError("the built-in service has no capture_continuity")

    # -- helpers -------------------------------------------------------------

    def _revision(self, target: str, entries: List[str]) -> w.CompositeRevision:
        digest = hashlib.sha256(("\n§\n".join(entries)).encode("utf-8")).hexdigest()[:32]
        return w.CompositeRevision(provider_epoch="builtin", visibility_revision="builtin", scope_revisions=(w.ScopeRevision(scope=self._scope, revision=f"{target}:{digest}"),))

    def _hidden_state(self, target: str, revision: Optional[w.CompositeRevision]) -> w.HiddenPreservationState:
        seed = w.canonical_json(revision.to_wire()) if revision is not None else b""
        token = _b64url(hashlib.sha256(b"hidden:" + seed).digest())
        return w.HiddenPreservationState(target=target, complete_for_scopes=(self._scope,), opaque_state_b64url=token)

    def _provenance(self) -> w.AcceptedProvenance:
        return w.AcceptedProvenance(actor_kind="hermes", principal_id=self._identity.principal_id, logical_session_id=None, surface="builtin", source_entry_ids=(), source_commit=None, transaction_id=None)

    def _stored_entries(self, target: str, entries: List[str]) -> List[w.StoredEntry]:
        prov = self._provenance()
        return [w.StoredEntry(id=entry_id(t), text=t, origin_scope=self._scope, target=target, record_channel=_CHANNEL[target], lane=_LANE[target], lifecycle="active", policy_key=None, provenance=prov) for t in entries]

    def _snapshot(self, target: str, entries: List[str]) -> w.CuratedSnapshot:
        stored = self._stored_entries(target, entries)
        delivered = tuple(w.DeliveredEntry(id=e.id, text=e.text, origin_scope=e.origin_scope, target=e.target, record_channel=e.record_channel, lane=e.lane, delivery_tier="principal_global", policy_key=None, provenance=e.provenance) for e in stored)
        revision = self._revision(target, entries)
        limit = self._store.memory_char_limit if target == "memory" else self._store.user_char_limit
        # max_entries=0 means the native store imposes no entry-count cap: its
        # only budget is the per-target character limit, so max_entry_chars
        # equals that whole-target budget rather than a smaller per-entry cap.
        return w.CuratedSnapshot(
            api_version=1,
            status="ok",
            frozen_identity=self._wire_identity(),
            target=target,
            visible_scopes=(self._scope,),
            default_write_scope=self._scope,
            eligible_write_scopes=(self._scope,),
            complete_for_scopes=(self._scope,),
            revision=revision,
            limits=w.CuratedLimits(memory_chars=self._store.memory_char_limit, user_chars=self._store.user_char_limit, initial_general_chars=0, max_entry_chars=limit, max_entries=0),
            mutation_entries=tuple(stored),
            delivery_entries=delivered,
            hidden_preservation_state=self._hidden_state(target, revision),
        )

    def _wire_identity(self) -> w.FrozenIdentityWire:
        # CuratedSnapshot.frozen_identity is typed as wire.FrozenIdentityWire,
        # whose provider_mode is Literal["authoritative"] -- the wire schema
        # has no slot for "additive". This method exists ONLY to satisfy that
        # typed field; its output is a wire-shaped ARTIFACT, not the host
        # identity, and hosts MUST NOT branch on snapshot.frozen_identity's
        # provider_mode. The real, host-visible identity is `service.identity`
        # (FrozenMemoryIdentity: provider="builtin", provider_mode="additive"),
        # and disposition is `service.disposition` (MemoryDisposition.BUILTIN).
        data = {**self._identity.__dict__, "provider_mode": "authoritative", "opaque_binding_b64url": _b64url(bytes(32))}
        return w.FrozenIdentityWire(**data)

    def _to_wire_stage_request(self, request: MutationRequest) -> w.StageRequest:
        if request.intent is None or request.provenance is None or request.expected_revision is None or request.hidden_preservation_state is None:
            raise ProviderError(code="invalid_request", outcome="not_committed", details=None, operation="stage_curated")
        wire_request = w.StageRequest(
            expected_provider_epoch="builtin",
            frozen_identity=self._wire_identity(),
            target=request.target,
            expected_revision=request.expected_revision,
            hidden_preservation_state=request.hidden_preservation_state,
            request_id=request.request_id,
            requested_write_scopes=request.requested_write_scopes,
            intent=request.intent,
            mutation_delta=request.mutation_delta,
            candidate_entries=request.candidate_entries,
            provenance=request.provenance,
        )
        try:
            wire_request.validate()
        except w.WireError as exc:
            raise ProviderError(code="invalid_request", outcome="not_committed", details=None, operation="stage_curated") from exc
        return wire_request

    def _approval_binding(self, unsigned: w.StageResult, request: w.StageRequest) -> str:
        without_hash = {k: v for k, v in unsigned.to_wire().items() if k != "approval_binding_sha256"}
        semantics = {
            "api_version": 1,
            "expected_provider_epoch": request.expected_provider_epoch,
            "frozen_identity": request.frozen_identity.to_wire(),
            "hidden_preservation_state": request.hidden_preservation_state.to_wire(),
            "intent": request.intent.to_wire(),
            "mutation_delta": [d.to_wire() for d in request.mutation_delta],
            "import_source_identities": [c.import_source_identity.to_wire() if c.import_source_identity else None for c in request.candidate_entries],
            "provenance": request.provenance.to_wire(),
        }
        payload = {"stage_result_without_binding_hash": without_hash, "request_semantics": semantics}
        return hashlib.sha256(w.canonical_json(payload)).hexdigest()

    def _lookup_stage(self, handle: str, request_id: str, operation: str):
        staged = self._stages.get(handle)
        if staged is None or staged[1].request_id != request_id:
            receipt = self._receipts.get(request_id)
            if receipt is not None and receipt[0] == handle:
                raise ProviderError(code="stage_not_found", outcome="not_committed", details={"state": "committed", "tx_id": receipt[1].tx_id}, operation=operation)
            raise ProviderError(code="stage_not_found", outcome="not_committed", details=None, operation=operation)
        return staged

    def _expire_stages(self) -> None:
        now = _now()
        for handle, staged in list(self._stages.items()):
            if staged[3] <= now:
                self._stages.pop(handle, None)
                self._expired_requests.add(staged[1].request_id)

    def _require_not_expired(self, request_id: str, operation: str) -> None:
        if request_id in self._expired_requests:
            raise ProviderError(code="stage_expired", outcome="not_committed", details=None, operation=operation)

    @staticmethod
    def _apply_to_entries(entries: List[str], retire: Tuple[str, ...], candidates) -> List[str]:
        retired = set(retire)
        result = [t for t in entries if t not in retired]
        for cand in candidates:
            text = cand.text.strip()
            if text not in result:
                result.append(text)
        return result
