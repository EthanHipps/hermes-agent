"""Stateful in-memory fake of an authoritative memory provider (R36, S5).

The fake models **ygg's** provider-side contract (spec §9.3, §9.5, §9.6,
§9.10) behind the ``AuthoritativeBackend`` protocol so that rows 37–48 have
a fixture that behaves like ygg without being ygg. It is test support only:
it lives under ``tests/`` and must never move under ``agent/`` — a shipped
in-memory "authoritative" store would be one plugin wrapper away from being
selectable by configuration and losing data while reporting success.

What the fake is and is not:

* It is codec-honest. Every request is ``canonical_json`` → strict
  ``REQUEST_TYPES`` decode, every result is encoded and strictly decoded
  through ``RESULT_TYPES`` before the service sees it, and every typed
  failure envelope is validated through ``WireFailure`` and
  ``decode_error_details`` before it is raised. An illegal emission of the
  fake's own fails inside the fake as a ``WireError``, never in the service.
* It never natively emits ``unavailable`` or ``outcome: unknown`` (§9.3
  L1020, L1489): both are adapter-synthesized. A test that scripts either
  through ``fail_typed`` is simulating the adapter, and must say so.
* The test-supplied ``admission_classifier`` that can produce ``withheld_raw``
  is a **fixture seam** per ygg ruling R11-1 = (a) (Checkpoint A D-3.1: no v1
  wire field carries prompt-likeness and ygg may not classify content). It is
  not a §9.3 schema claim; R40 and later rows must not read it as one.
* Likewise ``secret_detector`` is a fixture seam, not §5.4: the fake runs the
  supplied detector over the canonical request container and over every
  candidate/continuity text, in §5.4's order, and nothing more.
* Delivery and recall ordering is simple and deterministic by ruling R36-K
  (general records first, then ``visible_scopes`` order, then id) and is
  **not** ygg's §8 order. No test may assert §8 ordering from this fake.

One ``FakeProviderStore`` holds the shared state behind a single
``threading.RLock``; any number of ``FakeAuthoritativeBackend`` instances (one
per service) share it, which is what cross-session CAS and idempotency need.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.memory_service import wire as w
from agent.memory_service.backend import ProviderResult
from agent.memory_service.errors import ProviderError, ProviderTransportError

MUTATION_OPERATIONS = ("stage_curated", "commit_curated")
#: Ruling R36-E: provider_epoch_changed / binding_* on a stage or commit that
#: published nothing carry not_committed (§9.3 L1020 is per-operation).
EPOCH_BINDING_OUTCOME_ON_MUTATION = "not_committed"
#: Ruling R36-F: authorized_write_scopes differs from the staged requested set.
COMMIT_SCOPE_MISMATCH_CODE = "unauthorized_scope"
#: Ruling R36-H: store_blocked refuses stage and a first commit only.
SERVE_READS_WHILE_BLOCKED = True
#: Ruling R36-J: the import-source tuple index is modelled minimally.
ENABLE_IMPORT_SOURCE_INDEX = True
#: Ruling R36-L: a failure of the store that holds staged bytes and receipts
#: (§9.5 L1539) is stage_integrity_error, the one v1 code naming stage-store
#: integrity; outcome is not_committed because nothing was published.
APPROVAL_STORE_FAILURE_CODE = "stage_integrity_error"

DEFAULT_LIMITS = w.Limits(max_request_bytes=1_000_000, max_response_bytes=4_000_000, max_stage_bytes=200_000, stage_ttl_seconds=3600, max_opaque_binding_bytes=32, max_continuity_bytes=65_536)
DEFAULT_CURATED_LIMITS = w.CuratedLimits(memory_chars=2200, user_chars=1375, initial_general_chars=3000, max_entry_chars=2200, max_entries=100)
CONTINUITY_TTL_SECONDS = 86400

_CHANNEL = {"memory": "hermes_memory", "user": "hermes_user"}
_DEFAULT_DISPOSITION = {"memory": "scoped_evidence", "user": "trusted_instruction"}
_SESSION_BOUND = ("validate_session", "load_curated", "stage_curated", "inspect_staged", "commit_curated", "recall_context", "capture_continuity")


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def timestamp(dt: datetime) -> str:
    """RFC 3339 ``Z`` form of a UTC datetime (seconds precision)."""
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _scope_key(scope: w.ScopeRef) -> Tuple[str, str]:
    return (scope.kind, scope.id)


def _walk(path: str):
    """Split ``a.b[0].c`` into ``["a", "b", 0, "c"]``."""
    parts: List[Any] = []
    for segment in path.split("."):
        head, *indexes = segment.split("[")
        if head:
            parts.append(head)
        parts.extend(int(index.rstrip("]")) for index in indexes)
    return parts


def _resolve_parent(data: Any, path: str):
    parts = _walk(path)
    node = data
    for part in parts[:-1]:
        node = node[part]
    return node, parts[-1]


def drop_key(path: str) -> Callable[[dict], dict]:
    """``corrupt_result`` helper: delete the encoded key at ``path``."""

    def mutate(data: dict) -> dict:
        parent, last = _resolve_parent(data, path)
        del parent[last]
        return data

    return mutate


def set_key(path: str, value: Any) -> Callable[[dict], dict]:
    """``corrupt_result`` helper: overwrite the encoded key at ``path``."""

    def mutate(data: dict) -> dict:
        parent, last = _resolve_parent(data, path)
        parent[last] = value
        return data

    return mutate


def approval_binding_sha256(unsigned: w.StageResult, request: w.StageRequest) -> str:
    """§9.5 L1543 binding: the StageResult without its hash plus request semantics.

    Independent of ``BuiltinMemoryService._approval_binding`` (which imports
    the native store); ``test_binding_hash_shape_matches_builtin`` pins the
    two to the same shape.
    """
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


class FakeClock:
    """Injected UTC clock: ``now()`` and ``advance(seconds)``; no test sleeps."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("FakeClock needs an aware UTC datetime")
        self._now = start.astimezone(timezone.utc)

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: int) -> None:
        self._now = self._now + timedelta(seconds=seconds)


@dataclass(frozen=True)
class ResolvedBinding:
    org_id: Optional[str]
    project_id: Optional[str]
    repo_id: Optional[str]
    workspace_id: Optional[str]
    visible_scopes: Tuple[w.ScopeRef, ...]
    default_scope: Optional[w.ScopeRef]
    eligible_scopes: Tuple[w.ScopeRef, ...]
    user_scope: w.ScopeRef
    degraded: bool


class FakeRegistry:
    """A tiny activated registry: directories, projects, workspaces, one owner.

    ``revision`` is the activated binding revision (``rev-N``); ``change()``
    models a registry activation (§7.3 L786), after which every handle bound
    at an older revision is ``binding_revoked`` on its next validation.
    """

    def __init__(self, principal_id: str = "ethan", owner_principal_id: Optional[str] = None, directories: Optional[Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]]] = None, projects: Optional[Dict[str, Optional[str]]] = None, workspaces: Optional[Dict[str, Tuple[str, Optional[str], Optional[str], Optional[str]]]] = None, grants: Tuple[Any, ...] = ()) -> None:
        self.principal_id = principal_id
        self.owner_principal_id = owner_principal_id or principal_id
        self.directories = dict(directories) if directories is not None else {"C:\\work\\repo": ("repo-1", "proj-1", None)}
        self.projects = dict(projects) if projects is not None else {"proj-1": None}
        self.workspaces = dict(workspaces) if workspaces is not None else {}
        self.grants = tuple(grants)
        self._revision = 1

    @property
    def revision(self) -> str:
        return f"rev-{self._revision}"

    def change(self) -> None:
        self._revision += 1

    # -- resolution ----------------------------------------------------------

    def _repos(self) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
        out: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        for repo_id, project_id, org_id in self.directories.values():
            if repo_id is not None:
                out[repo_id] = (project_id, org_id)
        for _, repo_id, project_id, org_id in self.workspaces.values():
            if repo_id is not None:
                out[repo_id] = (project_id, org_id)
        return out

    def _agrees(self, ctx: w.RequestedContext, repo_id, project_id, org_id) -> bool:
        return all(supplied is None or supplied == actual for supplied, actual in ((ctx.repo_id, repo_id), (ctx.project_id, project_id), (ctx.org_id, org_id)))

    def resolve(self, ctx: w.RequestedContext) -> Optional[ResolvedBinding]:
        """The authorized binding for a coherent assertion, or None (invalid)."""
        if ctx.principal_id != self.principal_id:
            return None
        if ctx.resolution_source == "directory":
            mapping = self.directories.get(ctx.canonical_directory)
            if mapping is None:
                return None
            repo_id, project_id, org_id = mapping
            if not self._agrees(ctx, repo_id, project_id, org_id):
                return None
            if ctx.workspace_id is not None and ctx.workspace_id not in self.workspaces:
                return None
        elif ctx.resolution_source == "workspace":
            mapping = self.workspaces.get(ctx.workspace_id)
            if mapping is None or mapping[0] != ctx.canonical_directory:
                return None
            _, repo_id, project_id, org_id = mapping
            if not self._agrees(ctx, repo_id, project_id, org_id):
                return None
        else:  # explicit_ids: one registered ancestor chain, or nothing at all
            if ctx.workspace_id is not None and ctx.workspace_id not in self.workspaces:
                return None
            repo_id, project_id, org_id = ctx.repo_id, ctx.project_id, ctx.org_id
            if repo_id is not None:
                known = self._repos().get(repo_id)
                if known is None or project_id != known[0] or org_id != known[1]:
                    return None
            elif project_id is not None:
                if project_id not in self.projects or org_id != self.projects[project_id]:
                    return None
            elif org_id is not None:
                if org_id not in set(self.projects.values()):
                    return None
        return self._binding(repo_id, project_id, org_id, ctx.workspace_id)

    def _binding(self, repo_id, project_id, org_id, workspace_id) -> ResolvedBinding:
        user_scope = w.ScopeRef(kind="principal_global", id=self.principal_id)
        chain: List[w.ScopeRef] = []
        if repo_id is not None:
            chain.append(w.ScopeRef(kind="repository", id=repo_id))
        if project_id is not None:
            chain.append(w.ScopeRef(kind="project", id=project_id))
        if org_id is not None:
            chain.append(w.ScopeRef(kind="organization", id=org_id))
        degraded = not chain
        visible = tuple(chain) + (user_scope,)
        default = chain[0] if chain else None
        return ResolvedBinding(org_id=org_id, project_id=project_id, repo_id=repo_id, workspace_id=workspace_id, visible_scopes=visible, default_scope=default, eligible_scopes=tuple(chain), user_scope=user_scope, degraded=degraded)


@dataclass
class HandleRecord:
    identity: w.FrozenIdentityWire
    epoch: str
    visible_scopes: Tuple[w.ScopeRef, ...]
    default_scope: Optional[w.ScopeRef]
    eligible_scopes: Tuple[w.ScopeRef, ...]
    user_scope: w.ScopeRef
    revoked: bool
    degraded: bool
    binding_revision: str


@dataclass
class FakeRecord:
    id: str
    text: str
    target: Optional[str]
    record_channel: str
    lane: str  # trusted_instruction | scoped_evidence | raw
    lifecycle: str  # active | superseded | retired
    origin_scope: w.ScopeRef
    policy_key: Optional[str]
    provenance: w.AcceptedProvenance
    epoch: str

    @property
    def hidden(self) -> bool:
        return self.lane == "raw" or self.lifecycle != "active"


@dataclass
class Stage:
    handle: str  # the session handle that staged it
    request: w.StageRequest
    result: w.StageResult
    fingerprint: bytes
    expires_at: datetime
    base_snapshot: w.CuratedSnapshot
    identity: w.FrozenIdentityWire
    retire_ids: Tuple[str, ...]
    hidden_retire_ids: Tuple[str, ...]
    affected_scopes: Tuple[w.ScopeRef, ...]


@dataclass
class StagedRequest:
    """The ``(epoch, request_id)`` index entry; it outlives the live stage."""

    fingerprint: bytes
    stage_handle: str
    result: w.StageResult


@dataclass
class Receipt:
    handle: str
    stage_handle: str
    fingerprint: bytes  # commit request fingerprint
    tx_id: str
    admissions: Tuple[w.AdmissionDecision, ...]
    identity: w.FrozenIdentityWire
    stage_result: w.StageResult
    target: str


@dataclass
class ContinuityBuffer:
    buffer_id: str
    fingerprint: bytes
    kind: str
    size: int
    expires_at: datetime


class FakeProviderStore:
    """Shared provider state, fault plan and assertion views (one RLock)."""

    def __init__(self, *, epoch: str = "ep-1", clock: Optional[FakeClock] = None, registry: Optional[FakeRegistry] = None, limits: Optional[w.Limits] = None, curated_limits: Optional[w.CuratedLimits] = None, secret_detector: Optional[Callable[[str], Optional[str]]] = None, admission_classifier: Optional[Callable[[w.CandidateEntry], str]] = None) -> None:
        self.epoch = epoch
        self.clock = clock or FakeClock(datetime(2026, 9, 17, 0, 0, 0, tzinfo=timezone.utc))
        self.registry = registry or FakeRegistry()
        self.limits = limits or DEFAULT_LIMITS
        self.curated_limits = curated_limits or DEFAULT_CURATED_LIMITS
        self.secret_detector = secret_detector
        self.admission_classifier = admission_classifier
        self.lock = threading.RLock()
        self.handles: Dict[str, HandleRecord] = {}
        self.records: Dict[str, FakeRecord] = {}
        self.scope_revisions: Dict[Tuple[str, str], int] = {}
        self.visibility_revision = 1
        self.tokens: Dict[Tuple[str, str], Tuple[str, w.CompositeRevision]] = {}
        self.stages: Dict[str, Stage] = {}
        self.stage_by_request: Dict[Tuple[str, str], StagedRequest] = {}
        self.receipts: Dict[Tuple[str, str], Receipt] = {}
        self.tombstones: Dict[Tuple[str, str], str] = {}
        self.bind_receipts: Dict[Tuple[str, str], Tuple[bytes, w.BindResult]] = {}
        self.continuity: Dict[Tuple[str, str], ContinuityBuffer] = {}
        self.continuity_directory_bytes: Optional[int] = None
        self.import_index: Dict[Tuple[Any, ...], str] = {}
        self.import_acknowledgements: List[Tuple[str, str]] = []  # (tx_id, assigned_id), body-free
        self.store_state = "normal"
        self.policy_conflicts: Dict[str, int] = {}
        self.events: List[Tuple[str, str, bool]] = []
        self.before_publish: Optional[Callable[[], None]] = None
        # fault plan
        self._transport_faults: Dict[Tuple[str, str], List[str]] = {}
        self._typed_faults: Dict[str, List[Tuple[str, Optional[str], Optional[dict]]]] = {}
        self._corruptions: Dict[str, List[Callable[[dict], dict]]] = {}
        self._epoch_overrides: Dict[str, List[str]] = {}
        self._stage_persistence_failures = 0
        self._receipt_persistence_failures = 0
        self._audit_failures = 0

    # -- knobs ---------------------------------------------------------------

    def fail_transport(self, operation: str, reason: str = "timeout", *, times: int = 1, phase: str = "before") -> None:
        if phase not in ("before", "during", "after_publish"):
            raise ValueError(f"unknown fault phase {phase!r}")
        self._transport_faults.setdefault((operation, phase), []).extend([reason] * times)

    def fail_stage_persistence(self, *, times: int = 1) -> None:
        self._stage_persistence_failures += times

    def fail_receipt_persistence(self, *, times: int = 1) -> None:
        self._receipt_persistence_failures += times

    def fail_typed(self, operation: str, code: str, *, outcome: Optional[str] = None, details: Optional[dict] = None, times: int = 1) -> None:
        self._typed_faults.setdefault(operation, []).extend([(code, outcome, details)] * times)

    def corrupt_result(self, operation: str, mutate: Callable[[dict], dict], *, times: int = 1) -> None:
        self._corruptions.setdefault(operation, []).extend([mutate] * times)

    def envelope_epoch_override(self, operation: str, epoch: str, *, times: int = 1) -> None:
        self._epoch_overrides.setdefault(operation, []).extend([epoch] * times)

    def set_epoch(self, new_epoch: str) -> None:
        """An epoch change voids every token, stage, receipt, tombstone and buffer."""
        with self.lock:
            self.epoch = new_epoch
            self.tokens.clear()
            self.stages.clear()
            self.stage_by_request.clear()
            self.receipts.clear()
            self.tombstones.clear()
            self.bind_receipts.clear()
            self.continuity.clear()
            self.import_index.clear()

    def registry_change(self) -> None:
        with self.lock:
            self.registry.change()

    def revoke(self, handle_b64url: str) -> None:
        with self.lock:
            record = self.handles[handle_b64url]
            record.revoked = True
            for key in [k for k in self.tokens if k[0] == handle_b64url]:
                del self.tokens[key]
            for stage_handle, stage in list(self.stages.items()):
                if stage.handle == handle_b64url:
                    del self.stages[stage_handle]
                    self.stage_by_request.pop((self.epoch, stage.request.request_id), None)

    def block_store(self, reason: str) -> None:
        if reason not in ("git_dirty", "maintenance", "restore_cutover"):
            raise ValueError(f"unknown store_blocked reason {reason!r}")
        self.store_state = reason

    def unblock(self) -> None:
        self.store_state = "normal"

    def reconcile(self) -> None:
        self.store_state = "normal"

    def fail_next_audit(self) -> None:
        self._audit_failures += 1

    def external_write(self, scope: w.ScopeRef, target: Optional[str], text: str, *, lane: str = "scoped_evidence", policy_key: Optional[str] = None, lifecycle: str = "active") -> FakeRecord:
        """A record published outside any session (bumps that scope's revision)."""
        channel = "general" if target is None else _CHANNEL[target]
        return self._add_record(scope, target, channel, text, lane=lane, policy_key=policy_key, lifecycle=lifecycle)

    def seed_record(self, scope: w.ScopeRef, target: str, text: str, *, lane: str = "scoped_evidence", record_channel: Optional[str] = None, policy_key: Optional[str] = None, lifecycle: str = "active") -> FakeRecord:
        channel = record_channel or _CHANNEL[target]
        return self._add_record(scope, target, channel, text, lane=lane, policy_key=policy_key, lifecycle=lifecycle)

    def seed_general(self, scope: w.ScopeRef, text: str, policy_key: Optional[str] = None, *, lane: str = "trusted_instruction") -> FakeRecord:
        return self._add_record(scope, None, "general", text, lane=lane, policy_key=policy_key, lifecycle="active")

    def hidden_change(self, scope: w.ScopeRef, target: Optional[str] = None) -> None:
        """A hidden-only change (e.g. a human adopt) still bumps the scope revision (§9.3 L1240)."""
        with self.lock:
            self._bump(scope)

    def visibility_change(self) -> None:
        with self.lock:
            self.visibility_revision += 1

    def add_policy_conflict(self, policy_key: str, count: int = 2) -> None:
        self.policy_conflicts[policy_key] = count

    # -- views ---------------------------------------------------------------

    def records_for(self, scope: w.ScopeRef, target: Optional[str], *, include_hidden: bool = False) -> List[FakeRecord]:
        with self.lock:
            return [r for r in self.records.values() if r.origin_scope == scope and r.target == target and (include_hidden or not r.hidden)]

    def token_for(self, handle: str, target: str) -> Optional[str]:
        entry = self.tokens.get((handle, target))
        return entry[0] if entry else None

    def revision_for(self, handle: str) -> w.CompositeRevision:
        return self._composite(self.handles[handle].visible_scopes)

    def stage_state(self, request_id: str) -> str:
        with self.lock:
            key = (self.epoch, request_id)
            staged = self.stage_by_request.get(key)
            if staged is not None and staged.stage_handle in self.stages:
                return "live"
            if key in self.receipts:
                return "committed"
            if key in self.tombstones:
                return "expired"
            return "absent"

    # -- internals shared by every backend ----------------------------------

    def _expire_stages(self) -> None:
        """TTL: replace expired stage bytes with a content-free tombstone (§9.3 L1356)."""
        now = self.clock.now()
        for stage_handle, stage in list(self.stages.items()):
            if stage.expires_at <= now:
                del self.stages[stage_handle]
                key = (self.epoch, stage.request.request_id)
                self.stage_by_request.pop(key, None)
                self.tombstones[key] = timestamp(stage.expires_at)

    def _add_record(self, scope, target, channel, text, *, lane, policy_key, lifecycle) -> FakeRecord:
        with self.lock:
            record = FakeRecord(id="r" + secrets.token_hex(12), text=text, target=target, record_channel=channel, lane=lane, lifecycle=lifecycle, origin_scope=scope, policy_key=policy_key, provenance=w.AcceptedProvenance(actor_kind="human", principal_id=self.registry.principal_id, logical_session_id=None, surface="fake", source_entry_ids=(), source_commit=None, transaction_id=None), epoch=self.epoch)
            self.records[record.id] = record
            self._bump(scope)
            return record

    def _bump(self, scope: w.ScopeRef) -> None:
        key = _scope_key(scope)
        self.scope_revisions[key] = self.scope_revisions.get(key, 0) + 1

    def _composite(self, visible_scopes: Tuple[w.ScopeRef, ...]) -> w.CompositeRevision:
        return w.CompositeRevision(provider_epoch=self.epoch, visibility_revision=str(self.visibility_revision), scope_revisions=tuple(w.ScopeRevision(scope=s, revision=str(self.scope_revisions.get(_scope_key(s), 0))) for s in visible_scopes))

    def _pop_transport_fault(self, operation: str, phase: str) -> Optional[str]:
        queue = self._transport_faults.get((operation, phase))
        return queue.pop(0) if queue else None

    def _pop_typed_fault(self, operation: str):
        queue = self._typed_faults.get(operation)
        return queue.pop(0) if queue else None

    def _pop_corruption(self, operation: str):
        queue = self._corruptions.get(operation)
        return queue.pop(0) if queue else None

    def _pop_epoch_override(self, operation: str) -> Optional[str]:
        queue = self._epoch_overrides.get(operation)
        return queue.pop(0) if queue else None


class FakeAuthoritativeBackend:
    """One provider transport bound to a shared :class:`FakeProviderStore`.

    Operation order inside every call: ``_receive`` (strict decode, record the
    call) → scripted transport (``before``) or typed fault → epoch check →
    handle check (contract C7 order) → operation body → ``_send`` (strict
    re-decode, optional corruption, envelope epoch, ``after_publish`` fault).
    Everything up to and including the body runs under ``store.lock``.
    """

    def __init__(self, store: FakeProviderStore, *, provider: str = "example", operations: Optional[List[str]] = None, recall: bool = True, continuity: bool = True, api_version: int = 1) -> None:
        self._store = store
        self.provider = provider
        self.recall = recall
        self.continuity = continuity
        self.api_version = api_version
        self.operations = list(operations) if operations is not None else ["bind_session", "validate_session", "load_curated", "stage_curated", "inspect_staged", "commit_curated"] + (["recall_context"] if recall else []) + (["capture_continuity"] if continuity else [])
        self.calls: List[Tuple[str, Any]] = []
        self.shutdown_calls = 0
        self._closed = False

    @property
    def store(self) -> FakeProviderStore:
        return self._store

    def count(self, operation: str) -> int:
        return sum(1 for op, _ in self.calls if op == operation)

    # -- protocol -------------------------------------------------------------

    def negotiate(self, request: w.NegotiateRequest) -> ProviderResult:
        return self._invoke("negotiate", request, self._do_negotiate)

    def bind_session(self, request: w.BindRequest) -> ProviderResult:
        return self._invoke("bind_session", request, self._do_bind)

    def validate_session(self, request: w.ValidateSessionRequest) -> ProviderResult:
        return self._invoke("validate_session", request, self._do_validate)

    def load_curated(self, request: w.LoadRequest) -> ProviderResult:
        return self._invoke("load_curated", request, self._do_load)

    def stage_curated(self, request: w.StageRequest) -> ProviderResult:
        return self._invoke("stage_curated", request, self._do_stage)

    def inspect_staged(self, request: w.InspectStageRequest) -> ProviderResult:
        return self._invoke("inspect_staged", request, self._do_inspect)

    def commit_curated(self, request: w.CommitRequest) -> ProviderResult:
        return self._invoke("commit_curated", request, self._do_commit)

    def recall_context(self, request: w.RecallRequest) -> ProviderResult:
        return self._invoke("recall_context", request, self._do_recall)

    def capture_continuity(self, request: w.ContinuityRequest) -> ProviderResult:
        return self._invoke("capture_continuity", request, self._do_continuity)

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self._closed = True

    # -- transport ------------------------------------------------------------

    def _invoke(self, operation: str, request: Any, body: Callable[[Any], Any]) -> ProviderResult:
        store = self._store
        with store.lock:
            decoded = self._receive(operation, request)
            reason = store._pop_transport_fault(operation, "before")
            if reason is not None:
                store.events.append((operation, "transport", False))
                raise ProviderTransportError(reason=reason, operation=operation, mutation_outcome_unknown=operation in MUTATION_OPERATIONS)
            fault = store._pop_typed_fault(operation)
            if fault is not None:
                code, outcome, details = fault
                self._raise(operation, code, details, outcome=outcome)
            if operation != "negotiate":
                self._check_epoch(operation, decoded.expected_provider_epoch)
            handle = self._check_handle(operation, decoded.frozen_identity) if operation in _SESSION_BOUND else None
            result = body(decoded) if handle is None else body(decoded, handle)
            store.events.append((operation, "ok", handle is not None))
        return self._send(operation, result)

    def _receive(self, operation: str, request: Any) -> Any:
        if self._closed:
            raise ProviderTransportError(reason="backend shut down", operation=operation, mutation_outcome_unknown=False)
        decoded = w.REQUEST_TYPES[operation].from_wire(json.loads(w.canonical_json(request.to_wire())))
        self.calls.append((operation, decoded))
        return decoded

    def _send(self, operation: str, result: Any) -> ProviderResult:
        store = self._store
        encoded = json.loads(w.canonical_json(result.to_wire()))
        mutate = store._pop_corruption(operation)
        if mutate is not None:
            encoded = mutate(encoded)
        decoded = w.RESULT_TYPES[operation].from_wire(encoded, "$.result")
        override = store._pop_epoch_override(operation)
        reason = store._pop_transport_fault(operation, "after_publish")
        if reason is not None:
            raise ProviderTransportError(reason=reason, operation=operation, mutation_outcome_unknown=operation in MUTATION_OPERATIONS)
        return ProviderResult(decoded, override if override is not None else store.epoch)

    def _raise(self, operation: str, code: str, details: Optional[dict], *, outcome: Optional[str] = None) -> None:
        """Build, validate and raise a typed failure envelope (never returns)."""
        if outcome is None:
            outcome = "not_committed" if operation in MUTATION_OPERATIONS else "not_applicable"
        envelope = {"wire_version": 1, "call_id": secrets.token_hex(8), "operation": operation, "provider_epoch": self._store.epoch, "status": "error", "error": {"code": code, "outcome": outcome, "details": details}}
        failure = w.WireFailure.from_wire(json.loads(w.canonical_json(envelope)))
        w.decode_error_details(failure.error.code, failure.error.details)
        self._store.events.append((operation, code, False))
        raise ProviderError(code=failure.error.code, outcome=failure.error.outcome, details=failure.error.details, operation=operation)

    def _check_epoch(self, operation: str, expected: str) -> None:
        current = self._store.epoch
        if expected != current:
            self._raise(operation, "provider_epoch_changed", {"expected_provider_epoch": expected, "current_provider_epoch": current})

    def _check_handle(self, operation: str, identity: w.FrozenIdentityWire) -> HandleRecord:
        """Contract C7 order: unknown → tampered → revoked → older revision."""
        record = self._store.handles.get(identity.opaque_binding_b64url)
        if record is None or record.identity != identity:
            self._raise(operation, "binding_invalid", None)
        if record.revoked or record.epoch != self._store.epoch or record.binding_revision != self._store.registry.revision:
            self._raise(operation, "binding_revoked", None)
        return record

    # -- operation bodies -----------------------------------------------------

    def _do_negotiate(self, request: w.NegotiateRequest) -> w.Negotiation:
        return w.Negotiation(provider=self.provider, selected_api_version=self.api_version, operations=tuple(self.operations), capabilities=w.Capabilities(recall_context=self.recall, capture_continuity=self.continuity), limits=self._store.limits)

    def _do_bind(self, request: w.BindRequest) -> w.BindResult:
        store = self._store
        fingerprint = w.canonical_json(request.to_wire())
        prior = store.bind_receipts.get((store.epoch, request.binding_request_id))
        if prior is not None:
            if prior[0] != fingerprint:
                self._raise("bind_session", "idempotency_mismatch", None)
            return prior[1]
        if request.prior_identity is not None:
            known = store.handles.get(request.prior_identity.opaque_binding_b64url)
            if known is None or known.identity != request.prior_identity:
                self._raise("bind_session", "binding_invalid", None)
        resolved = store.registry.resolve(request.requested_context)
        if resolved is None:
            self._raise("bind_session", "invalid_request", None)
        if request.prior_identity is not None:
            store.revoke(request.prior_identity.opaque_binding_b64url)
        ctx = request.requested_context
        handle = _b64url(secrets.token_bytes(32))
        identity = w.FrozenIdentityWire(provider=self.provider, provider_mode="authoritative", principal_id=ctx.principal_id, profile_id=ctx.profile_id, logical_session_id=ctx.logical_session_id, org_id=resolved.org_id, project_id=resolved.project_id, repo_id=resolved.repo_id, workspace_id=resolved.workspace_id, platform=ctx.platform, binding_revision=store.registry.revision, opaque_binding_b64url=handle)
        store.handles[handle] = HandleRecord(identity=identity, epoch=store.epoch, visible_scopes=resolved.visible_scopes, default_scope=resolved.default_scope, eligible_scopes=resolved.eligible_scopes, user_scope=resolved.user_scope, revoked=False, degraded=resolved.degraded, binding_revision=store.registry.revision)
        result = w.BindResult(frozen_identity=identity, visible_scopes=resolved.visible_scopes, memory_default_write_scope=resolved.default_scope, memory_eligible_write_scopes=resolved.eligible_scopes, user_write_scope=resolved.user_scope)
        store.bind_receipts[(store.epoch, request.binding_request_id)] = (fingerprint, result)
        return result

    def _do_validate(self, request: w.ValidateSessionRequest, handle: HandleRecord) -> w.ValidateSessionResult:
        return w.ValidateSessionResult(valid=True, frozen_identity=request.frozen_identity, visible_scopes=handle.visible_scopes)

    def _do_load(self, request: w.LoadRequest, handle: HandleRecord) -> w.CuratedSnapshot:
        self._check_policy_conflicts("load_curated")
        return self._snapshot(handle, request.target)

    # -- snapshots, records, tokens -------------------------------------------

    def _check_policy_conflicts(self, operation: str) -> None:
        conflicts = self._store.policy_conflicts
        if conflicts:
            self._raise(operation, "ambiguous_policy", {"ambiguities": [{"policy_key": key, "tier": "dependency", "candidate_count": count} for key, count in conflicts.items()]})

    def _scopes_for(self, handle: HandleRecord, target: str):
        """(status, visible, default, eligible) for one target of one handle."""
        if target == "memory":
            status = "degraded_global_only" if handle.degraded else "ok"
            return status, handle.visible_scopes, handle.default_scope, handle.eligible_scopes
        return "ok", handle.visible_scopes, handle.user_scope, (handle.user_scope,)

    def _ordered(self, records: List[FakeRecord], visible: Tuple[w.ScopeRef, ...]) -> List[FakeRecord]:
        """Ruling R36-K: visible-scope order, then id. Not ygg's §8 order."""
        order = {_scope_key(s): i for i, s in enumerate(visible)}
        return sorted((r for r in records if _scope_key(r.origin_scope) in order), key=lambda r: (order[_scope_key(r.origin_scope)], r.id))

    @staticmethod
    def _stored(record: FakeRecord) -> w.StoredEntry:
        return w.StoredEntry(id=record.id, text=record.text, origin_scope=record.origin_scope, target=record.target, record_channel=record.record_channel, lane=record.lane, lifecycle="active", policy_key=record.policy_key, provenance=record.provenance)

    @staticmethod
    def _delivered(record: FakeRecord) -> w.DeliveredEntry:
        return w.DeliveredEntry(id=record.id, text=record.text, origin_scope=record.origin_scope, target=record.target, record_channel=record.record_channel, lane=record.lane, delivery_tier=record.origin_scope.kind, policy_key=record.policy_key, provenance=record.provenance)

    def _mutation_records(self, handle: HandleRecord, target: str) -> List[FakeRecord]:
        _, visible, _, eligible = self._scopes_for(handle, target)
        channel = _CHANNEL[target]
        addressable = {_scope_key(s) for s in eligible}
        return self._ordered([r for r in self._store.records.values() if r.record_channel == channel and not r.hidden and _scope_key(r.origin_scope) in addressable], visible)

    def _delivery_records(self, handle: HandleRecord, target: str) -> List[FakeRecord]:
        _, visible, _, _ = self._scopes_for(handle, target)
        active = [r for r in self._store.records.values() if not r.hidden]
        if target == "memory":
            general = self._ordered([r for r in active if r.record_channel == "general"], visible)
            return general + self._ordered([r for r in active if r.record_channel == "hermes_memory"], visible)
        return self._ordered([r for r in active if r.record_channel == "hermes_user"], visible)

    def _mint_or_reuse_token(self, handle: HandleRecord, target: str, revision: w.CompositeRevision) -> str:
        key = (handle.identity.opaque_binding_b64url, target)
        current = self._store.tokens.get(key)
        if current is not None and current[1] == revision:
            return current[0]
        token = _b64url(secrets.token_bytes(32))
        self._store.tokens[key] = (token, revision)
        return token

    def _snapshot(self, handle: HandleRecord, target: str) -> w.CuratedSnapshot:
        status, visible, default, eligible = self._scopes_for(handle, target)
        revision = self._store._composite(visible)
        token = self._mint_or_reuse_token(handle, target, revision)
        return w.CuratedSnapshot(
            api_version=1,
            status=status,
            frozen_identity=handle.identity,
            target=target,
            visible_scopes=visible,
            default_write_scope=default,
            eligible_write_scopes=eligible,
            complete_for_scopes=eligible,
            revision=revision,
            limits=self._store.curated_limits,
            mutation_entries=tuple(self._stored(r) for r in self._mutation_records(handle, target)),
            delivery_entries=tuple(self._delivered(r) for r in self._delivery_records(handle, target)),
            hidden_preservation_state=w.HiddenPreservationState(target=target, complete_for_scopes=eligible, opaque_state_b64url=token),
        )

    def _do_stage(self, request: w.StageRequest, handle: HandleRecord) -> w.StageResult:
        op = "stage_curated"
        store = self._store
        target = request.target
        store._expire_stages()
        key = (store.epoch, request.request_id)
        fingerprint = w.canonical_json(request.to_wire())
        fingerprint_digest = hashlib.sha256(fingerprint).digest()
        # replay lookup by (epoch, request_id): reads only, writes nothing
        if key in store.tombstones:
            self._raise(op, "stage_expired", None)
        staged = store.stage_by_request.get(key)
        if staged is not None:
            if staged.fingerprint != fingerprint_digest:
                self._raise(op, "idempotency_mismatch", None)
            return staged.result
        if store.store_state != "normal":
            self._raise(op, "store_blocked", {"reason": store.store_state})
        self._scan(op, fingerprint, [c.text for c in request.candidate_entries])
        current = self._snapshot(handle, target)
        # limits
        limits, curated = store.limits, store.curated_limits
        if len(fingerprint) > limits.max_request_bytes:
            self._raise(op, "limit_exceeded", {"limit": "max_request_bytes"})
        if sum(len(c.text.encode("utf-8")) for c in request.candidate_entries) > limits.max_stage_bytes:
            self._raise(op, "limit_exceeded", {"limit": "max_stage_bytes"})
        if any(len(c.text) > curated.max_entry_chars for c in request.candidate_entries):
            self._raise(op, "limit_exceeded", {"limit": "max_entry_chars"})
        retiring = sum(1 for d in request.mutation_delta if d.action in ("retire", "supersede"))
        if len(current.mutation_entries) - retiring + len(request.candidate_entries) > curated.max_entries:
            self._raise(op, "limit_exceeded", {"limit": "max_entries"})
        # scope resolution and eligibility (§9.2 L970, §9.3 L1289)
        if target == "memory" and handle.degraded:
            self._raise(op, "scope_unresolved", None)
        _, _, default, eligible = self._scopes_for(handle, target)
        eligible_keys = {_scope_key(s) for s in eligible}
        affected: List[w.ScopeRef] = []

        def touch(scope: w.ScopeRef) -> None:
            if scope not in affected:
                affected.append(scope)

        for cand in request.candidate_entries:
            if _scope_key(cand.destination_scope) not in eligible_keys:
                self._raise(op, "unauthorized_scope", None)
        reset_scopes = tuple(request.intent.reset_scopes or ()) if request.intent.kind == "reset" else ()
        for scope in reset_scopes:
            if _scope_key(scope) not in eligible_keys:
                self._raise(op, "unauthorized_scope", None)
            touch(scope)
        # CAS before token (contract C2, D-R10-2)
        if request.expected_revision != current.revision:
            self._raise(op, "version_conflict", {"current_snapshot": current.to_wire()})
        if request.hidden_preservation_state != current.hidden_preservation_state:
            self._raise(op, "invalid_request", None)
        # delta IDs are confined to the exact snapshot (§9.3 L1293)
        by_id = {e.id: e for e in current.mutation_entries}
        by_ref = {c.client_ref: c for c in request.candidate_entries}
        superseded: Dict[str, str] = {}
        retire_ids: List[str] = []
        for item in request.mutation_delta:
            if item.action == "add":
                touch(by_ref[item.client_ref].destination_scope)
                continue
            old_id = item.record_id if item.action == "retire" else item.old_record_id if item.action == "supersede" else None
            if old_id is None:
                continue
            old = by_id.get(old_id)
            if old is None:
                self._raise(op, "invalid_request", None)
            if item.action == "supersede":
                if by_ref[item.replacement_client_ref].destination_scope != old.origin_scope:
                    self._raise(op, "invalid_request", None)
                superseded[item.replacement_client_ref] = old_id
            retire_ids.append(old_id)
            touch(old.origin_scope)
        # reset: mechanically enumerate hidden state, body-free (§9.3 L1293)
        hidden_retire_ids: List[str] = []
        hidden_effects: List[w.HiddenEffect] = []
        if reset_scopes:
            reset_keys = {_scope_key(s) for s in reset_scopes}
            counts: Dict[Tuple[Any, ...], int] = {}
            for record in store.records.values():
                if record.epoch != store.epoch or record.target != target or record.lifecycle == "retired" or _scope_key(record.origin_scope) not in reset_keys:
                    continue
                if record.hidden:
                    hidden_retire_ids.append(record.id)
                    group = (record.origin_scope, record.record_channel, record.lane)
                    counts[group] = counts.get(group, 0) + 1
                else:
                    retire_ids.append(record.id)
            order = {_scope_key(s): i for i, s in enumerate(reset_scopes)}
            for scope, channel, lane in sorted(counts, key=lambda g: (order[_scope_key(g[0])], g[1], g[2])):
                hidden_effects.append(w.HiddenEffect(scope=scope, target=target, record_channel=channel, lane=lane, action="retire", count=counts[(scope, channel, lane)]))
        # admissions: one decision per candidate
        admissions: List[w.AdmissionDecision] = []
        hashes: List[w.CandidateHash] = []
        new_import_tuples: set = set()
        for cand in request.candidate_entries:
            sha = hashlib.sha256(cand.text.encode("utf-8")).hexdigest()
            hashes.append(w.CandidateHash(client_ref=cand.client_ref, canonical_sha256=sha))
            disposition = store.admission_classifier(cand) if store.admission_classifier else _DEFAULT_DISPOSITION[target]
            effect = "create_record"
            assigned = "r" + secrets.token_hex(12)
            if request.intent.kind == "import" and ENABLE_IMPORT_SOURCE_INDEX:
                import_tuple = self._import_tuple(handle, cand, sha)
                reused = store.import_index.get(import_tuple)
                if reused is not None:
                    effect, assigned = "reuse_existing_import", reused
                elif import_tuple in new_import_tuples:
                    self._raise(op, "invalid_request", None)
                else:
                    new_import_tuples.add(import_tuple)
            old_id = superseded.get(cand.client_ref)
            if target == "user" and old_id is None and cand.proposed_policy_key is not None:
                self._raise(op, "invalid_request", None)
            if disposition == "withheld_raw":
                policy_key = None
            elif target == "user":
                inherited = store.records[old_id].policy_key if old_id else None
                policy_key = cand.proposed_policy_key or inherited or f"profile:{assigned}"
            else:
                policy_key = cand.proposed_policy_key
            admissions.append(w.AdmissionDecision(client_ref=cand.client_ref, assigned_id=assigned, target=target, record_channel=_CHANNEL[target], origin_scope=cand.destination_scope, disposition=disposition, publication_effect=effect, superseded_id=old_id, policy_key=policy_key))
        if tuple(affected) != request.requested_write_scopes:
            self._raise(op, "invalid_request", None)
        # approval requirements, mechanically and in §9.3 L1326 order
        requirements: List[str] = []
        if target == "user":
            requirements.append("target_user")
        if any(scope != default for scope in affected):
            requirements.append("non_default_scope")
        for kind in ("bulk_edit", "reset", "import"):
            if request.intent.kind == kind:
                requirements.append(kind)
        if request.provenance.threat_decision_id is not None:
            requirements.append("threat")
        expires_at = store.clock.now() + timedelta(seconds=limits.stage_ttl_seconds)
        stage_handle = _b64url(secrets.token_bytes(32))
        unsigned = w.StageResult(stage_handle_b64url=stage_handle, request_id=request.request_id, target=target, expected_revision=request.expected_revision, requested_write_scopes=request.requested_write_scopes, eligible_write_scopes=eligible, expires_at=timestamp(expires_at), approval_binding_sha256="0" * 64, approval_requirements=tuple(requirements), candidate_hashes=tuple(hashes), admissions=tuple(admissions), hidden_effects=tuple(hidden_effects))
        result = replace(unsigned, approval_binding_sha256=approval_binding_sha256(unsigned, request))
        # persist: the one point where request_id-keyed state is written (ruling R36-L)
        if store._stage_persistence_failures > 0:
            store._stage_persistence_failures -= 1
            self._raise(op, APPROVAL_STORE_FAILURE_CODE, None)
        store.stages[stage_handle] = Stage(handle=handle.identity.opaque_binding_b64url, request=request, result=result, fingerprint=fingerprint_digest, expires_at=expires_at, base_snapshot=current, identity=request.frozen_identity, retire_ids=tuple(retire_ids), hidden_retire_ids=tuple(hidden_retire_ids), affected_scopes=tuple(affected))
        store.stage_by_request[key] = StagedRequest(fingerprint=fingerprint_digest, stage_handle=stage_handle, result=result)
        self._fault_during(op)
        return result

    def _import_tuple(self, handle: HandleRecord, cand: w.CandidateEntry, sha: str) -> Tuple[Any, ...]:
        """§9.3 L1335 uniqueness tuple."""
        ident = cand.import_source_identity
        return (self._store.epoch, handle.identity.principal_id, ident.source_kind, ident.parser_version, ident.source_id, ident.item_key, cand.target, _scope_key(cand.destination_scope), sha)

    def _scan(self, operation: str, container: bytes, texts: List[str]) -> None:
        """§5.4 order: the whole container first, then each content field (fixture seam)."""
        detector = self._store.secret_detector
        if detector is None:
            return
        for candidate in [container.decode("utf-8"), *texts]:
            code = detector(candidate)
            if code:
                self._raise(operation, "secret_rejected", {"event_id": _b64url(secrets.token_bytes(16)), "detector_code": code})

    def _fault_during(self, operation: str) -> None:
        """A crash after the durable write and before the response is built."""
        reason = self._store._pop_transport_fault(operation, "during")
        if reason is not None:
            self._store.events.append((operation, "transport", True))
            raise ProviderTransportError(reason=reason, operation=operation, mutation_outcome_unknown=operation in MUTATION_OPERATIONS)

    def _do_inspect(self, request: w.InspectStageRequest, handle: HandleRecord) -> w.StageInspection:
        op = "inspect_staged"
        self._store._expire_stages()
        stage = self._lookup_stage(op, request.stage_handle_b64url, request.request_id, request.frozen_identity, request.target)
        return w.StageInspection(summary=stage.result, canonical_candidates=stage.request.candidate_entries, visible_before=stage.base_snapshot.mutation_entries, visible_after=self._projected(stage))

    def _lookup_stage(self, operation: str, stage_handle: str, request_id: str, identity: w.FrozenIdentityWire, target: str) -> Stage:
        """The three stage states (§9.3 L1356) plus D-R36-7 identity binding."""
        store = self._store
        key = (store.epoch, request_id)
        if key in store.tombstones:
            self._raise(operation, "stage_expired", None)
        stage = store.stages.get(stage_handle)
        if stage is None or stage.request.request_id != request_id:
            receipt = store.receipts.get(key)
            if receipt is not None and receipt.stage_handle == stage_handle:
                self._raise(operation, "stage_not_found", {"state": "committed", "tx_id": receipt.tx_id})
            self._raise(operation, "stage_not_found", None)
        if stage.identity != identity or stage.request.target != target:
            self._raise(operation, "invalid_request", None)
        return stage

    @staticmethod
    def _accepted_provenance(request: w.StageRequest, tx_id: Optional[str]) -> w.AcceptedProvenance:
        p = request.provenance
        return w.AcceptedProvenance(actor_kind=p.actor_kind, principal_id=p.principal_id, logical_session_id=p.logical_session_id, surface=p.initiating_surface, source_entry_ids=p.source_entry_ids, source_commit=p.source_commit, transaction_id=tx_id)

    def _projected(self, stage: Stage) -> Tuple[w.StoredEntry, ...]:
        """visible_after: the base mutation list minus retirements plus new visible records."""
        retired = set(stage.retire_ids)
        after = [e for e in stage.base_snapshot.mutation_entries if e.id not in retired]
        by_ref = {c.client_ref: c for c in stage.request.candidate_entries}
        provenance = self._accepted_provenance(stage.request, None)
        for admission in stage.result.admissions:
            if admission.disposition == "withheld_raw" or admission.publication_effect != "create_record":
                continue
            cand = by_ref[admission.client_ref]
            after.append(w.StoredEntry(id=admission.assigned_id, text=cand.text, origin_scope=admission.origin_scope, target=admission.target, record_channel=admission.record_channel, lane=admission.disposition, lifecycle="active", policy_key=admission.policy_key, provenance=provenance))
        return tuple(after)

    def _do_commit(self, request: w.CommitRequest, handle: HandleRecord) -> w.CommitResult:
        op = "commit_curated"
        store = self._store
        store._expire_stages()
        key = (store.epoch, request.request_id)
        fingerprint = w.canonical_json(request.to_wire())
        # receipt replay: served even while blocked (ruling R36-H); writes nothing
        receipt = store.receipts.get(key)
        if receipt is not None:
            if receipt.fingerprint != fingerprint:
                self._raise(op, "idempotency_mismatch", None)
            return w.CommitResult(outcome="idempotent_replay", request_id=request.request_id, tx_id=receipt.tx_id, snapshot=self._snapshot(handle, receipt.target), admissions=receipt.admissions)
        stage = self._lookup_stage(op, request.stage_handle_b64url, request.request_id, request.frozen_identity, request.target)
        if store.store_state != "normal":
            self._raise(op, "store_blocked", {"reason": store.store_state})
        # approval codes (§9.3 L1394)
        result = stage.result
        auth = request.authorization
        if result.approval_requirements and auth.kind == "not_required":
            self._raise(op, "approval_required", None)
        if request.approval_binding_sha256 != result.approval_binding_sha256:
            self._raise(op, "approval_invalid", None)
        if auth.kind == "approved":
            if auth.approval_binding_sha256 != result.approval_binding_sha256 or auth.approved_by_principal_id != store.registry.owner_principal_id:
                self._raise(op, "approval_invalid", None)
            expires = _parse_timestamp(auth.expires_at)
            if expires > stage.expires_at:
                self._raise(op, "approval_invalid", None)
            if expires < store.clock.now():
                self._raise(op, "approval_expired", None)
        # scopes (§9.3 L1392; ruling R36-F for the code)
        _, _, _, eligible = self._scopes_for(handle, request.target)
        authorized = list(request.authorized_write_scopes)
        if authorized != list(result.requested_write_scopes):
            self._raise(op, COMMIT_SCOPE_MISMATCH_CODE, None)
        eligible_keys = {_scope_key(s) for s in eligible}
        authorized_keys = {_scope_key(s) for s in authorized}
        if not authorized_keys <= eligible_keys or any(_scope_key(s) not in authorized_keys for s in stage.affected_scopes):
            self._raise(op, "unauthorized_scope", None)
        # CAS re-check inside the same critical section as publication (D-R36-11)
        current = self._snapshot(handle, request.target)
        if current.revision != stage.request.expected_revision:
            self._raise(op, "version_conflict", {"current_snapshot": current.to_wire()})
        if store.before_publish is not None:
            store.before_publish()
        # receipt-store failure fires before anything durable (ruling R36-L)
        if store._receipt_persistence_failures > 0:
            store._receipt_persistence_failures -= 1
            self._raise(op, APPROVAL_STORE_FAILURE_CODE, None)
        tx_id = "tx-" + secrets.token_hex(8)
        self._publish(stage, tx_id, fingerprint)
        self._fault_during(op)
        if store._audit_failures > 0:
            store._audit_failures -= 1
            store.store_state = "git_dirty"
            outcome = "committed_audit_pending"
        else:
            outcome = "committed_audit_clean"
        return w.CommitResult(outcome=outcome, request_id=request.request_id, tx_id=tx_id, snapshot=self._snapshot(handle, request.target), admissions=result.admissions)

    def _publish(self, stage: Stage, tx_id: str, commit_fingerprint: bytes) -> None:
        """Publish the exact staged bytes and write the receipt as one step under the lock."""
        store = self._store
        request, result = stage.request, stage.result
        superseded_ids = {a.superseded_id for a in result.admissions if a.superseded_id is not None}
        for record_id in stage.retire_ids:
            store.records[record_id].lifecycle = "superseded" if record_id in superseded_ids else "retired"
        for record_id in stage.hidden_retire_ids:
            store.records[record_id].lifecycle = "retired"
        by_ref = {c.client_ref: c for c in request.candidate_entries}
        provenance = self._accepted_provenance(request, tx_id)
        for admission in result.admissions:
            if admission.publication_effect == "reuse_existing_import":
                store.import_acknowledgements.append((tx_id, admission.assigned_id))
                continue
            cand = by_ref[admission.client_ref]
            lane = "raw" if admission.disposition == "withheld_raw" else admission.disposition
            store.records[admission.assigned_id] = FakeRecord(id=admission.assigned_id, text=cand.text, target=admission.target, record_channel=admission.record_channel, lane=lane, lifecycle="active", origin_scope=admission.origin_scope, policy_key=admission.policy_key, provenance=provenance, epoch=store.epoch)
            if request.intent.kind == "import" and ENABLE_IMPORT_SOURCE_INDEX:
                sha = next(h.canonical_sha256 for h in result.candidate_hashes if h.client_ref == admission.client_ref)
                store.import_index[self._import_tuple(store.handles[stage.handle], cand, sha)] = admission.assigned_id
        for scope in stage.affected_scopes:
            store._bump(scope)
        store.receipts[(store.epoch, request.request_id)] = Receipt(handle=stage.handle, stage_handle=result.stage_handle_b64url, fingerprint=commit_fingerprint, tx_id=tx_id, admissions=result.admissions, identity=stage.identity, stage_result=result, target=request.target)
        del store.stages[result.stage_handle_b64url]

    def _do_recall(self, request: w.RecallRequest, handle: HandleRecord) -> w.TypedRecall:
        """Read-only, non-durable, target-specific (§9.3 L1424); ruling R36-K ordering."""
        op = "recall_context"
        store = self._store
        self._check_policy_conflicts(op)
        _, visible, _, _ = self._scopes_for(handle, request.target)
        if request.source_revision != store._composite(visible):
            self._raise(op, "version_conflict", None)
        visible_keys = {_scope_key(s) for s in visible}
        channels = set(request.include_channels)
        excluded = set(request.exclude_entry_ids)
        needle = request.query.lower()
        matches = sorted((r for r in store.records.values() if not r.hidden and r.record_channel in channels and _scope_key(r.origin_scope) in visible_keys and r.id not in excluded and (needle == "" or needle in r.text.lower())), key=lambda r: r.id)
        budget = request.budget
        trusted: List[w.DeliveredEntry] = []
        evidence: List[w.DeliveredEntry] = []
        used = {"trusted_instruction": 0, "scoped_evidence": 0}
        caps = {"trusted_instruction": budget.trusted_chars, "scoped_evidence": budget.evidence_chars}
        for record in matches:
            if len(trusted) + len(evidence) >= budget.max_entries:
                break
            if used[record.lane] + len(record.text) > caps[record.lane]:
                continue
            used[record.lane] += len(record.text)
            (trusted if record.lane == "trusted_instruction" else evidence).append(self._delivered(record))
        return w.TypedRecall(frozen_identity=request.frozen_identity, target=request.target, source_revision=request.source_revision, trusted_instructions=tuple(trusted), scoped_evidence=tuple(evidence))

    def _do_continuity(self, request: w.ContinuityRequest, handle: HandleRecord) -> w.ContinuityResult:
        """One temporary raw buffer (§9.3 L1445): no lock, revision, approval or acknowledgement effect."""
        op = "capture_continuity"
        store = self._store
        now = store.clock.now()
        key = (store.epoch, request.request_id)
        fingerprint = w.canonical_json(request.to_wire())
        existing = store.continuity.get(key)
        if existing is not None:
            if existing.expires_at <= now:
                self._raise(op, "stage_expired", None)
            if existing.fingerprint != fingerprint:
                self._raise(op, "idempotency_mismatch", None)
            return w.ContinuityResult(buffer_id=existing.buffer_id, request_id=request.request_id, kind=existing.kind, expires_at=timestamp(existing.expires_at), outcome="idempotent_replay")
        self._scan(op, fingerprint, [request.text])
        size = len(request.text.encode("utf-8"))
        if size > store.limits.max_continuity_bytes:
            self._raise(op, "limit_exceeded", {"limit": "continuity_buffer_bytes"})
        cap = store.continuity_directory_bytes
        if cap is not None and sum(b.size for b in store.continuity.values() if b.expires_at > now) + size > cap:
            self._raise(op, "limit_exceeded", {"limit": "continuity_directory_bytes"})
        buffer = ContinuityBuffer(buffer_id="buf-" + secrets.token_hex(8), fingerprint=fingerprint, kind=request.kind, size=size, expires_at=now + timedelta(seconds=CONTINUITY_TTL_SECONDS))
        store.continuity[key] = buffer
        return w.ContinuityResult(buffer_id=buffer.buffer_id, request_id=request.request_id, kind=request.kind, expires_at=timestamp(buffer.expires_at), outcome="stored")


def fake_backend_factory(store: FakeProviderStore, **backend_kwargs) -> Callable[[Any], FakeAuthoritativeBackend]:
    """A ``backend_factory`` for ``select_memory_service`` (D-R35-4 shape)."""

    def factory(cfg) -> FakeAuthoritativeBackend:
        kwargs = dict(backend_kwargs)
        kwargs.setdefault("provider", cfg.provider)
        return FakeAuthoritativeBackend(store, **kwargs)

    return factory


__all__ = [
    "APPROVAL_STORE_FAILURE_CODE",
    "COMMIT_SCOPE_MISMATCH_CODE",
    "CONTINUITY_TTL_SECONDS",
    "DEFAULT_CURATED_LIMITS",
    "DEFAULT_LIMITS",
    "ENABLE_IMPORT_SOURCE_INDEX",
    "EPOCH_BINDING_OUTCOME_ON_MUTATION",
    "MUTATION_OPERATIONS",
    "SERVE_READS_WHILE_BLOCKED",
    "ContinuityBuffer",
    "FakeAuthoritativeBackend",
    "FakeClock",
    "FakeProviderStore",
    "FakeRecord",
    "FakeRegistry",
    "HandleRecord",
    "Receipt",
    "ResolvedBinding",
    "Stage",
    "approval_binding_sha256",
    "drop_key",
    "fake_backend_factory",
    "set_key",
    "timestamp",
]
