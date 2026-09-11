"""Provider API v1 wire schemas (spec §9.3) as strict host-side types.

Every schema is a frozen dataclass whose fields are exactly the §9.3 fields
in §9.3 order. One generic codec decodes JSON objects into them (unknown
fields, missing fields, wrong types, unknown enum values, out-of-range
integers are all :class:`WireError`) and encodes them back. Per-schema
invariants live in ``validate()``.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, Union, get_args, get_origin, get_type_hints

logger = logging.getLogger(__name__)

MAX_SAFE_INTEGER = 9_007_199_254_740_991

_B64URL_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_SURFACE_RE = re.compile(r"^[a-z0-9_.-]{1,64}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_ID_RE = re.compile(r"^[a-z0-9-]{1,64}$")
_ITEM_KEY_RE = re.compile(r"^[A-Za-z0-9._/-]{1,256}$")
_RFC3339_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")


class WireError(ValueError):
    """Wire data violates §9.3. ``path`` locates the offending field."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message


def canonical_json(obj: Any) -> bytes:
    """RFC 8785 bytes for the integer-only, ASCII-key subset (§9.3).

    ``json.dumps`` with sorted keys, no whitespace, and ``ensure_ascii=False``
    produces exactly the JCS form for this subset: ASCII keys sort identically
    by code point and by UTF-16 unit, control characters use the RFC's short
    escapes and lowercase ``\\u00xx``, and integers print in plain decimal.
    Floats and non-ASCII keys are rejected, never canonicalized.
    """

    def check(value: Any, path: str) -> None:
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return
        if isinstance(value, float):
            raise ValueError(f"{path}: floating-point values are outside the canonical subset")
        if isinstance(value, int):
            if abs(value) > MAX_SAFE_INTEGER:
                raise ValueError(f"{path}: integer outside the ±(2^53-1) range")
            return
        if isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                check(item, f"{path}/{i}")
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str) or not key.isascii():
                    raise ValueError(f"{path}: non-ASCII or non-string object key")
                check(item, f"{path}/{key}")
            return
        raise ValueError(f"{path}: unsupported type {type(value).__name__}")

    check(obj, "$")
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _is_optional(tp: Any) -> Tuple[bool, Any]:
    if get_origin(tp) is Union:
        args = get_args(tp)
        inner = [a for a in args if a is not type(None)]
        if len(inner) == 1 and len(args) == 2:
            return True, inner[0]
    return False, tp


def _decode(tp: Any, value: Any, path: str) -> Any:
    optional, inner = _is_optional(tp)
    if value is None:
        if optional:
            return None
        raise WireError(path, "must not be null")
    origin = get_origin(inner)
    if isinstance(inner, type) and issubclass(inner, WireModel):
        return inner.from_wire(value, path)
    if inner is str:
        if not isinstance(value, str):
            raise WireError(path, "must be a string")
        return value
    if inner is bool:
        if not isinstance(value, bool):
            raise WireError(path, "must be a boolean")
        return value
    if inner is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise WireError(path, "must be an integer")
        if abs(value) > MAX_SAFE_INTEGER:
            raise WireError(path, "integer outside the 2^53 safe range")
        return value
    if inner is dict or origin is dict:
        if not isinstance(value, dict):
            raise WireError(path, "must be an object")
        return dict(value)
    if origin is Literal:
        options = get_args(inner)
        message = "must be one of " + ", ".join(repr(o) for o in options)
        bool_options = any(isinstance(o, bool) for o in options)
        if isinstance(value, bool) != bool_options:
            raise WireError(path, message)  # 1 is not True and True is not 1 on the wire
        if value not in options:
            raise WireError(path, message)
        return value
    if origin in (list, tuple, List, Tuple):
        if not isinstance(value, list):
            raise WireError(path, "must be an array")
        item_tp = get_args(inner)[0]
        return tuple(_decode(item_tp, item, f"{path}[{i}]") for i, item in enumerate(value))
    raise TypeError(f"unsupported wire annotation {tp!r} at {path}")


def _encode(value: Any) -> Any:
    if isinstance(value, WireModel):
        return value.to_wire()
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    return value


class WireModel:
    """Base for every §9.3 schema. Subclasses are frozen dataclasses."""

    _hints_cache: ClassVar[Dict[type, Dict[str, Any]]] = {}

    @classmethod
    def _hints(cls) -> Dict[str, Any]:
        cached = WireModel._hints_cache.get(cls)
        if cached is None:
            cached = get_type_hints(cls)
            WireModel._hints_cache[cls] = cached
        return cached

    @classmethod
    def from_wire(cls, data: Any, path: str = "$"):
        if not isinstance(data, dict):
            raise WireError(path, "must be an object")
        hints = cls._hints()
        names = [f.name for f in fields(cls)]  # type: ignore[arg-type]
        unknown = [k for k in data if k not in names]
        if unknown:
            raise WireError(path, f"unknown field {unknown[0]!r}")
        kwargs = {}
        for name in names:
            if name not in data:
                raise WireError(f"{path}.{name}", "missing required field")
            kwargs[name] = _decode(hints[name], data[name], f"{path}.{name}")
        obj = cls(**kwargs)  # type: ignore[call-arg]
        obj.validate(path)
        return obj

    def to_wire(self) -> Dict[str, Any]:
        return {f.name: _encode(getattr(self, f.name)) for f in fields(self)}  # type: ignore[arg-type]

    def validate(self, path: str = "$") -> None:
        """Per-schema invariants; raise :class:`WireError`."""


def _require(cond: bool, path: str, message: str) -> None:
    if not cond:
        raise WireError(path, message)


def _require_b64url(value: str, path: str, exact_bytes: Optional[int] = None) -> None:
    _require(bool(_B64URL_RE.match(value)), path, "must be unpadded base64url")
    if exact_bytes is not None:
        padded = value + "=" * (-len(value) % 4)
        try:
            raw = base64.urlsafe_b64decode(padded)
        except Exception:
            raise WireError(path, "must be unpadded base64url") from None
        _require(len(raw) == exact_bytes, path, f"must decode to exactly {exact_bytes} bytes")


def _require_surface(value: str, path: str) -> None:
    _require(bool(_SURFACE_RE.match(value)), path, "must be 1-64 characters of [a-z0-9_.-]")


def _require_nonneg(value: int, path: str) -> None:
    _require(value >= 0, path, "must be a non-negative integer")


def _require_positive(value: int, path: str) -> None:
    _require(value > 0, path, "must be a positive integer")


def _require_timestamp(value: str, path: str) -> None:
    _require(bool(_RFC3339_RE.match(value)), path, "must be an RFC 3339 UTC timestamp")
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise WireError(path, "must be a valid RFC 3339 UTC timestamp") from None
    _require(dt.utcoffset() == timedelta(0), path, "must be a UTC timestamp (Z or +00:00)")


def _same_scopes(a, b) -> bool:
    return [(s.kind, s.id) for s in a] == [(s.kind, s.id) for s in b]


Operation = Literal[
    "negotiate", "bind_session", "validate_session", "load_curated",
    "stage_curated", "inspect_staged", "commit_curated", "recall_context", "capture_continuity",
]
OPERATIONS: Tuple[str, ...] = get_args(Operation)
ErrorCode = Literal[
    "incompatible_api", "invalid_request", "unavailable",
    "provider_epoch_changed", "binding_invalid", "binding_revoked",
    "ambiguous_policy",
    "scope_unresolved", "unauthorized_scope", "store_blocked",
    "approval_required", "approval_invalid", "approval_expired",
    "secret_rejected",
    "stage_not_found", "stage_expired", "stage_integrity_error",
    "version_conflict", "idempotency_mismatch", "limit_exceeded",
]
ERROR_CODES: Tuple[str, ...] = get_args(ErrorCode)
Outcome = Literal["not_applicable", "not_committed", "unknown"]
ScopeKind = Literal["principal_global", "organization", "project", "repository"]
Target = Literal["memory", "user"]
RecordChannel = Literal["general", "hermes_memory", "hermes_user"]
HermesChannel = Literal["hermes_memory", "hermes_user"]
Lane = Literal["trusted_instruction", "scoped_evidence"]
DeliveryTier = Literal["repository", "project", "organization", "principal_global", "dependency"]
ActorKind = Literal["human", "hermes", "claude_code", "codex", "import", "scan", "reflection"]
MutationActorKind = Literal["human", "hermes", "claude_code", "codex", "import"]
SourceKind = Literal["native_memory", "legacy_archive"]
ParserVersion = Literal["hermes-native-v0.20.6", "hermes-legacy-archive-v1"]
ResolutionSource = Literal["explicit_ids", "workspace", "directory"]
BindIntent = Literal["new_session", "session_reset", "explicit_rebind"]
SnapshotStatus = Literal["ok", "degraded_global_only"]
IntentKind = Literal["add", "replace", "remove", "bulk_edit", "reset", "import"]
DeltaAction = Literal["add", "retire", "supersede"]
Disposition = Literal["trusted_instruction", "scoped_evidence", "withheld_raw"]
PublicationEffect = Literal["create_record", "reuse_existing_import"]
HiddenLane = Literal["trusted_instruction", "scoped_evidence", "raw"]
ApprovalRequirement = Literal["target_user", "non_default_scope", "bulk_edit", "reset", "import", "threat"]
CommitOutcome = Literal["committed_audit_clean", "committed_audit_pending", "idempotent_replay"]
ContinuityKind = Literal["compression_snapshot", "turn_excerpt"]
ContinuityOutcome = Literal["stored", "idempotent_replay"]
StoreBlockedReason = Literal["git_dirty", "maintenance", "restore_cutover"]
LimitName = Literal[
    "max_request_bytes", "max_response_bytes", "max_stage_bytes",
    "max_entry_chars", "max_entries", "max_opaque_binding_bytes",
    "continuity_buffer_bytes", "continuity_directory_bytes",
]


@dataclass(frozen=True)
class ScopeRef(WireModel):
    kind: ScopeKind
    id: str

    def validate(self, path: str = "$") -> None:
        _require(bool(self.id), f"{path}.id", "must be non-empty")


@dataclass(frozen=True)
class FrozenIdentityWire(WireModel):
    provider: str
    provider_mode: Literal["authoritative"]
    principal_id: str
    profile_id: str
    logical_session_id: str
    org_id: Optional[str]
    project_id: Optional[str]
    repo_id: Optional[str]
    workspace_id: Optional[str]
    platform: str
    binding_revision: str
    opaque_binding_b64url: str

    def validate(self, path: str = "$") -> None:
        _require_surface(self.platform, f"{path}.platform")
        _require_b64url(self.opaque_binding_b64url, f"{path}.opaque_binding_b64url", exact_bytes=32)
        for name in ("provider", "principal_id", "profile_id", "logical_session_id", "binding_revision"):
            _require(bool(getattr(self, name)), f"{path}.{name}", "must be non-empty")


@dataclass(frozen=True)
class ScopeRevision(WireModel):
    scope: ScopeRef
    revision: str


@dataclass(frozen=True)
class CompositeRevision(WireModel):
    provider_epoch: str
    visibility_revision: str
    scope_revisions: Tuple[ScopeRevision, ...]


@dataclass(frozen=True)
class AcceptedProvenance(WireModel):
    actor_kind: ActorKind
    principal_id: str
    logical_session_id: Optional[str]
    surface: str
    source_entry_ids: Tuple[str, ...]
    source_commit: Optional[str]
    transaction_id: Optional[str]

    def validate(self, path: str = "$") -> None:
        _require_surface(self.surface, f"{path}.surface")


def _validate_entry_channel(entry, path: str) -> None:
    if entry.record_channel == "general":
        _require(entry.target is None, f"{path}.target", "general records have target null")
    else:
        expected = "memory" if entry.record_channel == "hermes_memory" else "user"
        _require(entry.target == expected, f"{path}.target", f"{entry.record_channel} records require target {expected}")


@dataclass(frozen=True)
class StoredEntry(WireModel):
    id: str
    text: str
    origin_scope: ScopeRef
    target: Optional[Target]
    record_channel: RecordChannel
    lane: Lane
    lifecycle: Literal["active"]
    policy_key: Optional[str]
    provenance: AcceptedProvenance

    def validate(self, path: str = "$") -> None:
        _validate_entry_channel(self, path)


@dataclass(frozen=True)
class DeliveredEntry(WireModel):
    id: str
    text: str
    origin_scope: ScopeRef
    target: Optional[Target]
    record_channel: RecordChannel
    lane: Lane
    delivery_tier: DeliveryTier
    policy_key: Optional[str]
    provenance: AcceptedProvenance

    def validate(self, path: str = "$") -> None:
        _validate_entry_channel(self, path)


@dataclass(frozen=True)
class ImportSourceIdentity(WireModel):
    source_kind: SourceKind
    parser_version: ParserVersion
    source_id: str
    item_key: str

    def validate(self, path: str = "$") -> None:
        _require(bool(_SOURCE_ID_RE.match(self.source_id)), f"{path}.source_id", "must be 1-64 lowercase ASCII letters, digits, or hyphens")
        segments = self.item_key.split("/")
        ok = bool(_ITEM_KEY_RE.match(self.item_key)) and all(seg not in ("", ".", "..") for seg in segments)
        _require(ok, f"{path}.item_key", "must be a relative slash-normalized key with no empty, dot, or dot-dot segment")


@dataclass(frozen=True)
class CandidateEntry(WireModel):
    client_ref: str
    text: str
    destination_scope: ScopeRef
    target: Target
    proposed_policy_key: Optional[str]
    import_source_identity: Optional[ImportSourceIdentity]

    def validate(self, path: str = "$") -> None:
        _require(bool(self.client_ref), f"{path}.client_ref", "must be non-empty")


@dataclass(frozen=True)
class PolicyAmbiguity(WireModel):
    policy_key: str
    tier: Literal["dependency"]
    candidate_count: int

    def validate(self, path: str = "$") -> None:
        _require(self.candidate_count > 1, f"{path}.candidate_count", "must be greater than one")


@dataclass(frozen=True)
class CuratedLimits(WireModel):
    memory_chars: int
    user_chars: int
    initial_general_chars: int
    max_entry_chars: int
    max_entries: int

    def validate(self, path: str = "$") -> None:
        for f in fields(self):
            _require_nonneg(getattr(self, f.name), f"{path}.{f.name}")


@dataclass(frozen=True)
class HiddenPreservationState(WireModel):
    target: Target
    complete_for_scopes: Tuple[ScopeRef, ...]
    opaque_state_b64url: str

    def validate(self, path: str = "$") -> None:
        _require_b64url(self.opaque_state_b64url, f"{path}.opaque_state_b64url")


@dataclass(frozen=True)
class Capabilities(WireModel):
    recall_context: bool
    capture_continuity: bool


@dataclass(frozen=True)
class Limits(WireModel):
    max_request_bytes: int
    max_response_bytes: int
    max_stage_bytes: int
    stage_ttl_seconds: int
    max_opaque_binding_bytes: int
    max_continuity_bytes: int

    def validate(self, path: str = "$") -> None:
        for f in fields(self):
            _require_positive(getattr(self, f.name), f"{path}.{f.name}")
        _require(self.stage_ttl_seconds <= 86400, f"{path}.stage_ttl_seconds", "must not exceed 86400")
        _require(self.max_opaque_binding_bytes >= 32, f"{path}.max_opaque_binding_bytes", "must be >= 32")


@dataclass(frozen=True)
class NegotiateRequest(WireModel):
    host: Literal["hermes"]
    supported_api_versions: Tuple[int, ...]
    required_operations: Tuple[Operation, ...]


@dataclass(frozen=True)
class Negotiation(WireModel):
    provider: str
    selected_api_version: int
    operations: Tuple[Operation, ...]
    capabilities: Capabilities
    limits: Limits

    def validate(self, path: str = "$") -> None:
        # Any positive integer decodes. §9.1 assigns the version decision to
        # the host: an incompatible provider API MUST be a configuration error
        # and MUST NOT be reinterpreted, so the host has to SEE the selected
        # version to report it (ProviderAuthoritativeMemoryService._negotiate).
        # Rejecting it here would turn an incompatible API into a WireError,
        # which under authoritative_failure_policy: stateless would silently
        # start a stateless session instead. Same treatment as `provider`,
        # which the host checks rather than the codec.
        _require_positive(self.selected_api_version, f"{path}.selected_api_version")


@dataclass(frozen=True)
class RequestedContext(WireModel):
    principal_id: str
    profile_id: str
    logical_session_id: str
    platform: str
    org_id: Optional[str]
    project_id: Optional[str]
    repo_id: Optional[str]
    workspace_id: Optional[str]
    resolution_source: ResolutionSource
    canonical_directory: Optional[str]

    def validate(self, path: str = "$") -> None:
        _require_surface(self.platform, f"{path}.platform")
        if self.resolution_source == "explicit_ids":
            _require(self.canonical_directory is None, f"{path}.canonical_directory", "must be null for explicit_ids")
        elif self.resolution_source == "workspace":
            _require(self.workspace_id is not None and self.canonical_directory is not None, f"{path}.workspace_id", "workspace resolution requires workspace_id and canonical_directory")
        else:
            _require(self.canonical_directory is not None, f"{path}.canonical_directory", "directory resolution requires canonical_directory")


@dataclass(frozen=True)
class BindRequest(WireModel):
    expected_provider_epoch: str
    binding_request_id: str
    bind_intent: BindIntent
    requested_context: RequestedContext
    prior_identity: Optional[FrozenIdentityWire]

    def validate(self, path: str = "$") -> None:
        if self.bind_intent == "new_session":
            _require(self.prior_identity is None, f"{path}.prior_identity", "new_session requires prior_identity null")
        if self.bind_intent == "explicit_rebind":
            _require(self.prior_identity is not None, f"{path}.prior_identity", "explicit_rebind requires prior_identity")
        if self.requested_context.resolution_source == "directory":
            _require(self.bind_intent in ("new_session", "explicit_rebind"), f"{path}.requested_context.resolution_source", "directory resolution is permitted only for new_session or explicit_rebind")


@dataclass(frozen=True)
class BindResult(WireModel):
    frozen_identity: FrozenIdentityWire
    visible_scopes: Tuple[ScopeRef, ...]
    memory_default_write_scope: Optional[ScopeRef]
    memory_eligible_write_scopes: Tuple[ScopeRef, ...]
    user_write_scope: ScopeRef

    def validate(self, path: str = "$") -> None:
        _require(self.user_write_scope.kind == "principal_global", f"{path}.user_write_scope", "must be the principal-global scope")


@dataclass(frozen=True)
class ValidateSessionRequest(WireModel):
    expected_provider_epoch: str
    frozen_identity: FrozenIdentityWire


@dataclass(frozen=True)
class ValidateSessionResult(WireModel):
    valid: Literal[True]
    frozen_identity: FrozenIdentityWire
    visible_scopes: Tuple[ScopeRef, ...]


@dataclass(frozen=True)
class LoadRequest(WireModel):
    expected_provider_epoch: str
    frozen_identity: FrozenIdentityWire
    target: Target


@dataclass(frozen=True)
class CuratedSnapshot(WireModel):
    api_version: int
    status: SnapshotStatus
    frozen_identity: FrozenIdentityWire
    target: Target
    visible_scopes: Tuple[ScopeRef, ...]
    default_write_scope: Optional[ScopeRef]
    eligible_write_scopes: Tuple[ScopeRef, ...]
    complete_for_scopes: Tuple[ScopeRef, ...]
    revision: CompositeRevision
    limits: CuratedLimits
    mutation_entries: Tuple[StoredEntry, ...]
    delivery_entries: Tuple[DeliveredEntry, ...]
    hidden_preservation_state: HiddenPreservationState

    def validate(self, path: str = "$") -> None:
        _require(self.api_version == 1, f"{path}.api_version", "must be 1")
        _require(_same_scopes(self.complete_for_scopes, self.eligible_write_scopes), f"{path}.complete_for_scopes", "must equal eligible_write_scopes")
        _require(_same_scopes([r.scope for r in self.revision.scope_revisions], self.visible_scopes), f"{path}.revision.scope_revisions", "must list every visible scope in order")
        hps = self.hidden_preservation_state
        _require(hps.target == self.target and _same_scopes(hps.complete_for_scopes, self.complete_for_scopes), f"{path}.hidden_preservation_state", "must carry the snapshot's target and complete_for_scopes")
        channel = "hermes_memory" if self.target == "memory" else "hermes_user"
        for i, e in enumerate(self.mutation_entries):
            _require(e.record_channel == channel and e.target == self.target, f"{path}.mutation_entries[{i}]", f"must be a {channel} record for target {self.target}")
        allowed = ("general", "hermes_memory") if self.target == "memory" else ("hermes_user",)
        for i, e in enumerate(self.delivery_entries):
            _require(e.record_channel in allowed, f"{path}.delivery_entries[{i}]", f"{self.target} snapshots deliver only {' or '.join(allowed)} records")
        if self.status == "degraded_global_only":
            ok = (
                len(self.visible_scopes) == 1 and self.visible_scopes[0].kind == "principal_global"
                and self.default_write_scope is None and not self.eligible_write_scopes
                and not self.complete_for_scopes and not self.mutation_entries
            )
            _require(ok, f"{path}.status", "degraded_global_only requires visible_scopes=[principal_global], null default scope, and empty eligible/complete/mutation lists")


@dataclass(frozen=True)
class MutationIntent(WireModel):
    kind: IntentKind
    matched_entry_id: Optional[str] = None
    reset_scopes: Optional[Tuple[ScopeRef, ...]] = None
    import_run_id: Optional[str] = None
    source_kind: Optional[SourceKind] = None

    _ALLOWED: ClassVar[Dict[str, Tuple[str, ...]]] = {
        "add": (), "bulk_edit": (),
        "replace": ("matched_entry_id",), "remove": ("matched_entry_id",),
        "reset": ("reset_scopes",), "import": ("import_run_id", "source_kind"),
    }

    @classmethod
    def from_wire(cls, data: Any, path: str = "$"):
        if not isinstance(data, dict):
            raise WireError(path, "must be an object")
        kind = _decode(IntentKind, data.get("kind"), f"{path}.kind")
        allowed = cls._ALLOWED[kind]
        for key in data:
            if key != "kind" and key not in allowed:
                raise WireError(path, f"unknown field {key!r} for intent {kind}")
        kwargs: Dict[str, Any] = {"kind": kind}
        for key in allowed:
            if key not in data:
                raise WireError(f"{path}.{key}", f"missing required field for intent {kind}")
            kwargs[key] = _decode(cls._hints()[key], data[key], f"{path}.{key}")
            if kwargs[key] is None:
                raise WireError(f"{path}.{key}", "must not be null")
        obj = cls(**kwargs)
        obj.validate(path)
        return obj

    def to_wire(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"kind": self.kind}
        for key in self._ALLOWED[self.kind]:
            out[key] = _encode(getattr(self, key))
        return out

    def validate(self, path: str = "$") -> None:
        # Also reachable on a DIRECTLY constructed intent (StageRequest.validate
        # calls this), so a missing variant field fails as a WireError instead
        # of crashing a later consumer with TypeError/KeyError.
        for key in self._ALLOWED[self.kind]:
            value = getattr(self, key)
            _require(value is not None, f"{path}.{key}", f"must not be null for intent {self.kind}")
            if isinstance(value, str):
                _require(bool(value), f"{path}.{key}", f"must not be empty for intent {self.kind}")


@dataclass(frozen=True)
class MutationDeltaItem(WireModel):
    action: DeltaAction
    client_ref: Optional[str] = None
    record_id: Optional[str] = None
    old_record_id: Optional[str] = None
    replacement_client_ref: Optional[str] = None

    _ALLOWED: ClassVar[Dict[str, Tuple[str, ...]]] = {
        "add": ("client_ref",), "retire": ("record_id",), "supersede": ("old_record_id", "replacement_client_ref"),
    }

    @classmethod
    def from_wire(cls, data: Any, path: str = "$"):
        if not isinstance(data, dict):
            raise WireError(path, "must be an object")
        action = _decode(DeltaAction, data.get("action"), f"{path}.action")
        allowed = cls._ALLOWED[action]
        for key in data:
            if key != "action" and key not in allowed:
                raise WireError(path, f"unknown field {key!r} for action {action}")
        kwargs: Dict[str, Any] = {"action": action}
        for key in allowed:
            if key not in data or data[key] is None:
                raise WireError(f"{path}.{key}", f"missing required field for action {action}")
            kwargs[key] = _decode(str, data[key], f"{path}.{key}")
        obj = cls(**kwargs)
        obj.validate(path)
        return obj

    def to_wire(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"action": self.action}
        for key in self._ALLOWED[self.action]:
            out[key] = getattr(self, key)
        return out

    def validate(self, path: str = "$") -> None:
        # Also reachable on a DIRECTLY constructed item (StageRequest.validate
        # calls this), so a missing or empty variant field fails as a WireError
        # instead of crashing a later consumer with TypeError/KeyError.
        for key in self._ALLOWED[self.action]:
            value = getattr(self, key)
            _require(value is not None, f"{path}.{key}", f"must not be null for action {self.action}")
            _require(bool(value), f"{path}.{key}", f"must not be empty for action {self.action}")


@dataclass(frozen=True)
class MutationProvenance(WireModel):
    actor_kind: MutationActorKind
    principal_id: str
    logical_session_id: str
    initiating_surface: str
    source_entry_ids: Tuple[str, ...]
    source_commit: Optional[str]
    threat_decision_id: Optional[str]

    def validate(self, path: str = "$") -> None:
        _require_surface(self.initiating_surface, f"{path}.initiating_surface")


@dataclass(frozen=True)
class StageRequest(WireModel):
    expected_provider_epoch: str
    frozen_identity: FrozenIdentityWire
    target: Target
    expected_revision: CompositeRevision
    hidden_preservation_state: HiddenPreservationState
    request_id: str
    requested_write_scopes: Tuple[ScopeRef, ...]
    intent: MutationIntent
    mutation_delta: Tuple[MutationDeltaItem, ...]
    candidate_entries: Tuple[CandidateEntry, ...]
    provenance: MutationProvenance

    def validate(self, path: str = "$") -> None:
        # Per-item shape first: the cardinality rules below read variant fields
        # (client_ref, record_id, old_record_id) and must never see a null or
        # empty one, which a directly constructed item can carry.
        self.intent.validate(f"{path}.intent")
        for i, item in enumerate(self.mutation_delta):
            item.validate(f"{path}.mutation_delta[{i}]")
        kind = self.intent.kind
        delta = self.mutation_delta
        cands = self.candidate_entries
        refs = [c.client_ref for c in cands]
        _require(len(refs) == len(set(refs)), f"{path}.candidate_entries", "client_ref values must be unique")
        used = [d.client_ref or d.replacement_client_ref for d in delta if d.action in ("add", "supersede")]
        _require(sorted(used) == sorted(refs) and len(used) == len(set(used)), f"{path}.mutation_delta", "every candidate must be referenced exactly once")
        olds = [d.record_id or d.old_record_id for d in delta if d.action in ("retire", "supersede")]
        _require(len(olds) == len(set(olds)), f"{path}.mutation_delta", "old record IDs must be unique")
        for i, c in enumerate(cands):
            _require(c.target == self.target, f"{path}.candidate_entries[{i}].target", "must equal the request target")
            if kind == "import":
                _require(c.import_source_identity is not None and c.import_source_identity.source_kind == self.intent.source_kind, f"{path}.candidate_entries[{i}].import_source_identity", "import candidates need an identity whose source_kind matches the intent")
            else:
                _require(c.import_source_identity is None, f"{path}.candidate_entries[{i}].import_source_identity", "non-import candidates must have null identity")
        if kind == "add":
            _require(len(cands) == 1 and len(delta) == 1 and delta[0].action == "add", f"{path}.intent", "add requires one candidate and one add item")
        elif kind == "replace":
            ok = len(cands) == 1 and len(delta) == 1 and delta[0].action == "supersede" and delta[0].old_record_id == self.intent.matched_entry_id
            _require(ok, f"{path}.intent", "replace requires one candidate and one supersede of matched_entry_id")
        elif kind == "remove":
            ok = not cands and len(delta) == 1 and delta[0].action == "retire" and delta[0].record_id == self.intent.matched_entry_id
            _require(ok, f"{path}.intent", "remove requires no candidates and one retire of matched_entry_id")
        elif kind == "bulk_edit":
            _require(len(delta) >= 2, f"{path}.intent", "bulk_edit requires an explicit delta of at least two items")
        elif kind == "import":
            _require(bool(cands) and all(d.action == "add" for d in delta), f"{path}.intent", "import is add-only with at least one candidate")
        elif kind == "reset":
            _require(not cands and not delta, f"{path}.intent", "reset has neither candidates nor delta")


@dataclass(frozen=True)
class AdmissionDecision(WireModel):
    client_ref: str
    assigned_id: str
    target: Target
    record_channel: HermesChannel
    origin_scope: ScopeRef
    disposition: Disposition
    publication_effect: PublicationEffect
    superseded_id: Optional[str]
    policy_key: Optional[str]


@dataclass(frozen=True)
class HiddenEffect(WireModel):
    scope: ScopeRef
    target: Target
    record_channel: HermesChannel
    lane: HiddenLane
    action: Literal["retire"]
    count: int

    def validate(self, path: str = "$") -> None:
        _require_nonneg(self.count, f"{path}.count")


@dataclass(frozen=True)
class CandidateHash(WireModel):
    client_ref: str
    canonical_sha256: str

    def validate(self, path: str = "$") -> None:
        _require(bool(_HEX64_RE.match(self.canonical_sha256)), f"{path}.canonical_sha256", "must be lowercase hex sha256")


@dataclass(frozen=True)
class StageResult(WireModel):
    stage_handle_b64url: str
    request_id: str
    target: Target
    expected_revision: CompositeRevision
    requested_write_scopes: Tuple[ScopeRef, ...]
    eligible_write_scopes: Tuple[ScopeRef, ...]
    expires_at: str
    approval_binding_sha256: str
    approval_requirements: Tuple[ApprovalRequirement, ...]
    candidate_hashes: Tuple[CandidateHash, ...]
    admissions: Tuple[AdmissionDecision, ...]
    hidden_effects: Tuple[HiddenEffect, ...]

    def validate(self, path: str = "$") -> None:
        _require_b64url(self.stage_handle_b64url, f"{path}.stage_handle_b64url")
        _require_timestamp(self.expires_at, f"{path}.expires_at")
        _require(bool(_HEX64_RE.match(self.approval_binding_sha256)), f"{path}.approval_binding_sha256", "must be lowercase hex sha256")
        _require(len(self.approval_requirements) == len(set(self.approval_requirements)), f"{path}.approval_requirements", "must be an ordered subset without repeats")


@dataclass(frozen=True)
class InspectStageRequest(WireModel):
    expected_provider_epoch: str
    frozen_identity: FrozenIdentityWire
    target: Target
    request_id: str
    stage_handle_b64url: str


@dataclass(frozen=True)
class StageInspection(WireModel):
    summary: StageResult
    canonical_candidates: Tuple[CandidateEntry, ...]
    visible_before: Tuple[StoredEntry, ...]
    visible_after: Tuple[StoredEntry, ...]


@dataclass(frozen=True)
class ApprovalAuthorization(WireModel):
    kind: Literal["not_required", "approved"]
    approval_id: Optional[str] = None
    approved_by_principal_id: Optional[str] = None
    approved_at: Optional[str] = None
    expires_at: Optional[str] = None
    approval_binding_sha256: Optional[str] = None

    _APPROVED_FIELDS: ClassVar[Tuple[str, ...]] = ("approval_id", "approved_by_principal_id", "approved_at", "expires_at", "approval_binding_sha256")

    @classmethod
    def from_wire(cls, data: Any, path: str = "$"):
        if not isinstance(data, dict):
            raise WireError(path, "must be an object")
        kind = _decode(Literal["not_required", "approved"], data.get("kind"), f"{path}.kind")
        allowed = cls._APPROVED_FIELDS if kind == "approved" else ()
        for key in data:
            if key != "kind" and key not in allowed:
                raise WireError(path, f"unknown field {key!r} for authorization {kind}")
        kwargs: Dict[str, Any] = {"kind": kind}
        for key in allowed:
            if key not in data or data[key] is None:
                raise WireError(f"{path}.{key}", "missing required field for approved authorization")
            kwargs[key] = _decode(str, data[key], f"{path}.{key}")
        obj = cls(**kwargs)
        if kind == "approved":
            _require_timestamp(obj.approved_at, f"{path}.approved_at")
            _require_timestamp(obj.expires_at, f"{path}.expires_at")
            _require(bool(_HEX64_RE.match(obj.approval_binding_sha256)), f"{path}.approval_binding_sha256", "must be lowercase hex sha256")
        return obj

    def to_wire(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"kind": self.kind}
        if self.kind == "approved":
            for key in self._APPROVED_FIELDS:
                out[key] = getattr(self, key)
        return out


@dataclass(frozen=True)
class CommitRequest(WireModel):
    expected_provider_epoch: str
    frozen_identity: FrozenIdentityWire
    target: Target
    request_id: str
    stage_handle_b64url: str
    approval_binding_sha256: str
    authorized_write_scopes: Tuple[ScopeRef, ...]
    authorization: ApprovalAuthorization

    def validate(self, path: str = "$") -> None:
        _require_b64url(self.stage_handle_b64url, f"{path}.stage_handle_b64url")
        _require(bool(_HEX64_RE.match(self.approval_binding_sha256)), f"{path}.approval_binding_sha256", "must be lowercase hex sha256")


@dataclass(frozen=True)
class CommitResult(WireModel):
    outcome: CommitOutcome
    request_id: str
    tx_id: str
    snapshot: CuratedSnapshot
    admissions: Tuple[AdmissionDecision, ...]


@dataclass(frozen=True)
class RecallBudget(WireModel):
    trusted_chars: int
    evidence_chars: int
    max_entries: int

    def validate(self, path: str = "$") -> None:
        for f in fields(self):
            _require_nonneg(getattr(self, f.name), f"{path}.{f.name}")


@dataclass(frozen=True)
class RecallRequest(WireModel):
    expected_provider_epoch: str
    frozen_identity: FrozenIdentityWire
    target: Target
    source_revision: CompositeRevision
    query: str
    include_channels: Tuple[RecordChannel, ...]
    exclude_entry_ids: Tuple[str, ...]
    budget: RecallBudget

    def validate(self, path: str = "$") -> None:
        chans = list(self.include_channels)
        if self.target == "memory":
            ok = bool(chans) and len(chans) == len(set(chans)) and all(c in ("general", "hermes_memory") for c in chans)
            _require(ok, f"{path}.include_channels", "memory recall takes an ordered nonempty subset of [general, hermes_memory]")
        else:
            _require(chans == ["hermes_user"], f"{path}.include_channels", "user recall takes exactly [hermes_user]")


@dataclass(frozen=True)
class TypedRecall(WireModel):
    frozen_identity: FrozenIdentityWire
    target: Target
    source_revision: CompositeRevision
    trusted_instructions: Tuple[DeliveredEntry, ...]
    scoped_evidence: Tuple[DeliveredEntry, ...]


@dataclass(frozen=True)
class ContinuityRequest(WireModel):
    expected_provider_epoch: str
    frozen_identity: FrozenIdentityWire
    request_id: str
    kind: ContinuityKind
    text: str
    initiating_surface: str

    def validate(self, path: str = "$") -> None:
        _require_surface(self.initiating_surface, f"{path}.initiating_surface")


@dataclass(frozen=True)
class ContinuityResult(WireModel):
    buffer_id: str
    request_id: str
    kind: ContinuityKind
    expires_at: str
    outcome: ContinuityOutcome

    def validate(self, path: str = "$") -> None:
        _require_timestamp(self.expires_at, f"{path}.expires_at")


@dataclass(frozen=True)
class WireErrorBody(WireModel):
    code: ErrorCode
    outcome: Outcome
    details: Optional[dict]


@dataclass(frozen=True)
class WireRequest(WireModel):
    wire_version: int
    call_id: str
    operation: Operation
    payload: dict

    def validate(self, path: str = "$") -> None:
        _require(self.wire_version == 1, f"{path}.wire_version", "must be 1")


@dataclass(frozen=True)
class WireSuccess(WireModel):
    wire_version: int
    call_id: str
    operation: Operation
    provider_epoch: str
    status: Literal["ok"]
    result: dict

    def validate(self, path: str = "$") -> None:
        _require(self.wire_version == 1, f"{path}.wire_version", "must be 1")


@dataclass(frozen=True)
class WireFailure(WireModel):
    wire_version: int
    call_id: str
    operation: Operation
    provider_epoch: Optional[str]
    status: Literal["error"]
    error: WireErrorBody

    def validate(self, path: str = "$") -> None:
        _require(self.wire_version == 1, f"{path}.wire_version", "must be 1")


@dataclass(frozen=True)
class ProviderEpochChangedDetails(WireModel):
    expected_provider_epoch: str
    current_provider_epoch: str


@dataclass(frozen=True)
class VersionConflictDetails(WireModel):
    current_snapshot: CuratedSnapshot


@dataclass(frozen=True)
class AmbiguousPolicyDetails(WireModel):
    ambiguities: Tuple[PolicyAmbiguity, ...]

    def validate(self, path: str = "$") -> None:
        _require(bool(self.ambiguities), f"{path}.ambiguities", "must be nonempty")


@dataclass(frozen=True)
class SecretRejectedDetails(WireModel):
    event_id: str
    detector_code: str


@dataclass(frozen=True)
class StoreBlockedDetails(WireModel):
    reason: StoreBlockedReason


@dataclass(frozen=True)
class StageNotFoundDetails(WireModel):
    state: Literal["committed"]
    tx_id: str


@dataclass(frozen=True)
class LimitExceededDetails(WireModel):
    limit: LimitName


#: Codes whose details are typed. version_conflict details are null for recall.
_DETAIL_TYPES: Dict[str, type] = {
    "provider_epoch_changed": ProviderEpochChangedDetails,
    "version_conflict": VersionConflictDetails,
    "ambiguous_policy": AmbiguousPolicyDetails,
    "secret_rejected": SecretRejectedDetails,
    "store_blocked": StoreBlockedDetails,
    "stage_not_found": StageNotFoundDetails,
    "limit_exceeded": LimitExceededDetails,
}
_DETAILS_MAY_BE_NULL = frozenset({"version_conflict", "stage_not_found"})


def decode_error_details(code: str, details: Optional[dict], path: str = "$.error.details"):
    """Decode ``WireFailure.error.details`` for ``code`` (§9.3 ErrorDetails)."""
    if code not in ERROR_CODES:
        raise WireError(path, "unknown error code")
    detail_type = _DETAIL_TYPES.get(code)
    if detail_type is None:
        _require(details is None, path, f"details must be null for {code}")
        return None
    if details is None:
        _require(code in _DETAILS_MAY_BE_NULL, path, f"details are required for {code}")
        return None
    return detail_type.from_wire(details, path)


REQUEST_TYPES: Dict[str, type] = {
    "negotiate": NegotiateRequest,
    "bind_session": BindRequest,
    "validate_session": ValidateSessionRequest,
    "load_curated": LoadRequest,
    "stage_curated": StageRequest,
    "inspect_staged": InspectStageRequest,
    "commit_curated": CommitRequest,
    "recall_context": RecallRequest,
    "capture_continuity": ContinuityRequest,
}
RESULT_TYPES: Dict[str, type] = {
    "negotiate": Negotiation,
    "bind_session": BindResult,
    "validate_session": ValidateSessionResult,
    "load_curated": CuratedSnapshot,
    "stage_curated": StageResult,
    "inspect_staged": StageInspection,
    "commit_curated": CommitResult,
    "recall_context": TypedRecall,
    "capture_continuity": ContinuityResult,
}
