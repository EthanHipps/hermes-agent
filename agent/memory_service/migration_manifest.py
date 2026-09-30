"""Host migration state of the §9.9 native-memory migration (R45, contract C7F-1; ledger K-9, contract C9).

One canonical-JSON file per import run at ``<home>/migrations/<provider>/<provider-epoch>/<run-id>.json``.
It is first written only after a secret-clean ``StageResult`` (§9.5 L1560, L1562) and then progresses
``active -> completed | rolled_back`` by atomic replacement at the same path (§9.9 L1655, L1659).
``state`` is the top-level member C9's reader keys on (``hermes_cli/backup_memory.active_migration_manifests``,
ruling R44-9): anything but a readable ``completed``/``rolled_back`` document blocks archives.

Provider-generic (D5): the provider segment and the schema prefix render the configured
``memory.provider``, so with the spec's provider configured the path and schema are exactly §9.9's
(the R44-4 precedent). The member sets below are a cross-repository contract pinned by
``tests/fixtures/hermes-migration/`` in both repositories (rulings R45-6, R45-7; contract C7F-2); change them
only as a coordinated PR pair (§14.1 L2267). Active state holds no candidate body, excerpt, absolute path or
binding handle; receipts hold no stage handle, binding hash, authorization, digest or session identity
(§9.9 L1659). Both carry ``created_at``/``updated_at`` (Checkpoint A, R45-6: set at the first write, then on
every replacement including compaction), and each batch holds exactly one item (one item per target per run).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Tuple, Union, get_args

from agent.memory_service import wire as w
from agent.memory_service.errors import MemoryServiceError

SCHEMA_SUFFIX = ".hermes-migration/v1"
ACTIVE, COMPLETED, ROLLED_BACK = "active", "completed", "rolled_back"
STATES = (ACTIVE, COMPLETED, ROLLED_BACK)
RUN_OUTCOMES = ("completed", "operator_denied", "provider_rejected", "operator_rollback", "reconciled", "discarded")
BATCH_STATUSES = ("pending", "staged", "approved", "committed")
BATCH_OUTCOMES = ("committed", "not_committed", "unknown")
PRIOR_OUTCOMES = ("stage_expired", "version_conflict")
SOURCE_PARSERS = {"native_memory": "hermes-native-v0.20.6", "legacy_archive": "hermes-legacy-archive-v1"}
LOCK_NAME = ".lock"

ACTIVE_KEYS = frozenset({"schema", "state", "provider_epoch", "import_run_id", "created_at", "updated_at", "identity",
                         "source", "batches"})
IDENTITY_KEYS = frozenset({"principal_id", "profile_id", "logical_session_id", "platform", "org_id", "project_id",
                           "repo_id", "workspace_id", "binding_revision"})
SOURCE_KEYS = frozenset({"source_kind", "parser_version", "source_id"})
ACTIVE_BATCH_KEYS = frozenset({"batch_id", "target", "destination_scope", "requested_write_scopes", "items", "status",
                               "request_id", "prior_requests", "stage", "candidate_hashes", "admissions",
                               "authorization", "result"})
ITEM_KEYS = frozenset({"item_key", "source_sha256"})
STAGE_KEYS = frozenset({"stage_handle_b64url", "approval_binding_sha256", "expected_revision", "expires_at",
                        "approval_requirements"})
ADMISSION_KEYS = frozenset({"client_ref", "assigned_id", "origin_scope", "disposition", "publication_effect",
                            "policy_key"})
RESULT_KEYS = frozenset({"tx_id", "commit_outcome", "revision"})
PRIOR_KEYS = frozenset({"request_id", "outcome"})
RECEIPT_KEYS = frozenset({"schema", "state", "outcome", "provider_epoch", "import_run_id", "created_at", "updated_at",
                          "principal_id", "source", "batches"})
RECEIPT_BATCH_KEYS = frozenset({"target", "destination_scope", "item_keys", "request_ids", "candidate_hashes",
                                "assigned", "tx_id", "outcome"})
ASSIGNED_KEYS = frozenset({"client_ref", "assigned_id", "publication_effect"})

_PROVIDER_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_EPOCH_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_HEX16_RE = re.compile(r"^[0-9a-f]{16}$")
_HEX32_RE = re.compile(r"^[0-9a-f]{32}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{1,256}$")
_B64URL_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_STAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_EXPIRY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,9})?Z$")
_TARGETS = ("memory", "user")
_ITEM_FILE = {"memory": "MEMORY.md", "user": "USER.md"}


class MigrationStateError(MemoryServiceError):
    """A migration file is unreadable, not canonical, or violates the schema (§9.9 L1655: blocks resume)."""

    code = "migration_state_invalid"


class MigrationBusyError(MemoryServiceError):
    """Another ``hermes memory migrate`` command holds the migration lock (ruling R45-4)."""

    code = "MIGRATION_BUSY"


def schema_name(provider: str) -> str:
    return f"{provider}{SCHEMA_SUFFIX}"


def valid_provider(value: Any) -> bool:
    return isinstance(value, str) and bool(_PROVIDER_RE.match(value)) and value not in (".", "..")


def valid_epoch(value: Any) -> bool:
    return isinstance(value, str) and bool(_EPOCH_RE.match(value))


@dataclass(frozen=True)
class Identity:
    principal_id: str
    profile_id: str
    logical_session_id: str
    platform: str
    org_id: Optional[str]
    project_id: Optional[str]
    repo_id: Optional[str]
    workspace_id: Optional[str]
    binding_revision: str

    @classmethod
    def of(cls, frozen: Any) -> "Identity":
        """Assertions only (§9.9 L1655); never the opaque binding handle (§9.3 L991)."""
        return cls(frozen.principal_id, frozen.profile_id, frozen.logical_session_id, frozen.platform, frozen.org_id,
                   frozen.project_id, frozen.repo_id, frozen.workspace_id, frozen.binding_revision)


@dataclass(frozen=True)
class Source:
    source_kind: str
    parser_version: str
    source_id: str


@dataclass(frozen=True)
class Item:
    item_key: str
    source_sha256: Optional[str] = None        # raw-byte digest, set when the batch stages clean (ruling R45-12)


@dataclass(frozen=True)
class StageRef:
    stage_handle_b64url: str
    approval_binding_sha256: str
    expected_revision: w.CompositeRevision
    expires_at: str
    approval_requirements: Tuple[str, ...]

    @classmethod
    def of(cls, stage: w.StageResult) -> "StageRef":
        return cls(stage.stage_handle_b64url, stage.approval_binding_sha256, stage.expected_revision,
                   stage.expires_at, tuple(stage.approval_requirements))


@dataclass(frozen=True)
class Admission:
    client_ref: str
    assigned_id: str
    origin_scope: w.ScopeRef
    disposition: str
    publication_effect: str
    policy_key: Optional[str]

    @classmethod
    def of(cls, a: w.AdmissionDecision) -> "Admission":
        return cls(a.client_ref, a.assigned_id, a.origin_scope, a.disposition, a.publication_effect, a.policy_key)


@dataclass(frozen=True)
class CommitRef:
    tx_id: str
    commit_outcome: Optional[str]              # None when proven by a stage_not_found {committed} probe (D-R45-4)
    revision: Optional[w.CompositeRevision]


@dataclass(frozen=True)
class Prior:
    request_id: str
    outcome: str


@dataclass(frozen=True)
class Batch:
    batch_id: str
    target: str
    destination_scope: w.ScopeRef
    requested_write_scopes: Tuple[w.ScopeRef, ...]
    items: Tuple[Item, ...]
    status: str = "pending"
    request_id: Optional[str] = None
    prior_requests: Tuple[Prior, ...] = ()
    stage: Optional[StageRef] = None
    candidate_hashes: Tuple[w.CandidateHash, ...] = ()
    admissions: Tuple[Admission, ...] = ()
    authorization: Optional[w.ApprovalAuthorization] = None
    result: Optional[CommitRef] = None


@dataclass(frozen=True)
class ActiveManifest:
    provider: str                              # the path segment and schema prefix; never a document member
    provider_epoch: str
    import_run_id: str
    identity: Identity
    source: Source
    batches: Tuple[Batch, ...]
    created_at: str                            # set at the first write, never changed (R45-6)
    updated_at: str                            # set on every replacement (R45-6)

    @property
    def state(self) -> str:
        return ACTIVE


@dataclass(frozen=True)
class Assigned:
    client_ref: str
    assigned_id: str
    publication_effect: str


@dataclass(frozen=True)
class ReceiptBatch:
    target: str
    destination_scope: w.ScopeRef
    item_keys: Tuple[str, ...]
    request_ids: Tuple[str, ...]
    candidate_hashes: Tuple[w.CandidateHash, ...]
    assigned: Tuple[Assigned, ...]
    tx_id: Optional[str]
    outcome: str


@dataclass(frozen=True)
class Receipt:
    provider: str
    provider_epoch: str
    import_run_id: str
    state: str
    outcome: str
    principal_id: str
    source: Source
    batches: Tuple[ReceiptBatch, ...]
    created_at: str                            # the active manifest's, unchanged
    updated_at: str                            # the compaction time


def _scope(s: w.ScopeRef) -> dict:
    return s.to_wire()


def _hashes(hashes) -> list:
    return [h.to_wire() for h in hashes]


def to_document(doc: Union[ActiveManifest, Receipt]) -> Dict[str, Any]:
    source = {"source_kind": doc.source.source_kind, "parser_version": doc.source.parser_version,
              "source_id": doc.source.source_id}
    if isinstance(doc, Receipt):
        return {"schema": schema_name(doc.provider), "state": doc.state, "outcome": doc.outcome,
                "provider_epoch": doc.provider_epoch, "import_run_id": doc.import_run_id,
                "created_at": doc.created_at, "updated_at": doc.updated_at,
                "principal_id": doc.principal_id, "source": source,
                "batches": [{"target": b.target, "destination_scope": _scope(b.destination_scope),
                             "item_keys": list(b.item_keys), "request_ids": list(b.request_ids),
                             "candidate_hashes": _hashes(b.candidate_hashes),
                             "assigned": [{"client_ref": a.client_ref, "assigned_id": a.assigned_id,
                                           "publication_effect": a.publication_effect} for a in b.assigned],
                             "tx_id": b.tx_id, "outcome": b.outcome} for b in doc.batches]}
    i = doc.identity
    return {"schema": schema_name(doc.provider), "state": ACTIVE, "provider_epoch": doc.provider_epoch,
            "import_run_id": doc.import_run_id, "created_at": doc.created_at, "updated_at": doc.updated_at,
            "identity": {"principal_id": i.principal_id, "profile_id": i.profile_id,
                         "logical_session_id": i.logical_session_id, "platform": i.platform, "org_id": i.org_id,
                         "project_id": i.project_id, "repo_id": i.repo_id, "workspace_id": i.workspace_id,
                         "binding_revision": i.binding_revision},
            "source": source, "batches": [_batch_doc(b) for b in doc.batches]}


def _batch_doc(b: Batch) -> Dict[str, Any]:
    return {"batch_id": b.batch_id, "target": b.target, "destination_scope": _scope(b.destination_scope),
            "requested_write_scopes": [_scope(s) for s in b.requested_write_scopes],
            "items": [{"item_key": it.item_key, "source_sha256": it.source_sha256} for it in b.items],
            "status": b.status, "request_id": b.request_id,
            "prior_requests": [{"request_id": p.request_id, "outcome": p.outcome} for p in b.prior_requests],
            "stage": None if b.stage is None else {
                "stage_handle_b64url": b.stage.stage_handle_b64url,
                "approval_binding_sha256": b.stage.approval_binding_sha256,
                "expected_revision": b.stage.expected_revision.to_wire(), "expires_at": b.stage.expires_at,
                "approval_requirements": list(b.stage.approval_requirements)},
            "candidate_hashes": _hashes(b.candidate_hashes),
            "admissions": [{"client_ref": a.client_ref, "assigned_id": a.assigned_id,
                            "origin_scope": _scope(a.origin_scope), "disposition": a.disposition,
                            "publication_effect": a.publication_effect, "policy_key": a.policy_key}
                           for a in b.admissions],
            "authorization": b.authorization.to_wire() if b.authorization is not None else None,
            "result": None if b.result is None else {
                "tx_id": b.result.tx_id, "commit_outcome": b.result.commit_outcome,
                "revision": b.result.revision.to_wire() if b.result.revision is not None else None}}


def encode(document: Mapping[str, Any]) -> bytes:
    return w.canonical_json(dict(document))


def compact(manifest: ActiveManifest, *, state: str, outcome: str,
            settled: Mapping[str, Tuple[str, Optional[str]]], updated_at: str) -> Receipt:
    """§9.9 L1659: the body-free receipt, in place of the active manifest, at the same path.

    It keeps the manifest's ``created_at`` and stamps ``updated_at`` (compaction is a replacement; R45-6).
    """
    if state not in (COMPLETED, ROLLED_BACK) or outcome not in RUN_OUTCOMES or (state == COMPLETED) != (outcome == "completed"):
        raise ValueError("a receipt is completed/completed or rolled_back/<non-completed outcome>")
    batches = []
    for b in manifest.batches:
        batch_outcome, tx_id = settled[b.batch_id]
        committed = batch_outcome == "committed"
        batches.append(ReceiptBatch(
            target=b.target, destination_scope=b.destination_scope, item_keys=tuple(i.item_key for i in b.items),
            request_ids=tuple(p.request_id for p in b.prior_requests) + ((b.request_id,) if b.request_id else ()),
            candidate_hashes=tuple(b.candidate_hashes),
            assigned=tuple(Assigned(a.client_ref, a.assigned_id, a.publication_effect) for a in b.admissions)
            if committed else (),
            tx_id=tx_id if committed else None, outcome=batch_outcome))
    return Receipt(manifest.provider, manifest.provider_epoch, manifest.import_run_id, state, outcome,
                   manifest.identity.principal_id, manifest.source, tuple(batches),
                   created_at=manifest.created_at, updated_at=updated_at)


# -- strict decoding (a malformed document is reported, never guessed) --------------------------------------


def _need(condition: Any, what: str) -> None:
    if not condition:
        raise MigrationStateError(f"migration state is invalid: {what}")


def _members(value: Any, keys: frozenset, what: str) -> Dict[str, Any]:
    _need(isinstance(value, dict) and set(value) == keys, f"{what} has the wrong members")
    return value


def _text(value: Any, what: str) -> str:
    _need(isinstance(value, str) and value, f"{what} must be a non-empty string")
    return value


def _optional_text(value: Any, what: str) -> Optional[str]:
    return None if value is None else _text(value, what)


def _matching(value: Any, pattern: "re.Pattern[str]", what: str) -> str:
    _need(isinstance(value, str) and pattern.match(value), f"{what} is malformed")
    return value


def _array(value: Any, what: str) -> list:
    _need(isinstance(value, list), f"{what} must be an array")
    return value


def _stamp(value: Any, what: str) -> str:
    """RFC 3339, second precision, UTC ``Z`` (R45-6). No ordering is enforced between the two stamps."""
    _matching(value, _STAMP_RE, what)
    datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")       # ValueError is wrapped by from_document
    return value


def _expiry(value: Any, what: str) -> str:
    _matching(value, _EXPIRY_RE, what)
    datetime.fromisoformat(value[:-1] + "+00:00")
    return value


def _literal(value: Any, options: Tuple[str, ...], what: str) -> str:
    _need(isinstance(value, str) and value in options, f"{what} is not a known value")
    return value


def _source(value: Any) -> Source:
    data = _members(value, SOURCE_KEYS, "source")
    kind = _literal(data["source_kind"], tuple(SOURCE_PARSERS), "source_kind")
    _need(data["parser_version"] == SOURCE_PARSERS[kind], "parser_version does not match source_kind")
    return Source(kind, data["parser_version"], _text(data["source_id"], "source_id"))


def _item_key(source: Source, key: Any, target: str) -> str:
    w.ImportSourceIdentity.from_wire({"source_kind": source.source_kind, "parser_version": source.parser_version,
                                      "source_id": source.source_id, "item_key": key})
    _need(key.rsplit("/", 1)[-1] == _ITEM_FILE[target], "item_key does not name its batch's target")
    return key


def _identity(value: Any) -> Identity:
    data = _members(value, IDENTITY_KEYS, "identity")
    return Identity(*(
        _optional_text(data[name], name) if name in ("org_id", "project_id", "repo_id", "workspace_id")
        else _text(data[name], name)
        for name in ("principal_id", "profile_id", "logical_session_id", "platform", "org_id", "project_id",
                     "repo_id", "workspace_id", "binding_revision")))


def _stage(value: Any) -> StageRef:
    data = _members(value, STAGE_KEYS, "stage")
    requirements = tuple(_literal(r, get_args(w.ApprovalRequirement), "approval requirement")
                         for r in _array(data["approval_requirements"], "approval_requirements"))
    return StageRef(_matching(data["stage_handle_b64url"], _B64URL_RE, "stage handle"),
                    _matching(data["approval_binding_sha256"], _HEX64_RE, "binding hash"),
                    w.CompositeRevision.from_wire(data["expected_revision"]),
                    _expiry(data["expires_at"], "expires_at"), requirements)


def _admission(value: Any) -> Admission:
    data = _members(value, ADMISSION_KEYS, "admission")
    return Admission(_text(data["client_ref"], "client_ref"), _text(data["assigned_id"], "assigned_id"),
                     w.ScopeRef.from_wire(data["origin_scope"]),
                     _literal(data["disposition"], get_args(w.Disposition), "disposition"),
                     _literal(data["publication_effect"], get_args(w.PublicationEffect), "publication_effect"),
                     _optional_text(data["policy_key"], "policy_key"))


def _result(value: Any) -> Optional[CommitRef]:
    if value is None:
        return None
    data = _members(value, RESULT_KEYS, "result")
    outcome = data["commit_outcome"]
    if outcome is not None:
        _literal(outcome, get_args(w.CommitOutcome), "commit_outcome")
    revision = None if data["revision"] is None else w.CompositeRevision.from_wire(data["revision"])
    return CommitRef(_matching(data["tx_id"], _TOKEN_RE, "tx_id"), outcome, revision)


def _active_batch(value: Any, source: Source) -> Batch:
    data = _members(value, ACTIVE_BATCH_KEYS, "batch")
    target = _literal(data["target"], _TARGETS, "target")
    destination = w.ScopeRef.from_wire(data["destination_scope"])
    scopes = tuple(w.ScopeRef.from_wire(s) for s in _array(data["requested_write_scopes"], "requested_write_scopes"))
    _need(scopes == (destination,), "requested_write_scopes must be exactly the destination")
    raw_items = _array(data["items"], "items")
    _need(len(raw_items) == 1, "a batch holds exactly one item")             # one item per target per run (R45-6)
    items = []
    for raw in raw_items:
        item = _members(raw, ITEM_KEYS, "item")
        digest = item["source_sha256"]
        if digest is not None:
            _matching(digest, _HEX64_RE, "source_sha256")
        items.append(Item(_item_key(source, item["item_key"], target), digest))
    status = _literal(data["status"], BATCH_STATUSES, "status")
    request_id = data["request_id"]
    priors = tuple(Prior(_matching(p["request_id"], _HEX32_RE, "prior request_id"),
                         _literal(p["outcome"], PRIOR_OUTCOMES, "prior outcome"))
                   for p in (_members(p, PRIOR_KEYS, "prior request") for p in _array(data["prior_requests"],
                                                                                        "prior_requests")))
    stage = None if data["stage"] is None else _stage(data["stage"])
    hashes = tuple(w.CandidateHash.from_wire(h) for h in _array(data["candidate_hashes"], "candidate_hashes"))
    admissions = tuple(_admission(a) for a in _array(data["admissions"], "admissions"))
    authorization = (None if data["authorization"] is None
                     else w.ApprovalAuthorization.from_wire(data["authorization"]))
    result = _result(data["result"])
    if status == "pending":
        _need(request_id is None and stage is None and authorization is None and result is None
              and not priors and not hashes and not admissions
              and all(i.source_sha256 is None for i in items), "a pending batch carries stage state")
    else:
        _matching(request_id, _HEX32_RE, "request_id")
        refs = tuple(h.client_ref for h in hashes)
        _need(stage is not None and hashes and refs == tuple(a.client_ref for a in admissions)
              and len(set(refs)) == len(refs), "a staged batch needs its stage, hashes and admissions")
        _need(all(a.origin_scope == destination for a in admissions), "an admission left its destination")
        _need(all(i.source_sha256 is not None for i in items), "a staged batch records its source digest")
        if status == "staged":
            _need(authorization is None and result is None, "a staged batch has no authorization or result")
        else:
            _need(authorization is not None and authorization.kind == "approved"
                  and authorization.approval_binding_sha256 == stage.approval_binding_sha256,
                  "an approved batch needs the stage's authorization")
            _need((result is not None) == (status == "committed"), "a result belongs to a committed batch only")
    return Batch(_matching(data["batch_id"], _HEX16_RE, "batch_id"), target, destination, scopes, tuple(items),
                 status, request_id, priors, stage, hashes, admissions, authorization, result)


def _receipt_batch(value: Any, source: Source) -> ReceiptBatch:
    data = _members(value, RECEIPT_BATCH_KEYS, "receipt batch")
    target = _literal(data["target"], _TARGETS, "target")
    keys = _array(data["item_keys"], "item_keys")
    _need(len(keys) == 1, "a batch holds exactly one item")                  # one item per target per run (R45-6)
    request_ids = tuple(_matching(r, _HEX32_RE, "request_id") for r in _array(data["request_ids"], "request_ids"))
    hashes = tuple(w.CandidateHash.from_wire(h) for h in _array(data["candidate_hashes"], "candidate_hashes"))
    assigned = tuple(Assigned(_text(a["client_ref"], "client_ref"), _text(a["assigned_id"], "assigned_id"),
                              _literal(a["publication_effect"], get_args(w.PublicationEffect), "publication_effect"))
                     for a in (_members(a, ASSIGNED_KEYS, "assigned") for a in _array(data["assigned"], "assigned")))
    outcome = _literal(data["outcome"], BATCH_OUTCOMES, "batch outcome")
    tx_id = data["tx_id"]
    if outcome == "committed":
        _matching(tx_id, _TOKEN_RE, "tx_id")
        _need(request_ids and tuple(a.client_ref for a in assigned) == tuple(h.client_ref for h in hashes)
              and assigned, "a committed batch names its requests and one assignment per candidate")
    else:
        _need(tx_id is None and not assigned, "only a committed batch has a transaction or assignments")
    return ReceiptBatch(target, w.ScopeRef.from_wire(data["destination_scope"]),
                        tuple(_item_key(source, k, target) for k in keys), request_ids, hashes, assigned, tx_id,
                        outcome)


def _decode(data: Any, provider: str, provider_epoch: str, import_run_id: str) -> Union[ActiveManifest, Receipt]:
    _need(isinstance(data, dict), "not an object")
    state = _literal(data.get("state"), STATES, "state")
    _members(data, ACTIVE_KEYS if state == ACTIVE else RECEIPT_KEYS, "document")
    _need(valid_provider(provider) and data["schema"] == schema_name(provider), "wrong schema")
    _need(data["provider_epoch"] == provider_epoch and valid_epoch(provider_epoch), "epoch disagrees with the path")
    _need(isinstance(import_run_id, str) and _HEX32_RE.match(import_run_id)
          and data["import_run_id"] == import_run_id, "run id disagrees with the path")
    created_at, updated_at = _stamp(data["created_at"], "created_at"), _stamp(data["updated_at"], "updated_at")
    source = _source(data["source"])
    raw_batches = _array(data["batches"], "batches")
    _need(raw_batches, "a run has at least one batch")
    if state == ACTIVE:
        batches = tuple(_active_batch(b, source) for b in raw_batches)
        _need(len({b.target for b in batches}) == len(batches), "one batch per target")
        _need(len({b.batch_id for b in batches}) == len(batches), "batch ids are unique")
        return ActiveManifest(provider, provider_epoch, import_run_id, _identity(data["identity"]), source, batches,
                              created_at, updated_at)
    outcome = _literal(data["outcome"], RUN_OUTCOMES, "outcome")
    _need((state == COMPLETED) == (outcome == "completed"), "state and outcome disagree")
    receipt_batches = tuple(_receipt_batch(b, source) for b in raw_batches)
    _need(len({b.target for b in receipt_batches}) == len(receipt_batches), "one batch per target")
    _need(state != COMPLETED or all(b.outcome == "committed" for b in receipt_batches),
          "a completed run has every batch committed")
    return Receipt(provider, provider_epoch, import_run_id, state, outcome, _text(data["principal_id"], "principal_id"),
                   source, receipt_batches, created_at, updated_at)


def from_document(data: Any, *, provider: str, provider_epoch: str,
                  import_run_id: str) -> Union[ActiveManifest, Receipt]:
    """Strict and total: every defect is :class:`MigrationStateError`, with a content-free message."""
    try:
        return _decode(data, provider, provider_epoch, import_run_id)
    except MigrationStateError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:     # w.WireError is a ValueError
        raise MigrationStateError(f"migration state is invalid: {type(exc).__name__}") from exc
