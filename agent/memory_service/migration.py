"""The §9.9 native-memory migration state machine over MemoryService (R45; entry points are contract C7F-3).

A run moves the selected native items into the configured provider as one ``import`` batch per target
(memory, then user; ruling R45-9). Each batch is a state machine ``pending -> staged -> approved ->
committed`` persisted in the active manifest (``migration_manifest``) after every transition. Hermes
persists nothing before a secret-clean ``StageResult`` (§9.5 L1560), fsyncs the manifest before showing
``inspect_staged`` (§9.9 L1663), writes the approved authorization ahead of the first commit send (the
R39-7 pattern; §9.5 L1564), commits with the manifest's request ID, records the provider's IDs and
transaction, and only then moves on. After every batch is committed it checks conformance (ruling R45-13),
re-verifies native digests (R45-12), switches ``provider_mode`` (R45-14, through the injected
``switch_to_authoritative``) and compacts to a ``completed`` receipt (§9.9 L1666).

The engine drives the service directly instead of through ``mutation.run_curated_mutation`` (ruling R45-3):
C5 inspects immediately after staging, re-plans with new request IDs and has no resume entry; with an inline
prompt it keeps no write-ahead record for the identical retry (EDD-109, fork-B3); §9.9 needs a manifest fsync
between stage and inspect and the recorded request across restarts. It reuses C6b-1 and C5's requirement
prediction unchanged. Administrative identity comes from C6b-5 ``admin_service`` over an in-memory
authoritative overlay of the home's own ``memory:`` section (ruling R45-2): the file stays additive until
step 6.
"""

from __future__ import annotations

import logging
import secrets
import uuid
from contextlib import ExitStack
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from agent.memory_service import migration_manifest as mm
from agent.memory_service import wire as w
from agent.memory_service.admin import ADMIN_FAILURES, AdminContext, admin_service, format_scope
from agent.memory_service.approval import PromptFn, rfc3339, utcnow
from agent.memory_service.errors import MemoryBlockedError, MemoryServiceError, ProviderError, ProviderTransportError
from agent.memory_service.migration_source import SourceError, SourceItem, read_live_items, scan_items, target_for
from agent.memory_service.mutation import PlannedMutation, predict_approval_requirements
from agent.memory_service.service import MutationRequest

logger = logging.getLogger(__name__)

SURFACE = "memory_migrate"
MAX_STAGE_ATTEMPTS = 3
# Ruling R45-14 (a): the first approval header of a live-native run announces the switch, and a switched
# completion tells the operator that running sessions keep the old mode until they restart (§9.6 L1568).
SWITCH_NOTICE = "  When every batch is committed and verified, memory.provider_mode is switched to authoritative."
RESTART_NOTICE = ("Hermes sessions and gateways that are already running keep using MEMORY.md/USER.md until they "
                  "restart (the memory mode is fixed per session); restart them now.")
_TARGET_ORDER = ("memory", "user")
_DETAIL_KEY = {"store_blocked": "reason", "secret_rejected": "detector_code", "limit_exceeded": "limit"}


class MigrationStatus(str, Enum):
    COMPLETED = "completed"
    DENIED = "denied"
    REJECTED = "rejected"
    REFUSED = "refused"
    STOPPED = "stopped"
    ROLLED_BACK = "rolled_back"
    RECONCILED = "reconciled"


@dataclass(frozen=True)
class SourceSelection:
    kind: str                            # "native_memory" | "legacy_archive"
    source_id: str
    item_keys: Tuple[str, ...]
    archive: Optional[Path] = None


@dataclass(frozen=True)
class MigrationPlan:
    source: SourceSelection
    memory_scope: Optional[w.ScopeRef]
    context: AdminContext


@dataclass(frozen=True)
class BatchSummary:
    target: str
    item_keys: Tuple[str, ...]
    created: int
    reused: int
    withheld: int
    status: str


@dataclass(frozen=True)
class MigrationReport:
    status: MigrationStatus
    code: Optional[str]
    message: str
    run_id: Optional[str] = None
    batches: Tuple[BatchSummary, ...] = ()
    switched: bool = False
    warnings: Tuple[str, ...] = ()


class _Refuse(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


class _Stop(_Refuse):
    """Stop with the active manifest (if any) left for resume or reconcile."""


class _Rejected(Exception):
    def __init__(self, code: str, detail: Optional[str] = None, *, proven: Optional[str] = None) -> None:
        super().__init__(code)
        self.code, self.detail, self.proven = code, detail, proven


class _Denied(Exception):
    pass


class _Unknown(Exception):
    pass


def _overlay(raw_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Ruling R45-2: the home's own memory section, authoritative in memory only; the file is not touched."""
    section = raw_config.get("memory") if isinstance(raw_config, Mapping) else None
    return {"memory": {**(dict(section) if isinstance(section, Mapping) else {}), "provider_mode": "authoritative"}}


def _provider(raw_config: Mapping[str, Any]) -> Optional[str]:
    section = raw_config.get("memory") if isinstance(raw_config, Mapping) else None
    value = section.get("provider") if isinstance(section, Mapping) else None
    value = value.strip() if isinstance(value, str) else ""
    return value if mm.valid_provider(value) else None


def _requests_authoritative(home: Path) -> bool:
    from agent.memory_service.bootstrap import requests_authoritative_mode
    from hermes_cli.backup_memory import home_memory_section
    return requests_authoritative_mode({"memory": dict(home_memory_section(home))})


def _detail(exc: ProviderError) -> Optional[str]:
    key = _DETAIL_KEY.get(exc.code)
    value = exc.details.get(key) if key and isinstance(exc.details, dict) else None
    return str(value) if value else None


def _exact(service: Any, target: str, call: Callable[[], Any]) -> Any:
    """R38-8 / D-R45-5: one identical retry after a fresh load clears the fail-closed latch."""
    try:
        return call()
    except ProviderTransportError:
        pass
    except MemoryBlockedError as exc:
        if not isinstance(exc.__cause__, ProviderTransportError):
            raise
    try:
        service.load_curated(target)
        return call()
    except (ProviderTransportError, MemoryBlockedError) as exc:
        raise _Unknown() from exc


def _ask(prompt: PromptFn, approval: Any) -> Optional[bool]:
    try:
        return prompt(approval)
    except Exception:
        logger.warning("memory migration approval prompt failed; treated as no answer")
        return None


def _hex8() -> str:
    return secrets.token_hex(8)


def _start_refusal(raw_config: Mapping[str, Any], plan: MigrationPlan, home: Path) -> Optional[MigrationReport]:
    """Everything decidable before any provider contact (exit 2 at the CLI)."""
    from agent.memory_service.config import MemoryConfigurationError, resolve_memory_service_config
    from hermes_cli.backup_memory import MIGRATION_IN_PROGRESS, active_migration_manifests

    def refuse(code: str, message: str) -> MigrationReport:
        return MigrationReport(MigrationStatus.REFUSED, code, message)

    if _provider(raw_config) is None:
        return refuse("configuration_error", "memory.provider must name the provider as a plain lowercase name.")
    try:
        resolve_memory_service_config(_overlay(raw_config))
    except MemoryConfigurationError as exc:
        return refuse("configuration_error", f"The memory configuration cannot serve a migration: {exc}")
    selection = plan.source
    if selection.kind not in mm.SOURCE_PARSERS:
        return refuse("source_invalid", "Unknown source kind.")
    try:
        w.ImportSourceIdentity.from_wire({"source_kind": selection.kind, "parser_version": mm.SOURCE_PARSERS[selection.kind],
                                          "source_id": selection.source_id, "item_key": "MEMORY.md"})
    except w.WireError:
        return refuse("source_invalid", "--source-id must be 1-64 lowercase letters, digits or hyphens.")
    if (selection.kind == "legacy_archive") != (selection.archive is not None):
        return refuse("source_invalid", "--archive is required for, and only for, a legacy archive source.")
    if not selection.item_keys:
        return refuse("source_invalid", "Select at least one item.")
    per_target = [target_for(k) for k in selection.item_keys]
    if len(per_target) != len(set(per_target)):                       # one item per target per run (R45-6)
        return refuse("source_invalid", "Select at most one MEMORY file and one USER file per run; "
                                        "run again for another profile's file.")
    if ("memory" in per_target) != (plan.memory_scope is not None):
        return refuse("scope_required", "Name the memory destination with --scope when, and only when, MEMORY.md is selected.")
    if selection.kind == "native_memory" and _requests_authoritative(home):
        return refuse("native_dormant", "This home is authoritative, so its MEMORY.md/USER.md are dormant (§9.1). "
                      "Roll back with 'hermes memory migrate rollback' first, then migrate.")
    if active_migration_manifests(home):                                  # ruling R45-17; C9
        return refuse(MIGRATION_IN_PROGRESS,
                      "A memory migration is already in progress: resume it, roll it back or reconcile it.")
    return None


class _Run:
    def __init__(self, *, home: Path, provider: str, provider_epoch: str, run_id: str, identity: mm.Identity,
                 source: mm.Source, batches: Sequence[mm.Batch], archive: Optional[Path], clock, locks: ExitStack,
                 items: Sequence[SourceItem] = (), exists: bool = False, created_at: Optional[str] = None,
                 updated_at: Optional[str] = None) -> None:
        self.home, self.provider, self.epoch, self.run_id = Path(home), provider, provider_epoch, run_id
        self.identity, self.source, self.batches = identity, source, list(batches)
        self.archive, self.clock, self.locks, self.exists = archive, clock, locks, exists
        self.created_at, self.updated_at = created_at, updated_at
        self.path = mm.manifest_path(self.home, provider, provider_epoch, run_id)
        self._items: Dict[str, SourceItem] = {i.item_key: i for i in items}
        self._restages: Dict[str, int] = {}
        self.warnings: List[str] = []

    # -- the manifest --------------------------------------------------------------------------------------

    def manifest(self) -> mm.ActiveManifest:
        return mm.ActiveManifest(self.provider, self.epoch, self.run_id, self.identity, self.source, tuple(self.batches),
                                 created_at=self.created_at, updated_at=self.updated_at)

    def persist(self, index: int, batch: mm.Batch) -> None:
        """The only writer of active state. The first write takes the migration lock (ruling R45-4) and
        re-checks C9 under it (R45-22); every write stamps updated_at, the first also created_at (R45-6)."""
        from hermes_cli.backup_memory import MIGRATION_IN_PROGRESS, active_migration_manifests
        previous, stamps = self.batches[index], (self.created_at, self.updated_at)
        now = rfc3339(self.clock())
        self.batches[index] = batch
        self.created_at, self.updated_at = (self.created_at or now), now
        try:
            if not self.exists:
                self.locks.enter_context(mm.migration_lock(self.home))
                if active_migration_manifests(self.home):
                    raise _Stop(MIGRATION_IN_PROGRESS, "Another memory migration became active meanwhile; nothing "
                                "was recorded. Resume, roll back or reconcile it; this run's stage expires on its own.")
            mm.write_document(self.path, mm.to_document(self.manifest()))
        except (OSError, mm.MigrationBusyError, _Stop) as exc:
            self.batches[index] = previous
            self.created_at, self.updated_at = stamps
            if isinstance(exc, _Stop):
                raise
            code = getattr(exc, "code", None) or "state_not_recorded"
            raise _Stop(code, "The migration state could not be recorded; nothing new was saved. "
                              + ("Run the command again." if not self.exists else
                                 "Run 'hermes memory migrate resume'.")) from exc
        self.exists = True

    # -- preflight (§9.9 steps 1-2) ------------------------------------------------------------------------

    @classmethod
    def start(cls, service: Any, plan: MigrationPlan, *, provider: str, home: Path, clock, locks: ExitStack) -> "_Run":
        epoch = service.session_state.provider_epoch
        if not mm.valid_epoch(epoch):
            raise _Refuse("provider_epoch_unusable", "The provider's epoch cannot name a migration directory; "
                                                     "nothing was sent.")
        keys = tuple(plan.source.item_keys)
        targets = [t for t in _TARGET_ORDER if any(target_for(k) == t for k in keys)]
        snapshots: Dict[str, w.CuratedSnapshot] = {}
        for target in targets:
            if not service.target_enabled(target):
                flag = "memory_enabled" if target == "memory" else "user_profile_enabled"
                raise _Refuse("target_disabled", f"memory.{flag} is off; nothing was sent.")
            try:
                snapshot = service.load_curated(target)
            except MemoryServiceError as exc:
                raise _Refuse(getattr(exc, "code", None) or "unavailable",
                              "The memory provider is unavailable; nothing was sent.") from exc
            if snapshot.revision.provider_epoch != epoch:
                raise _Refuse("provider_epoch_changed", "The provider epoch changed while the migration started; "
                                                        "nothing was sent.")
            snapshots[target] = snapshot
        destinations: Dict[str, w.ScopeRef] = {}
        if "memory" in snapshots:
            eligible = snapshots["memory"].eligible_write_scopes or ()
            if plan.memory_scope not in eligible:
                raise _Refuse("unauthorized_scope", "Eligible destinations for this identity: "
                              + (", ".join(format_scope(s) for s in eligible) or "none") + ". Nothing was sent.")
            destinations["memory"] = plan.memory_scope
        if "user" in snapshots:
            if snapshots["user"].default_write_scope is None:
                raise _Refuse("scope_unresolved", "The provider resolved no user-profile scope for this identity; "
                                                  "nothing was sent.")
            destinations["user"] = snapshots["user"].default_write_scope
        try:
            items = cls._read(plan.source.kind, home, plan.source.archive, keys)
            scan_items(items)
        except SourceError as exc:
            raise _Refuse(exc.code, str(exc)) from exc
        for item in items:
            limit = snapshots[item.target].limits.max_entry_chars
            for ordinal, text in enumerate(item.entries, 1):
                if len(text) > limit:
                    raise _Refuse("limit_exceeded", f"{item.item_key} entry {ordinal} is longer than {limit} "
                                                    "characters; nothing was sent.")
        batches = [mm.Batch(batch_id=_hex8(), target=t, destination_scope=destinations[t],
                            requested_write_scopes=(destinations[t],),
                            items=tuple(mm.Item(i.item_key) for i in items if i.target == t and i.entries))
                   for t in targets if any(i.target == t and i.entries for i in items)]
        if not batches:
            raise _Refuse("nothing_to_migrate", "The selected items hold no entries; nothing was sent.")
        source = mm.Source(plan.source.kind, mm.SOURCE_PARSERS[plan.source.kind], plan.source.source_id)
        return cls(home=home, provider=provider, provider_epoch=epoch, run_id=uuid.uuid4().hex,
                   identity=mm.Identity.of(service.identity), source=source, batches=batches,
                   archive=plan.source.archive, clock=clock, locks=locks, items=items)

    @staticmethod
    def _read(kind: str, home: Path, archive: Optional[Path], keys: Tuple[str, ...]) -> List[SourceItem]:
        return read_live_items(home, keys)

    # -- staging (§9.9 step 3) -----------------------------------------------------------------------------

    def candidates(self, batch: mm.Batch, items: Sequence[SourceItem],
                   refs: Optional[Tuple[str, ...]] = None) -> Tuple[w.CandidateEntry, ...]:
        texts = [(item.item_key, text) for item in items for text in item.entries]
        if refs is None:
            refs = ()
            while len(set(refs)) != len(texts):
                refs = tuple(_hex8() for _ in texts)
        if len(refs) != len(texts) or not texts:
            raise _Stop("source_changed", "The native source changed since it was staged; roll back and start again.")
        return tuple(w.CandidateEntry(
            client_ref=ref, text=text, destination_scope=batch.destination_scope, target=batch.target,
            proposed_policy_key=None,
            import_source_identity=w.ImportSourceIdentity(source_kind=self.source.source_kind,
                                                          parser_version=self.source.parser_version,
                                                          source_id=self.source.source_id, item_key=key))
            for ref, (key, text) in zip(refs, texts))

    def _request(self, snapshot: w.CuratedSnapshot, batch: mm.Batch,
                 candidates: Tuple[w.CandidateEntry, ...]) -> MutationRequest:
        return MutationRequest(
            target=batch.target, request_id=uuid.uuid4().hex, expected_revision=snapshot.revision,
            hidden_preservation_state=snapshot.hidden_preservation_state,
            requested_write_scopes=batch.requested_write_scopes,
            intent=w.MutationIntent(kind="import", import_run_id=self.run_id, source_kind=self.source.source_kind),
            mutation_delta=tuple(w.MutationDeltaItem(action="add", client_ref=c.client_ref) for c in candidates),
            candidate_entries=candidates,
            provenance=w.MutationProvenance(actor_kind="import", principal_id=self.identity.principal_id,
                                            logical_session_id=self.identity.logical_session_id,
                                            initiating_surface=SURFACE, source_entry_ids=(), source_commit=None,
                                            threat_decision_id=None))

    def stage(self, service: Any, batch: mm.Batch, candidates) -> Tuple[w.StageResult, MutationRequest]:
        snapshot = self._load(service, batch.target)
        for _ in range(MAX_STAGE_ATTEMPTS):
            request = self._request(snapshot, batch, candidates)
            predicted = predict_approval_requirements(snapshot, PlannedMutation(
                intent=request.intent, mutation_delta=request.mutation_delta, candidate_entries=candidates,
                requested_write_scopes=batch.requested_write_scopes, projected_texts=(), message=""))
            try:
                stage = _exact(service, batch.target, lambda: service.stage_curated(request))
            except ProviderError as exc:
                if exc.code == "version_conflict":           # §9.5 L1558: reload, new request ID
                    snapshot = self._load(service, batch.target)
                    continue
                if exc.outcome == "unknown":
                    raise _Stop("outcome_unknown", "The provider did not confirm the stage. Nothing was recorded "
                                                   "for this batch; any stage it made expires on its own.") from exc
                raise _Rejected(exc.code, _detail(exc)) from exc
            except _Unknown:
                raise _Stop("outcome_unknown", "The provider did not confirm the stage. Nothing was recorded for "
                                               "this batch; any stage it made expires on its own.") from None
            except MemoryBlockedError as exc:
                raise _Stop(exc.code or "unavailable", "The memory provider is unavailable; nothing new was saved.") from exc
            if not set(stage.approval_requirements) <= set(predicted):     # K-5; ruling R39-10's rule
                raise _Rejected("approval_requirements_mismatch")
            refs = tuple(c.client_ref for c in candidates)
            if (tuple(h.client_ref for h in stage.candidate_hashes) != refs
                    or tuple(a.client_ref for a in stage.admissions) != refs
                    or any(a.origin_scope != batch.destination_scope for a in stage.admissions)):
                raise _Stop("invalid_reply", "The provider's stage reply does not match the request; nothing was approved.")
            return stage, request
        raise _Rejected("conflict_exhausted")

    def _load(self, service: Any, target: str) -> w.CuratedSnapshot:
        try:
            snapshot = service.load_curated(target)
        except MemoryServiceError as exc:
            raise _Stop(getattr(exc, "code", None) or "unavailable", "The memory provider is unavailable.") from exc
        if snapshot.revision.provider_epoch != self.epoch:
            raise _Stop("provider_epoch_changed", "The provider epoch changed; run 'hermes memory migrate reconcile'.")
        return snapshot

    def stage_pending(self, service: Any, index: int, prompt: PromptFn) -> None:
        # A later batch's first stage is not write-ahead: its request ID is recorded only after a clean
        # StageResult. A crash in that window leaves an expiring provider stage and resume stages the batch anew
        # with a new request ID (§9.5 L1562's pre-manifest rule, per batch); the source tuple deduplicates.
        batch = self.batches[index]
        items = self.items_for(batch, fresh=False)
        stage, request = self.stage(service, batch, self.candidates(batch, items))
        digests = {i.item_key: i.source_sha256 for i in items}
        self.persist(index, replace(
            batch, status="staged", request_id=request.request_id, stage=mm.StageRef.of(stage),
            candidate_hashes=tuple(stage.candidate_hashes),
            admissions=tuple(mm.Admission.of(a) for a in stage.admissions),
            items=tuple(mm.Item(i.item_key, digests[i.item_key]) for i in batch.items)))

    def items_for(self, batch: mm.Batch, *, fresh: bool) -> List[SourceItem]:
        keys = tuple(i.item_key for i in batch.items)
        if not fresh and all(k in self._items for k in keys):
            items = [self._items[k] for k in keys]
        else:
            items = self._reread(keys)
        recorded = {i.item_key: i.source_sha256 for i in batch.items}
        if any(recorded[i.item_key] not in (None, i.source_sha256) for i in items) or not any(i.entries for i in items):
            raise _Stop("source_changed", "The native source changed after it was staged. Nothing was switched; "
                                          "run 'hermes memory migrate rollback' and start again (already imported "
                                          "entries are recognized, never duplicated).")
        return items

    def _reread(self, keys: Tuple[str, ...]) -> List[SourceItem]:
        if self.source.source_kind == "native_memory" and _requests_authoritative(self.home):
            raise _Stop("native_dormant", "The home is authoritative now; its native files are dormant.")
        try:
            items = self._read(self.source.source_kind, self.home, self.archive, keys)
            scan_items(items)
        except SourceError as exc:
            raise _Stop(exc.code, str(exc)) from exc
        self._items.update({i.item_key: i for i in items})
        return items


def start_migration(raw_config: Mapping[str, Any], plan: MigrationPlan, *, home: Path, prompt: PromptFn,
                    switch_to_authoritative: Callable[[], None], backend_factory: Any = None,
                    clock: Callable[[], datetime] = utcnow) -> MigrationReport:
    """C7F-3: preflight, then drive a new run to completion, denial, a stop or a rejection."""
    home = Path(home)
    refusal = _start_refusal(raw_config, plan, home)
    if refusal is not None:
        return refusal
    try:
        with admin_service(_overlay(raw_config), context=plan.context, backend_factory=backend_factory,
                           hermes_home=home) as service, ExitStack() as locks:
            try:
                run = _Run.start(service, plan, provider=_provider(raw_config), home=home, clock=clock, locks=locks)
            except _Refuse as refused:
                return MigrationReport(MigrationStatus.REFUSED, refused.code, refused.message)
            return _drive(run, service, prompt, switch_to_authoritative)
    except ADMIN_FAILURES as exc:
        return MigrationReport(MigrationStatus.REFUSED, getattr(exc, "code", None) or "unavailable",
                               f"The memory provider could not be reached for the migration ({type(exc).__name__}); "
                               "nothing was sent.")


def _drive(run: _Run, service: Any, prompt: PromptFn, switch: Callable[[], None]) -> MigrationReport:
    try:
        for index in range(len(run.batches)):
            while run.batches[index].status != "committed":
                _STEPS[run.batches[index].status](run, service, index, prompt)
        return run.finish(service, switch)
    except _Denied:
        return run.close("operator_denied", MigrationStatus.DENIED, None, "Denied. Nothing more was saved.")
    except _Rejected as rejected:
        detail = f" ({rejected.detail})" if rejected.detail else ""
        if not run.exists:
            return MigrationReport(MigrationStatus.REJECTED, rejected.code,
                                   f"The memory provider rejected the import: {rejected.code}{detail}. Nothing was recorded.")
        return run.close("provider_rejected", MigrationStatus.REJECTED, rejected.proven,
                         f"The memory provider rejected a batch: {rejected.code}{detail}.", code=rejected.code)
    except _Stop as stop:
        return MigrationReport(MigrationStatus.STOPPED, stop.code, stop.message,
                               run_id=run.run_id if run.exists else None)
    except (MemoryServiceError, w.WireError) as exc:
        return MigrationReport(MigrationStatus.STOPPED, getattr(exc, "code", None) or "unavailable",
                               "The memory provider failed mid-migration; run 'hermes memory migrate resume'.",
                               run_id=run.run_id if run.exists else None)


_STEPS: Dict[str, Callable[..., None]] = {"pending": _Run.stage_pending}
