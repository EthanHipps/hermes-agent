"""Explicit rollback, reconciliation and status of §9.9 migrations (R45; rulings R45-15, R45-16, R45-18,
R45-23; entry points are contract C7F-3).

A batch's commit state is settled by local proof first: a batch never approved never had a commit sent,
because every send is preceded by the write-ahead authorization under the migration lock. Only an approved,
unrecorded batch needs the provider's ``inspect_staged`` probe (§9.3 L1361): live, ``stage_expired`` or a
receipt-less ``stage_not_found`` prove no acknowledgement (committed receipts remain through the epoch,
§9.5 L1544; the probe is never made across epochs or providers); ``stage_not_found {committed}`` proves one.
Anything else is unprovable. Rollback refuses whole on an unprovable batch; reconcile records it ``unknown``
only with ``discard`` (§9.5 L1546: explicit reconciliation). Reconcile reaches a run under any provider
directory, because C9 blocks on every one of them (R44-9); a run under another provider than
``memory.provider`` is settled by local proof only. Neither command creates the lock file in a home without
migration state. The provider-mode write and the stale-native warning are the CLI's
(``hermes_cli/memory_migrate.py``).
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from agent.memory_service import migration_manifest as mm
from agent.memory_service.admin import ADMIN_FAILURES, AdminContext, admin_service
from agent.memory_service.approval import rfc3339, utcnow
from agent.memory_service.errors import MemoryServiceError, ProviderError
from agent.memory_service.migration import (BatchSummary, MigrationReport, MigrationStatus, _overlay, _provider,
                                            _same_identity)
from agent.memory_service.service import InspectRequest

Settled = Dict[str, Tuple[str, Optional[str]]]


class _Unprovable(Exception):
    pass


def _settle(manifest: mm.ActiveManifest, service: Any, *, discard: bool = False) -> Settled:
    settled: Settled = {}
    for b in manifest.batches:
        if b.status == "committed":
            settled[b.batch_id] = ("committed", b.result.tx_id)
        elif b.authorization is None:
            settled[b.batch_id] = ("not_committed", None)
        else:
            try:
                settled[b.batch_id] = _probe(service, b)
            except _Unprovable:
                if not discard:
                    raise
                settled[b.batch_id] = ("unknown", None)
    return settled


def _probe(service: Any, b: mm.Batch) -> Tuple[str, Optional[str]]:
    if service is None:
        raise _Unprovable()
    try:
        service.inspect_staged(InspectRequest(target=b.target, request_id=b.request_id,
                                              stage_handle_b64url=b.stage.stage_handle_b64url))
        return ("not_committed", None)
    except ProviderError as exc:
        details = exc.details if isinstance(exc.details, dict) else {}
        if exc.code == "stage_not_found" and details.get("state") == "committed":
            return ("committed", str(details["tx_id"]))
        if exc.code in ("stage_expired", "stage_not_found") and exc.outcome != "unknown":
            return ("not_committed", None)
        raise _Unprovable() from exc
    except MemoryServiceError as exc:
        raise _Unprovable() from exc


def _needs_probe(manifest: mm.ActiveManifest) -> bool:
    return any(b.status != "committed" and b.authorization is not None for b in manifest.batches)


def _service_for(raw_config: Mapping[str, Any], run: mm.RunFile, home: Path, backend_factory: Any,
                 stack: ExitStack) -> Optional[Any]:
    """The run's own administrative identity, when a probe is needed and provable; else None (local proof only)."""
    manifest = run.document
    if not _needs_probe(manifest) or run.provider != _provider(raw_config):     # R45-23: a foreign provider
        return None
    context = AdminContext(manifest.identity.org_id, manifest.identity.project_id, manifest.identity.repo_id)
    try:
        service = stack.enter_context(admin_service(_overlay(raw_config), context=context,
                                                    backend_factory=backend_factory, hermes_home=home))
    except ADMIN_FAILURES:
        return None
    if (service.session_state.provider_epoch != manifest.provider_epoch
            or not _same_identity(service.identity, manifest.identity)):
        return None
    return service


def _active(runs: List[mm.RunFile]) -> List[mm.RunFile]:
    return [r for r in runs if r.document is None or r.document.state == mm.ACTIVE]


def _foreign_note(home: Path, provider: str) -> Tuple[str, ...]:
    from hermes_cli.backup_memory import active_migration_manifests
    ours = {r.path for r in _active(mm.list_runs(home, provider))}
    others = [p for p in active_migration_manifests(home) if p not in ours]
    if not others:
        return ()
    names = ", ".join(sorted(p.stem for p in others))
    return (f"Other migration state still blocks archives ({names}); settle each with "
            "'hermes memory migrate reconcile <run>'.",)


def rollback_migrations(raw_config: Mapping[str, Any], *, home: Path, backend_factory: Any = None,
                        clock: Callable[[], datetime] = utcnow) -> MigrationReport:
    """C7F-3 / R45-15: settle every active run of ``memory.provider`` or refuse whole; compact each to operator_rollback."""
    home, provider = Path(home), _provider(raw_config)
    if provider is None:
        return MigrationReport(MigrationStatus.REFUSED, "configuration_error", "memory.provider is not usable.")
    if not _active(mm.list_runs(home, provider)):                   # no state: no lock file either (R45-23)
        return MigrationReport(MigrationStatus.ROLLED_BACK, "nothing_to_roll_back",
                               "No memory migration was in progress.", warnings=_foreign_note(home, provider))
    try:
        with ExitStack() as stack:
            stack.enter_context(mm.migration_lock(home))
            runs = _active(mm.list_runs(home, provider))            # re-read under the lock
            unreadable = [r.import_run_id for r in runs if r.document is None]
            if unreadable:
                return MigrationReport(MigrationStatus.REFUSED, "reconcile_required",
                                       "Unreadable migration state: " + ", ".join(unreadable) + ". Run "
                                       "'hermes memory migrate reconcile <run> --discard' first; nothing was changed.")
            plans = []
            for run in runs:
                try:
                    plans.append((run, _settle(run.document, _service_for(raw_config, run, home, backend_factory,
                                                                          stack))))
                except _Unprovable:
                    return MigrationReport(MigrationStatus.REFUSED, "unprovable",
                                           f"The commit state of run {run.import_run_id} cannot be proven, so nothing "
                                           "was changed. Retry when the memory provider is reachable, or run "
                                           f"'hermes memory migrate reconcile {run.import_run_id} --discard'.")
            now = rfc3339(clock())
            done: List[str] = []
            for run, settled in plans:
                receipt = mm.compact(run.document, state=mm.ROLLED_BACK, outcome="operator_rollback",
                                     settled=settled, updated_at=now)
                try:
                    mm.write_document(run.path, mm.to_document(receipt))
                except OSError:
                    remaining = ", ".join(r.import_run_id for r, _ in plans if r.import_run_id not in done)
                    return MigrationReport(MigrationStatus.STOPPED, "rollback_not_complete",
                                           f"The rollback could not be recorded; still in progress: {remaining}. "
                                           "Nothing else was changed; run 'hermes memory migrate rollback' again.")
                done.append(run.import_run_id)
            return MigrationReport(MigrationStatus.ROLLED_BACK, None,
                                   "Rolled back migration(s): " + ", ".join(done) + ".",
                                   warnings=_foreign_note(home, provider))
    except mm.MigrationBusyError as exc:
        return MigrationReport(MigrationStatus.STOPPED, exc.code, "Another memory migration command is running.")


def reconcile_migration(raw_config: Mapping[str, Any], run_id: str, *, home: Path, discard: bool = False,
                        backend_factory: Any = None, clock: Callable[[], datetime] = utcnow) -> MigrationReport:
    """C7F-3 / R45-16: settle one run by local proof or a probe and compact it to reconciled (or discarded)."""
    home = Path(home)

    def matches() -> List[mm.RunFile]:
        return [r for r in mm.list_runs(home) if r.import_run_id == run_id]

    if not matches():                                              # no state: no lock file either (R45-23)
        return MigrationReport(MigrationStatus.REFUSED, "not_found", f"No migration state for run {run_id}.")
    try:
        with ExitStack() as stack:
            stack.enter_context(mm.migration_lock(home))
            found = matches()                                      # re-read under the lock
            if len(found) != 1:
                return MigrationReport(MigrationStatus.REFUSED, "not_found" if not found else "ambiguous_migration",
                                       f"Run {run_id} is not exactly one migration file; nothing was changed.")
            run = found[0]
            if run.document is None:
                if not discard:
                    return MigrationReport(MigrationStatus.REFUSED, "reconcile_needs_discard",
                                           f"The state of run {run_id} is unreadable. Rerun with --discard to "
                                           "delete it; nothing was changed.", run_id=run_id)
                try:
                    run.path.unlink()
                except OSError:
                    return MigrationReport(MigrationStatus.STOPPED, "reconcile_not_complete",
                                           f"The unreadable state of run {run_id} could not be deleted.", run_id=run_id)
                return MigrationReport(MigrationStatus.RECONCILED, "discarded",
                                       f"Deleted the unreadable state of run {run_id}.", run_id=run_id)
            if run.document.state != mm.ACTIVE:
                return MigrationReport(MigrationStatus.REFUSED, "already_compacted",
                                       f"Run {run_id} is already {run.document.state}.", run_id=run_id)
            try:
                settled = _settle(run.document, _service_for(raw_config, run, home, backend_factory, stack),
                                  discard=discard)
            except _Unprovable:
                return MigrationReport(MigrationStatus.REFUSED, "unprovable",
                                       f"The commit state of run {run_id} cannot be proven. Retry when the memory "
                                       "provider is reachable, or rerun with --discard to record it as unknown; "
                                       "nothing was changed.", run_id=run_id)
            outcome = "discarded" if any(o == "unknown" for o, _ in settled.values()) else "reconciled"
            receipt = mm.compact(run.document, state=mm.ROLLED_BACK, outcome=outcome, settled=settled,
                                 updated_at=rfc3339(clock()))
            try:
                mm.write_document(run.path, mm.to_document(receipt))
            except OSError:
                return MigrationReport(MigrationStatus.STOPPED, "reconcile_not_complete",
                                       f"The reconciliation of run {run_id} could not be recorded.", run_id=run_id)
            return MigrationReport(MigrationStatus.RECONCILED, None, f"Run {run_id} reconciled ({outcome}).",
                                   run_id=run_id)
    except mm.MigrationBusyError as exc:
        return MigrationReport(MigrationStatus.STOPPED, exc.code, "Another memory migration command is running.")


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    state: Optional[str]
    outcome: Optional[str]
    error: Optional[str]
    batches: Tuple[BatchSummary, ...]
    blocks_archives: bool


def _summary(run: mm.RunFile) -> RunSummary:
    doc = run.document
    if doc is None:
        return RunSummary(run.import_run_id, None, None, run.error, (), True)
    if isinstance(doc, mm.ActiveManifest):
        batches = tuple(BatchSummary(
            b.target, tuple(i.item_key for i in b.items),
            created=sum(a.publication_effect == "create_record" and a.disposition != "withheld_raw" for a in b.admissions),
            reused=sum(a.publication_effect == "reuse_existing_import" for a in b.admissions),
            withheld=sum(a.disposition == "withheld_raw" for a in b.admissions), status=b.status) for b in doc.batches)
        return RunSummary(run.import_run_id, doc.state, None, None, batches, True)
    batches = tuple(BatchSummary(
        b.target, b.item_keys, created=sum(a.publication_effect == "create_record" for a in b.assigned),
        reused=sum(a.publication_effect == "reuse_existing_import" for a in b.assigned), withheld=None,
        status=b.outcome) for b in doc.batches)
    return RunSummary(run.import_run_id, doc.state, doc.outcome, None, batches, False)


def migration_summaries(home: Path, provider: str) -> List[RunSummary]:
    """R45-18: offline and content-free (no epoch, handle, hash, revision or text)."""
    return [_summary(run) for run in mm.list_runs(home, provider)]
