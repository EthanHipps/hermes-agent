"""``hermes memory migrate start|resume|status|rollback|reconcile`` (§9.9; R45; ruling R45-1; contract C7F-5).

CLI only and interactive: the import approval is mandatory (§9.5 L1537) and cannot be waived, so
``start``/``resume`` refuse a non-interactive stdin before any provider contact. The approval text,
which holds the candidate bodies, goes to this terminal only (§9.3 L1359). This module owns the one
config.yaml writer of ``memory.provider_mode`` (rulings R45-14, R45-15), guarded like ``hermes config set``
and comment-preserving (``utils.atomic_roundtrip_yaml_update``).
Exit codes: 0 = the operator's decision was carried out; 1 = stopped or rejected, and state may remain;
2 = refused; the command changed no migration state, configuration or memory (for start, nothing was staged):
arguments, non-interactive, configuration, ``MIGRATION_IN_PROGRESS``, native dormant, a host scan hit, an
ineligible scope, nothing to migrate; for resume, rollback and reconcile also an ambiguous, unreadable or
unprovable run, which stays as it was.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

STALE_NATIVE_AFTER_ROLLBACK = ("Memory is additive again. The dormant MEMORY.md/USER.md are stale: authoritative "
                               "changes made since the migration are not in them, and nothing was copied back "
                               "from the provider.")
STATELESS_ROLLBACK_REFUSED = ("Refused: remove memory.authoritative_failure_policy from config.yaml (stateless is "
                              "valid only in authoritative mode), then rerun 'hermes memory migrate rollback'. "
                              "Nothing was changed.")
_NOT_INTERACTIVE = ("  'hermes memory migrate {sub}' needs an interactive terminal: every import needs your "
                    "approval (§9.5), and it cannot be waived. Nothing was sent.")
_EXIT = {"completed": 0, "denied": 0, "rolled_back": 0, "reconciled": 0, "refused": 2, "rejected": 1, "stopped": 1}


def _interactive() -> bool:
    return bool(getattr(sys.stdin, "isatty", lambda: False)())


def _ask(question: str) -> bool:
    try:
        return input(question).strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def _console_prompt(approval) -> Optional[bool]:
    print("\n" + approval.text)
    return _ask("\n  Approve this import? [y/N] ")


def _home() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home()


def _raw_config(home: Path) -> dict:
    from hermes_cli.backup_memory import home_memory_section
    return {"memory": dict(home_memory_section(home))}


def mode_switch_refusal(home: Path) -> Optional[str]:
    """Why ``memory.provider_mode`` cannot be written here, or None (ruling R45-14; never on a last-known-good read)."""
    from hermes_cli import managed_scope
    from hermes_cli.config import is_managed, require_readable_config_before_write
    if is_managed():
        return "This Hermes install is package-managed; config.yaml cannot be changed by this command."
    if managed_scope.is_key_managed("memory.provider_mode"):
        return "memory.provider_mode is set by your administrator's managed configuration."
    try:
        require_readable_config_before_write(Path(home) / "config.yaml")
    except RuntimeError as exc:
        return str(exc)
    return None


def write_provider_mode(home: Path, mode: str) -> None:
    from utils import atomic_roundtrip_yaml_update
    atomic_roundtrip_yaml_update(Path(home) / "config.yaml", "memory.provider_mode", mode)


def _switcher(home: Path) -> Callable[[], None]:
    """The engine's ``switch_to_authoritative``: re-checks the guards at switch time (a resume may come later)."""
    def switch() -> None:
        refusal = mode_switch_refusal(home)
        if refusal is not None:
            raise RuntimeError(refusal)
        write_provider_mode(home, "authoritative")
    return switch


def _print_report(report) -> int:
    print(f"\n  {report.message}")
    if report.code and report.status.value not in ("completed", "denied", "rolled_back", "reconciled"):
        print(f"  ({report.code})")
    for b in report.batches:
        withheld = "" if b.withheld is None else f", {b.withheld} withheld raw"
        print(f"    {b.target}: {b.created} new, {b.reused} already present{withheld} ({b.status})")
    for warning in report.warnings:
        print(f"  Warning: {warning}")
    print()
    return _EXIT[report.status.value]


def _start(args: Any) -> int:
    from agent.memory_service.admin import AdminContext, AdminIdentityError, parse_scope_selector
    from agent.memory_service.migration import MigrationPlan, SourceSelection, start_migration
    home = _home()
    if not _interactive():
        print(_NOT_INTERACTIVE.format(sub="start"))
        return 2
    try:
        context = AdminContext.from_fields(args.organization, args.project, args.repository)
        scope = parse_scope_selector(args.scope) if args.scope else None
    except AdminIdentityError as exc:
        print(f"\n  {exc}. Nothing was sent.\n")
        return 2
    archive = Path(args.archive) if args.archive else None
    kind = "legacy_archive" if archive is not None else "native_memory"
    if kind == "native_memory":
        refusal = mode_switch_refusal(home)
        if refusal is not None:
            print(f"\n  {refusal} Nothing was sent.\n")
            return 2
    plan = MigrationPlan(SourceSelection(kind, args.source_id, tuple(args.item or ()), archive), scope, context)
    report = start_migration(_raw_config(home), plan, home=home, prompt=_console_prompt,
                             switch_to_authoritative=_switcher(home))
    return _print_report(report)


def _resume(args: Any) -> int:
    from agent.memory_service.migration import resume_migration
    home = _home()
    if not _interactive():
        print(_NOT_INTERACTIVE.format(sub="resume"))
        return 2
    report = resume_migration(_raw_config(home), home=home, prompt=_console_prompt,
                              switch_to_authoritative=_switcher(home), run_id=args.run_id,
                              archive=Path(args.archive) if args.archive else None)
    return _print_report(report)


def _status(args: Any) -> int:
    from agent.memory_service.migration_recovery import migration_summaries
    from hermes_cli.backup_memory import active_migration_manifests
    home = _home()
    provider = _raw_config(home)["memory"].get("provider")
    provider = provider.strip() if isinstance(provider, str) else ""
    summaries = migration_summaries(home, provider) if provider else []
    print("\n  Memory migrations" + (f" ({provider})" if provider else ""))
    if not summaries:
        print("    none")
    for s in summaries:
        state = s.state or "unreadable"
        outcome = f"/{s.outcome}" if s.outcome else ""
        blocks = "; blocks archives" if s.blocks_archives else ""
        print(f"    {s.run_id}: {state}{outcome}{blocks}" + (f" ({s.error})" if s.error else ""))
        for b in s.batches:
            withheld = "" if b.withheld is None else f", {b.withheld} withheld raw"
            # A receipt keeps no disposition (R45-impl-3), so its create count includes withheld-raw admissions.
            new = "new" if b.withheld is not None else "new incl. any withheld raw"
            print(f"      {b.target} ({', '.join(b.item_keys)}): {b.created} {new}, {b.reused} already present"
                  f"{withheld} ({b.status})")
    blocking = len(active_migration_manifests(home))
    if blocking != sum(s.blocks_archives for s in summaries):
        print(f"    {blocking} migration file(s) in progress block archives in total "
              "(including other providers' directories); settle each with 'hermes memory migrate reconcile <run>'.")
    print()
    return 0


def _rollback(args: Any) -> int:
    from agent.memory_service.bootstrap import requests_authoritative_mode
    from agent.memory_service.migration import MigrationStatus
    from agent.memory_service.migration_recovery import rollback_migrations
    home = _home()
    section = _raw_config(home)["memory"]
    authoritative = requests_authoritative_mode({"memory": section})
    if authoritative:
        refusal = mode_switch_refusal(home)
        if refusal is not None:
            print(f"\n  {refusal} Nothing was changed.\n")
            return 2
        if section.get("authoritative_failure_policy") == "stateless":        # ruling R45-15 (Checkpoint A)
            print(f"\n  {STATELESS_ROLLBACK_REFUSED}\n")
            return 2
    if not getattr(args, "yes", False):
        if not _interactive():
            print("\n  'hermes memory migrate rollback' needs an interactive terminal or --yes. Nothing was changed.\n")
            return 2
        if not _ask("\n  Close every active migration and return memory to additive mode? [y/N] "):
            print("  Nothing was changed.\n")
            return 0
    report = rollback_migrations(_raw_config(home), home=home)
    code = _print_report(report)
    if report.status is not MigrationStatus.ROLLED_BACK or not authoritative:
        return code
    try:
        write_provider_mode(home, "additive")
    except Exception as exc:
        print(f"  memory.provider_mode could not be written ({type(exc).__name__}); set it to additive yourself.\n")
        return 1
    print(f"  {STALE_NATIVE_AFTER_ROLLBACK}\n")
    return 0


def _reconcile(args: Any) -> int:
    from agent.memory_service.migration_recovery import reconcile_migration
    home = _home()
    if args.discard and not getattr(args, "yes", False):
        if not _interactive():
            print("\n  --discard needs an interactive terminal or --yes. Nothing was changed.\n")
            return 2
        if not _ask("\n  Record unprovable batches as unknown (or delete unreadable state)? [y/N] "):
            print("  Nothing was changed.\n")
            return 0
    report = reconcile_migration(_raw_config(home), args.run_id, home=home, discard=bool(args.discard))
    return _print_report(report)


_HANDLERS: Dict[str, Callable[[Any], int]] = {"start": _start, "resume": _resume, "status": _status,
                                              "rollback": _rollback, "reconcile": _reconcile}


def migrate_command(args: Any) -> int:
    handler = _HANDLERS.get(getattr(args, "migrate_command", None))
    if handler is None:
        print("\n  Use: hermes memory migrate start|resume|status|rollback|reconcile\n")
        return 2
    return handler(args)
