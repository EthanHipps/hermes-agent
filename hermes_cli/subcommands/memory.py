"""``hermes memory`` subcommand parser."""

from __future__ import annotations

from typing import Callable

from hermes_cli.subcommands._shared import add_yes_flag


def build_memory_parser(subparsers, *, cmd_memory: Callable) -> None:
    """Attach the ``memory`` subcommand to ``subparsers``."""
    memory_parser = subparsers.add_parser(
        "memory", help="Configure external memory provider",
        description="Set up and manage external memory provider plugins.\n\n"
            "Available providers: honcho, openviking, mem0, hindsight,\n"
            "holographic, retaindb, byterover.\n\n"
            "Only one external provider can be active at a time.\n"
            "Built-in memory (MEMORY.md/USER.md) is always active.")
    memory_sub = memory_parser.add_subparsers(dest="memory_command")
    _setup_parser = memory_sub.add_parser(
        "setup", help="Interactive provider selection and configuration")
    _setup_parser.add_argument(
        "provider", nargs="?", default=None,
        help="Provider to configure directly (e.g. honcho), skipping the picker")
    _status_parser = memory_sub.add_parser("status", help="Show current memory provider config")
    _status_parser.add_argument(
        "--session", default=None,
        help="authoritative only: show this Hermes session's frozen identity and scopes")
    memory_sub.add_parser("off", help="Disable external provider (built-in only)")
    _reset_parser = memory_sub.add_parser(
        "reset", help="Erase all built-in memory (MEMORY.md and USER.md)")
    add_yes_flag(_reset_parser)
    _reset_parser.add_argument(
        "--target", choices=["all", "memory", "user"], default="all",
        help="Which store to reset: 'all' (default), 'memory', or 'user'")
    _reset_parser.add_argument(
        "--scope", action="append", metavar="SCOPE", default=None,
        help="authoritative only: a scope to reset — global:<principal>, organization:<id>, project:<id> "
             "or repository:<id>; repeatable")
    _migrate_parser = memory_sub.add_parser(
        "migrate", help="Move native MEMORY.md/USER.md into the authoritative memory provider (§9.9)",
        description="Exit codes: 0 = your decision was carried out; 1 = stopped or rejected (state may remain); "
                    "2 = refused: the command changed no migration state, configuration or memory "
                    "(for start: nothing was staged).")
    migrate_sub = _migrate_parser.add_subparsers(dest="migrate_command")
    start = migrate_sub.add_parser("start", help="Stage, review and commit a migration, then switch to authoritative")
    start.add_argument("--source-id", required=True, help="Stable id for this source, e.g. home-default (a-z, 0-9, -)")
    start.add_argument("--item", action="append", required=True, metavar="KEY",
                       help="MEMORY.md or USER.md (with --archive: memories/MEMORY.md, "
                            "profiles/<name>/memories/USER.md); repeatable, at most one per target")
    start.add_argument("--scope", default=None,
                       help="Destination for MEMORY.md: repository:<id>, project:<id> or organization:<id>")
    _add_context_flags(start)
    start.add_argument("--archive", default=None, help="Import from a legacy Hermes backup zip instead of this home")
    resume = migrate_sub.add_parser("resume", help="Continue an interrupted migration")
    resume.add_argument("run_id", nargs="?", default=None)
    resume.add_argument("--archive", default=None, help="The same archive, when a legacy-archive run must restage")
    migrate_sub.add_parser("status", help="List migrations and their state (offline)")
    rollback = migrate_sub.add_parser("rollback", help="Close active migrations and return memory to additive mode")
    add_yes_flag(rollback)
    reconcile = migrate_sub.add_parser("reconcile", help="Settle a migration that cannot resume")
    reconcile.add_argument("run_id")
    reconcile.add_argument("--discard", action="store_true",
                           help="Record unprovable batches as unknown, or delete an unreadable file")
    add_yes_flag(reconcile)
    memory_parser.set_defaults(func=cmd_memory)


def _add_context_flags(parser) -> None:
    """Contract C7F-4: the explicit administrative chain (C6b-5 ``AdminContext.from_fields``); reset reuses it."""
    parser.add_argument("--organization", default=None, help="Registry organization id of the destination chain")
    parser.add_argument("--project", default=None, help="Registry project id of the destination chain")
    parser.add_argument("--repository", default=None, help="Registry repository id of the destination chain")
