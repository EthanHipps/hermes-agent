"""Archive policy for provider-managed curated memory (§9.8, §9.9, §12.1).

A sibling of ``backup.py`` that both it and ``profiles.py`` import, so the two
facades gain call-site lines only (root ``AGENTS.md``, "Don't recreate god files").
Every rule here is about *names at a Hermes home root*, never about paths below one:

* **Per home, from that home's own config** (ruling R44-2). ``run_backup`` archives
  the whole root, which holds every profile, and profiles are independent islands
  (``hermes_cli/AGENTS.md``). Deciding once for the whole archive would either copy
  an authoritative profile's dormant files (§9.1 L948) or strip an additive
  profile's live memory.
* **Prune before descent.** An authoritative home's ``memories`` is dropped from
  ``os.walk``'s ``dirnames`` or from a ``copytree`` ignore at the root, so no path
  at or below it is ever listed or stat'd. §9.1 L948 forbids stat, and §9.10 L1685
  requires "not used" to be proven, not inferred.
* **Three names are pruned in every mode**: ``migrations/`` (§9.9 L1647/L1651 host
  migration state, which an archive must not carry because a restore cannot
  reconstruct its restart state), the disposition record (regenerated per archive,
  so a stale one — e.g. left by a rollback to additive — never propagates), and
  ``memory_service/``, R40's persisted host session state (ruling X-1 (a),
  contract C3; §9.8 L1639 lets a restored configuration reach the provider only
  through a new ``new_session``).
* **Requested mode, not validated mode** (ruling R44-3), reusing R37's
  never-raising predicate so a broken authoritative install can still be archived.

Nothing here contacts the provider (R44-10): the disposition is config-derived.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Callable, Collection, Iterable, List, NamedTuple, Optional, Tuple

logger = logging.getLogger(__name__)

NATIVE_MEMORY_DIRNAME = "memories"          # tools.memory_tool.get_memory_dir(); the R37 sentinel's directory
MIGRATION_STATE_DIRNAME = "migrations"      # §9.9 L1647 host migration state; never archived (L1647, L1651)
HOST_STATE_DIRNAME = "memory_service"       # R40's persisted host session state; never archived, never restored
#                                             (ruling X-1 (a), contract C3; == host_state.HOST_STATE_ROOT_DIRNAME)
MIGRATION_IN_PROGRESS = "MIGRATION_IN_PROGRESS"


def home_memory_section(home: Path) -> dict:
    """``memory:`` of *home*'s own config.yaml: raw read + env expansion + managed overlay. Never raises.

    Same pipeline as ``doctor_state._doctor_memory_config`` but for any home: an archive
    spans homes that are not the active one (root + profiles/*; ruling R44-2), while
    doctor's helper is private and bound to ``hermes_cli.doctor.HERMES_HOME`` (D-R44-e).
    """
    try:
        from hermes_cli.config import _expand_env_vars, read_user_config_raw
        from hermes_cli.managed_scope import apply_managed_overlay
        config = apply_managed_overlay(_expand_env_vars(read_user_config_raw(Path(home) / "config.yaml")))
        section = config.get("memory") if isinstance(config, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def home_disposition(home: Path):
    """The :class:`ArchiveDisposition` for *home*, or ``None`` when it is not authoritative."""
    from agent.memory_service.archive import archive_disposition
    return archive_disposition({"memory": home_memory_section(home)})


def archive_prune_names(home: Path) -> frozenset:
    """Names dropped from every archive at *home*'s root, before anything descends into them."""
    from agent.memory_service.archive import DISPOSITION_RECORD_NAME
    # Mode-independent (ruling X-1 (a)): host migration state, the regenerated record, and R40's
    # host session state, which §9.8 L1639 keeps out of every archive in every mode.
    names = {MIGRATION_STATE_DIRNAME, DISPOSITION_RECORD_NAME, HOST_STATE_DIRNAME}
    if home_disposition(home) is not None:
        names.add(NATIVE_MEMORY_DIRNAME)
    return frozenset(names)


def archive_homes(root: Path) -> List[Tuple[str, Path]]:
    """``[("", root), ("profiles/<name>/", dir), ...]`` — every Hermes home an archive of *root* spans."""
    root = Path(root)
    homes: List[Tuple[str, Path]] = [("", root)]
    try:
        entries = sorted((p for p in (root / "profiles").iterdir() if p.is_dir() and not p.name.startswith(".")),
                         key=lambda p: p.name)
    except OSError:
        entries = []
    homes.extend((f"profiles/{p.name}/", p) for p in entries)
    return homes


def root_pruning_ignore(
    root: Path,
    names: Collection[str],
    inner: Optional[Callable[[str, list], Iterable[str]]] = None,
) -> Callable[[str, list], set]:
    """A ``copytree`` ignore that drops *names* at *root* only, composing with *inner*."""
    root_key = os.path.normcase(os.path.abspath(root))

    def _ignore(directory: str, contents: list) -> set:
        ignored = set(inner(directory, contents)) if inner else set()
        if os.path.normcase(os.path.abspath(directory)) == root_key:
            ignored.update(name for name in contents if name in names)
        return ignored

    return _ignore


_TERMINAL_MIGRATION_STATES = frozenset({"completed", "rolled_back"})


class MigrationInProgressError(ValueError):
    """§9.8 L1637: backup/export/clone refuse while a memory migration is active.

    A ValueError so existing profile entry points (CLI, slash, REST 400, TUI 4062)
    already report it as a refused request (D-R44-c). The message is content-free.
    """

    code = MIGRATION_IN_PROGRESS

    def __init__(self, manifests: Iterable[Path]) -> None:
        self.manifests = tuple(manifests)
        super().__init__(
            f"{MIGRATION_IN_PROGRESS}: {len(self.manifests)} memory migration(s) in progress. Finish or "
            "roll back the migration before archiving; an archive would omit its restart state.")


def active_migration_manifests(home: Path) -> List[Path]:
    """Migration files under *home* not provably compacted (ruling R44-9: top-level ``state``).

    Cross-repository contract C9, pinned in the ygg ledger as K-9: R45's writer and
    ygg R55's purge inventory must keep ``state`` in {active, completed, rolled_back}.
    Every provider subdirectory is scanned, never just ``memory.provider``'s, so an
    active manifest stays visible after the operator renames the configured provider.
    Anything not provably terminal blocks, because §9.9 L1647 says missing or corrupt
    state blocks resume — an archive of it would omit the restart state.
    """
    root = Path(home) / MIGRATION_STATE_DIRNAME
    if not root.is_dir():
        return []
    try:
        candidates = sorted(root.glob("*/*/*.json"))
    except OSError:
        return [root]  # unlistable state cannot prove "no active migration" (§9.9 L1647)
    active: List[Path] = []
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        if not (isinstance(data, dict) and data.get("state") in _TERMINAL_MIGRATION_STATES):
            active.append(path)
    return active


def refuse_if_migration_in_progress(homes: Iterable[Path]) -> None:
    """Raise :class:`MigrationInProgressError` if any of *homes* has an active migration."""
    active = [manifest for home in homes for manifest in active_migration_manifests(home)]
    if active:
        raise MigrationInProgressError(active)


def disposition_line(home: Path, *, kind: str) -> Optional[str]:
    """The §9.8 L1637 sentence for *home*, or ``None`` in additive mode (nothing is printed)."""
    disposition = home_disposition(home)
    if disposition is None:
        return None
    if kind == "restore":
        return disposition.restore_message()
    return disposition.archive_message(what=kind) + "."
