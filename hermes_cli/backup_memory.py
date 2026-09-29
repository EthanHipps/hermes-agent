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

    Doctor, the home skeleton and every archive surface share this one pipeline (D-R44-e, unified by
    R41; ruling R41-10); an archive spans homes that are not the active one (root + profiles/*;
    ruling R44-2). An unparseable or unreadable config.yaml falls back to the newest last-known-good
    copy that ``load_config()`` left in ``backups/config/`` (ruling R41-9), so a broken edit cannot
    revive an authoritative home's dormant native memory (§9.1 L950). Only the requested mode comes
    from that copy: unless the result requests authoritative mode, an unparseable config.yaml reads as
    ``{}`` (additive), as before (EDD-67-A1). A failing managed overlay keeps the home's own section, as
    doctor always did. Name and signature are pinned (C6b-10).
    """
    path = Path(home) / "config.yaml"
    parsed = True
    try:
        from hermes_cli.config import read_user_config_raw
        raw = read_user_config_raw(path)
    except Exception:
        parsed = False
        raw = _last_known_good(path)
    try:
        from hermes_cli.config import _expand_env_vars
        config = _expand_env_vars(raw)
    except Exception:
        return {}
    try:
        from hermes_cli.managed_scope import apply_managed_overlay
        config = apply_managed_overlay(config)
    except Exception:
        logger.debug("managed overlay unavailable for a memory-mode read; using the home's own config")
    section = config.get("memory") if isinstance(config, dict) else None
    section = section if isinstance(section, dict) else {}
    if not parsed:
        try:
            from agent.memory_service.bootstrap import requests_authoritative_mode
        except Exception:
            return {}
        if not requests_authoritative_mode({"memory": section}):
            return {}
    return section


def _last_known_good(path: Path) -> dict:
    """The newest ``good`` backup of *path* (ruling R41-9), or ``{}``. Never raises."""
    try:
        from hermes_cli.config_backups import load_newest_good_backup
        return load_newest_good_backup(path) or {}
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
    candidates: List[Path] = []
    active: List[Path] = []  # an unlistable directory cannot prove "no active migration" (§9.9 L1647)
    _walk_migration_state(root, 2, candidates, active)
    active.extend(path for path in sorted(candidates) if not _is_terminal_manifest(path))
    return active


def _walk_migration_state(directory: Path, depth: int, candidates: List[Path], unlistable: List[Path]) -> None:
    """Collect ``*.json`` *depth* directory levels below *directory*; a listing error is never skipped.

    Not ``Path.glob``: CPython 3.11's ``pathlib._WildcardSelector._select_from`` ends with
    ``except PermissionError: return``, so an access-denied level would read as empty (fail open).
    """
    try:
        with os.scandir(directory) as it:
            entries = list(it)
    except OSError:
        unlistable.append(directory)
        return
    for entry in entries:
        path = directory / entry.name
        if depth == 0:
            if os.path.normcase(entry.name).endswith(".json"):  # glob's case rule on this host
                candidates.append(path)
            continue
        try:
            is_dir = entry.is_dir()
        except OSError:
            unlistable.append(path)
            continue
        if is_dir:
            _walk_migration_state(path, depth - 1, candidates, unlistable)


def _is_terminal_manifest(path: Path) -> bool:
    """True only for a readable manifest whose top-level ``state`` is compacted. Never raises.

    ``create_pre_update_backup`` must never raise (R44-8 (a)), so a decoder failure of any kind
    (``RecursionError`` on deep nesting, ``TypeError`` on an unhashable ``state``) blocks instead.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return isinstance(data, dict) and data.get("state") in _TERMINAL_MIGRATION_STATES
    except Exception:
        return False


def refuse_if_migration_in_progress(homes: Iterable[Path]) -> None:
    """Raise :class:`MigrationInProgressError` if any of *homes* has an active migration."""
    active = [manifest for home in homes for manifest in active_migration_manifests(home)]
    if active:
        raise MigrationInProgressError(active)


class RestorePlan(NamedTuple):
    """What a restore writes, what it withholds, and what it must say (§9.8 L1639-1641)."""

    restore: List[str]                                   # members to hand to _import_members
    withheld_native: List[str]
    withheld_external: List[str]
    dispositions: List[Tuple[str, object]]               # (home label, ArchiveDisposition) restored authoritative
    switched_to_additive: List[str]                      # home labels flipped authoritative -> additive
    withheld_host_state: List[str]                       # X-1 (a): <prefix>memory_service/** in EVERY mode


def _archive_home_disposition(zf, member: str):
    """The disposition an archived ``config.yaml`` member declares, or ``None``. Never raises."""
    from agent.memory_service.archive import archive_disposition
    try:
        from hermes_cli.config import _expand_env_vars
        from hermes_cli.managed_scope import apply_managed_overlay
        from utils import fast_safe_load
        data = fast_safe_load(zf.read(member).decode("utf-8")) or {}
        return archive_disposition(apply_managed_overlay(_expand_env_vars(data)))
    except Exception:
        return None


def plan_restore(zf, members: List[str], prefix: str, target_root: Path, *, external_prefix: str) -> RestorePlan:
    """Decide every member's fate BEFORE anything is written (§9.8 L1639).

    Each home's mode comes from that home's own ``config.yaml`` *inside the archive*,
    falling back to the target's current config when the archive carries none
    (ruling R44-2). Planning ahead of the first write is what makes the sentinel's
    "never created" provable: an authoritative home's native members are never
    handed to ``_import_members`` at all, rather than written and removed.
    """
    rels = {member: (member[len(prefix):] if prefix and member.startswith(prefix) else member)
            for member in members}

    home_prefixes = [""]
    for rel in rels.values():
        parts = rel.split("/")
        if len(parts) > 2 and parts[0] == "profiles" and parts[1]:
            candidate = f"profiles/{parts[1]}/"
            if candidate not in home_prefixes:
                home_prefixes.append(candidate)
    # Longest first, so a profile member is never attributed to the root home.
    ordered = sorted(home_prefixes, key=len, reverse=True)

    after_by_home = {}
    dispositions: List[Tuple[str, object]] = []
    switched: List[str] = []
    for home_prefix in home_prefixes:
        config_member = f"{prefix}{home_prefix}config.yaml"
        has_config = config_member in rels
        before = home_disposition(Path(target_root) / home_prefix)
        after = _archive_home_disposition(zf, config_member) if has_config else before
        after_by_home[home_prefix] = after
        if after is not None:
            dispositions.append((home_prefix, after))
        elif before is not None and has_config:
            # §9.9 L1662: an explicit rollback to additive must warn that the dormant
            # native files are stale. R44-7 (a) allows the flip; it does not hide it.
            switched.append(home_prefix)

    restore: List[str] = []
    withheld_native: List[str] = []
    withheld_external: List[str] = []
    withheld_host_state: List[str] = []
    for member, rel in rels.items():
        if rel.startswith(external_prefix):
            # §9.8 L1639 forbids reporting memory restoration, and _external/ restore prints
            # exactly that line, so an authoritative root withholds provider state too (R44-6).
            (withheld_external if after_by_home[""] is not None else restore).append(member)
            continue
        home_prefix = next((h for h in ordered if rel.startswith(h)), "")
        if _is_under(rel, home_prefix, HOST_STATE_DIRNAME):
            # X-1 (a): host session state never restores, in EVERY mode. §9.8 L1639 lets a
            # restored configuration reach the provider only through a new ``new_session``.
            withheld_host_state.append(member)
        elif after_by_home[home_prefix] is not None and _is_under(rel, home_prefix, NATIVE_MEMORY_DIRNAME):
            withheld_native.append(member)
        else:
            restore.append(member)
    return RestorePlan(restore, withheld_native, withheld_external, dispositions, switched, withheld_host_state)


def _is_under(rel: str, home_prefix: str, dirname: str) -> bool:
    base = f"{home_prefix}{dirname}"
    return rel == base or rel.startswith(base + "/")


def withhold_native_memory(staged_home: Path) -> bool:
    """Drop ``memories/`` from an authoritative *staged* copy before it is published (R44-6).

    Only ever called on a temporary staging directory, never on a live home.
    """
    if home_disposition(staged_home) is None:
        return False
    native = Path(staged_home) / NATIVE_MEMORY_DIRNAME
    if not native.is_dir():
        return False
    _drop_staged(native)
    return True


def withhold_host_state(staged_home: Path) -> bool:
    """Drop ``memory_service/`` from a *staged* copy in EVERY mode (ruling X-1 (a), contract C3).

    No ``home_disposition`` check: host session state never restores, whatever the mode
    (§9.8 L1639). Only ever called on a temporary staging directory.
    """
    host_state = Path(staged_home) / HOST_STATE_DIRNAME
    if not host_state.is_dir():
        return False
    _drop_staged(host_state)
    return True


def _drop_staged(directory: Path) -> None:
    """Delete a withheld *directory* from a staging copy, or refuse the import.

    Extraction applies the tar mode, so a legacy 0o444 member is read-only here; the house
    handler clears that and retries. A failure it cannot clear (a scanner's handle) leaves a
    survivor, which would be published into the live profile (§9.8 L1639). A ValueError is a
    clean refusal on every import entry point (D-R44-c); the staging tree is then discarded.
    """
    from hermes_cli.profiles import _rmtree_make_writable
    try:
        try:
            shutil.rmtree(directory, onexc=_rmtree_make_writable)
        except TypeError:  # ``onexc`` is 3.12+; 3.11 has ``onerror``
            shutil.rmtree(directory, onerror=_rmtree_make_writable)
    except OSError:
        pass
    if os.path.lexists(directory):
        raise ValueError(f"Could not withhold {directory.name}/ from the imported profile "
                         "(a file in it could not be deleted); nothing was imported.")


def stale_native_warning(label: str = "") -> str:
    """§9.9 L1662 (ruling R44-7): a restore that flips a home authoritative -> additive says so.

    One sentence for every restore surface R44-1 lists (``hermes import`` and ``/snapshot restore``).
    """
    return (f"Warning: this restore switches {label.rstrip('/') or 'this home'} from authoritative to "
            "additive memory. Its dormant MEMORY.md/USER.md are stale; authoritative changes are not in them.")


def disposition_line(home: Path, *, kind: str) -> Optional[str]:
    """The §9.8 L1637 sentence for *home*, or ``None`` in additive mode (nothing is printed)."""
    disposition = home_disposition(home)
    if disposition is None:
        return None
    if kind == "restore":
        return disposition.restore_message()
    return disposition.archive_message(what=kind) + "."
