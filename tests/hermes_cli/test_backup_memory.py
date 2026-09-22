"""S6 archive suite B: per-home archive policy (rulings R44-2, R44-5, R44-8, R44-9)."""

import json
import os
from pathlib import Path

import pytest

from agent.memory_service.archive import DISPOSITION_RECORD_NAME


def _mod():
    import importlib
    return importlib.import_module("hermes_cli.backup_memory")


def _authoritative(home: Path, provider: str = "example") -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        f"memory:\n  provider: {provider}\n  provider_mode: authoritative\n", encoding="utf-8")


def test_each_home_reads_its_own_config(tmp_path):
    root, prof = tmp_path / "root", tmp_path / "root" / "profiles" / "coder"
    root.mkdir()
    (root / "config.yaml").write_text("model: x\n", encoding="utf-8")
    _authoritative(prof)
    m = _mod()
    assert m.home_disposition(root) is None
    assert m.home_disposition(prof).provider == "example"


def test_missing_or_unparseable_config_is_not_authoritative(tmp_path):
    (tmp_path / "config.yaml").write_text("memory: [unclosed\n", encoding="utf-8")
    assert _mod().home_disposition(tmp_path) is None
    assert _mod().home_disposition(tmp_path / "absent") is None


def test_migration_state_record_and_host_state_are_pruned_in_every_mode(tmp_path):
    """Ruling X-1 (a): the host-state directory joins migrations/ and the record as mode-independent."""
    m = _mod()
    additive = tmp_path / "a"
    additive.mkdir()
    authoritative = tmp_path / "b"
    _authoritative(authoritative)
    for home in (additive, authoritative):
        names = m.archive_prune_names(home)
        assert {m.MIGRATION_STATE_DIRNAME, DISPOSITION_RECORD_NAME, m.HOST_STATE_DIRNAME} <= names
    assert m.NATIVE_MEMORY_DIRNAME not in m.archive_prune_names(additive)
    assert m.NATIVE_MEMORY_DIRNAME in m.archive_prune_names(authoritative)


def test_host_state_dirname_is_the_name_r40_writes(tmp_path):
    """Contract C3: one literal, spelled here and in R40's host_state.py. R40 pins the equality when it merges."""
    assert _mod().HOST_STATE_DIRNAME == "memory_service"


def test_archive_homes_are_the_root_then_each_profile_dir(tmp_path):
    (tmp_path / "profiles" / "b").mkdir(parents=True)
    (tmp_path / "profiles" / "a").mkdir()
    (tmp_path / "profiles" / ".deleted").mkdir()
    assert _mod().archive_homes(tmp_path) == [
        ("", tmp_path), ("profiles/a/", tmp_path / "profiles" / "a"), ("profiles/b/", tmp_path / "profiles" / "b")]


def test_root_pruning_ignore_drops_names_only_at_the_root(tmp_path):
    ignore = _mod().root_pruning_ignore(tmp_path, {"memories"})
    assert ignore(str(tmp_path), ["memories", "skills"]) == {"memories"}
    assert ignore(os.path.join(str(tmp_path), "skills"), ["memories"]) == set()


def test_root_pruning_ignore_composes_with_an_inner_ignore(tmp_path):
    ignore = _mod().root_pruning_ignore(tmp_path, {"memories"}, inner=lambda d, names: {".env"} & set(names))
    assert ignore(str(tmp_path), ["memories", ".env", "config.yaml"]) == {"memories", ".env"}


def test_disposition_line_is_none_for_additive_and_says_not_included_otherwise(tmp_path):
    m = _mod()
    additive = tmp_path / "a"
    additive.mkdir()
    authoritative = tmp_path / "b"
    _authoritative(authoritative, provider="ygg")
    assert m.disposition_line(additive, kind="archive") is None
    assert "authoritative ygg memory is provider-managed and not included" in m.disposition_line(
        authoritative, kind="archive")
