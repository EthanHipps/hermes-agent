"""R45: native sources (§9.9 L1657), the host scan (§9.5 L1542; R45-10) and source digests (R45-12)."""

import hashlib

import pytest

from agent.memory_service import migration_source as ms
from tests.agent.memory_service.migration_support import write_native


def test_live_items_parse_with_native_semantics(tmp_path):
    native = write_native(tmp_path, memory=["one", "two\r\nlines", "one", "  "], user=None)
    (native / "USER.md").write_bytes("﻿profile".encode("utf-8"))
    items = ms.read_live_items(tmp_path, ("MEMORY.md", "USER.md"))
    assert [(i.item_key, i.target, i.entries) for i in items] == [
        ("MEMORY.md", "memory", ("one", "two\nlines")), ("USER.md", "user", ("profile",))]
    assert items[0].source_sha256 == hashlib.sha256((native / "MEMORY.md").read_bytes()).hexdigest()


def test_a_crlf_native_file_parses_like_the_native_store(tmp_path):
    """MemoryStore._write_file writes through text-mode atomic_write_text, so a Windows file holds CRLF delimiters."""
    from tools.memory_tool_store import MemoryStore
    native = tmp_path / "memories"
    native.mkdir()
    path = native / "MEMORY.md"
    path.write_bytes("one\r\n§\r\ntwo".encode("utf-8"))
    [item] = ms.read_live_items(tmp_path, ("MEMORY.md",))
    assert item.entries == ("one", "two")
    assert item.entries == tuple(MemoryStore._parse_entries(path.read_text(encoding="utf-8-sig")))
    assert item.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()     # the raw-byte digest stays


def test_a_missing_item_is_empty_and_an_undecodable_one_refuses(tmp_path):
    native = write_native(tmp_path, memory=["x"])
    assert ms.read_live_items(tmp_path, ("USER.md",))[0].entries == ()
    (native / "USER.md").write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(ms.SourceError) as refused:
        ms.read_live_items(tmp_path, ("USER.md",))
    assert refused.value.code == "source_unreadable"


def test_reading_creates_nothing(tmp_path):
    assert ms.read_live_items(tmp_path, ("MEMORY.md",))[0].entries == ()
    assert not (tmp_path / "memories").exists()


@pytest.mark.parametrize("key", ["../MEMORY.md", "memories/MEMORY.md", "NOTES.md", ""])
def test_only_the_two_live_item_keys(tmp_path, key):
    with pytest.raises(ms.SourceError):
        ms.read_live_items(tmp_path, (key,))


def test_a_threat_or_secret_hit_refuses_content_free(tmp_path):
    write_native(tmp_path, memory=["fine", "api_key = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ012345'"])
    items = ms.read_live_items(tmp_path, ("MEMORY.md",))
    with pytest.raises(ms.SourceError) as refused:
        ms.scan_items(items)
    assert refused.value.code == "source_threat"
    assert "MEMORY.md" in str(refused.value) and "entry 2" in str(refused.value) and "ABCDEF" not in str(refused.value)


# -- Task 11: legacy Hermes archives (R45-8; §9.8 L1649, §9.9 L1657) ---------------------------------------

import zipfile  # noqa: E402


def _zip(path, members):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def _archive(tmp_path, prefix=".hermes/"):
    return _zip(tmp_path / "backup.zip", {
        f"{prefix}config.yaml": "memory: {}\n",
        f"{prefix}memories/MEMORY.md": "alpha\n§\nbeta".encode("utf-8"),
        f"{prefix}profiles/work/memories/USER.md": "gamma".encode("utf-8")})


@pytest.mark.parametrize("prefix", [".hermes/", "hermes/", ""])
def test_archive_items_parse_with_native_semantics(tmp_path, prefix):
    archive = _archive(tmp_path, prefix)
    items = ms.read_archive_items(archive, ("memories/MEMORY.md", "profiles/work/memories/USER.md"))
    assert [(i.item_key, i.target, i.entries) for i in items] == [
        ("memories/MEMORY.md", "memory", ("alpha", "beta")), ("profiles/work/memories/USER.md", "user", ("gamma",))]
    with zipfile.ZipFile(archive) as zf:
        assert items[0].source_sha256 == hashlib.sha256(zf.read(f"{prefix}memories/MEMORY.md")).hexdigest()


@pytest.mark.parametrize("key", ["MEMORY.md", "../memories/MEMORY.md", "profiles/a/b/memories/MEMORY.md",
                                 "memories/NOTES.md", "config.yaml"])
def test_only_archive_memory_keys(tmp_path, key):
    with pytest.raises(ms.SourceError) as refused:
        ms.read_archive_items(_archive(tmp_path), (key,))
    assert refused.value.code == "source_invalid"


def test_a_missing_member_refuses(tmp_path):
    with pytest.raises(ms.SourceError) as refused:
        ms.read_archive_items(_archive(tmp_path), ("profiles/home/memories/USER.md",))
    assert refused.value.code == "source_invalid"


def test_an_oversized_member_refuses(tmp_path):
    archive = _zip(tmp_path / "big.zip", {".hermes/config.yaml": "x: 1\n",
                                          ".hermes/memories/MEMORY.md": b"x" * (ms.MAX_ITEM_BYTES + 1)})
    with pytest.raises(ms.SourceError) as refused:
        ms.read_archive_items(archive, ("memories/MEMORY.md",))
    assert refused.value.code == "source_invalid"


def test_an_unreadable_archive_refuses(tmp_path):
    (tmp_path / "broken.zip").write_bytes(b"not a zip")
    with pytest.raises(ms.SourceError) as refused:
        ms.read_archive_items(tmp_path / "broken.zip", ("memories/MEMORY.md",))
    assert refused.value.code == "source_unreadable"


def test_reading_an_archive_changes_and_extracts_nothing(tmp_path):
    archive = _archive(tmp_path)
    before = (archive.read_bytes(), archive.stat().st_mtime_ns)
    listing = sorted(tmp_path.rglob("*"))
    ms.read_archive_items(archive, ("memories/MEMORY.md",))
    assert (archive.read_bytes(), archive.stat().st_mtime_ns) == before
    assert sorted(tmp_path.rglob("*")) == listing


@pytest.mark.parametrize("kind,key,valid", [
    ("native_memory", "MEMORY.md", True), ("native_memory", "memories/MEMORY.md", False),
    ("legacy_archive", "memories/USER.md", True), ("legacy_archive", "USER.md", False)])
def test_item_keys_are_valid_per_source_kind(kind, key, valid):
    assert ms.valid_item_key(kind, key) is valid
