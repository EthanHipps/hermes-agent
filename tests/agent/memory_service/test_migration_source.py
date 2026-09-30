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
