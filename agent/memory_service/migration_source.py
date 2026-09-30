"""Native-memory sources of the §9.9 migration (R45): parsers, the host scan and source digests.

``hermes-native-v0.20.6`` reads the active home's ``memories/MEMORY.md`` and ``USER.md`` with the fork's own
MemoryStore semantics (§9.9 L1657): strict UTF-8 with a BOM strip and universal newlines (as
``MemoryStore._read_raw_checked`` reads; a Windows-written file holds CRLF delimiters), the full ``\\n§\\n``
delimiter, stripped non-empty entries, first-wins de-duplication and ``normalize_entry``.
``hermes-legacy-archive-v1`` applies the same parse to a native-memory member of a Hermes backup zip.
Nothing here creates, writes or renames anything (D-R45-7), and nothing here contacts a provider.
Every refusal is content-free: an item key, an entry ordinal and a pattern ID, never text.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from agent.memory_service.errors import MemoryServiceError

NATIVE_PARSER = "hermes-native-v0.20.6"
ARCHIVE_PARSER = "hermes-legacy-archive-v1"
LIVE_ITEMS = {"MEMORY.md": "memory", "USER.md": "user"}
MAX_ITEM_BYTES = 1 << 20
_ARCHIVE_ITEM_RE = re.compile(r"^(?:profiles/[A-Za-z0-9._-]+/)?memories/(?:MEMORY|USER)\.md$")


class SourceError(MemoryServiceError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SourceItem:
    item_key: str
    target: str
    entries: Tuple[str, ...]
    source_sha256: str                  # raw-byte digest (ruling R45-12)


def target_for(item_key: str) -> str:
    return "user" if item_key.rsplit("/", 1)[-1] == "USER.md" else "memory"


def parse_entries(raw: bytes, item_key: str) -> Tuple[str, ...]:
    """Decode exactly as ``MemoryStore._read_raw_checked`` reads (utf-8-sig, universal newlines), then parse natively."""
    from tools.memory_tool_store import MemoryStore
    try:
        text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError:
        raise SourceError("source_unreadable", f"{item_key} is not valid UTF-8; nothing was sent.") from None
    normalized = (MemoryStore.normalize_entry(e) for e in MemoryStore._parse_entries(text))
    return tuple(dict.fromkeys(e for e in normalized if e))


def _item(item_key: str, raw: Optional[bytes]) -> SourceItem:
    data = raw or b""
    return SourceItem(item_key, target_for(item_key), parse_entries(data, item_key) if data else (),
                      hashlib.sha256(data).hexdigest())


def read_live_items(home: Path, item_keys: Sequence[str]) -> List[SourceItem]:
    from hermes_cli.backup_memory import NATIVE_MEMORY_DIRNAME      # C3's name for the native directory
    if not item_keys or len(set(item_keys)) != len(item_keys) or any(k not in LIVE_ITEMS for k in item_keys):
        raise SourceError("source_invalid", "Select MEMORY.md and/or USER.md, each once.")
    items = []
    for key in item_keys:
        path = Path(home) / NATIVE_MEMORY_DIRNAME / key
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            raw = None
        except OSError:
            raise SourceError("source_unreadable", f"{key} could not be read; nothing was sent.") from None
        if raw is not None and len(raw) > MAX_ITEM_BYTES:
            raise SourceError("source_invalid", f"{key} is larger than {MAX_ITEM_BYTES} bytes; nothing was sent.")
        items.append(_item(key, raw))
    return items


def scan_items(items: Iterable[SourceItem]) -> None:
    """Ruling R45-10: the memory tool's strict scan (R38-12) on every entry, already stripped and normalized."""
    from tools.threat_patterns import scan_for_threats
    for item in items:
        for ordinal, entry in enumerate(item.entries, 1):
            findings = scan_for_threats(entry, scope="strict")
            if findings:
                raise SourceError("source_threat", f"{item.item_key} entry {ordinal} matches threat pattern "
                                  f"'{findings[0]}'; nothing was sent. Edit the file and start again.")
