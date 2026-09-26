"""Persisted host session state (§4.1 L250; ruling R40-4a).

One owner-only JSON record per Hermes session id holds the provider epoch and the
complete FrozenMemoryIdentity (HostSessionState), or marks the session stateless.
Resume, branch and compression read it; they never bind (§9.3 L1233). The record is
host state under §9.3 L987: never logged, never shown, and excluded from every
Hermes archive (R44 prunes the same literal, spelled independently as
hermes_cli/backup_memory.HOST_STATE_DIRNAME; see HOST_STATE_ROOT_DIRNAME).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

from agent.memory_service.errors import BindingInvalidError
from agent.memory_service.identity import HostSessionState

logger = logging.getLogger(__name__)

HOST_STATE_SCHEMA = "hermes.memory-host-session/v1"
#: Contract C2 / ruling X-1 (a): the home-root directory name R44 prunes from every
#: archive in every mode and withholds on every restore (hermes_cli/backup_memory.py
#: spells the same literal as HOST_STATE_DIRNAME; the later merger pins the equality).
HOST_STATE_ROOT_DIRNAME = "memory_service"
_DISPOSITIONS = ("provider_authoritative", "stateless")


@dataclass(frozen=True)
class HostStateRecord:
    session_id: str
    disposition: str
    state: Optional[HostSessionState]
    prompt_sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"schema": HOST_STATE_SCHEMA, "session_id": self.session_id, "disposition": self.disposition,
                "state": self.state.to_dict() if self.state else None, "prompt_sha256": self.prompt_sha256}

    @classmethod
    def from_dict(cls, data: Any, *, session_id: str) -> "HostStateRecord":
        if not isinstance(data, dict) or data.get("schema") != HOST_STATE_SCHEMA:
            raise BindingInvalidError("host session state has an unknown schema")
        if data.get("session_id") != session_id or data.get("disposition") not in _DISPOSITIONS:
            raise BindingInvalidError("host session state does not match this session")
        raw_state = data.get("state")
        state = HostSessionState.from_dict(raw_state) if raw_state is not None else None
        if (state is None) != (data["disposition"] == "stateless"):
            raise BindingInvalidError("host session state is missing a member")
        digest = data.get("prompt_sha256")
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64):
            raise BindingInvalidError("host session state has a malformed prompt digest")
        return cls(session_id=session_id, disposition=data["disposition"], state=state, prompt_sha256=digest)


def host_state_dir(hermes_home: Optional[Path] = None) -> Path:
    if hermes_home is None:
        from hermes_constants import get_hermes_home

        hermes_home = get_hermes_home()
    return Path(hermes_home) / HOST_STATE_ROOT_DIRNAME / "sessions"


def _path(session_id: str, hermes_home: Optional[Path]) -> Path:
    return host_state_dir(hermes_home) / (hashlib.sha256(session_id.encode("utf-8")).hexdigest() + ".json")


def load_host_state(session_id: str, *, hermes_home: Optional[Path] = None) -> Optional[HostStateRecord]:
    path = _path(session_id, hermes_home)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BindingInvalidError("host session state is unreadable") from exc
    return HostStateRecord.from_dict(data, session_id=session_id)


def save_host_state(record: HostStateRecord, *, hermes_home: Optional[Path] = None) -> None:
    from utils import atomic_json_write

    atomic_json_write(_path(record.session_id, hermes_home), record.to_dict(), mode=0o600)


def inherit_host_state(parent_session_id: str, child_session_id: str, *, hermes_home: Optional[Path] = None) -> Optional[HostStateRecord]:
    parent = load_host_state(parent_session_id, hermes_home=hermes_home)
    if parent is None:
        return None
    child = replace(parent, session_id=child_session_id)
    save_host_state(child, hermes_home=hermes_home)
    return child
