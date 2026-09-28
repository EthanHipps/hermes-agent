"""Handle-only approval records (§9.4 step 8; §9.5 L1540, L1546; rulings R39-2, R39-3; contract C6b-3).

One owner-only JSON record per deferred or in-flight approval, under
``<home>/memory_service/approvals/<pending-id>.json``. It holds the stage handle,
binding hash, request and approval IDs, revision, scopes, decision and expiry. It
also holds the wire-required target and the reference to the staging session's
host-state record. It never holds a candidate body, a visible-entry body, a
candidate hash, an excerpt or a summary (§9.3 L1357). The handle and binding hash
are host state under §9.3 L987, so the directory sits under ``memory_service/``.
R44 keeps that directory out of every archive and withholds it on every restore,
in every mode (ruling X-1 (a)).
"""

from __future__ import annotations

import json
import logging
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.memory_service import wire as w
from agent.memory_service.errors import MemoryServiceError
from agent.memory_service.host_state import HOST_STATE_ROOT_DIRNAME

logger = logging.getLogger(__name__)

APPROVAL_SCHEMA = "hermes.memory-approval/v1"
APPROVALS_DIRNAME = "approvals"
_PENDING_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_TARGETS = frozenset({"memory", "user"})
_INTENTS = frozenset({"add", "replace", "remove", "bulk_edit", "reset", "import"})
_DECISIONS = frozenset({"pending", "approved"})
_STRINGS = ("session_id", "logical_session_id", "provider_epoch", "request_id", "stage_handle_b64url",
            "approval_binding_sha256", "expires_at", "origin", "created_at")


class ApprovalStoreError(MemoryServiceError):
    """The approval store could not be read or written (§9.10 L1689 "approval-store failure")."""


@dataclass(frozen=True)
class PendingApproval:
    pending_id: str
    session_id: str                     # the Hermes session whose host-state record (C2) holds the identity
    logical_session_id: str
    provider_epoch: str
    target: str
    intent_kind: str
    request_id: str
    stage_handle_b64url: str
    approval_binding_sha256: str
    expected_revision: w.CompositeRevision
    requested_write_scopes: Tuple[w.ScopeRef, ...]
    approval_requirements: Tuple[str, ...]
    expires_at: str
    decision: str = "pending"           # pending | approved
    authorization: Optional[w.ApprovalAuthorization] = None   # with "approved": replayed identically (§9.5 L1562)
    outcome: Optional[str] = None       # None | "unknown"
    origin: str = "foreground"
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": APPROVAL_SCHEMA, "pending_id": self.pending_id, "session_id": self.session_id,
            "logical_session_id": self.logical_session_id, "provider_epoch": self.provider_epoch,
            "target": self.target, "intent_kind": self.intent_kind, "request_id": self.request_id,
            "stage_handle_b64url": self.stage_handle_b64url,
            "approval_binding_sha256": self.approval_binding_sha256,
            "expected_revision": self.expected_revision.to_wire(),
            "requested_write_scopes": [scope.to_wire() for scope in self.requested_write_scopes],
            "approval_requirements": list(self.approval_requirements), "expires_at": self.expires_at,
            "decision": self.decision,
            "authorization": self.authorization.to_wire() if self.authorization is not None else None,
            "outcome": self.outcome, "origin": self.origin, "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Any, *, pending_id: str) -> "PendingApproval":
        try:
            if (not isinstance(data, dict) or data.get("schema") != APPROVAL_SCHEMA
                    or data.get("pending_id") != pending_id):
                raise ValueError("schema or id mismatch")
            if not all(isinstance(data.get(key), str) and data[key] for key in _STRINGS):
                raise ValueError("missing member")
            raw_auth, requirements = data["authorization"], data["approval_requirements"]
            if (data["target"] not in _TARGETS or data["intent_kind"] not in _INTENTS
                    or data["decision"] not in _DECISIONS or data["outcome"] not in (None, "unknown")
                    or (raw_auth is None) != (data["decision"] == "pending")
                    or not isinstance(requirements, list) or not all(isinstance(r, str) for r in requirements)):
                raise ValueError("invalid member")
            return cls(
                pending_id=pending_id, session_id=data["session_id"],
                logical_session_id=data["logical_session_id"], provider_epoch=data["provider_epoch"],
                target=data["target"], intent_kind=data["intent_kind"], request_id=data["request_id"],
                stage_handle_b64url=data["stage_handle_b64url"],
                approval_binding_sha256=data["approval_binding_sha256"],
                expected_revision=w.CompositeRevision.from_wire(data["expected_revision"], "$.expected_revision"),
                requested_write_scopes=tuple(w.ScopeRef.from_wire(scope, "$.requested_write_scopes")
                                             for scope in data["requested_write_scopes"]),
                approval_requirements=tuple(requirements), expires_at=data["expires_at"],
                decision=data["decision"],
                authorization=(w.ApprovalAuthorization.from_wire(raw_auth, "$.authorization")
                               if raw_auth is not None else None),
                outcome=data["outcome"], origin=data["origin"], created_at=data["created_at"])
        except (KeyError, TypeError, ValueError) as exc:  # WireError is a ValueError
            raise ApprovalStoreError("approval record is malformed") from exc


def approvals_dir(hermes_home: Optional[Path] = None) -> Path:
    if hermes_home is None:
        from hermes_constants import get_hermes_home

        hermes_home = get_hermes_home()
    return Path(hermes_home) / HOST_STATE_ROOT_DIRNAME / APPROVALS_DIRNAME


def valid_pending_id(value: str) -> bool:
    return isinstance(value, str) and bool(_PENDING_ID_RE.match(value))


def new_pending_id() -> str:
    return secrets.token_hex(6)


def _path(pending_id: str, hermes_home: Optional[Path]) -> Path:
    if not valid_pending_id(pending_id):  # user input: never a path fragment (D-R39-6)
        raise ApprovalStoreError("malformed approval id")
    return approvals_dir(hermes_home) / f"{pending_id}.json"


def save_pending_approval(record: PendingApproval, *, hermes_home: Optional[Path] = None) -> None:
    from utils import atomic_json_write

    path = _path(record.pending_id, hermes_home)
    try:
        atomic_json_write(path, record.to_dict(), mode=0o600)
    except OSError as exc:
        raise ApprovalStoreError("approval record could not be written") from exc


def load_pending_approval(pending_id: str, *, hermes_home: Optional[Path] = None) -> Optional[PendingApproval]:
    path = _path(pending_id, hermes_home)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ApprovalStoreError("approval record is unreadable") from exc
    return PendingApproval.from_dict(data, pending_id=pending_id)


def list_pending_approvals(*, hermes_home: Optional[Path] = None) -> List[Tuple[str, Optional[PendingApproval]]]:
    """Every record, oldest first; ``None`` stands for an unreadable one (reported, never guessed)."""
    directory = approvals_dir(hermes_home)
    if not directory.is_dir():
        return []
    entries: List[Tuple[str, Optional[PendingApproval]]] = []
    for path in directory.glob("*.json"):
        if not valid_pending_id(path.stem):
            continue
        try:
            entries.append((path.stem, load_pending_approval(path.stem, hermes_home=hermes_home)))
        except ApprovalStoreError:
            entries.append((path.stem, None))
    return sorted(entries, key=lambda e: (e[1] is None, e[1].created_at if e[1] is not None else "", e[0]))


def drop_pending_approval(pending_id: str, *, hermes_home: Optional[Path] = None) -> bool:
    try:
        _path(pending_id, hermes_home).unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ApprovalStoreError("approval record could not be removed") from exc
    return True
