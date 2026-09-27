"""Explicit administrative memory identity for non-session surfaces (contract C6b-5).

The desktop/web memory panel, the journey ("learning graph") and the REST
memory routes are not Hermes sessions, so they never infer identity from the
process directory: §9.2 L968 gives non-session administration an explicit
identity and scope, and §4.2 L271 an explicit administrative scope selector
(§9.7 L1602: a non-agent path uses a live service or an explicit
administrative identity). :class:`AdminContext` is that explicit
org/project/repository chain; all ``None`` is the principal-global identity
(ruling R42-4). :func:`admin_service` binds an ordinary ``new_session`` with
``resolution_source: explicit_ids``, ``canonical_directory: null`` and
``platform: admin``, persists that binding's host state once per context under
``<home>/memory_service/admin/`` (inside the host-state root every archive
already excludes, X-1) and resumes it with negotiate plus validate afterwards,
so a polled surface does not mint a live handle per request (K-2).

This module is the host's ONE owner of non-session administrative identity
(ruling X6b-3; R42-2, R42-3). Administrative surfaces never adopt ``stateless``
(§9.6 L1566, L1586; D-R42-c): a failure is ``unavailable`` or
``configuration_error``. Scheduled surfaces bind through
``agent/memory_service/explicit.py`` (C6b-8) instead. Later rows import this
module and add their own siblings; none edits it.

It also carries the shared scoped-reset planner (§9.3 L1256, L1294), the §11.2
L1837 scope selectors, the config-derived authority report (ruling R42-8) and a
surface-neutral, content-free outcome mapping.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

from agent.memory_service import wire as w
from agent.memory_service.archive import archive_disposition
from agent.memory_service.config import (MemoryConfigurationError, MemoryMode, MemoryServiceConfig,
                                         resolve_memory_service_config)
from agent.memory_service.errors import MemoryBlockedError, MemoryServiceError, ProviderTransportError
from agent.memory_service.host_state import HOST_STATE_ROOT_DIRNAME, HostStateRecord
from agent.memory_service.mutation import MutationOutcome, MutationStatus

logger = logging.getLogger(__name__)

ADMIN_PLATFORM = "admin"
ADMIN_STATE_DIRNAME = "admin"
#: Everything opening or using an administrative identity may raise instead of returning a typed outcome.
ADMIN_FAILURES = (MemoryConfigurationError, MemoryServiceError, w.WireError)
_KIND_FOR_PREFIX = {"global": "principal_global", "organization": "organization", "project": "project",
                    "repository": "repository"}
_PREFIX_FOR_KIND = {kind: prefix for prefix, kind in _KIND_FOR_PREFIX.items()}
_MAX_ID_CHARS = 256


class AdminIdentityError(ValueError):
    """Malformed administrative input (an ID or a scope selector); never a provider failure."""


def _identifier(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AdminIdentityError(f"{name} must be a string")
    text = value.strip()
    if not text:
        return None
    if len(text) > _MAX_ID_CHARS or any(ord(ch) < 32 for ch in text):
        raise AdminIdentityError(f"{name} is not a stable identifier")
    return text


@dataclass(frozen=True)
class AdminContext:
    """An explicit ancestor chain for a non-session surface (§4.2 L271); all None = principal-global."""

    org_id: Optional[str] = None
    project_id: Optional[str] = None
    repo_id: Optional[str] = None

    @classmethod
    def from_fields(cls, org_id: Any = None, project_id: Any = None, repo_id: Any = None) -> "AdminContext":
        return cls(_identifier(org_id, "org_id"), _identifier(project_id, "project_id"),
                   _identifier(repo_id, "repo_id"))

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {"org_id": self.org_id, "project_id": self.project_id, "repo_id": self.repo_id}

    def record_key(self, config: MemoryServiceConfig) -> str:
        """``admin:<sha256>`` over provider, principal and the chain: one persisted identity per context."""
        material = {"provider": config.provider, "principal_id": config.principal_id, **self.as_dict()}
        return "admin:" + hashlib.sha256(w.canonical_json(material)).hexdigest()

    def requested_context(self, config: MemoryServiceConfig, *, profile_id: str,
                          logical_session_id: str) -> w.RequestedContext:
        return w.RequestedContext(principal_id=config.principal_id, profile_id=profile_id,
                                  logical_session_id=logical_session_id, platform=ADMIN_PLATFORM,
                                  org_id=self.org_id, project_id=self.project_id, repo_id=self.repo_id,
                                  workspace_id=None, resolution_source="explicit_ids", canonical_directory=None)


def parse_scope_selector(text: str) -> w.ScopeRef:
    """§11.2 L1837's canonical encoding: global:<principal>, organization:<id>, project:<id>, repository:<id>."""
    prefix, sep, ident = (text if isinstance(text, str) else "").strip().partition(":")
    scope_id = _identifier(ident, "scope id") if sep else None
    if prefix not in _KIND_FOR_PREFIX or scope_id is None:
        raise AdminIdentityError(f"{text!r} is not a scope (use global:, organization:, project: or repository:<id>)")
    return w.ScopeRef(kind=_KIND_FOR_PREFIX[prefix], id=scope_id)


def format_scope(scope: w.ScopeRef) -> str:
    return f"{_PREFIX_FOR_KIND[scope.kind]}:{scope.id}"


def admin_state_path(key: str, hermes_home: Optional[Path] = None) -> Path:
    """``<home>/memory_service/admin/<sha256>.json``: host state (§9.3 L987), inside the X-1 archive exclusion."""
    if hermes_home is None:
        from hermes_constants import get_hermes_home

        hermes_home = get_hermes_home()
    return Path(hermes_home) / HOST_STATE_ROOT_DIRNAME / ADMIN_STATE_DIRNAME / (key.split(":", 1)[1] + ".json")


def _load_record(key: str, hermes_home: Optional[Path]) -> Optional[HostStateRecord]:
    path = admin_state_path(key, hermes_home)
    if not path.exists():
        return None
    try:
        record = HostStateRecord.from_dict(json.loads(path.read_text(encoding="utf-8")), session_id=key)
    except (OSError, ValueError, MemoryServiceError):  # BindingInvalidError is a MemoryServiceError
        logger.warning("administrative memory identity record is unreadable; binding a new one")
        return None
    return record if record.state is not None else None


def _save_record(key: str, state, hermes_home: Optional[Path]) -> None:
    from utils import atomic_json_write

    atomic_json_write(admin_state_path(key, hermes_home),
                      HostStateRecord(key, "provider_authoritative", state).to_dict(), mode=0o600)


def _binding_refused(exc: MemoryBlockedError) -> bool:
    """Revoked/invalid/epoch-changed/identity-mismatch: re-bindable. A transport failure is not."""
    return not isinstance(exc.__cause__, ProviderTransportError)


@contextmanager
def admin_service(raw_config: Any, *, context: AdminContext, backend_factory: Any = None,
                  hermes_home: Optional[Path] = None) -> Iterator[Any]:
    """The explicit administrative identity for *context* (ruling R42-2); the caller owns nothing to close.

    The first open binds a ``new_session`` (``explicit_ids``, ``platform: admin``) and persists its host
    state; later opens resume it with negotiate plus validate (C4 ``open_session_view``) and never bind.
    A binding the provider refuses (revoked, invalid, epoch changed, identity mismatch) is re-bound; a
    transport failure propagates and keeps the record. Never stateless (D-R42-c). Two concurrent first
    opens may both bind and the last record wins, leaving one orphan handle (bounded; K-2, R28).
    """
    from agent.memory_service.authoritative import ProviderAuthoritativeMemoryService
    from agent.memory_service.bootstrap import open_session_view
    from agent.memory_service.service import _default_backend_factory
    from hermes_cli.profiles import get_active_profile_name

    config = resolve_memory_service_config(raw_config)
    if config.provider_mode is not MemoryMode.AUTHORITATIVE:
        raise MemoryConfigurationError("an administrative memory identity requires provider_mode: authoritative")
    key = context.record_key(config)
    record = _load_record(key, hermes_home)
    service = None
    if record is not None:
        try:
            service = open_session_view(config, record.state, backend_factory=backend_factory)
        except MemoryBlockedError as exc:
            if not _binding_refused(exc):
                raise
            logger.info("administrative memory identity was refused by the provider; binding a new one")
    if service is None:
        factory = backend_factory or _default_backend_factory(config)
        service = ProviderAuthoritativeMemoryService(config, factory(config))
        try:
            service.start(context.requested_context(config, profile_id=get_active_profile_name(),
                                                    logical_session_id="admin-" + uuid.uuid4().hex))
            _save_record(key, service.session_state, hermes_home)
        except BaseException:
            service.shutdown()
            raise
    try:
        yield service
    finally:
        service.shutdown()


def authority_report(raw_config: Any) -> Optional[Dict[str, Any]]:
    """Config-derived provider-managed status (ruling R42-8); None unless authoritative mode is requested.

    Never contacts the provider (R44-10 precedent: health is a status surface's job) and never
    stats native files (§9.1 L950).
    """
    disposition = archive_disposition(raw_config)
    if disposition is None:
        return None
    report: Dict[str, Any] = {**disposition.as_mapping(), "provider_mode": "authoritative"}
    try:
        config = resolve_memory_service_config(raw_config)
    except MemoryConfigurationError as exc:
        return {**report, "failure_policy": None, "targets": None, "configuration_error": str(exc)}
    return {**report, "failure_policy": config.failure_policy.value,
            "targets": {"memory": config.memory_enabled, "user": config.user_profile_enabled},
            "configuration_error": None}


# -- outcome payloads (content-free: codes, requirement names and host messages only) -------------------


def _committed(outcome: MutationOutcome) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"ok": True, "message": outcome.plan.message if outcome.plan else "Saved."}
    admissions = outcome.commit.admissions if outcome.commit else ()
    if any(a.disposition == "withheld_raw" for a in admissions):
        payload["withheld"] = True
    if outcome.commit is not None and outcome.commit.outcome == "committed_audit_pending":
        payload["warning"] = "Saved and published; the memory provider's audit trail is pending."
    return payload


def _short_circuit(outcome: MutationOutcome) -> Dict[str, Any]:
    return dict(outcome.short_circuit.response) if outcome.short_circuit else {"ok": True, "message": "no changes"}


def _approval_unavailable(outcome: MutationOutcome) -> Dict[str, Any]:
    # R42 ships fail-closed (R38-1; ruling X6b-4) until the R42 follow-up adopts R39's flow (C6b-9).
    required = list(outcome.approval_requirements)
    return {"ok": False, "code": "approval_unavailable", "approval_required": required,
            "message": f"This change needs explicit approval ({', '.join(required)}) and approval is not "
                       "available for provider-managed memory yet, so nothing was changed."}


def _rejected(outcome: MutationOutcome) -> Dict[str, Any]:
    code = outcome.error_code or "rejected"
    detail = f": {outcome.error_detail}" if outcome.error_detail else ""
    return {"ok": False, "code": code,
            "message": f"The memory provider rejected this change ({code}{detail}). Nothing was changed."}


def _conflict_exhausted(outcome: MutationOutcome) -> Dict[str, Any]:
    return {"ok": False, "code": "version_conflict",
            "message": "Memory changed repeatedly while this change was being saved, so nothing was changed. "
                       "Refresh and retry."}


def _unavailable(outcome: MutationOutcome) -> Dict[str, Any]:
    code = outcome.error_code or "unavailable"
    return {"ok": False, "code": code, "message": f"Curated memory is unavailable ({code}). Nothing was changed."}


def _outcome_unknown(outcome: MutationOutcome) -> Dict[str, Any]:
    # X-4 / correction R-5: never "nothing was changed" -- the write may have landed.
    return {"ok": False, "code": "outcome_unknown",
            "message": "The memory provider did not confirm whether this change was saved. "
                       "Refresh to see the current state."}


#: Correction C68-5: keyed by MutationStatus, with a content-free fallback for members added later
#: (R39's DENIED and PENDING_APPROVAL, C6b-2).
_OUTCOME_PAYLOADS: Dict[MutationStatus, Callable[[MutationOutcome], Dict[str, Any]]] = {
    MutationStatus.COMMITTED: _committed,
    MutationStatus.SHORT_CIRCUIT: _short_circuit,
    MutationStatus.APPROVAL_UNAVAILABLE: _approval_unavailable,
    MutationStatus.REJECTED: _rejected,
    MutationStatus.CONFLICT_EXHAUSTED: _conflict_exhausted,
    MutationStatus.UNAVAILABLE: _unavailable,
    MutationStatus.OUTCOME_UNKNOWN: _outcome_unknown,
}


def _fallback(outcome: MutationOutcome) -> Dict[str, Any]:
    return {"ok": False, "code": outcome.status.value, "message": f"This change was not saved ({outcome.status.value})."}


def outcome_payload(outcome: MutationOutcome) -> Dict[str, Any]:
    """The surface-neutral answer for one C5 outcome: ``{"ok", "code"?, "message", ...}``, content-free."""
    return _OUTCOME_PAYLOADS.get(outcome.status, _fallback)(outcome)


def unavailable_payload(exc: BaseException) -> Dict[str, Any]:
    """An open or load that raised instead of returning an outcome; codes only, never provider details."""
    if isinstance(exc, MemoryConfigurationError):
        return {"ok": False, "code": "configuration_error", "message": f"Memory configuration error: {exc}"}
    code = getattr(exc, "code", None) or "unavailable"
    return {"ok": False, "code": code, "message": f"Curated memory is unavailable ({code})."}
