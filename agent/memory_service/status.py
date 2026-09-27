"""Health probe and content-free status over MemoryService (R41; contract C6b-13).

Administrative identity, the reset planner and administrative scope selection belong to
``agent/memory_service/admin.py`` (R42, contract C6b-5; ruling X6b-3); ``format_scope_ref``/
``parse_scope_ref`` encode §11.2 L1837 exactly as C6b-5's selectors do and are removed by the R41
follow-up in favour of them.

* ``probe_provider`` negotiates and nothing else (§9.10 L1683: negotiate publishes no state).
* ``collect_memory_status`` reports mode, identity, scopes and degraded state from a live or
  resumed service. It is content-free: no provider epoch, opaque binding handle, revision token,
  stage handle or approval value enters it (§9.3 L987). ``binding_revision`` is the one revision
  shown, because §9.3 L1167 makes it a stable, human-typeable identifier (ruling R41-3).

Provider-generic (D5).
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from agent.memory_service import wire as w
from agent.memory_service.config import PROVIDER_API_VERSION, MemoryServiceConfig
from agent.memory_service.errors import MemoryServiceError, ProviderTransportError

logger = logging.getLogger(__name__)

_PREFIX = {"principal_global": "global", "organization": "organization", "project": "project",
           "repository": "repository"}
_KIND = {prefix: kind for kind, prefix in _PREFIX.items()}


def format_scope_ref(scope: w.ScopeRef) -> str:
    """§11.2 L1837's canonical CLI encoding of a ``ScopeRef``."""
    return f"{_PREFIX[scope.kind]}:{scope.id}"


def parse_scope_ref(text: str) -> w.ScopeRef:
    """The inverse of :func:`format_scope_ref`; ``ValueError`` on anything else."""
    prefix, sep, ident = (text or "").strip().partition(":")
    kind = _KIND.get(prefix)
    if not sep or kind is None or not ident.strip():
        raise ValueError(f"{text!r} is not a scope reference; use global:<principal>, "
                         "organization:<id>, project:<id> or repository:<id>")
    return w.ScopeRef(kind=kind, id=ident.strip())


@dataclass(frozen=True)
class ProviderProbe:
    ok: bool
    provider: str
    api_version: Optional[int] = None
    operations: Tuple[str, ...] = ()
    recall_context: bool = False
    capture_continuity: bool = False
    error: Optional[str] = None


def _failure(exc: BaseException) -> str:
    """Content-free: a transport reason, a typed code, or a configuration message (config values only)."""
    if isinstance(exc, ProviderTransportError):
        return f"transport failure ({exc.reason})"
    code = getattr(exc, "code", None)
    return f"{type(exc).__name__}: {code}" if code else f"{type(exc).__name__}: {exc}"


def probe_provider(config: MemoryServiceConfig, *,
                   backend_factory: Optional[Callable[[MemoryServiceConfig], Any]] = None) -> ProviderProbe:
    """Negotiate only (D-R41-2). Never raises: doctor and status report what failed."""
    from agent.memory_service.authoritative import ProviderAuthoritativeMemoryService
    from agent.memory_service.service import _default_backend_factory

    try:
        backend = (backend_factory or _default_backend_factory(config))(config)
    except Exception as exc:  # a plugin that cannot build a backend is itself the finding
        return ProviderProbe(False, config.provider, error=_failure(exc))
    service = ProviderAuthoritativeMemoryService(config, backend)
    try:
        negotiation = service.negotiate_only()
    except Exception as exc:
        return ProviderProbe(False, config.provider, error=_failure(exc))
    finally:
        service.shutdown()
    caps = negotiation.capabilities
    return ProviderProbe(True, negotiation.provider, negotiation.selected_api_version, tuple(negotiation.operations),
                         bool(caps.recall_context), bool(caps.capture_continuity))


@dataclass(frozen=True)
class TargetStatus:
    target: str
    enabled: bool
    status: Optional[str] = None                 # "ok" | "degraded_global_only"
    visible_scopes: Tuple[str, ...] = ()         # §11.2 encoding
    default_write_scope: Optional[str] = None
    eligible_write_scopes: Tuple[str, ...] = ()
    entries: Optional[int] = None                # len(mutation_entries)
    error: Optional[str] = None                  # content-free code


@dataclass(frozen=True)
class MemoryStatus:
    mode: str                                    # the configured provider_mode value
    disposition: str                             # the MemoryDisposition value
    provider: str
    api_version: int
    failure_policy: str
    identity: Optional[Mapping[str, Optional[str]]]   # FrozenMemoryIdentity minus opaque_binding_b64url
    targets: Tuple[TargetStatus, ...]
    degraded: Optional[str]

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


_HIDDEN_IDENTITY_FIELDS = frozenset({"opaque_binding_b64url"})


def _target_status(service: Any, target: str) -> TargetStatus:
    try:
        snapshot = service.load_curated(target)
    except MemoryServiceError as exc:
        return TargetStatus(target, True, error=getattr(exc, "code", None) or "blocked")
    return TargetStatus(
        target, True, status=snapshot.status,
        visible_scopes=tuple(format_scope_ref(s) for s in snapshot.visible_scopes),
        default_write_scope=format_scope_ref(snapshot.default_write_scope) if snapshot.default_write_scope else None,
        eligible_write_scopes=tuple(format_scope_ref(s) for s in snapshot.eligible_write_scopes),
        entries=len(snapshot.mutation_entries))


def collect_memory_status(service: Any) -> MemoryStatus:
    """Mode, identity (never the handle), per-target scopes and degraded state (§9.7 L1605)."""
    config = service.config
    disposition = getattr(service.disposition, "value", str(service.disposition))
    identity = service.identity
    shown = None if identity is None else {
        key: value for key, value in dataclasses.asdict(identity).items() if key not in _HIDDEN_IDENTITY_FIELDS}
    targets = []
    for target in ("memory", "user"):
        if not config.target_enabled(target):
            targets.append(TargetStatus(target, False))
        elif disposition != "provider_authoritative":
            targets.append(TargetStatus(target, True, error=disposition))
        else:
            targets.append(_target_status(service, target))
    return MemoryStatus(mode=config.provider_mode.value, disposition=disposition, provider=config.provider,
                        api_version=PROVIDER_API_VERSION, failure_policy=config.failure_policy.value,
                        identity=shown, targets=tuple(targets), degraded=service.degraded_warning())
