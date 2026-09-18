"""What an authoritative provider must implement (§9.3 operations).

Row 36's fake backend and row 47's provider adapter implement this protocol. A
backend raises :class:`ProviderError` for a typed failure envelope and
:class:`ProviderTransportError` when the provider could not be reached or
answered malformed output. Every result carries the ``provider_epoch`` from
the success envelope so the service can enforce epoch continuity. The service
also tolerates a :class:`~agent.memory_service.wire.WireError` escaping a
backend, treating it the same as a :class:`ProviderTransportError` for
malformed output.

Backends validate their own request and result envelopes with the wire codecs,
and validate typed failure envelopes with ``WireFailure`` and
``decode_error_details`` before raising ``ProviderError``. The service relies
on that boundary; its correlation checks do not replace envelope validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from agent.memory_service import wire as w

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class ProviderResult(Generic[T]):
    result: T
    provider_epoch: str


class AuthoritativeBackend(Protocol):
    def negotiate(self, request: w.NegotiateRequest) -> ProviderResult[w.Negotiation]: ...
    def bind_session(self, request: w.BindRequest) -> ProviderResult[w.BindResult]: ...
    def validate_session(self, request: w.ValidateSessionRequest) -> ProviderResult[w.ValidateSessionResult]: ...
    def load_curated(self, request: w.LoadRequest) -> ProviderResult[w.CuratedSnapshot]: ...
    def stage_curated(self, request: w.StageRequest) -> ProviderResult[w.StageResult]: ...
    def inspect_staged(self, request: w.InspectStageRequest) -> ProviderResult[w.StageInspection]: ...
    def commit_curated(self, request: w.CommitRequest) -> ProviderResult[w.CommitResult]: ...
    def recall_context(self, request: w.RecallRequest) -> ProviderResult[w.TypedRecall]: ...
    def capture_continuity(self, request: w.ContinuityRequest) -> ProviderResult[w.ContinuityResult]: ...
    def shutdown(self) -> None: ...
