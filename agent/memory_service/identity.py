"""§4.1 frozen identity and the atomic host session state Hermes persists."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from agent.memory_service.errors import BindingInvalidError
from agent.memory_service.wire import FrozenIdentityWire, WireError


@dataclass(frozen=True)
class FrozenMemoryIdentity:
    provider: str
    provider_mode: str
    principal_id: str
    profile_id: str
    logical_session_id: str
    org_id: Optional[str]
    project_id: Optional[str]
    repo_id: Optional[str]
    workspace_id: Optional[str]
    platform: str
    binding_revision: str  # activated registry/config revision
    opaque_binding_b64url: str  # unpadded base64url provider handle; persisted without reinterpretation

    def to_wire(self) -> FrozenIdentityWire:
        if self.provider_mode != "authoritative":
            # A bare ValueError must never escape the seam: resume() calls
            # this on a caller-supplied HostSessionState and does not catch
            # ValueError, so a hand-built non-authoritative identity needs
            # the package's own typed error here (M14).
            raise BindingInvalidError("only an authoritative identity has a wire form")
        return FrozenIdentityWire.from_wire(asdict(self))

    @classmethod
    def from_wire(cls, wire: FrozenIdentityWire) -> "FrozenMemoryIdentity":
        return cls(**wire.to_wire())


@dataclass(frozen=True)
class HostSessionState:
    """Provider epoch (adjacent transport state) plus the complete identity."""

    provider_epoch: str
    identity: FrozenMemoryIdentity

    def to_dict(self) -> Dict[str, Any]:
        return {"provider_epoch": self.provider_epoch, "identity": asdict(self.identity)}

    @classmethod
    def from_dict(cls, data: Any) -> "HostSessionState":
        if not isinstance(data, dict):
            raise BindingInvalidError("host session state must be an object")
        epoch = data.get("provider_epoch")
        identity = data.get("identity")
        if not isinstance(epoch, str) or not epoch:
            raise BindingInvalidError("host session state is missing provider_epoch")
        if not isinstance(identity, dict):
            raise BindingInvalidError("host session state is missing identity")
        try:
            wire = FrozenIdentityWire.from_wire(identity, "$.identity")
        except WireError as exc:
            raise BindingInvalidError(f"host session state identity is invalid: {exc}") from exc
        return cls(provider_epoch=epoch, identity=FrozenMemoryIdentity.from_wire(wire))
