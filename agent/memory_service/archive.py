"""Generic archive disposition for provider-managed curated memory (§9.8, §12.1).

Every Hermes backup, profile export, clone, restore and diagnostic bundle made in
authoritative mode declares that curated memory is provider-managed and not
included. The block is generic: ``provider`` is the configured ``memory.provider``
(D5; ruling R44-4) and ``provider_api`` is the version Hermes requires, never a
negotiated value, because an archive never contacts the provider (R44-10).
Keyed on the REQUESTED mode, like R37's boot and doctor (R44-3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import yaml

from agent.memory_service.bootstrap import requests_authoritative_mode
from agent.memory_service.config import PROVIDER_API_VERSION

#: File name of the record inside an archive or a cloned/restored home (ruling R44-5).
DISPOSITION_RECORD_NAME = "curated-memory-disposition.yaml"


@dataclass(frozen=True)
class ArchiveDisposition:
    """The §9.8 L1629-1635 block, plus the sentences L1637 requires be said."""

    provider: str
    provider_api: int = PROVIDER_API_VERSION
    authority: str = "provider"
    included: bool = False
    disposition: str = "provider-managed"
    restore_action: str = "reconnect-provider"

    def as_mapping(self) -> dict:
        return {  # §9.8 L1629-1635 order
            "authority": self.authority, "provider": self.provider, "provider_api": self.provider_api,
            "included": self.included, "disposition": self.disposition, "restore_action": self.restore_action,
        }

    def yaml_block(self) -> str:
        return yaml.safe_dump({"curated_memory": self.as_mapping()}, sort_keys=False, default_flow_style=False)

    def subject(self) -> str:
        """The provider is absent when an authoritative config is requested but invalid (R44-3)."""
        return f"authoritative {self.provider} memory" if self.provider else "authoritative memory"

    def archive_message(self, *, what: str = "archive", complete: bool = True) -> str:
        state = "complete" if complete else "incomplete"
        return f"Hermes {what} {state}; {self.subject()} is provider-managed and not included"

    def restore_message(self) -> str:
        return (f"Hermes configuration restored; {self.subject()} is provider-managed and was not restored. "
                f"Start a new session to reconnect the provider (restore_action: {self.restore_action}).")

    def record_text(self) -> str:
        first = self.subject()
        return (f"# {first[:1].upper()}{first[1:]} is provider-managed and not included in this Hermes archive.\n"
                "# Restoring restores Hermes configuration and this record only; a new session reconnects the provider.\n"
                + self.yaml_block())


def archive_disposition(raw_config: object) -> Optional[ArchiveDisposition]:
    """The disposition for *raw_config*, or ``None`` unless authoritative mode is requested. Never raises."""
    if not requests_authoritative_mode(raw_config):
        return None
    try:
        provider = raw_config.get("memory").get("provider")  # type: ignore[union-attr]
    except Exception:
        provider = None
    return ArchiveDisposition(provider=provider.strip() if isinstance(provider, str) else "")
