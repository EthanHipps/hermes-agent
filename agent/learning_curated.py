"""Journey memory over provider-managed curated memory (§9.7 L1608-L1609; rulings R42-4, R42-5, R42-6).

In authoritative mode the journey ("learning graph") never reads, stats, rewrites or deletes
``MEMORY.md``/``USER.md``. Cards come from each enabled target's ``mutation_entries`` -- the
addressable stored entries, each with a stable entry ID and its origin scope (§9.3 L1237) -- read
through the explicit administrative identity (``agent/memory_service/admin.py``, contract C6b-5),
never the process directory. Node ids keep the vocabulary every front end parses,
``memory:<memory|profile>:<entry-id>``; the stable entry ID replaces the positional index, and
opaque provider values (handles, revisions, tokens) never reach a card (§9.3 L989).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from agent.memory_service import wire as w
from agent.memory_service.admin import ADMIN_FAILURES, AdminContext, admin_service, format_scope, unavailable_payload

logger = logging.getLogger(__name__)

_SOURCE = {"memory": "memory", "user": "profile"}
_TARGET = {source: target for target, source in _SOURCE.items()}
JOURNEY_SURFACE = "journey"


def _card(entry: w.StoredEntry, source: str) -> Dict[str, Any]:
    lines = entry.text.strip().splitlines()
    first = lines[0].strip().lstrip("# ").strip() if lines else ""
    return {"id": f"memory:{source}:{entry.id}", "source": source, "timestamp": None,
            "title": (first[:80] + "…") if len(first) > 80 else first, "body": entry.text[:1200],
            "entry_id": entry.id, "scope": format_scope(entry.origin_scope)}


def curated_graph_memory(raw_config: Any, context: Optional[AdminContext], *,
                         backend_factory: Any = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """``(cards, curated_memory status)`` for the journey graph; an unavailable provider shows no memory."""
    context = context or AdminContext()
    snapshots: Dict[str, str] = {}
    status: Dict[str, Any] = {"state": "ok", "context": context.as_dict(), "snapshots": snapshots}
    cards: List[Dict[str, Any]] = []
    try:
        with admin_service(raw_config, context=context, backend_factory=backend_factory) as service:
            for target in ("memory", "user"):
                if not service.target_enabled(target):
                    continue
                snapshot = service.load_curated(target)
                snapshots[target] = snapshot.status
                cards.extend(_card(entry, _SOURCE[target]) for entry in snapshot.mutation_entries)
    except ADMIN_FAILURES as exc:
        payload = unavailable_payload(exc)
        logger.warning("journey memory unavailable (%s)", payload["code"])
        state = "configuration_error" if payload["code"] == "configuration_error" else "unavailable"
        return [], {**status, "state": state, "code": payload["code"]}
    return cards, status
