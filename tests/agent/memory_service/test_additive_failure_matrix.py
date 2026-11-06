"""Additive evidence for the section 9.6 failure matrix (not authoritative evidence).

Every row below drives the legacy additive path on purpose: the built-in
MemoryStore stays the curated memory and one external ``memory.provider``
sidecar rides alongside it through ``MemoryManager``.  The contract is the same
for each cell of the "Additive" column: the sidecar's fault is reported (logged
or returned as a tool error) and never adopted, so the real built-in memory
tool results, the system-prompt memory parts and the native file bytes equal a
baseline run of the identical scenario with no sidecar at all.

Rows 1 and 16 are proven elsewhere (``test_failure_matrix`` row 1,
``test_bootstrap_sentinel`` and the cron/admin suites).  The sidecar interface
has no wire-level "binding", "epoch" or "stage" concepts, so such a row is carried by the
failure of the method that would observe it and its parameter id says "no-additive-concept".
"""

import json
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agent.memory_provider import MemoryProvider

POISON = "SIDECAR-POISON-7d1f"


class SidecarFault(Exception):
    """A scripted sidecar failure; the message carries a unique marker for the log assertions."""


def fault(marker: str) -> SidecarFault:
    return SidecarFault(f"[{marker}] {POISON}")


@dataclass(frozen=True)
class Returns:
    """A scripted return value (instead of a raise) for a sidecar method."""

    value: Any


class FaultySidecar(MemoryProvider):
    """A sidecar whose every method follows a per-method script.

    A script entry is an exception (raise it), ``Returns`` (hand back the value),
    or ``None`` (behave healthily).  An exhausted script is healthy, so
    ``[exc, None]`` means "fail once, then recover".
    """

    def __init__(self, name: str, script: Optional[Dict[str, list]] = None, *, available: bool = True,
                 reason: str = "") -> None:
        self._name = name
        self._script = {method: list(steps) for method, steps in (script or {}).items()}
        self._available = available
        self._reason = reason
        self.calls: List[tuple] = []  # (method, "raised" | "ok"), in call order
        self.raised_messages: List[str] = []

    @property
    def name(self) -> str:
        return self._name

    def _step(self, method: str, healthy: Any = None) -> Any:
        queue = self._script.get(method)
        step = queue.pop(0) if queue else None
        if isinstance(step, BaseException):
            self.calls.append((method, "raised"))
            self.raised_messages.append(str(step))
            raise step
        self.calls.append((method, "ok"))
        return step.value if isinstance(step, Returns) else healthy

    def outcomes(self, method: str) -> List[str]:
        return [outcome for called, outcome in self.calls if called == method]

    def is_available(self) -> bool:
        return self._available

    def unavailable_reason(self) -> str:
        return self._reason

    def initialize(self, session_id: str, **kwargs) -> None:
        self._step("initialize")

    def system_prompt_block(self) -> str:
        return self._step("system_prompt_block", "")

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return self._step("prefetch", "")

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        self._step("queue_prefetch")

    def recall_status(self):
        return self._step("recall_status")

    def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None, turn_author=None) -> None:
        self._step("sync_turn")

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [{"name": f"{self._name}_search", "description": "sidecar search",
                 "parameters": {"type": "object", "properties": {}}}]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        return self._step("handle_tool_call", json.dumps({"ok": True}))

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._step("on_turn_start")

    def on_session_end(self, messages) -> None:
        self._step("on_session_end")

    def on_session_switch(self, new_session_id, *, parent_session_id="", reset=False, rewound=False, **kwargs):
        self._step("on_session_switch")

    def on_pre_compress(self, messages) -> str:
        return self._step("on_pre_compress", "")

    def on_memory_write(self, action, target, content, metadata=None) -> None:
        self._step("on_memory_write")

    def backup_paths(self) -> List[str]:
        return self._step("backup_paths", [])

    def shutdown(self) -> None:
        self._step("shutdown")


# --- The scenario -------------------------------------------------------------------------------


def _tool_defs(*names):
    return [{"type": "function", "function": {"name": n, "description": f"{n} tool",
                                              "parameters": {"type": "object", "properties": {}}}} for n in names]


def _build_agent(home, sidecar: Optional[FaultySidecar], load_error: Optional[BaseException]):
    """A real AIAgent on ``home`` whose ``memory.provider`` names ``sidecar`` (none for the baseline)."""
    from run_agent import AIAgent

    section = {"provider": sidecar.name} if sidecar else {}
    (home / "config.yaml").write_text(yaml.safe_dump({"memory": section}), encoding="utf-8")

    def load(name, **_):
        if load_error is not None:
            raise load_error
        return sidecar

    with patch("plugins.memory.load_memory_provider", side_effect=load), \
         patch("model_tools.get_tool_definitions", return_value=_tool_defs("memory", "web_search")), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=False, session_id="sess-additive")
    agent.client = MagicMock()
    return agent


def _native_bytes(home) -> Dict[str, bytes]:
    root = home / "memories"
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()} if root.is_dir() else {}


def _memory(agent, **args) -> Any:
    return json.loads(agent._invoke_tool("memory", args, "task-1"))


def run_scenario(home, sidecar: Optional[FaultySidecar], *, load_error: Optional[BaseException] = None,
                 extra=None) -> SimpleNamespace:
    """Two turns of the real turn-loop calls: built-in writes (mirrored to the sidecar), recall, sync, a
    session switch, a pre-compress hook, a prompt rebuild after a native reload, and a clean shutdown."""
    from agent.system_prompt import _memory_parts

    agent = _build_agent(home, sidecar, load_error)
    manager = agent._memory_manager
    results = [_memory(agent, action="add", target="memory", content="first built-in fact")]
    prompts = [tuple(_memory_parts(agent))]
    recall: List[str] = []
    tool_replies: List[str] = []

    def turn(number: int, message: str) -> None:
        if manager is None:
            return
        manager.on_turn_start(number, message, remaining_tokens=1000)
        recall.append(manager.prefetch_all(message, session_id=agent.session_id))
        recall.append(manager.describe_recall())
        manager.queue_prefetch_all(message, session_id=agent.session_id)

    turn(1, "what do you remember about the build?")
    results.append(_memory(agent, action="add", target="user", content="prefers terse answers"))
    if manager is not None:
        manager.sync_all("question one", "answer one", session_id=agent.session_id)
        assert manager.flush_pending(5.0)
    turn(2, "and what about the tests?")
    results.append(_memory(agent, action="replace", target="memory", old_text="first built-in",
                           content="first built-in fact, revised"))
    if manager is not None:
        manager.sync_all("question two", "answer two", session_id=agent.session_id)
        assert manager.flush_pending(5.0)
        manager.on_session_switch("sess-additive-2", parent_session_id=agent.session_id, reset=True)
        manager.on_pre_compress([{"role": "user", "content": "hello"}])
        if manager.has_tool(f"{sidecar.name}_search"):
            tool_replies.append(agent._invoke_tool(f"{sidecar.name}_search", {}, "task-1"))
    agent._memory_store.load_from_disk()
    prompts.append(tuple(_memory_parts(agent)))
    results.append(_memory(agent, action="remove", target="user", old_text="prefers terse"))
    if extra is not None:
        extra(agent)
    if manager is not None:
        manager.on_session_end([{"role": "user", "content": "bye"}])
        manager.shutdown_all()
    return SimpleNamespace(results=results, prompts=prompts, native=_native_bytes(home), recall=recall,
                           tool_replies=tool_replies, manager=manager)


# --- The rows ------------------------------------------------------------------------------------
#
# Parameter ids say what each case proves.  "no-additive-concept" marks a cell whose condition has no
# counterpart in the sidecar interface (no scope resolution, conflict, binding or epoch); it is carried by the
# failure of the named method and takes the same code path as another row, so it is isolation evidence only.


@dataclass(frozen=True)
class Row:
    script: Optional[Dict[str, list]] = None
    available: bool = True
    load_error: Optional[BaseException] = None
    raised: tuple = ()          # (method, ...) the sidecar must have raised from, in the scenario
    recovers: tuple = ()        # methods that must succeed again after a raise (sidecar resumes)
    never_started: bool = False  # the sidecar was never adopted (no manager)
    extra: str = ""             # "backup", "status" or "reinit": an out-of-turn step the row reaches
    report: bool = False        # the cell says "report": a record at WARNING or above is required
    returns: tuple = ()         # (method, ...) the sidecar answered with a malformed value instead of raising
    report_text: str = ""       # with ``returns``: the WARNING text the host must log


def _scripted(method: str, marker: str, *, then_ok: int = 0, after_ok: int = 0) -> Dict[str, list]:
    return {method: [None] * after_ok + [fault(marker)] + [None] * then_ok}


ROWS: Dict[str, Row] = {
    # Row 2: the additive cell is a warned degrade for a missing sidecar; the configuration-error half of the
    # cell is proven at boot by the two ``test_*_is_a_configuration_error_at_boot`` tests below.
    "row2-sidecar-load-failure-is-a-warned-degrade": Row(load_error=fault("row2-load"), never_started=True,
                                                         report=True),
    "row2-sidecar-unavailable-is-a-warned-degrade": Row(available=False, never_started=True, report=True),
    # Row 3: negotiate / bind / validate / first load unavailable or malformed.
    "row3-bind-unavailable-initialize": Row(_scripted("initialize", "row3-bind"), raised=("initialize",),
                                            report=True),
    "row3-negotiate-malformed-initialize": Row({"initialize": [ValueError(f"[row3-malformed] {POISON}")]},
                                               raised=("initialize",), report=True),
    "row3-first-load-unavailable-system-prompt": Row(_scripted("system_prompt_block", "row3-load"),
                                                     raised=("system_prompt_block",), report=True),
    "row3-first-load-malformed-non-string-snapshot": Row({"system_prompt_block": [Returns(["bad", POISON])]},
                                                         returns=("system_prompt_block",), report=True,
                                                         report_text="system prompt block was malformed"),
    # Row 4: the snapshot omits a required field; the cell asks only that the sidecar be ignored.
    "row4-incomplete-snapshot-non-string-system-prompt": Row(
        {"system_prompt_block": [Returns({"revision": 1, "note": POISON})]}, returns=("system_prompt_block",)),
    "row4-incomplete-snapshot-raises-system-prompt": Row(_scripted("system_prompt_block", "row4-snapshot"),
                                                         raised=("system_prompt_block",)),
    # Row 5: no additive concept (the additive path resolves no scope); carried by initialize / prefetch failing.
    "row5-no-additive-concept-initialize-fails": Row(_scripted("initialize", "row5-initialize"),
                                                     raised=("initialize",)),
    "row5-no-additive-concept-prefetch-fails": Row(_scripted("prefetch", "row5-prefetch"), raised=("prefetch",)),
    # Row 6: the sidecar fails after an earlier success.
    "row6-after-success-prefetch": Row(_scripted("prefetch", "row6-prefetch", after_ok=1), raised=("prefetch",)),
    "row6-after-success-sync-turn": Row(_scripted("sync_turn", "row6-sync", after_ok=1), raised=("sync_turn",)),
    "row6-after-success-on-turn-start": Row(_scripted("on_turn_start", "row6-turn", after_ok=1),
                                            raised=("on_turn_start",)),
    "row6-after-success-mirror": Row(_scripted("on_memory_write", "row6-mirror", after_ok=1),
                                     raised=("on_memory_write",)),
    "row6-after-success-system-prompt": Row(_scripted("system_prompt_block", "row6-prompt", after_ok=1),
                                            raised=("system_prompt_block",)),
    # Row 7: optional recall unavailable after the required load.
    "row7-recall-unavailable-prefetch": Row(_scripted("prefetch", "row7-prefetch"), raised=("prefetch",)),
    "row7-recall-unavailable-recall-status": Row(_scripted("recall_status", "row7-status"),
                                                 raised=("recall_status",)),
    # Row 8: no additive concept (no ambiguous_policy result); carried by the load failing; same path as row 3.
    "row8-no-additive-concept-system-prompt-fails": Row(_scripted("system_prompt_block", "row8-ambiguous"),
                                                        raised=("system_prompt_block",), report=True),
    "row8-no-additive-concept-initialize-fails": Row(_scripted("initialize", "row8-ambiguous-init"),
                                                     raised=("initialize",), report=True),
    # Row 9: no additive concept; carried by recall failing; same path as row 7, but the cell requires a report.
    "row9-no-additive-concept-prefetch-fails": Row(_scripted("prefetch", "row9-prefetch"), raised=("prefetch",),
                                                   report=True),
    "row9-no-additive-concept-recall-status-fails": Row(_scripted("recall_status", "row9-status"),
                                                        raised=("recall_status",), report=True),
    # Row 11: version_conflict; a later write from the sidecar succeeds (it may reapply independently).
    "row11-version-conflict-then-a-later-write-succeeds": Row(
        _scripted("on_memory_write", "row11-conflict", then_ok=2), raised=("on_memory_write",),
        recovers=("on_memory_write",)),
    # Row 12: stage/commit transport outcome unknown.
    "row12-unknown-outcome-mirror": Row({"on_memory_write": [TimeoutError(f"[row12-mirror] {POISON}")]},
                                        raised=("on_memory_write",)),
    "row12-unknown-outcome-sync-turn": Row({"sync_turn": [TimeoutError(f"[row12-sync] {POISON}")]},
                                           raised=("sync_turn",)),
    # Row 13: no additive concept (no binding); carried by method failure.
    "row13-no-additive-concept-sync-turn-fails-then-resumes": Row(
        _scripted("sync_turn", "row13-revoked", then_ok=2), raised=("sync_turn",), recovers=("sync_turn",)),
    "row13-no-additive-concept-tool-call-fails": Row(_scripted("handle_tool_call", "row13-tool"),
                                                     raised=("handle_tool_call",)),
    # Row 14: no additive concept (no epoch); carried by method failure, then a test-driven initialize_all.
    "row14-no-additive-concept-session-switch-fails": Row(_scripted("on_session_switch", "row14-switch"),
                                                          raised=("on_session_switch",), extra="reinit"),
    "row14-no-additive-concept-pre-compress-fails": Row(_scripted("on_pre_compress", "row14-compress"),
                                                        raised=("on_pre_compress",), extra="reinit"),
    # Row 15: the provider recovers; the sidecar resumes with the native result unchanged.
    "row15-recovers-sync-turn": Row(_scripted("sync_turn", "row15-sync", then_ok=2), raised=("sync_turn",),
                                    recovers=("sync_turn",)),
    "row15-recovers-prefetch": Row(_scripted("prefetch", "row15-prefetch", then_ok=2), raised=("prefetch",),
                                   recovers=("prefetch",)),
    # Row 17: backup and status cannot reach the sidecar.
    "row17-backup-unreachable": Row(_scripted("backup_paths", "row17-backup"), raised=("backup_paths",),
                                    extra="backup", report=True),
    "row17-status-unreachable": Row(available=False, never_started=True, extra="status", report=True),
}

# Row 10: scope, approval, secret, limit, stage and store_blocked.  No additive concept for the kind; each is
# carried by the mirrored built-in write failing and by the sidecar's own tool call failing (two code paths).
for _kind in ("scope", "approval", "secret", "limit", "stage", "store_blocked"):
    ROWS[f"row10-no-additive-concept-{_kind}-mirror-fails"] = Row(
        _scripted("on_memory_write", f"row10-{_kind}-mirror"), raised=("on_memory_write",), report=True)
    ROWS[f"row10-no-additive-concept-{_kind}-tool-call-fails"] = Row(
        _scripted("handle_tool_call", f"row10-{_kind}-tool"), raised=("handle_tool_call",), report=True)


def _unique_name(case: str) -> str:
    return "faulty_" + case.replace("-", "_")


def _out_of_turn_step(kind: str, sidecar: FaultySidecar, out: dict):
    """The step a row reaches outside the turn loop: a sidecar reinitialize, the backup collector, or
    ``hermes memory status``."""

    def run(agent) -> None:
        if kind == "reinit":
            agent._memory_manager.initialize_all(agent.session_id)
        elif kind == "backup":
            from hermes_cli.backup import _collect_memory_provider_external_paths

            with patch("plugins.memory._get_active_memory_provider", return_value=sidecar.name), \
                 patch("plugins.memory.load_memory_provider", return_value=sidecar):
                out["backup_paths"] = _collect_memory_provider_external_paths()
        elif kind == "status":
            import contextlib
            import io

            from hermes_cli import memory_setup

            buffer = io.StringIO()
            with patch("hermes_cli.config.load_config", return_value={"memory": {"provider": sidecar.name}}), \
                 patch.object(memory_setup, "_get_available_providers",
                              return_value=[(sidecar.name, "faulty sidecar", sidecar)]), \
                 contextlib.redirect_stdout(buffer):
                memory_setup.cmd_status(SimpleNamespace())
            out["status"] = buffer.getvalue()

    return run


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    """The scenario with no sidecar; every row compares against this one run."""
    home = tmp_path_factory.mktemp("baseline")
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setenv("HERMES_HOME", str(home))
        run = run_scenario(home, None)
    assert run.manager is None
    return run


@pytest.mark.parametrize("case", sorted(ROWS))
def test_a_failing_additive_sidecar_never_damages_the_builtin_result(tmp_path, monkeypatch, caplog, baseline, case):
    row = ROWS[case]
    caplog.set_level(logging.DEBUG)
    sidecar_home = tmp_path / "sidecar"
    sidecar_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(sidecar_home))
    sidecar = FaultySidecar(_unique_name(case), row.script, available=row.available,
                            reason=f"[{case}] executable missing {POISON}")
    surfaces: dict = {}
    extra = _out_of_turn_step(row.extra, sidecar, surfaces) if row.extra else None
    observed = run_scenario(sidecar_home, sidecar, load_error=row.load_error, extra=extra)

    # The built-in result is the baseline's: real tool results, rebuilt prompt parts and native bytes.
    assert observed.results == baseline.results
    assert observed.prompts == baseline.prompts
    assert observed.native == baseline.native and "MEMORY.md" in observed.native
    assert all(POISON not in repr(part)
               for part in (observed.results, observed.prompts, observed.native, observed.recall))

    # The fault was exercised on the additive path, never adopted, and reported where the cell says so.
    if row.never_started:
        assert observed.manager is None and sidecar.calls == []
        expected = ["[row2-load]" if row.load_error else f"[{case}]"]
    else:
        assert observed.manager is not None and sidecar in observed.manager.providers
        expected = list(sidecar.raised_messages)
    if row.report_text:
        expected.append(row.report_text)
    for method in row.raised:
        assert "raised" in sidecar.outcomes(method)
    for method in row.returns:
        assert sidecar.outcomes(method)[0] == "ok"  # answered, with a malformed value
    assert expected or not row.report
    loud = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
    for message in expected:
        assert message in caplog.text
        if row.report:
            assert any(message in line for line in loud), f"{message!r} is not reported at WARNING or above"
    for method in row.recovers:
        outcomes = sidecar.outcomes(method)
        assert "ok" in outcomes[outcomes.index("raised"):]
    if "handle_tool_call" in row.raised:
        assert all(message in reply for message in sidecar.raised_messages for reply in observed.tool_replies)
        assert observed.tool_replies
    else:
        assert all(json.loads(reply) == {"ok": True} for reply in observed.tool_replies)
    if row.extra == "reinit":
        assert sidecar.outcomes("initialize") == ["ok", "ok"]  # the test re-ran initialize_all after the failure
    if row.extra == "backup":
        assert surfaces["backup_paths"] == []
    if row.extra == "status":
        assert "not available" in surfaces["status"]


# --- Row 2, configuration error: an explicitly named but invalid mode or policy is never additive -------


def _assert_boot_refuses(tmp_path, monkeypatch, section, match):
    from agent.memory_service.config import MemoryConfigurationError

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    built: list = []
    monkeypatch.setattr("tools.memory_tool.MemoryStore", lambda *a, **k: built.append("MemoryStore"))

    def load(name, **_):
        built.append(("sidecar", name))

    from run_agent import AIAgent

    (home / "config.yaml").write_text(yaml.safe_dump({"memory": section}), encoding="utf-8")
    with patch("plugins.memory.load_memory_provider", side_effect=load), \
         patch("model_tools.get_tool_definitions", return_value=_tool_defs("memory")), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI"):
        with pytest.raises(MemoryConfigurationError, match=match):
            AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                    skip_context_files=True, skip_memory=False, session_id="sess-bad-mode")
    assert built == []  # neither a native store nor a sidecar was built
    assert not (home / "memories" / "MEMORY.md").exists()


def test_unknown_provider_mode_is_a_configuration_error_at_boot(tmp_path, monkeypatch):
    _assert_boot_refuses(tmp_path, monkeypatch, {"provider": "example", "provider_mode": "sidecar"}, "provider_mode")


def test_stateless_policy_without_authoritative_is_a_configuration_error_at_boot(tmp_path, monkeypatch):
    _assert_boot_refuses(tmp_path, monkeypatch,
                         {"provider": "example", "authoritative_failure_policy": "stateless"}, "stateless")
