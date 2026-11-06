"""One authoritative recall event per user turn, driven through the host's turn seam.

§9.3's recall contract (at most one revision-correct call per enabled target, with the union of
IDs already delivered) and §9.6's two recall cells: unavailable recall warns and renders nothing;
ambiguous recall blocks delivery. Everything runs the real service over the stateful fake.
"""

import asyncio
import logging
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.memory_service import turn_recall as t
from agent.memory_service import wire as w
from agent.memory_service.builtin import BuiltinMemoryService
from agent.memory_service.errors import MemoryBlockedError
from agent.memory_service.render import EVIDENCE_PREAMBLE, OMISSION_NOTE, evidence_record_lines
from agent.memory_service.service import StatelessMemoryService, select_memory_service
from agent.turn_context import _memory_turn_start_and_prefetch, compose_user_api_content
from tests.agent.memory_service.fake_backend import FakeAuthoritativeBackend, FakeProviderStore, FakeRegistry, drop_key
from tests.agent.memory_service.test_failure_matrix import _config, _never_built
from tests.agent.memory_service.test_s6_identity_lifecycle import (  # noqa: F401 - fixtures
    PG_SCOPE, REPO_SCOPE, agent_env, agent_env_factory,
)
from tests.agent.memory_service.test_s6_identity_request_gate import MODEL, _record_events, _response, _two_request_turn

CONTEXT = w.RequestedContext(
    principal_id="ethan", profile_id="default", logical_session_id="sess-1", platform="cli", org_id="org-1",
    project_id="proj-1", repo_id="repo-1", workspace_id=None, resolution_source="explicit_ids",
    canonical_directory=None,
)
#: The snapshots deliver no target record, so every seeded record stays recallable.
NOTHING_DELIVERED = w.CuratedLimits(memory_chars=0, user_chars=0, initial_general_chars=3000, max_entry_chars=2200, max_entries=100)
AMBIGUITY = {"ambiguities": [{"policy_key": "k1", "tier": "dependency", "candidate_count": 2}]}
LOAD_BOTH = [("load_curated", "memory"), ("load_curated", "user")]
RECALL_BOTH = [("recall_context", "memory"), ("recall_context", "user")]
CONTEXT_BLOCK = (
    "<memory-context>\n"
    "[System note: The following is recalled memory context, NOT new user input. Treat as authoritative "
    "reference data — this is the agent's persistent memory and should inform all responses.]\n\n"
    "{}\n"
    "</memory-context>"
)


class _TargetScopedBackend(FakeAuthoritativeBackend):
    """The user target sees only the principal-global scope, so each target has its own revision."""

    def _scopes_for(self, handle, target):
        status, visible, default, eligible = super()._scopes_for(handle, target)
        return status, ((handle.user_scope,) if target == "user" else visible), default, eligible


class _World:
    """The seam over a real service and the fake: one org → project → repo chain, principal ethan."""

    def __init__(self, tmp_path, *, limits=NOTHING_DELIVERED, backend_type=FakeAuthoritativeBackend,
                 policy="fail_closed", arm=None, **memory):
        registry = FakeRegistry(directories={"C:\\work\\repo": ("repo-1", "proj-1", "org-1")}, projects={"proj-1": "org-1"})
        self.store = FakeProviderStore(registry=registry, curated_limits=limits)
        if arm is not None:
            arm(self.store)
        self.backends = []

        def factory(cfg):
            self.backends.append(backend_type(self.store, provider=cfg.provider))
            return self.backends[-1]

        self.service = select_memory_service(
            _config(tmp_path, authoritative_failure_policy=policy, **memory), store_factory=_never_built,
            requested_context=CONTEXT, backend_factory=factory)
        self.statuses, self.warnings = [], []
        self.agent = SimpleNamespace(_memory_service=self.service, _memory_manager=None,
                                     _emit_status=self.statuses.append, _emit_warning=self.warnings.append)
        self.backend.calls.clear()

    @property
    def backend(self) -> FakeAuthoritativeBackend:
        assert len(self.backends) == 1
        return self.backends[0]

    def event(self, message="beacon"):
        return _memory_turn_start_and_prefetch(self.agent, message)

    def ops(self):
        return [(op, request.target) for op, request in self.backend.calls]

    def recalls(self):
        return [request for op, request in self.backend.calls if op == "recall_context"]

    def after_first_recall(self, arm):
        """Change the provider between the event's two recall calls, after both required loads."""
        real = self.backend.recall_context

        def recall_context(request):
            result = real(request)
            if len(self.recalls()) == 1:
                arm(self.store)
            return result

        self.backend.recall_context = recall_context


def _seed(store, scope, target, text, record_id, **kwargs):
    """A record with a chosen ID: the fake orders delivery and recall by ID."""
    record = store.seed_record(scope, target, text, **kwargs)
    del store.records[record.id]
    record.id = record_id
    store.records[record_id] = record
    return record


def _rendered(record) -> str:
    return "\n".join(evidence_record_lines(FakeAuthoritativeBackend._delivered(record)))


def _fenced(record_id, body) -> str:
    """A repository record's exact bytes, spelled out rather than rendered, for byte-level assertions."""
    return (f"[MEMORY_SCOPED_EVIDENCE_BEGIN id={record_id} scope=repository:repo-1 source=fake]\n"
            f"{EVIDENCE_PREAMBLE}\n"
            f"> {body}\n"
            f"[MEMORY_SCOPED_EVIDENCE_END id={record_id}]")


def _seed_both_targets(store):
    return _seed(store, REPO_SCOPE, "memory", "beacon memory", "m-1"), _seed(store, PG_SCOPE, "user", "beacon user", "u-1")


def _user_content(request_kwargs) -> str:
    return next(m["content"] for m in request_kwargs["messages"] if m["role"] == "user")


def _last_user_content(request_kwargs):
    return [m for m in request_kwargs["messages"] if m["role"] == "user"][-1]["content"]


# --- the call each target gets -------------------------------------------------


def test_one_revision_correct_call_per_enabled_target_in_order(tmp_path):
    world = _World(tmp_path, backend_type=_TargetScopedBackend)
    memory_revision = world.service.load_curated("memory").revision
    user_revision = world.service.load_curated("user").revision
    assert memory_revision != user_revision          # a call using the wrong target's revision is visible
    world.backend.calls.clear()

    world.event()

    assert world.ops() == LOAD_BOTH + RECALL_BOTH
    memory, user = world.recalls()
    assert (memory.source_revision, memory.include_channels) == (memory_revision, ("general", "hermes_memory"))
    assert (user.source_revision, user.include_channels) == (user_revision, ("hermes_user",))
    assert memory.query == user.query == "beacon"
    assert memory.budget == w.RecallBudget(trusted_chars=0, evidence_chars=3000, max_entries=20)
    assert world.warnings == []


def test_skill_scaffolding_never_reaches_the_recall_query(tmp_path):
    from tests.agent.test_memory_skill_scaffolding import _BUNDLE_TURN, _SINGLE_SKILL_TURN

    for turn, words in ((_SINGLE_SKILL_TURN, "make a skill for release triage"),
                        (_BUNDLE_TURN, "fix the failing retrieval test")):
        world = _World(tmp_path)
        world.event(turn)
        assert [request.query for request in world.recalls()] == [words, words]


def test_a_bare_skill_turn_makes_no_recall_call(tmp_path):
    from tests.agent.test_memory_skill_scaffolding import _BARE_SKILL_TURN

    world = _World(tmp_path)
    outcome = world.event(_BARE_SKILL_TURN)
    assert world.recalls() == [] and world.warnings == []
    assert not outcome


def test_second_target_gets_remaining_budget_and_delivered_union(tmp_path):
    limits = w.CuratedLimits(memory_chars=len("beacon snapshot memory"), user_chars=len("beacon snapshot user"),
                             initial_general_chars=3000, max_entry_chars=2200, max_entries=100)
    world = _World(tmp_path, limits=limits)
    general = world.store.seed_general(PG_SCOPE, "beacon general packet")
    _seed(world.store, REPO_SCOPE, "memory", "beacon snapshot memory", "m-1")
    recalled_memory = _seed(world.store, REPO_SCOPE, "memory", "beacon recalled memory", "m-2")
    _seed(world.store, PG_SCOPE, "user", "beacon snapshot user", "u-1")
    recalled_user = _seed(world.store, PG_SCOPE, "user", "beacon recalled user", "u-2")

    text = world.event()

    assert world.ops() == LOAD_BOTH + RECALL_BOTH
    memory, user = world.recalls()
    # §9.3: the union of IDs delivered by both current snapshots and any earlier call in the event.
    assert memory.exclude_entry_ids == (general.id, "m-1", "u-1")
    assert user.exclude_entry_ids == (general.id, "m-1", "u-1", "m-2")
    first = _rendered(recalled_memory)
    # The separator between the two targets' text counts against the event budget.
    assert user.budget == w.RecallBudget(trusted_chars=0, evidence_chars=3000 - len(first) - 1, max_entries=19)
    assert text == first + "\n" + _rendered(recalled_user)


def test_event_output_never_exceeds_the_cap_and_memory_goes_first(tmp_path):
    world = _World(tmp_path)
    memory = [_seed(world.store, REPO_SCOPE, "memory", f"beacon {i} " + "m" * 700, f"m-{i}") for i in range(6)]
    _seed(world.store, PG_SCOPE, "user", "beacon user " + "u" * 40, "u-1")

    text = world.event()

    # The provider returns four records within its raw 3,000; only three fit once they are fenced.
    note = OMISSION_NOTE.format(count=1, label="memory", limit=3000)
    expected_memory = "\n".join(_rendered(record) for record in memory[:3]) + "\n" + note
    assert text == expected_memory
    assert len(text) <= t.RECALL_EVIDENCE_SCALARS
    _, user = world.recalls()
    assert user.exclude_entry_ids == ("m-0", "m-1", "m-2")
    assert user.budget == w.RecallBudget(trusted_chars=0, evidence_chars=3000 - len(expected_memory) - 1, max_entries=17)


def test_disabled_target_and_stateless_make_no_call(tmp_path):
    only_memory = _World(tmp_path, user_profile_enabled=False)
    only_memory.event()
    assert only_memory.ops() == [("load_curated", "memory"), ("recall_context", "memory")]

    only_user = _World(tmp_path, memory_enabled=False)
    only_user.event()
    assert only_user.ops() == [("load_curated", "user"), ("recall_context", "user")]

    neither = _World(tmp_path, memory_enabled=False, user_profile_enabled=False)
    assert neither.event() == "" and neither.ops() == [] and neither.warnings == []

    stateless = _World(tmp_path, policy="stateless", arm=lambda store: store.fail_transport("negotiate"))
    assert isinstance(stateless.service, StatelessMemoryService)
    assert stateless.event() == "" and stateless.ops() == [] and stateless.warnings == []


def test_provider_without_recall_makes_no_call_no_warning_no_block(tmp_path):
    """§9.6 L1574 is about `unavailable`; a provider that never negotiated recall is just absent (§9.10)."""
    world = _World(tmp_path, backend_type=partial(FakeAuthoritativeBackend, recall=False))
    _seed_both_targets(world.store)
    assert world.service.capabilities.recall_context is False
    assert world.event() == "" and world.ops() == [] and world.warnings == []


# --- failure semantics ----------------------------------------------------------


@pytest.mark.parametrize("arm", [
    lambda store: store.fail_transport("recall_context", times=2),
    lambda store: store.corrupt_result("recall_context", drop_key("target"), times=2),
], ids=["transport", "malformed"])
def test_unavailable_recall_warns_and_renders_nothing_after_required_load(tmp_path, arm):
    world = _World(tmp_path)
    _seed_both_targets(world.store)
    arm(world.store)

    text = world.event()

    assert world.ops() == LOAD_BOTH + RECALL_BOTH
    assert text == ""
    assert world.warnings == [t.RECALL_UNAVAILABLE_WARNING] and world.statuses == []
    assert world.service.blocked is False and world.service.degraded_warning() is None
    state = world.service.session_state
    for token in (state.provider_epoch, state.identity.binding_revision, state.identity.opaque_binding_b64url, "beacon"):
        assert token not in world.warnings[0]


class _StatusAdapter:
    """The Telegram and Slack shape: each status key owns one bubble, and the next status with that key edits it."""

    def __init__(self):
        self.bubbles = []

    async def send_or_update_status(self, chat_id, status_key, content, *, metadata=None):
        self.bubbles.append((status_key, content))


def test_unavailable_warning_reaches_the_gateway_status_path(agent_env):
    """§9.6 L1574 says show it: under the lifecycle key, the next lifecycle status would edit it away."""
    from gateway.config import Platform
    from gateway.run import _prepare_gateway_status_message
    from gateway.run_turn_runner import TurnRunner

    store, agent = agent_env.store, agent_env.agent
    store.curated_limits = NOTHING_DELIVERED
    _seed_both_targets(store)
    store.fail_transport("recall_context", times=2)
    adapter = _StatusAdapter()
    ctx = SimpleNamespace(source=SimpleNamespace(platform=Platform.TELEGRAM), _status_adapter=adapter,
                          _status_chat_id="chat-1", _status_thread_metadata=None, _cleanup_progress=False)
    runner = SimpleNamespace(_ctx=ctx, _status_live=lambda: True, _schedule=lambda coro, message: asyncio.run(coro))
    agent.status_callback = partial(TurnRunner._status_callback_sync, runner)     # as the gateway wires its agent
    create = agent.client.chat.completions.create
    create.return_value = _response("answer")

    result = agent.run_conversation("beacon")

    assert result.get("completed") is True and create.call_count == 1
    assert adapter.bubbles.count(("warn", t.RECALL_UNAVAILABLE_WARNING)) == 1
    assert ("lifecycle", t.RECALL_UNAVAILABLE_WARNING) not in adapter.bubbles
    # The chat-surface noise filter is text-based: no surface may drop or rewrite this wording.
    for platform in Platform:
        assert _prepare_gateway_status_message(platform, "warn", t.RECALL_UNAVAILABLE_WARNING) == t.RECALL_UNAVAILABLE_WARNING


def test_one_unavailable_target_renders_nothing_for_the_event_and_warns_once(tmp_path):
    world = _World(tmp_path)
    _seed_both_targets(world.store)
    world.after_first_recall(lambda store: store.fail_transport("recall_context"))
    assert world.event() == ""
    assert world.ops() == LOAD_BOTH + RECALL_BOTH and world.warnings == [t.RECALL_UNAVAILABLE_WARNING]


def test_a_transport_failure_never_hides_the_other_targets_ambiguity(tmp_path):
    world = _World(tmp_path)
    world.store.fail_transport("recall_context")             # consumed by the memory call
    world.store.fail_typed("recall_context", "ambiguous_policy", details=AMBIGUITY)
    with pytest.raises(MemoryBlockedError) as exc:
        world.event()
    assert exc.value.code == "ambiguous_policy" and world.warnings == []
    assert world.ops() == LOAD_BOTH + RECALL_BOTH


def test_a_failed_required_load_blocks_and_is_not_reported_as_unavailable_recall(tmp_path):
    # Fault on the user load: the memory load already succeeded, and no recall call is made.
    world = _World(tmp_path)
    _seed_both_targets(world.store)
    real = world.backend.load_curated

    def load_curated(request):
        if request.target == "user":
            world.store.fail_transport("load_curated")
        return real(request)

    world.backend.load_curated = load_curated
    with pytest.raises(MemoryBlockedError):
        world.event()
    assert world.ops() == LOAD_BOTH and world.warnings == []
    assert world.service.blocked is True                     # the gate's own latch, not an optional miss


@pytest.mark.parametrize("code,details", [
    ("version_conflict", None),
    ("invalid_request", None),
    ("secret_rejected", {"event_id": "e-1", "detector_code": "aws_access_key"}),
])
def test_other_typed_recall_refusals_warn_and_render_nothing(tmp_path, code, details):
    """A stale revision is reloaded by the next request's gate; at most one call per target, so no retry here."""
    world = _World(tmp_path)
    _seed_both_targets(world.store)
    world.store.fail_typed("recall_context", code, details=details)
    assert world.event() == ""
    assert world.ops() == LOAD_BOTH + RECALL_BOTH and world.warnings == [t.RECALL_UNAVAILABLE_WARNING]
    assert world.service.blocked is False


@pytest.mark.parametrize("code,details", [
    ("binding_revoked", None),
    ("provider_epoch_changed", {"expected_provider_epoch": "ep-1", "current_provider_epoch": "ep-2"}),
])
def test_binding_or_epoch_failure_during_recall_blocks(tmp_path, code, details):
    world = _World(tmp_path)
    world.store.fail_typed("recall_context", code, details=details)
    with pytest.raises(MemoryBlockedError) as exc:
        world.event()
    assert exc.value.code == code and world.warnings == []
    assert world.ops() == LOAD_BOTH + [("recall_context", "memory")]


def test_ambiguous_recall_blocks_delivery_and_is_not_swallowed(agent_env, caplog):
    store, agent = agent_env.store, agent_env.agent
    store.curated_limits = NOTHING_DELIVERED
    store.seed_record(REPO_SCOPE, "memory", "beacon SECRET-FACT")
    store.fail_typed("recall_context", "ambiguous_policy", details=AMBIGUITY)
    create = agent.client.chat.completions.create
    create.return_value = _response("sent")

    result = agent.run_conversation("beacon")

    assert result["failed"] and result.get("turn_exit_reason") == "curated_memory_blocked"
    create.assert_not_called()
    assert agent_env.count("recall_context") == 1            # the first blocker ends the event
    assert "SECRET-FACT" not in result["final_response"] and "k1" not in result["final_response"]
    assert "policy conflicts and must be resolved" in result["final_response"]
    assert "unavailable" not in result["final_response"] and not result["final_response"].endswith(" ")
    assert any("ambiguous_policy" in record.getMessage() and record.levelno == logging.WARNING
               for record in caplog.records)

    store.fail_typed("recall_context", "ambiguous_policy", details=AMBIGUITY)
    with pytest.raises(MemoryBlockedError) as exc:
        t.authoritative_turn_recall(agent._memory_service, "beacon")
    assert exc.value.code == "ambiguous_policy" and exc.value.provider_error.code == "ambiguous_policy"

    # Row 9 stays unlatched: the block belongs to that delivery, and the next clean turn proceeds.
    assert agent._memory_service.blocked is False
    assert agent.run_conversation("beacon").get("completed") is True


@pytest.mark.parametrize("exhausted", ["scalars", "entries"])
def test_zero_remaining_budget_still_surfaces_ambiguity(tmp_path, monkeypatch, exhausted):
    world = _World(tmp_path)
    record = _seed(world.store, REPO_SCOPE, "memory", "beacon", "m-1")
    if exhausted == "entries":
        monkeypatch.setattr(t, "RECALL_ENTRIES", 1)
    else:
        record.text = "beacon".ljust(3000 - (len(_rendered(record)) - len("beacon")), "x")
        assert len(_rendered(record)) == 3000                # memory takes the whole event budget
    world.after_first_recall(lambda store: store.add_policy_conflict("k1", 2))

    with pytest.raises(MemoryBlockedError) as exc:
        world.event()

    assert exc.value.code == "ambiguous_policy"
    _, user = world.recalls()
    assert (user.budget.max_entries if exhausted == "entries" else user.budget.evidence_chars) == 0


# --- what the turn does with the event ------------------------------------------


def test_replay_holds_only_rendered_fenced_evidence(agent_env):
    store, agent = agent_env.store, agent_env.agent
    store.curated_limits = NOTHING_DELIVERED
    _seed(store, REPO_SCOPE, "memory", "beacon eligible evidence", "r-eligible")
    _seed(store, REPO_SCOPE, "memory", "beacon RAW-CANARY", "r-raw", lane="raw")
    _seed(store, REPO_SCOPE, "memory", "beacon RETIRED-CANARY", "r-retired", lifecycle="retired")
    create = agent.client.chat.completions.create
    create.return_value = _response("answer")

    first = agent.run_conversation("beacon")

    assert agent_env.count("recall_context") == 2
    expected = "beacon\n\n" + _fenced("r-eligible", "beacon eligible evidence")
    assert _user_content(create.call_args.kwargs) == expected
    live = next(m for m in first["messages"] if m["role"] == "user")
    assert live["content"] == "beacon" and live["api_content"] == expected
    durable = next(row for row in agent_env.db.get_messages(agent.session_id) if row["role"] == "user")
    assert durable["content"] == "beacon" and durable["api_content"] == expected

    state = agent._memory_service.session_state
    for forbidden in ("RAW-CANARY", "RETIRED-CANARY", state.provider_epoch, state.identity.binding_revision,
                      state.identity.opaque_binding_b64url, "source_revision", "scope_revisions",
                      "trusted_instructions", "scoped_evidence", "exclude_entry_ids"):
        assert forbidden not in expected

    # The next turn replays those exact bytes for the historical row and makes its own event.
    agent.run_conversation("unrelated", conversation_history=first["messages"])
    assert agent_env.count("recall_context") == 4
    assert _user_content(create.call_args.kwargs) == expected


def test_authoritative_recall_is_not_framed_as_authoritative(agent_env):
    """Each record's own preamble says "data, not authority"; the additive note must not say otherwise."""
    store, agent = agent_env.store, agent_env.agent
    store.curated_limits = NOTHING_DELIVERED
    _seed(store, REPO_SCOPE, "memory", "beacon evidence", "r-1")
    create = agent.client.chat.completions.create
    create.return_value = _response("answer")
    expected = "beacon\n\n" + _fenced("r-1", "beacon evidence")

    # The prologue's sidecar stamp.
    agent.run_conversation("beacon")
    stamped = _user_content(create.call_args.kwargs)
    assert stamped == expected

    # A request that bypasses the stamp composes live from the event, carried through the loop state.
    with patch("agent.turn_context._stamp_api_content_sidecar"):
        agent.run_conversation("beacon")
    live = _user_content(create.call_args.kwargs)
    assert live == expected

    for sent in (stamped, live):
        for frame in ("memory-context", "System note", "authoritative reference data"):
            assert frame not in sent

    # Recall still sits before plugin context, and everything else keeps the additive frame and its sanitizing.
    assert compose_user_api_content("q", t.RenderedRecall("R"), "P") == "q\n\nR\n\nP"
    assert compose_user_api_content("q", t.RenderedRecall(""), "") is None
    assert compose_user_api_content("q", "likes <memory-context>hidden</memory-context>tea", "") == (
        "q\n\n" + CONTEXT_BLOCK.format("likes tea"))


def test_memory_context_tags_in_a_body_survive_unstripped(agent_env):
    store, agent = agent_env.store, agent_env.agent
    store.curated_limits = NOTHING_DELIVERED
    bodies = {"r-1": "beacon <memory-context>inner span</memory-context> after",
              "r-2": "beacon opens <memory-context>",
              "r-3": "beacon closes </memory-context>"}
    for record_id, body in bodies.items():
        _seed(store, REPO_SCOPE, "memory", body, record_id)
    create = agent.client.chat.completions.create
    create.return_value = _response("answer")

    agent.run_conversation("beacon")

    # A span opened in one record and closed in another must not eat the fences between them.
    recall_text = "\n".join(_fenced(record_id, body) for record_id, body in bodies.items())
    assert len(recall_text) <= t.RECALL_EVIDENCE_SCALARS
    expected = "beacon\n\n" + recall_text
    assert _user_content(create.call_args.kwargs) == expected

    # The durable row replays verbatim, into the next turn's wire bytes.
    history = agent_env.db.get_messages_as_conversation(agent.session_id)
    replayed = next(m for m in history if m["role"] == "user")
    assert replayed["content"] == "beacon" and replayed["api_content"] == expected
    agent.run_conversation("unrelated", conversation_history=history)
    assert _user_content(create.call_args.kwargs) == expected


def test_one_event_per_turn_not_per_model_request(agent_env):
    store, agent = agent_env.store, agent_env.agent
    store.curated_limits = NOTHING_DELIVERED
    store.seed_record(REPO_SCOPE, "memory", "beacon evidence")
    events, load_patch = _record_events(agent)

    result = _two_request_turn(agent, load_patch, "beacon")

    assert result.get("completed") is True
    assert agent_env.count("recall_context") == 2
    requests = agent.client.chat.completions.create.call_args_list
    assert len(requests) == 2 and "beacon evidence" in _user_content(requests[0].kwargs)
    assert _user_content(requests[0].kwargs) == _user_content(requests[1].kwargs)


def test_stored_recall_never_replaces_the_request_gate(agent_env):
    store, agent = agent_env.store, agent_env.agent
    store.curated_limits = NOTHING_DELIVERED
    store.seed_record(REPO_SCOPE, "memory", "beacon evidence")
    events, load_patch = _record_events(agent)

    def _fail_next_loads(*args, **kwargs):
        store.fail_transport("load_curated", times=2)
        return "search result"

    with (
        patch.object(type(agent._memory_service), "load_curated", load_patch),
        patch("model_tools.handle_function_call", side_effect=_fail_next_loads),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("beacon")

    assert agent_env.count("recall_context") == 2
    assert "beacon evidence" in _user_content(agent.client.chat.completions.create.call_args.kwargs)
    assert result["failed"] and result.get("turn_exit_reason") == "curated_memory_blocked"
    assert len([event for event in events if event[0] == MODEL]) == 1


@pytest.mark.parametrize("message", [
    [{"type": "text", "text": "beacon"}], "", "   ", "thanks", "/status",
], ids=["multimodal", "empty", "blank", "trivial", "slash"])
def test_non_string_empty_or_trivial_message_makes_no_recall_call(agent_env, message):
    store, agent = agent_env.store, agent_env.agent
    store.curated_limits = NOTHING_DELIVERED
    store.seed_record(REPO_SCOPE, "memory", "beacon evidence")
    create = agent.client.chat.completions.create
    create.return_value = _response("answer")
    first = agent.run_conversation("beacon")
    assert agent_env.count("recall_context") == 2            # the same session does recall on a real question
    recalls, loads = agent_env.count("recall_context"), agent_env.count("load_curated")
    create.reset_mock()
    create.return_value = _response("answer")

    second = agent.run_conversation(message, conversation_history=first["messages"])

    assert second.get("completed") is True and create.call_count == 1
    assert agent_env.count("recall_context") == recalls
    assert agent_env.count("load_curated") == loads + 2      # the request gate's own, one per enabled target
    assert "MEMORY_SCOPED_EVIDENCE" not in str(_last_user_content(create.call_args.kwargs))

    # With nothing to recall, the required loads alone still gate the request.
    store.fail_transport("load_curated", times=2)
    create.reset_mock()
    blocked = agent.run_conversation(message, conversation_history=second["messages"])
    assert blocked["failed"] and blocked.get("turn_exit_reason") == "curated_memory_blocked"
    create.assert_not_called()
    assert agent_env.count("recall_context") == recalls


# --- the additive path ----------------------------------------------------------


class _Manager:
    def __init__(self, *, fail=False):
        self.calls, self.fail = [], fail

    def on_turn_start(self, *args, **kwargs):
        self.calls.append(("on_turn_start", args, kwargs))

    def prefetch_all(self, *args, **kwargs):
        self.calls.append(("prefetch_all", args, kwargs))
        if self.fail:
            raise RuntimeError("sidecar down")
        return "PREFETCHED"

    def describe_recall(self):
        self.calls.append(("describe_recall", (), {}))
        return "Recalled memory"


def _additive_agent(service, manager, statuses):
    agent = SimpleNamespace(_memory_manager=manager, _user_turn_count=7, session_id="s-1", _emit_status=statuses.append)
    if service is not None:
        agent._memory_service = service
    return agent


@pytest.mark.parametrize("kind", ["absent", "builtin"])
def test_additive_path_is_byte_identical(kind):
    from tools.memory_tool import MemoryStore

    service = None if kind == "absent" else select_memory_service({}, store_factory=MemoryStore)
    forbidden = AssertionError("the additive path never reaches the service for recall")
    with (
        patch.object(BuiltinMemoryService, "load_curated", side_effect=forbidden),
        patch.object(BuiltinMemoryService, "recall_context", side_effect=forbidden),
    ):
        statuses, manager = [], _Manager()
        author = {"id": "u1", "name": "Ann", "is_bot": False}
        agent = _additive_agent(service, manager, statuses)
        assert _memory_turn_start_and_prefetch(agent, "a real question", author) == "PREFETCHED"
        assert manager.calls == [
            ("on_turn_start", (7, "a real question"), {"author_id": "u1", "author_name": "Ann", "author_is_bot": False}),
            ("prefetch_all", ("a real question",), {"session_id": "s-1"}),
            ("describe_recall", (), {}),
        ]
        assert statuses == ["Recalled memory"]
        assert compose_user_api_content("a real question", "PREFETCHED", "") == (
            "a real question\n\n" + CONTEXT_BLOCK.format("PREFETCHED"))

        # Trivial and non-string messages notify the provider and prefetch nothing.
        no_author = {"author_id": None, "author_name": None, "author_is_bot": False}
        for message, query in (("thanks", "thanks"), ([{"type": "text", "text": "x"}], "")):
            statuses, manager = [], _Manager()
            assert _memory_turn_start_and_prefetch(_additive_agent(service, manager, statuses), message) == ""
            assert manager.calls == [("on_turn_start", (7, query), no_author)] and statuses == []

        # A provider failure stays swallowed: additive recall is best effort.
        statuses, manager = [], _Manager(fail=True)
        assert _memory_turn_start_and_prefetch(_additive_agent(service, manager, statuses), "a real question") == ""
        assert statuses == []

        # No provider at all.
        assert _memory_turn_start_and_prefetch(_additive_agent(service, None, []), "a real question") == ""
