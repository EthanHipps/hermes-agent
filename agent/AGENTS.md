# agent/ — AIAgent, turn loop, prompt, compression

Applies on top of the root `AGENTS.md` (prompt-caching invariant, facade + siblings rules).

## Shape

`run_agent.py` is the public facade: `AIAgent` is assembled from mixins (`agent/turn_facade.py`,
`client_lifecycle.py`, `stream_delivery.py`, `session_persistence.py`, `compression_facade.py`, ...).
Construction runs `agent/agent_init.py::init_agent`; a turn is
`agent/conversation_loop.py::run_conversation`, which `AIAgent.run_conversation` forwards to after
taking the session turn lease (`turn_facade_lease.py`). `AIAgent.__init__` takes ~60 parameters
(credentials, routing, callbacks, session context, budget, credential pool, ...) — read
`run_agent.py` for the list; the subset you usually touch: `base_url`, `api_key`, `provider`,
`api_mode` (`"chat_completions" | "codex_responses" | ...`), `model` (empty → resolved from
config/provider later), `max_iterations` (default 500, shared with subagents),
`enabled_toolsets`/`disabled_toolsets`, `quiet_mode`, `save_trajectories`, `platform`
(`"cli"`, `"telegram"`, ...), `session_id`, `skip_context_files`, `skip_memory`, `credential_pool`.
`chat(message) -> str` is the simple interface; `run_conversation(user_message, system_message=None,
conversation_history=None, task_id=None) -> dict` returns `final_response` + `messages`.

## Agent loop (`agent/conversation_loop.py` + `agent/turn_*.py`)

Entirely synchronous, with interrupt checks, budget tracking, and a one-turn grace call:

```python
while (api_call_count < self.max_iterations and self.iteration_budget.remaining > 0) \
        or self._budget_grace_call:
    if self._interrupt_requested: break
    response = client.chat.completions.create(model=model, messages=messages, tools=tool_schemas)
    if response.tool_calls:
        for tc in response.tool_calls:
            messages.append(tool_result_message(handle_function_call(tc.name, tc.args, task_id)))
        api_call_count += 1
    else:
        return response.content
```

Each phase of an iteration is its own sibling, so a change to (say) overflow handling touches one
~600-line file: `turn_preflight*`, `turn_iteration_prep`, `turn_request_assembly`/`turn_api_request`,
`turn_api_call`, `turn_api_error`, `turn_response_intake`/`turn_response_check`,
`turn_empty_response`, `turn_tool_round`/`turn_tool_validation`, `turn_overflow`,
`turn_truncation`, `turn_context_compaction`, `turn_recovery`, `turn_retry_state`,
`turn_stop_gates`, `turn_liveness`, `turn_usage`, `turn_final_response`, `turn_finalizer`,
`turn_summary`. Find the phase with `grep -rn "def X" agent/turn_*.py`.

Messages use OpenAI format `{"role": "system|user|assistant|tool", ...}`; reasoning content is stored
in `assistant_msg["reasoning"]`.

**Agent-level tools** (`todo`, `memory`, ...) are intercepted by `agent/tool_executor.py` through the
`INLINE_TOOL_EXECUTORS` table in `agent/inline_tool_executors.py` before `handle_function_call()`.
Adding one: register in that table (no `if name == ...` chain); `tools/todo_tool.py` is the pattern.

## Message-flow invariants (every change is reviewed against these)

- **Prompt caching must not break.** Never alter past context, change toolsets, reload memories,
  or rebuild the system prompt mid-conversation. The system prompt is byte-stable for the life of
  a conversation; the ONLY context mutation is compression. Anything that must inject content
  mid-conversation rides a **user message or tool result**, never the system prompt: skill slash
  commands (`agent/skill_commands.py`) inject as a user message; subdirectory `AGENTS.md` hints
  (`agent/subdirectory_hints.py`) append to the tool result (head+tail truncated past `_MAX_HINT_CHARS = 32_000`, with a warning).
- **Strict role alternation.** Never two same-role messages in a row; never a synthetic user
  message injected mid-loop. The one exception is `/steer`, delivered as a standalone user row
  after a tool result (`assistant(tool_calls) → tool → user` is legal on every provider path) —
  never smeared onto the already-persisted tool row, which append-only persistence would leave
  divergent from the live request. Cron deliveries live in their own session for this reason.
- **Context files** (`agent/prompt_builder.py`) load from the CWD only at startup and are capped
  (`CONTEXT_FILE_MAX_CHARS` / dynamic cap from the context window / `context_file_max_chars`).
  Never load an install-tree `AGENTS.md` as project context (PR #64611); subdirectory hints reject
  paths outside the working dir so `~/.codex/AGENTS.md` / `~/.claude/CLAUDE.md` never mix in.
- **`_last_resolved_tool_names` is a process-global in `model_tools.py`.** `_run_single_child()`
  in `tools/delegate_tool.py` saves/restores it around subagent execution; code reading it may see
  a temporarily stale value during child runs.

## Compression (`agent/compression_facade.py`, `conversation_compression.py`, `turn_context_compaction.py`)

Two layers: gateway session hygiene (85% threshold) and the agent `ContextCompressor` (50%,
configurable; per-model overrides; failure cooldown after provider-proven overflow). The algorithm
prunes old tool results first (no LLM call), then picks boundaries, then generates a structured
summary with the `auxiliary` compression model. In-place compaction keeps a single stable session
id; native Responses/Codex compaction paths are provider-specific. Compression is the sanctioned
cache break — keep it the only one. The handoff prefix and head note a session EMITS are
`ContextCompressor.summary_prefix` / `.compression_note`; an authoritative or stateless session
swaps in the `CURATED_MEMORY_*` variants, which differ from the live texts only in the memory
clause (see "Memory, context engines, curator"). Full detail:
`website/docs/developer-guide/context-compression-and-caching.md`.

## Model and provider resolution

- Runtime provider/model resolution and its precedence: `website/docs/developer-guide/provider-runtime.md`.
  Provider profiles are plugins (`plugins/model-providers/<name>/`, see `plugins/AGENTS.md`);
  `agent/model_metadata.py` holds context lengths and capabilities.
- **Auxiliary (side-LLM) work** — curator, vision, embedding, title generation, session_search,
  compression — resolves through `agent/auxiliary_client.py::_resolve_auto_route`; each task can pin
  its own `provider/model/base_url/reasoning_effort` under `auxiliary:` in config.yaml.
- Fallback models and credential pools are resolution-chain code: E2E them with real imports
  against a temp `HERMES_HOME`, not mocks (root rubric).

## Memory, context engines, curator

`agent/memory_provider.py` (ABC) + `agent/memory_manager.py` (orchestrator) drive memory-provider
plugins; `agent/context_engine.py` drives context-engine plugins; `agent/image_gen_provider.py`
image-gen plugins (all in `plugins/AGENTS.md`). `agent/curator.py` + `curator_backup.py` implement
the skill curator (`skills/AGENTS.md`). Cron agents are built with `skip_memory=False` and
`platform="cron"` (`cron/scheduler.py::_construct_cron_agent`), so memory — and a configured external
memory provider — loads as in any other session; in a provider-managed session the memory tool
refuses cron calls before any load until cron receives an explicit frozen-identity service.

`agent/memory_service/` is the host-owned generic memory service: `agent_init._init_memory`
selects the router via `bootstrap.init_memory_service` **before** any native `MemoryStore` is
constructed, so an authoritative or stateless-fallback session never builds, stats, or reads
`MEMORY.md`/`USER.md` — `agent._memory_store` stays `None` in that case. `agent._memory_service`
is `None` when memory is skipped entirely (`skip_memory=True` with no `memory` toolset requested,
as delegate children, the curator and batch runs do; the store is `None` too) or when *additive*
service init fails, which degrades
to the native store alone exactly as before the router existed. A configuration error or
fail-closed provider failure in authoritative mode propagates out of init instead — a session
never switches mode because of failure. `tests/agent/memory_service/
native_sentinel.py` guards the native directory with an audit hook plus `os.stat`/`os.lstat`
interception (CPython raises no audit event for stat) so "not used" is proven, not inferred.
Home skeleton initialization and doctor suppress native storage based on the requested
authority mode, even when provider validation fails. Cold-process tests arm the sentinel
before imports/config loading. Bootstrap binds to `runtime_cwd.resolve_agent_cwd()` (unless
an explicit working directory is supplied), and the default agent platform binds as `cli`.

Archives follow the same dormancy: `hermes_cli/backup_memory.py` decides each Hermes home's mode
from its own config, so an authoritative home's `memories/` is never walked, copied, created or
restored by backup, export, clone or import; every such archive carries
`agent/memory_service/archive.py`'s generic `curated_memory` record
(`curated-memory-disposition.yaml`); `migrations/` and the host session-state directory
`memory_service/` never enter an archive in any mode and are never restored; and an active
migration manifest refuses backup/export/clone with `MIGRATION_IN_PROGRESS`.

In a provider-managed session (`memory_service.service.is_provider_managed`) the memory tool
dispatches only through `MemoryService`: `inline_tool_executors._memory` passes `service=` (never
`store=`) and never calls `notify_memory_tool_write`; `tools/memory_tool_curated.py` runs the native
`MemoryStore` semantics over one complete snapshot in memory (`SnapshotMemoryStore`) and turns the
result into an explicit intent and an ID-based delta; `memory_service/mutation.py::run_curated_mutation`
loads fresh, stages, commits, replays `version_conflict` with a new request ID and reloads every
enabled target after a commit. Mutations that need approval are staged and approved as described
below, and cron and background-review calls are refused before any load until they receive an
explicit frozen-identity service.

Approvals in a provider-managed session go through the `approval=` channel of
`memory_service/mutation.py::run_curated_mutation` (`memory_service/approval.py::ApprovalChannel`).
Hermes predicts which mutations need approval: `target:user`, a non-default scope, `bulk_edit`,
reset, import, threat, or `memory.write_approval`, whose unreadable or unrecognized value
counts as on. Such a mutation is staged, inspected with `inspect_staged`, and approved or
denied through the terminal approval callback (never inline in a background-review fork, as
with native `evaluate_gate`). When nobody can answer, it waits for `/memory pending`, but only
when the session's persisted host-state record equals the staging identity, so replay can
resume it. Without such an identity it fails closed: before staging when nobody can be asked,
and after staging (the stage then expires) when an inline prompt goes unanswered.
Only the stage handle, binding hash, IDs, revision, scopes, decision and expiry are persisted,
never candidate text or an entry body: one owner-only record per approval under
`<home>/memory_service/approvals/` (`approval_store.py`), which no archive carries. In a home
that requests authoritative mode, `/memory pending|approve|reject`
(`hermes_cli/write_approval_commands_curated.py`) renders the live inspection. It replays
the byte-free `commit_curated` from the staging session's persisted identity on its own
transport (`approval_replay.py`), and it retries an unknown outcome only with the identical
request. It drops a record on denial, expiry, conflict or a void binding. Native
`pending/memory/` records stay dormant. The memory tool still refuses threat-pattern content,
as native memory does. A caller that sets `PlannedMutation.threat_decision_id` gets
mandatory approval instead.

In authoritative or stateless mode the system prompt's curated region comes only from
`agent/memory_service/render.py`, through `lifecycle.curated_prompt_parts` — one structured
region, never concatenated with native data — and the stable-tier memory-tool guidance comes
from `lifecycle.memory_guidance_flags`, so `agent._memory_enabled`/`_user_profile_enabled` stay
`False` and are read, never written. The frozen identity is persisted per Hermes session under
`<home>/memory_service/sessions/` (`host_state.py`, owner-only JSON). It is resolved at agent
init and again, lazily, whenever `agent.session_id` has moved — at turn start, before any prompt
render and before any compression begins (`lifecycle.follow_session_binding` →
`bootstrap.resolve_session_binding`: resume, inherit, new, stateless, invalid) — so
compression, branch, rewind, `/resume` and a working-directory change never re-bind, and an
out-of-turn `/context` or `/compress` never uses the previous session's service; only `/new` and
a genuinely new logical session bind, with intent `new_session`.
A conversation that already has messages but no record fails `binding_invalid`. Each model
request passes `lifecycle.curated_request_gate` — a fresh load per enabled target, which
validates and blocks but never re-renders the byte-stable prompt — and a stored prompt is
reused only on an exact disposition-plus-digest match. Compression carries the record to a
rotated session id and, when the provider negotiated it, submits the committed summary through
`capture_continuity` on a daemon thread over a second transport. Authoritative compressors emit
`CURATED_MEMORY_SUMMARY_PREFIX` / `CURATED_MEMORY_COMPRESSION_NOTE`; retiring `SUMMARY_PREFIX`
retires the variant too, and the frozen historical recognizers are never edited.
**Per-turn authoritative recall is not wired; see ledger R40-8** — the entry in the ygg
revision ledger's "Wave 6 fork decisions: R38, R40, R44 (2026-09-22)" section, which names R48
as the owner of host recall wiring and of §9.6's host-visible recall warning (L1574) and
recall-ambiguity blocking (L1576) cells.

## Tests

Loop/phase tests go in `tests/agent/`; patch the binding the phase actually reads (siblings often
`from run_agent import X` inside the function — root "patch where production reads"). Assert
message-shape invariants (alternation, byte-stable system prompt) rather than snapshotting prompt
text.

Long-form: `website/docs/developer-guide/agent-loop.md`, `prompt-assembly.md`,
`context-compression-and-caching.md`, `provider-runtime.md`, `session-storage.md`,
`subagent-lifecycle-api.md`.
