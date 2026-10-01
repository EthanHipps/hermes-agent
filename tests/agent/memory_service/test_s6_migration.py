"""S6 migration suite, CLI end to end (§13.1 L2182, L2183, L2213; §9.9; rulings R45-1, R45-14, R45-15, R45-21).

``hermes memory migrate`` runs through ``hermes_cli.main_agent_cmds.cmd_memory`` exactly as the CLI dispatches it.
The provider arrives through the patched discovery seam ``plugins.memory.load_authoritative_backend_factory``
(R36's fake); the terminal is scripted through ``hermes_cli.memory_migrate._interactive`` and ``_ask``. "Nothing
native is written" is proven with R37's sentinel and a byte-and-mtime snapshot, never inferred from output.
"""

import argparse
from argparse import Namespace
from contextlib import contextmanager

import pytest
import yaml

from agent.memory_service import migration_manifest as mm
from agent.memory_service.migration import RESTART_NOTICE, SWITCH_NOTICE
from hermes_cli.backup_memory import active_migration_manifests, refuse_if_migration_in_progress
from hermes_cli.memory_migrate import STALE_NATIVE_AFTER_ROLLBACK
from tests.agent.memory_service.migration_support import (
    MARKER, REPO, WRITE_EVENTS, Crash, file_bytes_under, make_env, snapshot, write_memory_section, write_native)
from tests.agent.memory_service.native_sentinel import native_memory_sentinel


@pytest.fixture
def s6(tmp_path, monkeypatch):
    env = make_env(tmp_path)
    write_native(env.home, memory=[f"uses postgres {MARKER}", "prefers tabs"], user=["name is Ethan"])
    monkeypatch.setattr("plugins.memory.load_authoritative_backend_factory", lambda name: env.factory)
    monkeypatch.setattr("hermes_cli.memory_migrate._interactive", lambda: True)
    answers = []
    monkeypatch.setattr("hermes_cli.memory_migrate._ask", lambda question: answers.pop(0) if answers else False)
    env.answers = answers
    return env


def _args(**over):
    base = dict(memory_command="migrate", migrate_command="start", source_id="home-default",
                item=["MEMORY.md", "USER.md"], scope="repository:repo-1", organization=None, project="proj-1",
                repository="repo-1", archive=None, run_id=None, yes=False, discard=False)
    return Namespace(**{**base, **over})


def _run(args) -> int:
    from hermes_cli.main_agent_cmds import cmd_memory
    with pytest.raises(SystemExit) as exited:
        cmd_memory(args)
    return exited.value.code


@contextmanager
def _patched_write(write):
    real = mm.write_document
    mm.write_document = write(real)
    try:
        yield
    finally:
        mm.write_document = real


def _crash_after_first_write(real):
    fired = []

    def write(path, doc):
        real(path, doc)
        if not fired:
            fired.append(True)
            raise Crash()
    return write


def _config_bytes(env):
    return (env.home / "config.yaml").read_bytes()


def _mode(env):
    return yaml.safe_load((env.home / "config.yaml").read_text(encoding="utf-8"))["memory"].get("provider_mode")


def _provider_tokens(env):
    staged = list(env.store.stage_by_request.values())
    return ["ep-1", *(s.result.stage_handle_b64url for s in staged), *(s.result.approval_binding_sha256 for s in staged)]


def test_start_migrates_both_files_switches_and_writes_nothing_native(s6, capsys):
    s6.answers.extend([True, True])
    native = s6.home / "memories"
    before = snapshot(native)
    with native_memory_sentinel(native) as sentinel:
        code = _run(_args())
    out = capsys.readouterr().out
    assert code == 0
    memory = yaml.safe_load((s6.home / "config.yaml").read_text(encoding="utf-8"))["memory"]
    assert memory == {**s6.section, "provider_mode": "authoritative"}
    assert [event for event, _ in sentinel.accesses if event in WRITE_EVENTS] == []
    assert snapshot(native) == before
    [run] = mm.list_runs(s6.home)
    assert run.document.state == "completed" and run.import_run_id in out
    assert MARKER.encode() not in file_bytes_under(s6.home, skip=native)
    assert not (s6.home / "memory_service" / "approvals").exists()
    for token in _provider_tokens(s6):
        assert token not in out
    assert out.count(SWITCH_NOTICE.strip()) == 1 and out.count(RESTART_NOTICE) == 1


def test_non_interactive_start_refuses_before_anything(s6, monkeypatch, capsys):
    monkeypatch.setattr("hermes_cli.memory_migrate._interactive", lambda: False)
    listing = sorted(p.relative_to(s6.home).as_posix() for p in s6.home.rglob("*"))
    assert _run(_args()) == 2
    assert "interactive terminal" in capsys.readouterr().out
    assert s6.backends == []
    assert sorted(p.relative_to(s6.home).as_posix() for p in s6.home.rglob("*")) == listing


def test_authoritative_home_refuses_a_live_source_without_touching_native(s6, capsys):
    write_memory_section(s6.home, {**s6.section, "provider_mode": "authoritative"})
    with native_memory_sentinel(s6.home / "memories", deny=True) as sentinel:
        code = _run(_args())
    assert code == 2 and sentinel.accesses == []
    out = capsys.readouterr().out
    assert "dormant" in out and "native_dormant" in out
    assert s6.backends == []


def test_denial_exits_zero_and_unblocks_archives(s6):
    config = _config_bytes(s6)
    s6.answers.append(False)
    assert _run(_args()) == 0
    refuse_if_migration_in_progress([s6.home])
    assert _config_bytes(s6) == config
    [run] = mm.list_runs(s6.home)
    assert (run.document.state, run.document.outcome) == ("rolled_back", "operator_denied")


def test_status_is_content_free(s6, capsys):
    s6.answers.extend([True, True])
    assert _run(_args()) == 0
    write_memory_section(s6.home, s6.section)                   # additive again, by hand
    s6.answers.append(False)
    assert _run(_args()) == 0
    capsys.readouterr()
    assert _run(_args(migrate_command="status")) == 0
    out = capsys.readouterr().out
    runs = mm.list_runs(s6.home)
    assert len(runs) == 2 and all(r.import_run_id in out for r in runs)
    assert "completed/completed" in out and "rolled_back/operator_denied" in out
    for token in [MARKER, *_provider_tokens(s6)]:
        assert token not in out


def test_status_of_a_compacted_run_does_not_count_withheld_raw_as_plain_new(s6, capsys):
    """A receipt keeps no disposition (R45-impl-3), so its create count must not read as plain new memory."""
    s6.store.admission_classifier = lambda cand: "withheld_raw" if "tabs" in cand.text else (
        "scoped_evidence" if cand.target == "memory" else "trusted_instruction")
    s6.answers.extend([True, True])
    assert _run(_args()) == 0
    capsys.readouterr()
    assert _run(_args(migrate_command="status")) == 0
    [memory] = [line for line in capsys.readouterr().out.splitlines() if "memory (MEMORY.md)" in line]
    assert "2 new incl. any withheld raw, 0 already present (committed)" in memory


def test_rollback_switches_back_and_warns_stale(s6, capsys):
    s6.answers.extend([True, True])
    assert _run(_args()) == 0
    native = s6.home / "memories"
    before = snapshot(native)
    [run] = mm.list_runs(s6.home)
    receipt = run.path.read_bytes()
    capsys.readouterr()
    assert _run(_args(migrate_command="rollback", yes=True)) == 0
    assert _mode(s6) == "additive"
    assert STALE_NATIVE_AFTER_ROLLBACK in capsys.readouterr().out
    assert snapshot(native) == before and run.path.read_bytes() == receipt


def test_rollback_refuses_stateless_policy_first(s6, capsys):
    write_memory_section(s6.home, {**s6.section, "provider_mode": "authoritative",
                                   "authoritative_failure_policy": "stateless"})
    config = _config_bytes(s6)
    assert _run(_args(migrate_command="rollback", yes=True)) == 2
    out = capsys.readouterr().out
    assert "remove memory.authoritative_failure_policy" in out and "rerun 'hermes memory migrate rollback'" in out
    assert _config_bytes(s6) == config and not (s6.home / "migrations").exists() and s6.backends == []


def test_a_failed_rollback_compaction_exits_one_and_keeps_the_config(s6):
    s6.answers.extend([True, True])
    with _patched_write(_crash_after_first_write), pytest.raises(Crash):
        from hermes_cli.main_agent_cmds import cmd_memory
        cmd_memory(_args())
    write_memory_section(s6.home, {**s6.section, "provider_mode": "authoritative"})
    config = _config_bytes(s6)

    def failing(real):
        def write(path, doc):
            if doc["state"] == "rolled_back":
                raise OSError("disk full")
            real(path, doc)
        return write
    with _patched_write(failing):
        assert _run(_args(migrate_command="rollback", yes=True)) == 1
    assert _config_bytes(s6) == config and len(active_migration_manifests(s6.home)) == 1


def test_a_managed_provider_mode_refuses_start(s6, monkeypatch):
    monkeypatch.setattr("hermes_cli.managed_scope.is_key_managed", lambda key: key == "memory.provider_mode")
    assert _run(_args()) == 2
    assert s6.backends == [] and not (s6.home / "migrations").exists()


def test_off_names_the_rollback_command(s6, capsys):
    from hermes_cli.main_agent_cmds import _cmd_memory_off
    write_memory_section(s6.home, {**s6.section, "provider_mode": "authoritative"})
    config = _config_bytes(s6)
    with pytest.raises(SystemExit) as exited:
        _cmd_memory_off()
    assert exited.value.code == 2 and "hermes memory migrate rollback" in capsys.readouterr().out
    assert _config_bytes(s6) == config


def test_import_agent_then_migrate_composes(s6, tmp_path):
    """R46-6: import while additive, then the §9.9 migration; afterwards R46-1 skips the memory item."""
    from hermes_cli import agent_import
    from hermes_cli.agent_import import AgentImporter
    tree = tmp_path / ".claude"
    tree.mkdir()
    (tree / "CLAUDE.md").write_text("# Rules\n\n- Always use type hints\n", encoding="utf-8")
    AgentImporter("claude-code", tree, s6.home, execute=True).run()
    assert "Always use type hints" in (s6.home / "memories" / "MEMORY.md").read_text(encoding="utf-8")
    s6.answers.append(True)
    assert _run(_args(item=["MEMORY.md"])) == 0
    records = [r for r in s6.store.records.values() if "Always use type hints" in r.text]
    assert records and all(r.origin_scope == REPO for r in records)
    assert _mode(s6) == "authoritative"
    report = AgentImporter("claude-code", tree, s6.home, execute=True).run()
    [memory] = [i for i in report["items"] if i["kind"] == "claude-md"]
    assert (memory["status"], memory["reason"]) == ("skipped", agent_import.PROVIDER_MANAGED_MEMORY_REASON)


def test_resume_and_reconcile_commands_map_exit_codes(s6):
    s6.answers.extend([True, True])
    with _patched_write(_crash_after_first_write), pytest.raises(Crash):
        from hermes_cli.main_agent_cmds import cmd_memory
        cmd_memory(_args())
    s6.answers.extend([True, True])
    assert _run(_args(migrate_command="resume")) == 0
    corrupt = mm.manifest_path(s6.home, "example", "ep-1", "a" * 32)
    corrupt.write_bytes(b"{")
    assert _run(_args(migrate_command="reconcile", run_id="a" * 32)) == 2
    assert _run(_args(migrate_command="reconcile", run_id="a" * 32, discard=True, yes=True)) == 0
    assert not corrupt.exists() and active_migration_manifests(s6.home) == []


def test_existing_memory_subcommands_are_unchanged(monkeypatch):
    from hermes_cli.main_agent_cmds import cmd_memory
    calls = []
    monkeypatch.setattr("hermes_cli.memory_setup.memory_command", lambda args: calls.append(("setup", args.memory_command)))
    monkeypatch.setattr("hermes_cli.main_agent_cmds._cmd_memory_off", lambda: calls.append(("off",)))
    monkeypatch.setattr("hermes_cli.main_agent_cmds._cmd_memory_reset", lambda args: calls.append(("reset", args.target)))
    cmd_memory(Namespace(memory_command="status", session=None))
    cmd_memory(Namespace(memory_command="off"))
    cmd_memory(Namespace(memory_command="reset", target="memory", yes=False, scope=None))
    cmd_memory(Namespace(memory_command=None))
    assert calls == [("setup", "status"), ("off",), ("reset", "memory"), ("setup", None)]


def test_the_migrate_parser_carries_the_explicit_chain():
    """C7F-4/C7F-5: the parser the CLI and the console build; --source-id is required (R45-21)."""
    from hermes_cli.subcommands.memory import build_memory_parser

    def handler(args):
        return None
    parser = argparse.ArgumentParser(prog="hermes")
    build_memory_parser(parser.add_subparsers(dest="command"), cmd_memory=handler)
    ns = parser.parse_args(["memory", "migrate", "start", "--source-id", "home-default", "--item", "MEMORY.md",
                            "--scope", "repository:repo-1", "--project", "proj-1", "--repository", "repo-1"])
    assert (ns.memory_command, ns.migrate_command, ns.func) == ("migrate", "start", handler)
    assert (ns.organization, ns.project, ns.repository, ns.item, ns.archive) == (None, "proj-1", "repo-1",
                                                                                 ["MEMORY.md"], None)
    with pytest.raises(SystemExit):
        parser.parse_args(["memory", "migrate", "start", "--item", "MEMORY.md"])
    ns = parser.parse_args(["memory", "migrate", "reconcile", "a" * 32, "--discard", "--yes"])
    assert (ns.run_id, ns.discard, ns.yes) == ("a" * 32, True, True)


# -- Task 11: legacy Hermes archives (R45-8; §9.8 L1649, §9.9 L1657) ---------------------------------------

def _legacy_zip(path, memory_text):
    import zipfile
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(".hermes/config.yaml", "memory: {}\n")
        zf.writestr(".hermes/memories/MEMORY.md", memory_text.encode("utf-8"))
    return path


def _archive_args(archive, **over):
    return _args(archive=str(archive), item=["memories/MEMORY.md"], **over)


def _archive_tuples(env):
    return [key for key in env.store.import_index
            if key[2:6] == ("legacy_archive", "hermes-legacy-archive-v1", "home-default", "memories/MEMORY.md")]


def test_a_legacy_archive_migrates_in_an_authoritative_home_without_switching(s6, tmp_path):
    archive = _legacy_zip(tmp_path / "old.zip", "archived fact one\n§\narchived fact two")
    write_memory_section(s6.home, {**s6.section, "provider_mode": "authoritative"})
    config = _config_bytes(s6)
    s6.answers.append(True)
    with native_memory_sentinel(s6.home / "memories", deny=True) as sentinel:
        code = _run(_archive_args(archive))
    assert code == 0 and sentinel.accesses == []
    assert _config_bytes(s6) == config                          # an archive run never switches the mode
    assert len(_archive_tuples(s6)) == 2
    [run] = mm.list_runs(s6.home)
    assert (run.document.state, run.document.source.source_kind) == ("completed", "legacy_archive")


def test_rerunning_an_archive_with_a_new_run_id_dedupes(s6, tmp_path):
    archive = _legacy_zip(tmp_path / "old.zip", "archived fact one\n§\narchived fact two")
    write_memory_section(s6.home, {**s6.section, "provider_mode": "authoritative"})
    s6.answers.extend([True, True])
    assert _run(_archive_args(archive)) == 0
    [first] = mm.list_runs(s6.home)
    records = len(s6.store.records)
    assert _run(_archive_args(archive)) == 0
    [second] = [r for r in mm.list_runs(s6.home) if r.import_run_id != first.import_run_id]
    assert len(s6.store.records) == records
    effects = [a.publication_effect for b in second.document.batches for a in b.assigned]
    assert effects == ["reuse_existing_import", "reuse_existing_import"]


def test_an_archive_restage_needs_the_same_archive(s6, tmp_path, capsys):
    from tests.agent.memory_service.fake_backend import DEFAULT_LIMITS
    archive = _legacy_zip(tmp_path / "old.zip", "archived fact one\n§\narchived fact two")
    other = _legacy_zip(tmp_path / "other.zip", "a different fact")
    s6.answers.append(True)
    with _patched_write(_crash_after_first_write), pytest.raises(Crash):
        from hermes_cli.main_agent_cmds import cmd_memory
        cmd_memory(_archive_args(archive))
    s6.store.clock.advance(DEFAULT_LIMITS.stage_ttl_seconds + 1)
    [run] = mm.list_runs(s6.home)
    before = run.path.read_bytes()
    capsys.readouterr()
    assert _run(_args(migrate_command="resume")) == 1
    assert "(archive_required)" in capsys.readouterr().out and run.path.read_bytes() == before
    assert _run(_args(migrate_command="resume", archive=str(other))) == 1
    assert "(source_changed)" in capsys.readouterr().out and run.path.read_bytes() == before
    s6.answers.extend([True])
    assert _run(_args(migrate_command="resume", archive=str(archive))) == 0
    assert mm.list_runs(s6.home)[0].document.state == "completed"
