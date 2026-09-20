"""R37: host facts -> RequestedContext -> MemoryService, before any native store exists."""

import os

import pytest

from agent.memory_service.bootstrap import build_requested_context
from agent.memory_service.config import resolve_memory_service_config


def _authoritative(tmp_path, **extra):
    exe = tmp_path / "p.exe"
    exe.write_bytes(b"MZ")
    section = {"provider": "example", "provider_mode": "authoritative",
               "provider_executable": str(exe), "principal_id": "ethan"}
    section.update(extra)
    return resolve_memory_service_config({"memory": section})


def test_context_carries_host_facts_and_validates(tmp_path):
    cfg = _authoritative(tmp_path)
    ctx = build_requested_context(
        cfg, logical_session_id="sess-1", platform="cli",
        profile_id="default", working_directory=str(tmp_path),
    )
    ctx.validate()  # raises WireError if the shape is wrong
    assert ctx.principal_id == "ethan"
    assert ctx.logical_session_id == "sess-1"
    assert ctx.platform == "cli"
    assert ctx.profile_id == "default"


def test_scopes_are_provider_resolved_not_host_guessed(tmp_path):
    """D-R37-1 / §4.2: Hermes names the directory; the provider resolves org/project/repo."""
    cfg = _authoritative(tmp_path)
    ctx = build_requested_context(
        cfg, logical_session_id="s", platform="cli",
        profile_id="default", working_directory=str(tmp_path),
    )
    assert ctx.resolution_source == "directory"
    assert (ctx.org_id, ctx.project_id, ctx.repo_id, ctx.workspace_id) == (None, None, None, None)


def test_canonical_directory_is_fully_resolved(tmp_path):
    """§4.2 matches the longest canonical registered repo path; a relative or
    unnormalized path could match the wrong registration or none at all."""
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    cfg = _authoritative(tmp_path)
    ctx = build_requested_context(
        cfg, logical_session_id="s", platform="cli", profile_id="default",
        working_directory=str(tmp_path / "a" / ".." / "a" / "b"),
    )
    assert ctx.canonical_directory == os.path.realpath(str(nested))


def test_working_directory_defaults_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _authoritative(tmp_path)
    ctx = build_requested_context(cfg, logical_session_id="s", platform="cli", profile_id="default")
    assert ctx.canonical_directory == os.path.realpath(str(tmp_path))


def test_profile_id_defaults_to_the_active_profile(tmp_path):
    cfg = _authoritative(tmp_path)
    ctx = build_requested_context(cfg, logical_session_id="s", platform="cli",
                                  working_directory=str(tmp_path))
    assert ctx.profile_id  # non-empty; get_active_profile_name() never returns ""
