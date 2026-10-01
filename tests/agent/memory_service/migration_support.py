"""R45-local test support (wave-6 rule 10 keeps the shared support frozen; only R45's suites import this)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml

from agent.memory_service import wire as w
from tests.agent.memory_service.fake_backend import FakeClock, FakeProviderStore, fake_backend_factory

REPO = w.ScopeRef(kind="repository", id="repo-1")
PROJECT = w.ScopeRef(kind="project", id="proj-1")
GLOBAL = w.ScopeRef(kind="principal_global", id="ethan")
MARKER = "zq-marker-7f3a"            # appears only in native text; must never reach a host file other than the source
WRITE_EVENTS = frozenset({"os.remove", "os.rename", "os.rmdir", "os.chmod", "os.utime", "os.truncate",
                          "os.link", "os.symlink", "os.mkdir"})


class Crash(BaseException):
    """A simulated process death. BaseException, so no ``except Exception`` in production swallows it."""


@dataclass
class Env:
    home: Path
    store: FakeProviderStore
    factory: Callable
    section: Dict[str, object]
    backends: List = field(default_factory=list)         # every backend the factory built, in order
    switches: List[str] = field(default_factory=list)

    def calls(self, operation: str) -> List:
        return [request for b in self.backends for op, request in b.calls if op == operation]

    def raw(self) -> dict:
        from hermes_cli.backup_memory import home_memory_section
        return {"memory": dict(home_memory_section(self.home))}

    def switch(self) -> None:
        self.switches.append("authoritative")
        write_memory_section(self.home, {**self.section, "provider_mode": "authoritative"})


def write_memory_section(home: Path, section: dict) -> Path:
    path = home / "config.yaml"
    path.write_text(yaml.safe_dump({"memory": section}), encoding="utf-8")
    return path


def make_env(tmp_path: Path, *, mode: Optional[str] = None, store: Optional[FakeProviderStore] = None) -> Env:
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    exe = tmp_path / "provider.exe"
    exe.write_bytes(b"MZ")
    section = {"provider": "example", "provider_executable": str(exe), "principal_id": "ethan"}
    if mode:
        section["provider_mode"] = mode
    write_memory_section(home, section)
    store = store or FakeProviderStore(epoch="ep-1", clock=FakeClock(datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)))
    backends: List = []
    base = fake_backend_factory(store, provider="example")

    def factory(cfg):
        backends.append(base(cfg))
        return backends[-1]
    return Env(home=home, store=store, factory=factory, section=section, backends=backends)


def write_native(home: Path, memory: Optional[List[str]] = None, user: Optional[List[str]] = None) -> Path:
    """LF bytes exactly; ``write_text`` would emit CRLF on Windows. CRLF parity has its own test."""
    native = home / "memories"
    native.mkdir(exist_ok=True)
    for name, entries in (("MEMORY.md", memory), ("USER.md", user)):
        if entries is not None:
            (native / name).write_bytes("\n§\n".join(entries).encode("utf-8"))
    return native


def snapshot(root: Path):
    """Bytes and mtime of every file under root; taken outside any sentinel."""
    if not root.exists():
        return None
    return {p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file()}


def file_bytes_under(root: Path, *, skip: Optional[Path] = None) -> bytes:
    """Every file under root concatenated (for 'never persisted' byte searches), optionally skipping a subtree."""
    out = b""
    for p in sorted(root.rglob("*")):
        if p.is_file() and (skip is None or skip not in p.parents):
            out += p.read_bytes()
    return out


class Answers:
    """A scripted PromptFn: pops one answer per prompt and records every prompt."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, approval):
        self.prompts.append(approval)
        return self.answers.pop(0) if self.answers else None


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def plan(kind="native_memory", items=("MEMORY.md", "USER.md"), scope=REPO, archive=None, source_id="home-default"):
    from agent.memory_service.admin import AdminContext
    from agent.memory_service.migration import MigrationPlan, SourceSelection
    has_memory = any(k.endswith("MEMORY.md") for k in items)
    return MigrationPlan(source=SourceSelection(kind=kind, source_id=source_id, item_keys=tuple(items), archive=archive),
                         memory_scope=scope if has_memory else None,
                         context=AdminContext(repo_id="repo-1", project_id="proj-1"))


def start(env: Env, answers, **plan_kwargs):
    from agent.memory_service.migration import start_migration
    return start_migration(env.raw(), plan(**plan_kwargs), home=env.home, prompt=answers,
                           switch_to_authoritative=env.switch, backend_factory=env.factory, clock=env.store.clock.now)


def resume(env: Env, answers, **kwargs):
    from agent.memory_service.migration import resume_migration
    return resume_migration(env.raw(), home=env.home, prompt=answers, switch_to_authoritative=env.switch,
                            backend_factory=env.factory, clock=env.store.clock.now, **kwargs)


def run_files(home: Path):
    return sorted((home / "migrations").rglob("*.json")) if (home / "migrations").exists() else []
