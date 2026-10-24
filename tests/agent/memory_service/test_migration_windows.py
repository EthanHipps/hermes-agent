"""Native migration security: actual producer output, never preprotected epochs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

class _AfterFirstWrite(BaseException):
    pass


def _sid():
    import win32api
    import win32con
    import win32security
    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    try:
        return win32security.ConvertSidToStringSid(win32security.GetTokenInformation(token, win32security.TokenUser)[0])
    finally:
        token.Close()


def _native(path):
    import win32file
    handle = win32file.CreateFile(str(path), 0x20080, 7, None, 3, 0x02200000, None)
    try:
        return _native_handle(int(handle))
    finally:
        handle.Close()


def _native_handle(handle):
    # Independent Win32 auditor: does not call product validation helpers.
    import ctypes
    from ctypes import wintypes
    import struct
    import win32security
    query = ctypes.WinDLL("kernel32", use_last_error=True).GetFileInformationByHandleEx
    query.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    query.restype = wintypes.BOOL

    def info(kind, size):
        buf = ctypes.create_string_buffer(size)
        if not query(handle, kind, buf, size):
            raise ctypes.WinError(ctypes.get_last_error())
        return buf.raw

    sd = win32security.GetSecurityInfo(handle, win32security.SE_FILE_OBJECT, 5)
    acl = sd.GetSecurityDescriptorDacl()
    attrs, tag = struct.unpack("<II", info(9, 8))
    standard, basic, identity = info(1, 24), info(0, 40), info(18, 24)
    streams = []
    try:
        raw, offset = info(7, 65536), 0
    except OSError as exc:
        assert attrs & 0x10 and exc.winerror == 38
    else:
        while True:
            nxt, size = struct.unpack_from("<II", raw, offset)
            streams.append(raw[offset + 24:offset + 24 + size].decode("utf-16-le"))
            if not nxt:
                break
            offset += nxt
    return {"owner": win32security.ConvertSidToStringSid(sd.GetSecurityDescriptorOwner()),
            "protected": bool(sd.GetSecurityDescriptorControl()[0] & 0x1000),
            "aces": [] if acl is None else [[list(a[0]), a[1], win32security.ConvertSidToStringSid(a[2])]
                for a in (acl.GetAce(i) for i in range(acl.GetAceCount()))],
            "sddl": win32security.ConvertSecurityDescriptorToStringSecurityDescriptor(sd, 1, 5),
            "identity128": identity.hex(), "links": struct.unpack_from("<I", standard, 16)[0],
            "size": struct.unpack_from("<q", standard, 8)[0], "delete_pending": bool(standard[20]),
            "mtime100ns": struct.unpack_from("<q", basic, 16)[0], "streams": streams,
            "attributes": attrs, "tag": tag}


def _contender(pipe, home, owner):
    import msvcrt
    assert _sid() == owner
    with (Path(home) / "migrations" / ".lock").open("r+b", buffering=0) as handle:
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            pipe.send("busy")
        else:
            raise AssertionError("original byte-zero lock was not exclusive")
        assert pipe.recv() == "released"
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        pipe.send("acquired")
    pipe.close()


def _assert_e(facts, owner):
    assert facts["owner"] == owner
    assert facts["protected"], "actual producer epoch must have a protected DACL at birth"
    assert sorted(facts["aces"], key=lambda a: a[2]) == sorted(
        [[[0, 0], 0x001F01FF, owner], [[0, 0], 0x001F01FF, "S-1-5-18"]], key=lambda a: a[2])


@pytest.mark.windows_only
def test_actual_producer_creates_protected_migration_state(monkeypatch, tmp_path_factory):
    owner = _sid()
    from agent.memory_service import migration, migration_manifest as mm
    from tests.agent.memory_service.migration_support import Answers, make_env, start, write_native
    checkout = Path(__file__).resolve().parents[3]
    for name, module in tuple(sys.modules.items()):
        if name.startswith("agent.memory_service") and getattr(module, "__file__", None):
            assert Path(module.__file__).resolve().is_relative_to(checkout), (name, module.__file__)
    assert Path(migration.__file__).resolve().is_relative_to(checkout)
    fixture = tmp_path_factory.mktemp("b")
    assert fixture.resolve().is_relative_to(tmp_path_factory.getbasetemp().resolve()), (fixture, tmp_path_factory.getbasetemp())
    monkeypatch.setenv("HERMES_HOME", str(fixture / "home"))
    env = make_env(fixture)
    write_native(env.home, memory=["Use spaces for indentation."])
    source = env.home / "memories" / "MEMORY.md"
    before = (source.read_bytes(), source.stat().st_mtime_ns)
    writes = []
    real_write = mm.write_document
    from agent.memory_service import migration_windows as native
    births, prebytes, publications, handles = [], [], [], []
    real_init, real_relative = native._Handle.__init__, native._relative
    real_native_write, real_rename = native._write, native._rename
    temps = set()

    def tracked_init(handle, *args, **kwargs):
        real_init(handle, *args, **kwargs)
        handles.append(handle)

    def birth(parent, name, **kwargs):
        handle = real_relative(parent, name, **kwargs)
        if kwargs.get("create"):
            facts = _native_handle(handle.value)
            _assert_e(facts, owner)
            assert facts["tag"] == 0 and not facts["attributes"] & 0x400
            if not kwargs["directory"]:
                assert facts["size"] == 0 and facts["links"] == 1 and facts["streams"] == ["::$DATA"]
            births.append({"name": name, **facts})
            if name.startswith(".tmp_"):
                temps.add(handle)
        return handle

    def prebyte(handle, data):
        if handle in temps:
            facts = _native_handle(handle.value)
            _assert_e(facts, owner)
            assert facts["size"] == 0
            prebytes.append(facts)
        return real_native_write(handle, data)

    def publish(source_handle, parent, name, replace):
        before_publish = _native_handle(source_handle.value)
        real_rename(source_handle, parent, name, replace)
        after_publish = _native_handle(source_handle.value)
        assert before_publish["identity128"] == after_publish["identity128"]
        _assert_e(after_publish, owner)
        publications.append({"replace": replace, "before": before_publish, "after": after_publish})

    monkeypatch.setattr(native._Handle, "__init__", tracked_init)
    monkeypatch.setattr(native, "_relative", birth)
    monkeypatch.setattr(native, "_write", prebyte)
    monkeypatch.setattr(native, "_rename", publish)

    def first_write(path, document, **kwargs):
        real_write(path, document, **kwargs)
        writes.append(Path(path))
        raise _AfterFirstWrite()

    monkeypatch.setattr(mm, "write_document", first_write)
    with pytest.raises(_AfterFirstWrite):
        start(env, Answers(True), items=("MEMORY.md",))
    assert len(env.calls("stage_curated")) == 1 and len(writes) == 1
    assert env.calls("inspect_staged") == [] and env.calls("commit_curated") == []
    [manifest] = writes
    facts = {"token_user": owner, "fixture": str(fixture), "pytest_basetemp": str(tmp_path_factory.getbasetemp()), "module": migration.__file__,
             "epoch": _native(manifest.parent), "manifest": _native(manifest),
             "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
             "source_sha256": hashlib.sha256(before[0]).hexdigest(), "source_mtime_ns": before[1],
             "state": json.loads(manifest.read_bytes())["state"], "writes": len(writes), "stage_calls": 1}
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before
    (fixture / "native-proof.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print("EDD125_ACTUAL_FIRST_PUBLICATION " + json.dumps(facts, sort_keys=True), flush=True)
    _assert_e(facts["epoch"], owner)
    _assert_e(facts["manifest"], owner)
    assert births and len(prebytes) == len(publications) == 1
    assert not publications[0]["replace"]
    assert any(b["name"] == "ep-1" for b in births)
    assert all(handle.value == 0 for handle in handles)  # BaseException unwound first-acquisition scope
    import multiprocessing
    parent, child = multiprocessing.get_context("spawn").Pipe()
    process = multiprocessing.get_context("spawn").Process(target=_contender, args=(child, env.home, owner))
    try:
        with mm.migration_lock(env.home):
            process.start()
            child.close()
            assert parent.poll(20) and parent.recv() == "busy"
        parent.send("released")
        assert parent.poll(20) and parent.recv() == "acquired"
        process.join(20)
        assert process.exitcode == 0
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
            process.join(20)
    monkeypatch.setattr(mm, "write_document", real_write)
    from tests.agent.memory_service.migration_support import resume
    report = resume(env, Answers(True))
    assert report.status is migration.MigrationStatus.COMPLETED
    assert len(env.calls("commit_curated")) == 1
    assert json.loads(manifest.read_bytes())["state"] == "completed"
    _assert_e(_native(manifest), owner)
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before
    assert any(p["replace"] for p in publications)
    denied_fixture = fixture / "denied"
    denied_fixture.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(denied_fixture / "home"))
    denied = make_env(denied_fixture)
    write_native(denied.home, memory=["Use spaces for indentation."])
    denied_source = denied.home / "memories" / "MEMORY.md"
    denied_before = (denied_source.read_bytes(), denied_source.stat().st_mtime_ns)
    report = start(denied, Answers(False), items=("MEMORY.md",))
    assert report.status is migration.MigrationStatus.DENIED and denied.calls("commit_curated") == []
    [receipt] = (denied.home / "migrations").rglob("*.json")
    assert json.loads(receipt.read_bytes())["state"] == "rolled_back"
    _assert_e(_native(receipt), owner)
    assert (denied_source.read_bytes(), denied_source.stat().st_mtime_ns) == denied_before
    assert all(handle.value == 0 for handle in handles)
    facts.update(births=births, prebytes=prebytes, publications=publications, completed=_native(manifest),
                 denied=_native(receipt), children=1, homes=2, handles_closed=len(handles))
    (fixture / "native-proof.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")


class _RemoteBackend:
    """Test-only pipe to a parent-owned in-memory provider across real process death."""
    def __init__(self, pipe, config):
        self.pipe = pipe
        pipe.send(("factory", config))
        self.number = pipe.recv()

    def __getattr__(self, operation):
        def invoke(*args, **kwargs):
            self.pipe.send(("call", self.number, operation, args, kwargs))
            ok, value = self.pipe.recv()
            if not ok:
                raise value
            return value
        return invoke


def _migration_child(pipe, home, raw, section, now, owner, crash):
    from agent.memory_service import migration, migration_manifest as mm
    from tests.agent.memory_service.migration_support import Answers, plan, write_memory_section
    assert _sid() == owner
    assert Path(migration.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[3])
    os.environ["HERMES_HOME"] = str(home)
    if crash:
        real_write = mm.write_document

        def die_after_approval(path, document, **kwargs):
            real_write(path, document, **kwargs)
            if document["state"] == "active" and document["batches"][0]["status"] == "approved":
                pipe.send(("witness", str(path), document, _sid()))
                assert pipe.recv() == "recorded"
                os._exit(73)
        mm.write_document = die_after_approval
    common = dict(home=Path(home), prompt=Answers(True),
                  switch_to_authoritative=lambda: write_memory_section(Path(home), {**section, "provider_mode": "authoritative"}),
                  backend_factory=lambda config: _RemoteBackend(pipe, config), clock=lambda: now)
    if crash:
        report = migration.start_migration(raw, plan(items=("MEMORY.md",)), **common)
    else:
        report = migration.resume_migration(raw, **common)
    pipe.send(("report", report))
    pipe.close()


def _run_migration_child(env, owner, crash):
    import multiprocessing
    import time
    parent, child = multiprocessing.get_context("spawn").Pipe()
    process = multiprocessing.get_context("spawn").Process(target=_migration_child,
        args=(child, env.home, env.raw(), env.section, env.store.clock.now(), owner, crash))
    backends, witness, report = [], None, None
    process.start()
    child.close()
    deadline = time.monotonic() + 30
    try:
        while time.monotonic() < deadline:
            try:
                ready = parent.poll(0.05)
            except (BrokenPipeError, EOFError):
                break
            if ready:
                try:
                    message = parent.recv()
                except EOFError:
                    break
                if message[0] == "factory":
                    backends.append(env.factory(message[1]))
                    parent.send(len(backends) - 1)
                elif message[0] == "call":
                    _, number, operation, args, kwargs = message
                    try:
                        result = getattr(backends[number], operation)(*args, **kwargs)
                    except Exception as exc:
                        parent.send((False, exc))
                    else:
                        parent.send((True, result))
                elif message[0] == "witness":
                    witness = message[1:]
                    parent.send("recorded")
                elif message[0] == "report":
                    report = message[1]
                else:
                    raise AssertionError(message[0])
            elif not process.is_alive():
                break
        process.join(5)
        assert process.exitcode == (73 if crash else 0)
        return witness, report
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
            process.join(5)


@pytest.mark.windows_only
def test_actual_crash_replay_and_verified_failure_recovery(monkeypatch, tmp_path_factory):
    from agent.memory_service import migration, migration_manifest as mm, migration_windows as native
    from agent.memory_service.migration_recovery import rollback_migrations, reconcile_migration
    from tests.agent.memory_service.migration_support import Answers, make_env, start, resume, write_native
    import win32file
    import win32security
    owner = _sid()
    fixture = tmp_path_factory.mktemp("r")
    assert fixture.resolve().is_relative_to(tmp_path_factory.getbasetemp().resolve()), (fixture, tmp_path_factory.getbasetemp())
    proof = {"token_user": owner, "fixture": str(fixture), "pytest_basetemp": str(tmp_path_factory.getbasetemp()), "children": 2, "homes": 5, "cases": {}}

    def environment(name):
        base = fixture / name
        base.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(base / "h"))
        env = make_env(base)
        write_native(env.home, memory=["Use spaces for indentation."])
        return env

    def snapshot(path):
        return {"bytes": path.read_bytes().hex(), "native": _native(path), "mtime_ns": path.stat().st_mtime_ns}

    real_write = mm.write_document

    def stopped(env):
        def interrupt(path, document, **kwargs):
            real_write(path, document, **kwargs)
            raise _AfterFirstWrite()
        with monkeypatch.context() as patch:
            patch.setattr(mm, "write_document", interrupt)
            with pytest.raises(_AfterFirstWrite):
                start(env, Answers(True), items=("MEMORY.md",))
        [path] = (env.home / "migrations").rglob("*.json")
        return path

    env = environment("c")
    source = env.home / "memories" / "MEMORY.md"
    before_source = (source.read_bytes(), source.stat().st_mtime_ns)
    witness, report = _run_migration_child(env, owner, True)
    assert report is None and witness is not None and witness[2] == owner
    path, approved = Path(witness[0]), witness[1]
    assert env.calls("commit_curated") == []
    assert approved == json.loads(path.read_bytes())
    _assert_e(_native(path), owner)
    with mm.migration_lock(env.home, timeout_seconds=0):
        pass  # OS released the dead child's actual byte-zero lock and ancestors
    _, report = _run_migration_child(env, owner, False)
    assert report.status is migration.MigrationStatus.COMPLETED
    [request] = env.calls("commit_curated")
    batch = approved["batches"][0]
    actual = request.to_wire()
    assert actual["request_id"] == batch["request_id"]
    assert actual["stage_handle_b64url"] == batch["stage"]["stage_handle_b64url"]
    assert actual["approval_binding_sha256"] == batch["stage"]["approval_binding_sha256"]
    assert actual["authorization"] == batch["authorization"]
    assert len(env.calls("stage_curated")) == 1 and len(env.store.receipts) == 1
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before_source
    _assert_e(_native(path), owner)
    proof["cases"]["crash73_replay"] = {"path": str(path), "approved_sha256": hashlib.sha256(mm.encode(approved)).hexdigest(),
                                          "completed": snapshot(path), "commits": 1}

    env = environment("f")
    path = stopped(env)
    original = snapshot(path)
    # An exception before publication must dispose only its own unpublished temp.
    def abort_before_bytes(*args):
        raise _AfterFirstWrite()
    with monkeypatch.context() as patch:
        patch.setattr(native, "_write", abort_before_bytes)
        with pytest.raises(_AfterFirstWrite):
            resume(env, Answers(True))
    assert snapshot(path) == original
    assert list(path.parent.glob(".tmp_*.tmp")) == []
    # A genuine readonly replacement refusal reaches the one native rename call.
    attrs = win32file.GetFileAttributes(str(path))
    win32file.SetFileAttributes(str(path), attrs | 1)
    readonly = snapshot(path)
    report = resume(env, Answers(True))
    assert report.status is migration.MigrationStatus.STOPPED
    assert report.code == "migration_state_uncertain"
    assert "not confirmed" in report.message
    assert snapshot(path) == readonly and env.calls("commit_curated") == []
    retained = list(path.parent.glob(".tmp_*.tmp"))
    assert len(retained) == 1
    _assert_e(_native(retained[0]), owner)
    win32file.SetFileAttributes(str(path), attrs)  # test-owned attribute, never ACL repair
    # Interrupt after real rename to model uncertain postpublication verification.
    real_rename = native._rename
    def after_publication(*args):
        real_rename(*args)
        raise OSError("injected postpublication failure")
    with monkeypatch.context() as patch:
        patch.setattr(native, "_rename", after_publication)
        report = resume(env, Answers(True))
    assert report.code == "migration_state_uncertain" and "not confirmed" in report.message
    assert _native(path)["identity128"] != original["native"]["identity128"]
    assert path.exists() and env.calls("commit_curated") == []
    _assert_e(_native(path), owner)
    proof["cases"]["failure_preservation"] = {"old": original, "readonly": readonly, "uncertain": snapshot(path),
                                               "retained_temp": snapshot(retained[0])}
    report = rollback_migrations(env.raw(), home=env.home, backend_factory=env.factory, clock=env.store.clock.now)
    assert report.status is migration.MigrationStatus.ROLLED_BACK
    _assert_e(_native(path), owner)

    env = environment("r")
    path = stopped(env)
    report = reconcile_migration(env.raw(), path.stem, home=env.home, backend_factory=env.factory, clock=env.store.clock.now)
    assert report.status is migration.MigrationStatus.RECONCILED
    _assert_e(_native(path), owner)
    proof["cases"]["reconciled"] = snapshot(path)

    env = environment("d")
    path = stopped(env)
    path.write_bytes(b"unreadable")  # corrupt the actual protected producer object in place
    _assert_e(_native(path), owner)
    report = reconcile_migration(env.raw(), path.stem, home=env.home, discard=True, backend_factory=env.factory)
    assert report.status is migration.MigrationStatus.RECONCILED and report.code == "discarded"
    assert not path.exists() and env.calls("commit_curated") == []
    proof["cases"]["protected_discard"] = {"path": str(path), "absent": True}

    env = environment("l")
    path = stopped(env)
    # Deliberately invalidate this disposable epoch; production must never adopt/repair it.
    sd = win32security.GetFileSecurity(str(path.parent), win32security.DACL_SECURITY_INFORMATION)
    win32security.SetNamedSecurityInfo(str(path.parent), win32security.SE_FILE_OBJECT,
        win32security.DACL_SECURITY_INFORMATION | win32security.UNPROTECTED_DACL_SECURITY_INFORMATION,
        None, None, sd.GetSecurityDescriptorDacl(), None)
    legacy, parent_before, calls = snapshot(path), _native(path.parent), len(env.backends)
    assert not parent_before["protected"]
    results = [resume(env, Answers(True)),
               rollback_migrations(env.raw(), home=env.home, backend_factory=env.factory),
               reconcile_migration(env.raw(), path.stem, home=env.home, discard=True, backend_factory=env.factory)]
    assert all(r.status in (migration.MigrationStatus.REFUSED, migration.MigrationStatus.STOPPED) for r in results)
    assert len(env.backends) == calls and snapshot(path) == legacy and _native(path.parent) == parent_before
    assert env.calls("commit_curated") == []
    proof["cases"]["legacy_refused"] = {"file": legacy, "epoch": parent_before, "codes": [r.code for r in results]}
    (fixture / "native-recovery-proof.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")
    print("EDD125_NATIVE_RECOVERY_PROOF " + str(fixture / "native-recovery-proof.json"), flush=True)
