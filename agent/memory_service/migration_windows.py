"""Windows migration files: original-handle authority and protected creation.

Loaded only by the Windows arm of migration_manifest. This is a file flush and
atomic progression contract, not a power-loss namespace durability guarantee.
The Windows owner and SYSTEM are trusted. Sharing excludes replacement while
pinned, not ACL/attribute/stream mutation; checks never claim otherwise.
"""
from __future__ import annotations

import ctypes as c
from ctypes import wintypes as t
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path, PureWindowsPath
import re
import struct
import time
import uuid

import pywintypes
import win32api
import win32con
import win32file
import win32security as security

_READ_CONTROL, _SYNCHRONIZE, _DELETE = 0x20000, 0x100000, 0x10000
_READ_ATTRIBUTES, _READ_DATA, _WRITE_DATA = 0x80, 1, 2
_DIRECTORY, _REPARSE = 0x10, 0x400
_BUFFER = 65536
_FULL_ACCESS = 0x1F01FF
_SYSTEM = "S-1-5-18"
_PROVIDER = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")
_EPOCH = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_RUN = re.compile(r"[0-9a-f]{32}\.json\Z")


class WindowsMigrationError(OSError):
    """Fixed public reason; native diagnostics remain exception causes only."""
    code = "migration_state_unavailable"

    def __init__(self):
        super().__init__("Windows migration state could not be verified")


class WindowsPublicationUncertain(WindowsMigrationError):
    code = "migration_state_uncertain"


class MigrationLockBusy(WindowsMigrationError):
    pass


class _Unicode(c.Structure):
    _fields_ = [("Length", t.USHORT), ("MaximumLength", t.USHORT), ("Buffer", c.c_void_p)]


class _ObjectAttributes(c.Structure):
    _fields_ = [("Length", t.ULONG), ("RootDirectory", t.HANDLE),
                ("ObjectName", c.POINTER(_Unicode)), ("Attributes", t.ULONG),
                ("SecurityDescriptor", c.c_void_p), ("SecurityQualityOfService", c.c_void_p)]


class _Status(c.Union):
    _fields_ = [("Status", c.c_long), ("Pointer", c.c_void_p)]


class _IoStatus(c.Structure):
    _fields_ = [("status", _Status), ("Information", c.c_size_t)]


class _Rename(c.Structure):
    _fields_ = [("Flags", t.DWORD), ("RootDirectory", t.HANDLE),
                ("FileNameLength", t.DWORD), ("FileName", t.WCHAR * 1)]


_kernel = c.WinDLL("kernel32", use_last_error=True)
_nt = c.WinDLL("ntdll")
_advapi = c.WinDLL("advapi32", use_last_error=True)


def _bind(dll, name, args, result):
    fn = getattr(dll, name)
    fn.argtypes, fn.restype = args, result
    return fn


_create = _bind(_nt, "NtCreateFile", [c.POINTER(t.HANDLE), t.DWORD, c.POINTER(_ObjectAttributes),
                c.POINTER(_IoStatus), c.c_void_p, t.ULONG, t.ULONG, t.ULONG, t.ULONG,
                c.c_void_p, t.ULONG], c.c_long)
_rename_info = _bind(_nt, "NtSetInformationFile", [t.HANDLE, c.POINTER(_IoStatus), c.c_void_p, t.ULONG, c.c_int], c.c_int32)
_dos_error = _bind(_nt, "RtlNtStatusToDosError", [c.c_long], t.ULONG)
_get_info = _bind(_kernel, "GetFileInformationByHandleEx", [t.HANDLE, c.c_int, c.c_void_p, t.DWORD], t.BOOL)
_set_info = _bind(_kernel, "SetFileInformationByHandle", [t.HANDLE, c.c_int, c.c_void_p, t.DWORD], t.BOOL)
_close = _bind(_kernel, "CloseHandle", [t.HANDLE], t.BOOL)
_sddl = _bind(_advapi, "ConvertStringSecurityDescriptorToSecurityDescriptorW",
              [t.LPCWSTR, t.DWORD, c.POINTER(c.c_void_p), c.POINTER(t.ULONG)], t.BOOL)
_free = _bind(_kernel, "LocalFree", [c.c_void_p], c.c_void_p)
_volume = _bind(_kernel, "GetVolumeInformationByHandleW",
                [t.HANDLE, t.LPWSTR, t.DWORD, c.POINTER(t.DWORD), c.POINTER(t.DWORD),
                 c.POINTER(t.DWORD), t.LPWSTR, t.DWORD], t.BOOL)
_device = _bind(_kernel, "QueryDosDeviceW", [t.LPCWSTR, t.LPWSTR, t.DWORD], t.DWORD)


def _winerror():
    raise c.WinError(c.get_last_error())


def _component(name):
    if (not name or name in (".", "..") or name[-1:] in (".", " ")
            or any(ch in name for ch in '\\/:\x00') or len(name.encode("utf-16-le")) > 510):
        raise WindowsMigrationError()
    return name


def _info(handle, kind, size):
    buf = c.create_string_buffer(size)
    if not _get_info(handle, kind, buf, size):
        _winerror()
    return buf.raw


def _owner():
    token = security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    try:
        sid = security.ConvertSidToStringSid(security.GetTokenInformation(token, security.TokenUser)[0])
    finally:
        token.Close()
    try:
        thread = security.OpenThreadToken(win32api.GetCurrentThread(), win32con.TOKEN_QUERY, True)
    except pywintypes.error as exc:
        if exc.winerror != 1008:  # ERROR_NO_TOKEN, no impersonation
            raise
    else:
        try:
            if security.ConvertSidToStringSid(security.GetTokenInformation(thread, security.TokenUser)[0]) != sid:
                raise WindowsMigrationError()
        finally:
            thread.Close()
    if sid == _SYSTEM:
        raise WindowsMigrationError()
    return sid


@contextmanager
def _descriptor(owner):
    pointer, size = c.c_void_p(), t.ULONG()
    if not _sddl(f"O:{owner}D:P(A;;FA;;;{owner})(A;;FA;;;SY)", 1, c.byref(pointer), c.byref(size)):
        _winerror()
    try:
        if size.value > 4096:
            raise WindowsMigrationError()
        yield pointer
    finally:
        _free(pointer)


def _security(handle, owner=None, exact=False):
    # GetSecurityInfo allocates before our acceptance limit is checked.
    sd = security.GetSecurityInfo(handle, security.SE_FILE_OBJECT,
                                   security.OWNER_SECURITY_INFORMATION | security.DACL_SECURITY_INFORMATION)
    if sd.GetLength() > 4096:
        raise WindowsMigrationError()
    actual = sd.GetSecurityDescriptorOwner()
    if actual is None or actual.GetLength() > 68:
        raise WindowsMigrationError()
    if owner is not None and security.ConvertSidToStringSid(actual) != owner:
        raise WindowsMigrationError()
    if exact:
        acl = sd.GetSecurityDescriptorDacl()
        if not sd.GetSecurityDescriptorControl()[0] & 0x1000 or acl is None or acl.GetAceCount() != 2:
            raise WindowsMigrationError()
        found = set()
        for index in range(2):
            ace = acl.GetAce(index)
            if len(ace) != 3 or ace[0] != (0, 0) or ace[1] != _FULL_ACCESS:
                raise WindowsMigrationError()
            found.add(security.ConvertSidToStringSid(ace[2]))
        if found != {owner, _SYSTEM}:
            raise WindowsMigrationError()


def _streams(handle, directory):
    try:
        raw = _info(handle, 7, _BUFFER)  # FileStreamInfo; never follow a path
    except OSError as exc:
        if directory and getattr(exc, "winerror", None) == 38:  # documented no-stream result
            return
        raise
    offset, names = 0, []
    while True:
        if offset + 24 > len(raw):
            raise WindowsMigrationError()
        next_offset, length = struct.unpack_from("<II", raw, offset)
        if length % 2 or offset + 24 + length > len(raw):
            raise WindowsMigrationError()
        if length:
            names.append(raw[offset + 24:offset + 24 + length].decode("utf-16-le", errors="strict"))
        if not next_offset:
            break
        if next_offset < 24 + length or offset + next_offset >= len(raw):
            raise WindowsMigrationError()
        offset += next_offset
    if names not in (["::$DATA"], []) or (not directory and names != ["::$DATA"]):
        raise WindowsMigrationError()


class _Handle:
    def __init__(self, value, *, directory, owner=None, exact=False):
        self.value, self.directory, self.owner, self.exact = int(value), directory, owner, exact
        try:
            self.identity = self.check()
        except BaseException:
            self.close()
            raise

    def check(self):
        attributes, tag = struct.unpack("<II", _info(self.value, 9, 8))
        if bool(attributes & _DIRECTORY) != self.directory or attributes & _REPARSE or tag:
            raise WindowsMigrationError()
        standard = _info(self.value, 1, 24)
        if standard[20] or (not self.directory and struct.unpack_from("<I", standard, 16)[0] != 1):
            raise WindowsMigrationError()
        identity = _info(self.value, 18, 24)  # volume serial64 + FILE_ID_128
        if hasattr(self, "identity") and identity != self.identity:
            raise WindowsMigrationError()
        _security(self.value, self.owner, self.exact)
        _streams(self.value, self.directory)
        return identity

    def close(self):
        if self.value:
            _close(self.value)
            self.value = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _relative(parent, name, *, directory, owner=None, exact=False, create=False, write=False, delete=False):
    _component(name)
    parent.check()
    text = c.create_unicode_buffer(name)
    name_bytes = len(name.encode("utf-16-le"))
    string = _Unicode(name_bytes, name_bytes + 2, c.cast(text, c.c_void_p))
    access = _READ_CONTROL | _READ_ATTRIBUTES | _SYNCHRONIZE | _READ_DATA
    if directory:
        access |= 0x20  # FILE_TRAVERSE
    if write:
        access |= _WRITE_DATA
    if delete:
        access |= _DELETE
    attrs = _ObjectAttributes(c.sizeof(_ObjectAttributes), parent.value, c.pointer(string), 0x40 | 0x1000, None, None)
    result, status = t.HANDLE(), _IoStatus()
    options = 0x00200000 | 0x20 | (1 if directory else 0x40)
    with ExitStack() as stack:
        if create:
            attrs.SecurityDescriptor = stack.enter_context(_descriptor(owner))
        rc = _create(c.byref(result), access, c.byref(attrs), c.byref(status), None,
                     0x80, 3 if directory or not delete else 7, 2 if create else 1, options, None, 0)
    if rc != 0:
        if result.value:
            _close(result.value)
        raise c.WinError(_dos_error(rc))
    if status.status.Status != 0 or status.Information != (2 if create else 1) or not result.value:
        if result.value:
            _close(result.value)
        raise WindowsMigrationError()
    return _Handle(result.value, directory=directory, owner=owner, exact=exact or create)


def _directory(parent, name, owner, *, create=False, exact=False):
    try:
        return _relative(parent, name, directory=True, owner=owner, exact=exact)
    except OSError as exc:
        if not create or getattr(exc, "winerror", None) not in (2, 3):
            raise
    try:
        return _relative(parent, name, directory=True, owner=owner, exact=True, create=True)
    except OSError as exc:
        if getattr(exc, "winerror", None) not in (80, 183):
            raise
        return _relative(parent, name, directory=True, owner=owner, exact=exact)


def _read(handle):
    handle.check()
    length = struct.unpack_from("<q", _info(handle.value, 1, 24), 8)[0]
    if length < 0:
        raise WindowsMigrationError()
    win32file.SetFilePointer(handle.value, 0, 0)
    chunks, remaining = [], length
    while remaining:
        _, data = win32file.ReadFile(handle.value, min(remaining, _BUFFER))
        if not data:
            raise WindowsMigrationError()
        chunks.append(data)
        remaining -= len(data)
    handle.check()
    return b"".join(chunks)


def _write(handle, data):
    handle.check()
    for offset in range(0, len(data), _BUFFER):
        piece = data[offset:offset + _BUFFER]
        _, count = win32file.WriteFile(handle.value, piece)
        if count != len(piece):
            raise WindowsMigrationError()
    win32file.FlushFileBuffers(handle.value)


def _entries(handle):
    handle.check()
    restart, result = True, []
    while True:
        buf = c.create_string_buffer(_BUFFER)
        if not _get_info(handle.value, 15 if restart else 14, buf, len(buf)):
            if c.get_last_error() == 18:  # ERROR_NO_MORE_FILES
                break
            _winerror()
        restart, offset = False, 0
        raw = buf.raw
        while True:
            if offset + 68 > len(raw):
                raise WindowsMigrationError()
            next_offset = struct.unpack_from("<I", raw, offset)[0]
            attributes, length = struct.unpack_from("<II", raw, offset + 56)
            if length % 2 or offset + 68 + length > len(raw):
                raise WindowsMigrationError()
            name = raw[offset + 68:offset + 68 + length].decode("utf-16-le", errors="strict")
            if name not in (".", ".."):
                result.append((_component(name), bool(attributes & _DIRECTORY)))
            if not next_offset:
                break
            if next_offset < 68 + length or offset + next_offset >= len(raw):
                raise WindowsMigrationError()
            offset += next_offset
    handle.check()
    return sorted(result)


def _rename(source, parent, name, replace):
    encoded = _component(name).encode("utf-16-le")
    length = c.sizeof(_Rename) + len(encoded)
    buf = c.create_string_buffer(length)
    info = _Rename.from_buffer(buf)
    info.Flags, info.RootDirectory, info.FileNameLength = (3 if replace else 0), parent.value, len(encoded)
    c.memmove(c.addressof(buf) + _Rename.FileName.offset, encoded, len(encoded))
    status = _IoStatus()
    rc = _rename_info(source.value, c.byref(status), buf, length, 65)
    if rc != 0:
        raise c.WinError(_dos_error(rc))
    if status.status.Status != 0:
        raise WindowsMigrationError()


def _dispose(handle):
    handle.check()
    yes = t.BOOLEAN(1)
    if not _set_info(handle.value, 4, c.byref(yes), c.sizeof(yes)):
        _winerror()


class WindowsMigrationScope:
    """One home, original parent handles and byte-zero lock; no ambient authority."""

    def __init__(self, home, timeout_seconds=2.0):
        self.home = Path(home)
        self.timeout_seconds = timeout_seconds
        self._stack, self._anchors, self._locked, self._live = ExitStack(), [], False, False

    def __enter__(self):
        try:
            self.owner = _owner()
            path = PureWindowsPath(self.home)
            if not re.fullmatch(r"[A-Za-z]:", path.drive) or path.root != "\\" or len(path.parts) > 128:
                raise WindowsMigrationError()
            device = c.create_unicode_buffer(32768)
            if not _device(path.drive, device, len(device)) or not re.fullmatch(r"\\Device\\HarddiskVolume[0-9]+", device.value):
                raise WindowsMigrationError()
            raw = win32file.CreateFile(str(path.anchor), _READ_CONTROL | _READ_ATTRIBUTES | _SYNCHRONIZE | 0x21,
                                      3, None, 3, 0x02000000 | 0x00200000, None)
            try:
                root = _Handle(raw.Detach(), directory=True)
            except BaseException:
                raw.Close()
                raise
            self._anchors.append(self._stack.enter_context(root))
            serial, maximum, flags, fs = t.DWORD(), t.DWORD(), t.DWORD(), c.create_unicode_buffer(32)
            if not _volume(root.value, None, 0, c.byref(serial), c.byref(maximum), c.byref(flags), fs, len(fs)):
                _winerror()
            if fs.value not in ("NTFS", "ReFS") or not flags.value & 8:
                raise WindowsMigrationError()
            for index, part in enumerate(path.parts[1:]):
                node = _relative(self._anchors[-1], part, directory=True,
                                 owner=self.owner if index == len(path.parts) - 2 else None)
                self._anchors.append(self._stack.enter_context(node))
            self.root = self._stack.enter_context(_directory(self._anchors[-1], "migrations", self.owner, create=True))
            try:
                lock = _relative(self.root, ".lock", directory=False, owner=self.owner, write=True)
            except OSError as exc:
                if getattr(exc, "winerror", None) not in (2, 3):
                    raise
                try:
                    lock = _relative(self.root, ".lock", directory=False, owner=self.owner, create=True, write=True)
                except OSError as collision:
                    if getattr(collision, "winerror", None) not in (80, 183):
                        raise
                    lock = _relative(self.root, ".lock", directory=False, owner=self.owner, write=True)
            self.lock = self._stack.enter_context(lock)
            self._overlap = pywintypes.OVERLAPPED()
            deadline = time.monotonic() + max(0.0, self.timeout_seconds)
            while True:
                try:
                    win32file.LockFileEx(lock.value, 3, 1, 0, self._overlap)
                    self._locked = True
                    break
                except pywintypes.error as exc:
                    if exc.winerror != 33:
                        raise
                    if time.monotonic() >= deadline:
                        raise MigrationLockBusy() from None
                    time.sleep(0.05)
            if struct.unpack_from("<q", _info(lock.value, 1, 24), 8)[0] == 0:
                _write(lock, b" ")
            self._live = True
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        self._live = False
        try:
            if self._locked:
                self._locked = False
                win32file.UnlockFileEx(self.lock.value, 1, 0, self._overlap)
        finally:
            self._stack.close()

    def _check(self):
        if not self._live or _owner() != self.owner:
            raise WindowsMigrationError()
        for node in self._anchors:
            node.check()
        self.root.check()
        self.lock.check()

    def _parts(self, path):
        parts = Path(path).relative_to(self.home / "migrations").parts
        if len(parts) != 3 or not _PROVIDER.fullmatch(parts[0]) or not _EPOCH.fullmatch(parts[1]) or not _RUN.fullmatch(parts[2]):
            raise WindowsMigrationError()
        return parts

    @contextmanager
    def _epoch(self, path, *, create=False):
        self._check()
        provider, epoch, name = self._parts(path)
        with _directory(self.root, provider, self.owner, create=create) as parent:
            with _directory(parent, epoch, self.owner, create=create, exact=True) as boundary:
                yield boundary, name

    def verify(self, path):
        with self._epoch(path) as (epoch, name):
            with _relative(epoch, name, directory=False, owner=self.owner, exact=True):
                pass

    def read(self, path):
        with self._epoch(path) as (epoch, name):
            with _relative(epoch, name, directory=False, owner=self.owner, exact=True) as target:
                return _read(target)

    def paths(self, provider=None):
        self._check()
        paths = []
        for prov, is_dir in _entries(self.root):
            if not is_dir or (provider is not None and prov != provider):
                continue
            with _relative(self.root, prov, directory=True) as parent:
                for epoch, epoch_dir in _entries(parent):
                    if not epoch_dir:
                        continue
                    with _relative(parent, epoch, directory=True) as boundary:
                        for name, file_dir in _entries(boundary):
                            if name.lower().endswith(".json") and not file_dir:
                                paths.append(self.home / "migrations" / prov / epoch / name)
        return sorted(paths)

    def has_active(self):
        # Same C9 top-level terminal criterion, inspected through original handles.
        self._check()
        for prov, is_dir in _entries(self.root):
            if not is_dir:
                continue
            with _relative(self.root, prov, directory=True) as parent:
                for epoch, epoch_dir in _entries(parent):
                    if not epoch_dir:
                        continue
                    with _relative(parent, epoch, directory=True) as boundary:
                        for name, _ in _entries(boundary):
                            if name.lower().endswith(".json"):
                                try:
                                    with _relative(boundary, name, directory=False) as target:
                                        value = json.loads(_read(target).decode("utf-8"))
                                    if not isinstance(value, dict) or value.get("state") not in ("completed", "rolled_back"):
                                        return True
                                except (OSError, pywintypes.error, ValueError, TypeError, RecursionError):
                                    return True
        return False

    def write(self, path, data):
        with self._epoch(path, create=True) as (epoch, name), ExitStack() as stack:
            try:
                old = stack.enter_context(_relative(epoch, name, directory=False, owner=self.owner, exact=True, delete=True))
            except OSError as exc:
                if getattr(exc, "winerror", None) not in (2, 3):
                    raise
                old = None
            temporary_name = f".tmp_{uuid.uuid4().hex}.tmp"
            temporary = stack.enter_context(_relative(epoch, temporary_name, directory=False, owner=self.owner,
                                                       create=True, write=True, delete=True))
            attempted = False
            try:
                if struct.unpack_from("<q", _info(temporary.value, 1, 24), 8)[0] != 0:
                    raise WindowsMigrationError()
                _write(temporary, data)
                self._check()
                epoch.check()
                temporary.check()
                if old:
                    old.check()
                attempted = True
                _rename(temporary, epoch, name, old is not None)
                temporary.check()
                with _relative(epoch, name, directory=False, owner=self.owner, exact=True, delete=True) as final:
                    if final.identity != temporary.identity:
                        raise WindowsMigrationError()
                win32file.FlushFileBuffers(temporary.value)
            except (OSError, pywintypes.error, ValueError, TypeError) as exc:
                if attempted:
                    raise WindowsPublicationUncertain() from exc
                raise
            finally:
                if not attempted:
                    # Only our original still-temporary handle is disposable.
                    try:
                        epoch.check()
                        _dispose(temporary)
                    except (OSError, pywintypes.error):
                        pass

    def remove(self, path):
        with self._epoch(path) as (epoch, name):
            with _relative(epoch, name, directory=False, owner=self.owner, exact=True, delete=True) as target:
                self._check()
                epoch.check()
                _dispose(target)
