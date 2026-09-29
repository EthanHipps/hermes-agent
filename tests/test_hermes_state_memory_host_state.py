"""X-5 (ledger L452): deleting a Hermes session deletes its memory host-state record (ruling R41-13; C6b-11)."""

import time

import pytest

from agent.memory_service.host_state import HostStateRecord, load_host_state, save_host_state
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(db_path=tmp_path / "state.db")
    yield database, tmp_path
    database.close()


def _session(db, home, sid, **kw):
    db.create_session(sid, "cli", **kw)
    save_host_state(HostStateRecord(sid, "stateless", None), hermes_home=home)


@pytest.mark.parametrize("delete", [
    lambda d, sid: d.delete_session(sid),
    lambda d, sid: d.delete_sessions([sid]),
    lambda d, sid: (d.end_session(sid, "user"), d.delete_empty_sessions()),
    lambda d, sid: (d.end_session(sid, "user"), d.prune_sessions(older_than_days=None, last_active_before=time.time() + 3600)),
])
def test_every_delete_path_removes_the_record(db, delete):
    database, home = db
    _session(database, home, "s-1")
    delete(database, "s-1")
    assert load_host_state("s-1", hermes_home=home) is None


def test_a_branch_childs_record_survives_its_parents_delete(db):
    database, home = db
    _session(database, home, "parent")
    _session(database, home, "child", parent_session_id="parent")
    database.delete_session("parent")
    assert load_host_state("child", hermes_home=home) is not None
