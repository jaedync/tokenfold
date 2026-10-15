"""Link Pi child sessions (pi-extras subagents) to the session that started them.

Pi has no parent pointer in its events. pi-extras writes each child as an
ordinary Pi session at subagents/<parent raw session id>/<stamp>_<name>_<hash>.jsonl
and the reporter (dotfleet pi/tokenfold-push.py) sends that path as
session_file, which ingest keeps in sync_cursors. Ingest namespaces every Pi id
with sha256(class, machine, session_file, raw session id) (ingest._pi_namespace),
so a stored child id `pi:<ns>:<raw>` can be matched to its file exactly: hash
each candidate file and compare. No guessing by time or name alone.

Legacy subagent-artifacts transcripts carry no parent and stay unlinked.
"""

import json
import logging
import sqlite3
import time
from collections import defaultdict

from . import db
from .ingest import _pi_namespace

logger = logging.getLogger(__name__)

# Found links never change (a child keeps its parent), so they are kept across
# requests. A miss (a parent never pushed, e.g. an SDK harness) costs a hash of
# every child file of the machine, so it is kept for MISS_TTL_S only: the
# parent's events may still arrive.
_LINKS: dict = {}
_LINKS_MAX = 50_000
_MISSES: dict = {}
MISS_TTL_S = 600


def _remember(key, parent: str) -> None:
    if len(_LINKS) >= _LINKS_MAX:
        _LINKS.clear()
    _LINKS[key] = parent

CHILD_ROOT = "subagents"
MAX_DEPTH = 8          # a deeper chain is a loop or garbage; stop there
_PI_ORG_PREFIX = "dotfleet_"


def parse_child_file(session_file: str) -> tuple[str, str] | None:
    """(parent raw id, child name) for a pi-extras child path, else None.

    Mirrors dotfleet _pi_extras_child: the name is everything between the
    first and the last underscore of the file stem.
    """
    parts = session_file.split("/")
    if len(parts) != 3 or parts[0] != CHILD_ROOT or not parts[2].endswith(".jsonl"):
        return None
    _stamp, _, rest = parts[2][:-len(".jsonl")].partition("_")
    name, _, _suffix = rest.rpartition("_")
    return (parts[1], name or "subagent") if parts[1] else None


def split_pi_id(value: str) -> tuple[str, str] | None:
    """`pi:<40 hex>:<raw>` -> (namespace, raw); None for other ids."""
    if not value or not value.startswith("pi:") or len(value) < 45 or value[43] != ":":
        return None
    return value[:43], value[44:]


def account_class(org_type: str | None) -> str | None:
    """Pi rows carry org_type dotfleet_<class> (ingest._pi_account). The
    synthetic email holds the class too, but org_type is the stable column."""
    if not org_type or not org_type.startswith(_PI_ORG_PREFIX):
        return None
    return org_type[len(_PI_ORG_PREFIX):] or None


class PiLineage:
    """Parent lookup over one ledger request. Read-only, memoized."""

    def __init__(self, conn: sqlite3.Connection, known: dict[str, tuple]):
        """known: session id -> (class, machine, agent_id or None) for the
        Pi sessions already in hand; others are read on demand."""
        self._conn = conn
        self._known = dict(known)
        self._by_raw: dict[str, list[str]] = defaultdict(list)
        for sid in known:
            ids = split_pi_id(sid)
            if ids:
                self._by_raw[ids[1]].append(sid)
        self._index: dict | None = None
        self._stored: set | None = None
        self._parent: dict[str, str | None] = {}
        self._raw_memo: dict[tuple, str | None] = {}

    def _cursor_index(self) -> dict:
        """sync_cursors Pi files, indexed once per request:
        children[(cursor machine, name)] -> [(file, parent raw id)],
        child_files[cursor machine] -> [file],
        top[(cursor machine, raw id)] -> [file] (top-level <stamp>_<raw>.jsonl)."""
        if self._index is None:
            children, child_files, top = defaultdict(list), defaultdict(list), defaultdict(list)
            for machine, session_file in self._conn.execute(
                    "SELECT machine, session_file FROM sync_cursors WHERE machine LIKE 'pi:%'"):
                parsed = parse_child_file(session_file)
                if parsed:
                    children[(machine, parsed[1])].append((session_file, parsed[0]))
                    child_files[machine].append(session_file)
                elif session_file.endswith(".jsonl"):
                    stem = session_file.rsplit("/", 1)[-1][:-len(".jsonl")]
                    top[(machine, stem.partition("_")[2] or stem)].append(session_file)
            self._index = {"children": children, "child_files": child_files, "top": top}
        return self._index

    def _stored_raw_ids(self) -> set:
        """Raw ids of every stored Pi session (a few thousand ids; one
        covering-index range read). A raw id outside it cannot match any
        hashed candidate, so its miss costs no hashing and no probe."""
        if self._stored is None:
            rows = self._conn.execute(
                "SELECT DISTINCT session_id FROM events "
                "WHERE session_id >= 'pi:' AND session_id < 'pi;'")
            self._stored = {ids[1] for ids in map(split_pi_id, (r[0] for r in rows)) if ids}
        return self._stored

    def _identity(self, sid: str) -> tuple | None:
        if sid not in self._known:
            row = self._conn.execute(
                "SELECT org_type, source_machine, agent_id FROM events "
                "WHERE session_id=? LIMIT 1", (sid,)).fetchone()
            self._known[sid] = (None if row is None else (
                account_class(row[0]), row[1], row[2]))
        return self._known[sid]

    def agent_id(self, sid: str) -> str | None:
        ident = self._identity(sid)
        return ident[2] if ident else None

    def _child_file(self, sid: str) -> tuple[str, str, str] | None:
        """(class, machine, parent raw id) of a child session, else None."""
        ident, ids = self._identity(sid), split_pi_id(sid)
        if not ident or not ids or not ident[0] or ident[2] is None:
            return None
        cls, machine, name = ident[0], ident[1], agent_name(ident[2])
        ns, raw = ids
        for session_file, parent_raw in self._cursor_index()["children"].get(
                (f"pi:{cls}:{machine}", name), ()):
            if _pi_namespace(cls, machine, session_file, raw) == ns:
                return cls, machine, parent_raw
        return None

    def _session_for(self, cls: str, machine: str, raw: str) -> str | None:
        """The stored session id of raw Pi session `raw` on that machine.

        A session already in hand wins. Otherwise hash the cursor files:
        a top-level file name holds its raw id; a child file (the parent of a
        nested child) does not, so every child file of the machine is tried.
        Memoized: many siblings share one parent.
        """
        key = (cls, machine, raw)
        if key in self._raw_memo:
            return self._raw_memo[key]
        missed = _MISSES.get((db.DB_PATH, key))
        if missed is not None and time.monotonic() - missed < MISS_TTL_S:
            return None
        found = None
        for sid in self._by_raw.get(raw, ()):
            ident = self._known[sid]
            if ident and ident[:2] == (cls, machine):
                found = sid
                break
        if found is None and raw in self._stored_raw_ids():
            cm, index = f"pi:{cls}:{machine}", self._cursor_index()
            for group in (index["top"].get((cm, raw), ()), index["child_files"].get(cm, ())):
                found = self._first_stored(
                    [f"{_pi_namespace(cls, machine, f, raw)}:{raw}" for f in group])
                if found:
                    break
        self._raw_memo[key] = found
        if found is None:
            if len(_MISSES) >= _LINKS_MAX:
                _MISSES.clear()
            _MISSES[(db.DB_PATH, key)] = time.monotonic()
        return found

    def _first_stored(self, candidates: list[str]) -> str | None:
        if not candidates:
            return None
        row = self._conn.execute(
            "SELECT session_id FROM events WHERE session_id IN "
            "(SELECT value FROM json_each(?)) LIMIT 1", (json.dumps(candidates),)).fetchone()
        return row[0] if row else None

    def parent(self, sid: str) -> str | None:
        """Stored id of the session that started child `sid`, else None."""
        if sid not in self._parent:
            key = (db.DB_PATH, sid)
            if key in _LINKS:
                self._parent[sid] = _LINKS[key]
            else:
                found = self._child_file(sid)
                self._parent[sid] = self._session_for(*found) if found else None
                if self._parent[sid] is not None:
                    _remember(key, self._parent[sid])
        return self._parent[sid]

    def chain(self, sid: str) -> list[str]:
        """Ancestors of `sid`, nearest first, stopping at the first unknown."""
        out: list[str] = []
        current = sid
        for _ in range(MAX_DEPTH):
            up = self.parent(current)
            if up is None or up == sid or up in out:
                break
            out.append(up)
            current = up
        return out


def agent_name(agent_id: str | None) -> str | None:
    """Child name from a stored Pi agent id `pi:<ns>:<name>`."""
    if agent_id is None:
        return None
    ids = split_pi_id(agent_id)
    return ids[1] if ids else agent_id
