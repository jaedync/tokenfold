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
from collections import defaultdict

from .ingest import _pi_namespace

logger = logging.getLogger(__name__)

CHILD_ROOT = "subagents"
MAX_DEPTH = 8          # a deeper chain is a loop or garbage; stop there
_PI_ACCOUNT_SUFFIX = "@dotfleet.local"


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


def account_class(account_email: str | None) -> str | None:
    """Pi rows carry a synthetic pi-<class>@dotfleet.local identity."""
    if not account_email or not account_email.endswith(_PI_ACCOUNT_SUFFIX):
        return None
    local = account_email[:-len(_PI_ACCOUNT_SUFFIX)]
    return local[3:] if local.startswith("pi-") else None


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
        self._files: dict[str, list[str]] | None = None
        self._parent: dict[str, str | None] = {}

    def _cursor_files(self) -> dict[str, list[str]]:
        if self._files is None:
            files = defaultdict(list)
            for machine, session_file in self._conn.execute(
                    "SELECT machine, session_file FROM sync_cursors WHERE machine LIKE 'pi:%'"):
                files[machine].append(session_file)
            self._files = dict(files)
        return self._files

    def _identity(self, sid: str) -> tuple | None:
        if sid not in self._known:
            row = self._conn.execute(
                "SELECT account_email, source_machine, agent_id FROM events "
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
        for session_file in self._cursor_files().get(f"pi:{cls}:{machine}", ()):
            parsed = parse_child_file(session_file)
            if parsed and parsed[1] == name and \
                    _pi_namespace(cls, machine, session_file, raw) == ns:
                return cls, machine, parsed[0]
        return None

    def _session_for(self, cls: str, machine: str, raw: str) -> str | None:
        """The stored session id of raw Pi session `raw` on that machine.

        A session already in hand wins. Otherwise hash the cursor files:
        a top-level file name holds its raw id; a child file (the parent of a
        nested child) does not, so every child file of the machine is tried.
        """
        for sid in self._by_raw.get(raw, ()):
            ident = self._known[sid]
            if ident and ident[:2] == (cls, machine):
                return sid
        files = self._cursor_files().get(f"pi:{cls}:{machine}", ())
        named = [f for f in files if raw in f]
        for group in (named, [f for f in files if f.startswith(CHILD_ROOT + "/")]):
            found = self._first_stored(
                [f"{_pi_namespace(cls, machine, f, raw)}:{raw}" for f in group])
            if found:
                return found
        return None

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
            found = self._child_file(sid)
            self._parent[sid] = self._session_for(*found) if found else None
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
