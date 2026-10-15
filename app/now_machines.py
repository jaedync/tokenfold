"""Per-machine freshness for /api/now. No routes.

Machine names are canonical_machine() names, the normalization /api/stats
uses, so "Laptop.local" and "laptop" are one machine. A machine belongs to
a scope when its events in that scope appear in the last 30 days of
daily_summary (machine_json), which is cheap. Exact last-event times come
from two reads: a cheap last-hour index scan on every request (keeps
"online" exact), and a 30-day scan plus the ingest cursors, which take
~0.3 s on production and come from a background cache.
"""
import json
import sys
import time
from datetime import datetime, timedelta

from .aggregator import canonical_machine
from .bg_cache import BackgroundCache
from .db import read_conn

DIRECTORY_DAYS = 30
RECENT_WINDOW_S = 3600

_DIRECTORY = BackgroundCache(ttl=60, wait_s=0.05)


def _pred(scope: str) -> str:
    return sys.modules["app.config"].scope_predicate(scope)


def scope_machines(conn, scope: str, today: str) -> set[str]:
    """Canonical machines with usage in this scope over the last 30 days."""
    since = (datetime.strptime(today, "%Y-%m-%d")
             - timedelta(days=DIRECTORY_DAYS)).strftime("%Y-%m-%d")
    names: set[str] = set()
    for (blob,) in conn.execute(
            f"SELECT machine_json FROM daily_summary WHERE day >= ? AND {_pred(scope)}",
            (since,)):
        try:
            names.update(canonical_machine(m) for m in json.loads(blob or "{}"))
        except (TypeError, ValueError):
            continue
    names.discard("")
    names.discard(None)
    return names


def last_events(conn, scope: str, since: float) -> dict[str, float]:
    """canonical machine -> newest billable event since `since` (scope)."""
    out: dict[str, float] = {}
    for r in conn.execute(
            "SELECT source_machine, MAX(ts_epoch) AS ts FROM events "
            "INDEXED BY idx_events_billable_ts WHERE type='assistant' "
            f"AND request_id IS NOT NULL AND ts_epoch >= ? AND {_pred(scope)} "
            "GROUP BY source_machine", (since,)):
        name = canonical_machine(r["source_machine"])
        if name and r["ts"] and r["ts"] > out.get(name, 0):
            out[name] = r["ts"]
    return out


def _cursor_machine(raw: str) -> str:
    """sync_cursors machine -> canonical name. Pi cursors are keyed
    pi:<class>:<machine> (ingest_pi), Claude cursors by the bare machine."""
    if raw.startswith("pi:") and raw.count(":") >= 2:
        raw = raw.split(":", 2)[2]
    return canonical_machine(raw)


def last_ingests(conn) -> dict[str, float]:
    """canonical machine -> newest ingest cursor write (any scope: a cursor
    has no account; ingest freshness is a property of the machine)."""
    out: dict[str, float] = {}
    for r in conn.execute(
            "SELECT machine, MAX(updated_at) AS at FROM sync_cursors GROUP BY machine"):
        try:
            at = datetime.fromisoformat(r["at"]).timestamp()
        except (TypeError, ValueError):
            continue
        name = _cursor_machine(r["machine"] or "")
        if name and at > out.get(name, 0):
            out[name] = at
    return out


def build_directory(scope: str, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    with read_conn() as conn:
        return {"last_event": last_events(conn, scope, now - DIRECTORY_DAYS * 86400),
                "last_ingest": last_ingests(conn)}


def directory(scope: str, wait_s: float | None = None) -> dict | None:
    """Background-cached 30-day directory; None until the first build lands."""
    return _DIRECTORY.get(scope, lambda: build_directory(scope), wait_s)


def clear_cache() -> None:
    """Test support."""
    _DIRECTORY.clear()
