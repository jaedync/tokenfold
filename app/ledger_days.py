"""Per-day cache of the ledger's billable scan.

The billable scan (deduped, canonical, priced) is most of a cold /api/ledger
read: about 1.4 s for 30 days on the 2026-10-15 production copy, because every
assistant row is read from the table. A closed local day rarely changes, so
its result is kept per (scope, day) and reused until the day's daily_summary
rows change. Ingest and the twin sweep re-roll every day they touch
(summarize_days writes a fresh updated_at), so that fingerprint moves exactly
when the day's events or twin marks move. The open day is always scanned.

Requests are grouped by local day here, as the daily rollup does: a request
whose stream chunks straddle midnight is counted once per side.
"""

from collections import Counter
from datetime import datetime, timedelta

from . import db
from .aggregator import TZ
from .ledger_query import billable_groups
from .read_cache import ReadCache

# A price refresh (LiteLLM) does not re-roll old days; the TTL bounds how long
# a cached day can keep an old price.
DAY_TTL_S = 3600
_days = ReadCache(ttl=DAY_TTL_S, max_entries=400)


def _midnight(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, TZ).replace(hour=0, minute=0, second=0,
                                                      microsecond=0)


def split_range(start: float, end: float, now: float) -> tuple[list, list]:
    """([(day, lo, hi)] closed whole local days, [(lo, hi)] other pieces)."""
    today = _midnight(now).timestamp()
    days, rest = [], []
    cursor = start
    d = _midnight(start)
    if d.timestamp() < start:
        d = (d + timedelta(days=1, hours=2)).replace(hour=0)  # DST-safe next midnight
    while True:
        lo = d.timestamp()
        hi = (d + timedelta(days=1, hours=2)).replace(hour=0).timestamp()
        if hi > end or hi > today:
            break
        if lo > cursor:
            rest.append((cursor, lo))
        days.append((d.strftime("%Y-%m-%d"), lo, hi))
        cursor = hi
        d = datetime.fromtimestamp(hi, TZ)
    if cursor < end:
        rest.append((cursor, end))
    return days, rest


def fingerprints(conn, days: list[str]) -> dict:
    if not days:
        return {}
    out = {d: (0, None) for d in days}
    for row in conn.execute(
            "SELECT day, COUNT(*), MAX(updated_at) FROM daily_summary "
            "WHERE day >= ? AND day <= ? GROUP BY day", (min(days), max(days))):
        if row[0] in out:
            out[row[0]] = (row[1], row[2])
    return out


def merge_threads(parts) -> dict:
    """Sum per-(session, agent) threads; never mutates the (cached) parts."""
    out: dict = {}
    for part in parts:
        for key, t in part.items():
            cur = out.get(key)
            if cur is None:
                out[key] = {**t, "models": Counter(t["models"]),
                            "tokens": Counter(t["tokens"])}
                continue
            cur["cost"] += t["cost"]
            cur["models"].update(t["models"])
            cur["tokens"].update(t["tokens"])
            cur["start"] = min(cur["start"], t["start"])
            cur["end"] = max(cur["end"], t["end"])
            for k in ("machine", "project_dir", "account"):
                cur[k] = cur[k] or t[k]
    return out


def billable_threads(conn, pred: str, scope: str, start: float, end: float,
                     now: float, collect) -> dict:
    """collect(billable rows) -> threads, merged over cached closed days."""
    def scan(lo, hi):
        return collect(billable_groups(conn, pred, lo, hi))

    days, rest = split_range(start, end, now)
    marks = fingerprints(conn, [d for d, _lo, _hi in days])
    parts = [_days.get((db.DB_PATH, scope, day, marks[day]),
                       lambda lo=lo, hi=hi: scan(lo, hi))
             for day, lo, hi in days]
    parts += [scan(lo, hi) for lo, hi in rest]
    return merge_threads(parts)
