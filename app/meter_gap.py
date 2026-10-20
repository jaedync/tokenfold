"""Days when the work billing meter rose much more than the recorded spend.

The meter (extra_usage_readings, Anthropic's own count) bills every Claude
call on the work account. Tokenfold has only the calls whose transcripts
reached it. On 2026-10-19 `pi -p --session-dir` runs saved outside
~/.pi/agent/sessions, so the meter billed $354 and the ledger showed $126
with no sign that anything was missing. No routes (app/ledger.py serves it).
"""
from bisect import bisect_right
from datetime import datetime, timedelta, timezone

from . import db
from .cost_windows import compute_window_cost
from .read_cache import ReadCache

# The meter lags the calls a little and list prices round differently, so
# a day must miss both a fixed amount and a share of the bill to count.
GAP_MIN_USD = 20.0
GAP_MIN_SHARE = 0.25
# A fall of more than half, or any fall across a UTC month start, is the
# billing-cycle rollover (the meter restarts from zero). Any other fall is
# two machines' readings arriving out of order.
ROLLOVER_SHARE = 0.5
# Gaps cover every day with readings, so each work ledger body carries the
# same list whatever its period. Bounded so a cold build stays cheap.
HISTORY_S = 120 * 86400
# One cost query per day is ~12 ms. A closed day's window never moves, so its
# cost stays cached; only late ingest (a backfill) changes it, seen within
# the hour. A window that ended less than OPEN_S ago is priced each time:
# Claude events arrive up to 5 min late, and with no new reading today's
# window keeps its key.
_window_costs = ReadCache(ttl=3600, max_entries=1024)
OPEN_S = 15 * 60


def read_meter(conn, start: float, end: float) -> list[tuple[float, float]]:
    """Readings in [start, end] plus the last one before start.

    Identical readings are deduped at record time, so after a quiet spell
    the reading that opens the first day can be weeks old.
    """
    seed = conn.execute(
        "SELECT fetched_epoch, used_cents FROM extra_usage_readings "
        "WHERE fetched_epoch < ? ORDER BY fetched_epoch DESC LIMIT 1", (start,)).fetchall()
    rows = conn.execute(
        "SELECT fetched_epoch, used_cents FROM extra_usage_readings "
        "WHERE fetched_epoch >= ? AND fetched_epoch <= ? ORDER BY fetched_epoch",
        (start, end)).fetchall()
    return [(r["fetched_epoch"], r["used_cents"] / 100.0) for r in [*seed, *rows]]


def window_cost(conn, now: float):
    """Recorded work Claude spend over [t0, t1). The meter bills Anthropic
    only, so other providers on the work account do not count."""
    def price(t0, t1):
        return compute_window_cost(conn, t0, t1, "enterprise", anthropic_only=True)

    def cost(t0, t1):
        if t1 > now - OPEN_S:
            return price(t0, t1)
        return _window_costs.get((db.DB_PATH, db.generation(), t0, t1), lambda: price(t0, t1))
    return cost


def _midnight(day: str, tz, days: int = 0) -> float:
    d = datetime.strptime(day, "%Y-%m-%d") + timedelta(days=days)
    return d.replace(tzinfo=tz).timestamp()


def local_days(start: float, end: float, tz) -> list[str]:
    """Local dates from the day of `start` to the day of `end`, inclusive."""
    d = datetime.fromtimestamp(start, tz).date()
    last = datetime.fromtimestamp(end, tz).date()
    out = []
    while d <= last:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _month(epoch: float) -> tuple:
    d = datetime.fromtimestamp(epoch, timezone.utc)
    return d.year, d.month


def _billed(span) -> float:
    """Billed rise over [(epoch, usd), ...] readings, across a rollover too."""
    total = 0.0
    for (ta, a), (tb, b) in zip(span, span[1:]):
        rollover = b < a and (b < a * ROLLOVER_SHARE or _month(ta) != _month(tb))
        total += b if rollover else b - a
    return total


def meter_days(readings, days, tz, now: float, cost) -> list[dict]:
    """[{day, billed, recorded}] for the local days with a reading inside them.

    A day's bill runs from the last reading at or before its midnight (else
    its first reading) to
    the last reading before the next midnight, so a quiet weekend goes to the
    day of the next reading. `cost(t0, t1)` prices that same window, so
    spend the meter shows after midnight is counted on both sides.
    """
    if not readings:
        return []
    times = [t for t, _ in readings]
    out = []
    for day in days:
        # Without a reading before midnight, the day's first reading opens it.
        lo = max(bisect_right(times, _midnight(day, tz)) - 1, 0)
        hi = bisect_right(times, min(_midnight(day, tz, 1), now)) - 1
        if hi <= lo:
            continue  # fewer than two readings in reach of the day
        out.append({"day": day, "billed": _billed(readings[lo:hi + 1]),
                     "recorded": cost(readings[lo][0], readings[hi][0])})
    return out


def is_gap(billed: float, recorded: float) -> bool:
    gap = billed - recorded
    return gap > GAP_MIN_USD and gap > GAP_MIN_SHARE * billed


def meter_gaps(readings, days, tz, now: float, cost) -> list[dict]:
    """The meter_days whose recorded spend misses GAP_MIN_USD and GAP_MIN_SHARE of the bill."""
    return [{"day": d["day"], "billed": round(d["billed"], 2), "recorded": round(d["recorded"], 2)}
            for d in meter_days(readings, days, tz, now, cost) if is_gap(d["billed"], d["recorded"])]


def all_meter_gaps(conn, tz, now: float) -> list[dict]:
    """Gaps on every local day with meter readings (up to HISTORY_S back)."""
    first = conn.execute("SELECT MIN(fetched_epoch) FROM extra_usage_readings").fetchone()[0]
    if first is None:
        return []
    start = max(first, now - HISTORY_S)
    return meter_gaps(read_meter(conn, start, now), local_days(start, now, tz), tz, now,
                      window_cost(conn, now))
