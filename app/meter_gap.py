"""Days when the work billing meter rose much more than the recorded spend.

The meter (extra_usage_readings, Anthropic's own count) bills every Claude
call on the work account. Tokenfold has only the calls whose transcripts
reached it. On 2026-10-19 `pi -p --session-dir` runs saved outside
~/.pi/agent/sessions, so the meter billed $354 and the ledger showed $126
with no sign that anything was missing. No routes (app/ledger.py serves it).
"""
from bisect import bisect_right
from datetime import datetime, timedelta

from .cost_windows import compute_window_cost
from .extra_usage import _utc_month_start

# The meter lags the calls a little and list prices round differently, so
# a day must miss both a fixed amount and a share of the bill to count.
GAP_MIN_USD = 20.0
GAP_MIN_SHARE = 0.25


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


def window_cost(conn):
    """Recorded work Claude spend over [t0, t1), the meter's own window."""
    return lambda t0, t1: compute_window_cost(conn, t0, t1, "enterprise", anthropic_only=True)


def _midnight(day: str, tz, days: int = 0) -> float:
    d = datetime.strptime(day, "%Y-%m-%d") + timedelta(days=days)
    return d.replace(tzinfo=tz).timestamp()


def meter_gaps(readings, days, tz, now: float, cost) -> list[dict]:
    """[{day, billed, recorded}] for local days whose billed rise the
    recorded spend misses by more than GAP_MIN_USD and GAP_MIN_SHARE.

    A day's bill runs from the last reading at or before its midnight to
    the last reading before the next one. `cost(t0, t1)` prices that same
    window, so spend the meter shows after midnight is no false gap.
    """
    if not readings:
        return []
    times = [t for t, _ in readings]
    out = []
    for day in days:
        lo = bisect_right(times, _midnight(day, tz)) - 1
        hi = bisect_right(times, min(_midnight(day, tz, 1), now)) - 1
        if lo < 0 or hi <= lo:
            continue  # no reading before the day, or none inside it
        (t0, start), (t1, end) = readings[lo], readings[hi]
        billed = end - start
        if billed <= 0 or _utc_month_start(t0) != _utc_month_start(t1):
            continue  # the billing cycle rolled over inside the window
        recorded = cost(t0, t1)
        gap = billed - recorded
        if gap > GAP_MIN_USD and gap > GAP_MIN_SHARE * billed:
            out.append({"day": day, "billed": round(billed, 2), "recorded": round(recorded, 2)})
    return out
