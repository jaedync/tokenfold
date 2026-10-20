"""Days when the work billing meter rose much more than the recorded spend.

The meter (extra_usage_readings, Anthropic's own count) bills every Claude
call on the work account. The ledger has only the calls whose transcripts
reached Tokenfold. On 2026-10-19 `pi -p --session-dir` runs saved outside
~/.pi/agent/sessions, so the meter billed $354 and the ledger showed $126
with no sign that anything was missing. No routes (app/ledger.py serves it).
"""
from bisect import bisect_right
from datetime import datetime, timedelta

# The meter lags the calls a little and list prices round differently, so
# a day must miss both a fixed amount and a share of the bill to count.
GAP_MIN_USD = 20.0
GAP_MIN_SHARE = 0.25
LOOKBACK_S = 8 * 86400  # the meter is quiet over weekends: no new readings


def read_meter(conn, start: float, end: float) -> list[tuple[float, float]]:
    rows = conn.execute(
        "SELECT fetched_epoch, used_cents FROM extra_usage_readings "
        "WHERE fetched_epoch >= ? AND fetched_epoch <= ? ORDER BY fetched_epoch",
        (start - LOOKBACK_S, end)).fetchall()
    return [(r["fetched_epoch"], r["used_cents"] / 100.0) for r in rows]


def _midnight(day: str, tz, days: int = 0) -> float:
    d = datetime.strptime(day, "%Y-%m-%d") + timedelta(days=days)
    return d.replace(tzinfo=tz).timestamp()


def meter_gaps(readings, daily, tz, now: float) -> list[dict]:
    """[{day, billed, recorded}] for days whose billed rise the recorded
    Anthropic spend misses by more than GAP_MIN_USD and GAP_MIN_SHARE."""
    if not readings or not daily:
        return []
    times = [t for t, _ in readings]
    out = []
    for d in daily:
        lo = bisect_right(times, _midnight(d["day"], tz))
        hi = bisect_right(times, min(_midnight(d["day"], tz, 1), now))
        if lo == 0 or hi == 0:
            continue  # no reading before the day starts: the rise is unknown
        span = [usd for _, usd in readings[lo - 1:hi]]
        if any(b < a for a, b in zip(span, span[1:])):
            continue  # the billing cycle rolled over inside the day
        billed = span[-1] - span[0]
        recorded = sum(usd for (provider, _model), usd in d["by_model"].items()
                       if provider == "Anthropic")
        gap = billed - recorded
        if gap > GAP_MIN_USD and gap > GAP_MIN_SHARE * billed:
            out.append({"day": d["day"], "billed": round(billed, 2),
                        "recorded": round(recorded, 2)})
    return out
