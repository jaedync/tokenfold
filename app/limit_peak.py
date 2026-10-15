"""Peak-hour factor for the 5-hour limit, from fit residuals. No I/O.

limit-xray (REPORT.md section 3) found that the 5-hour meter moves about
1.35x faster per dollar on weekdays 05:00-11:00 Pacific, and the 7-day meter
shows no time effect. This module measures that per Pacific hour instead of
hard-coding the window, so a schedule change shows up in the data.

Cells pool all weekdays (Mon-Fri) and all weekends per hour: day-of-week
effects beyond that split were not significant, and 168 separate cells would
hold too few intervals each.
"""

from datetime import datetime
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

PEAK_TZ = "America/Los_Angeles"
_PT = ZoneInfo(PEAK_TZ)
# A cell is peak when it burns at least this much faster than normal hours.
PEAK_MIN_FACTOR = 1.15
# Expected points a cell needs before its factor is shown (whole-point
# rounding makes thinner cells mostly noise).
CELL_MIN_EXPECTED_PTS = 5.0
_MAX_ITER = 5
# Search horizon for the end of a peak or the next peak start.
_SEARCH_HOURS = 8 * 24


def _cell(t: float) -> tuple:
    d = datetime.fromtimestamp(t, _PT)
    return ("weekday" if d.weekday() < 5 else "weekend", d.hour)


def _ratio(pairs) -> Optional[float]:
    d = sum(p[0] for p in pairs)
    e = sum(p[1] for p in pairs)
    return d / e if e > 0 else None


def compute_peak(samples: Sequence) -> dict:
    """samples: (mid_epoch, points_gained, points_expected) per 5 h interval.

    Factor = residual ratio of a cell / residual ratio of the non-peak hours,
    so 1.0 means "normal" and the peak set is found from the data.
    """
    by_cell: dict = {}
    for mid, d, e in samples:
        by_cell.setdefault(_cell(mid), []).append((d, e))
    raw = {c: _ratio(v) for c, v in by_cell.items()
           if sum(p[1] for p in v) >= CELL_MIN_EXPECTED_PTS}
    raw = {c: r for c, r in raw.items() if r is not None}
    peak: set = set()
    base = None
    for _ in range(_MAX_ITER):
        base = _ratio([p for c, v in by_cell.items() if c not in peak for p in v])
        if not base:
            return empty_peak()
        found = {c for c, r in raw.items() if r / base >= PEAK_MIN_FACTOR}
        if found == peak:
            break
        peak = found
    factor = None
    if peak:
        pr = _ratio([p for c in peak for p in by_cell[c]])
        factor = round(pr / base, 4) if pr is not None else None
    hours = []
    for dow in range(7):
        cls = "weekday" if dow < 5 else "weekend"
        for hour in range(24):
            r = raw.get((cls, hour))
            hours.append({"dow": dow, "hour": hour,
                          "factor": round(r / base, 4) if r is not None else None})
    return {"factor": factor, "tz": PEAK_TZ, "hours": hours}


def empty_peak() -> dict:
    return {"factor": None, "tz": PEAK_TZ, "hours": []}


def peak_state(peak: dict, now: float) -> dict:
    """Serve-time state: is now a peak hour, when does it end or start.

    Evaluated per request from the cached hour table, so the flag flips on
    the hour even though the fit runs hourly. Pacific offsets are whole
    hours, so UTC hour boundaries are Pacific hour boundaries.
    """
    table = {(h["dow"], h["hour"]): h["factor"] for h in (peak or {}).get("hours") or []}

    def is_peak(t):
        d = datetime.fromtimestamp(t, _PT)
        f = table.get((d.weekday(), d.hour))
        return f is not None and f >= PEAK_MIN_FACTOR

    none = {"active": False, "until": None, "next_start": None}
    if not any(f is not None and f >= PEAK_MIN_FACTOR for f in table.values()):
        return none
    t = now - now % 3600 + 3600
    active = is_peak(now)
    for _ in range(_SEARCH_HOURS):
        if is_peak(t) != active:
            break
        t += 3600
    else:
        return dict(none, active=active)
    if active:
        return {"active": True, "until": t, "next_start": None}
    return {"active": False, "until": None, "next_start": t}
