"""The one pace rule for every limit gauge. No routes.

Port of pi-extras' lib/usage-pace.ts (the Pi `usage` tool), so the dashboard
and Pi give the same verdict for the same window. The window normally starts
at resets_at - window length. A granted mid-window reset (see
limit_windows._bucket_window_start) voids earlier usage while resets_at stays
put, so callers that know the grant pass window_start_epoch: the window then
runs from the grant to resets_at, and 100% must last that shorter span. Without
it, the average is diluted over the full window and reads low after a grant.
"""

import math

FIVE_HOUR_S = 5 * 3600
WEEK_S = 7 * 86400

# A short burst at the start gives unstable projections: wait for 5% of the
# window and 1% used before the average counts as evidence.
MIN_ELAPSED_FRACTION = 0.05
MIN_USED_PCT = 1.0
# +/-10 points around a full-window projection absorb rounding and bursts
# without hiding a sustained excess rate.
PACE_TOLERANCE_PCT = 10.0


def _num(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _round1(value):
    return math.floor(value * 10 + 0.5) / 10


def _effective_start(resets_at_epoch, window_s, now, window_start_epoch):
    """Granted start when it falls inside the nominal window and not in the
    future; otherwise the nominal start."""
    nominal = resets_at_epoch - window_s
    if _num(window_start_epoch) and nominal < window_start_epoch <= now:
        return window_start_epoch
    return nominal


def usage_pace(used_pct, resets_at_epoch, window_s, now, window_start_epoch=None):
    """Pace of one limit window, or None when the rule does not apply.

    window_start_epoch: the granted-reset start of the current window, when
    known; ignored unless it lies inside the nominal window.

    Returns pace_state ('above'|'on'|'below'|'too_early'), elapsed_fraction,
    expected_pct and, once there is enough evidence, projected_pct (the
    percent used at reset if the window average continues). When 'above' it
    also returns reaches_100_epoch (minute-floored, like every limit time that
    leaves the server) and reaches_100_in_s (0 once 100% is already reached).
    """
    if not (_num(used_pct) and _num(resets_at_epoch) and _num(window_s)
            and _num(now)):
        return None
    if used_pct < 0 or window_s <= 0 or resets_at_epoch <= now:
        return None
    start = _effective_start(resets_at_epoch, window_s, now, window_start_epoch)
    elapsed = now - start
    if elapsed < 0:
        return None
    fraction = elapsed / (resets_at_epoch - start)
    out = {"elapsed_fraction": fraction,
           "expected_pct": _round1(100 * fraction),
           "projected_pct": None, "reaches_100_epoch": None,
           "reaches_100_in_s": None}
    if fraction < MIN_ELAPSED_FRACTION or used_pct < MIN_USED_PCT:
        return {**out, "pace_state": "too_early"}
    projected = used_pct / fraction
    if not math.isfinite(projected * 10):
        return None
    if projected > 100 + PACE_TOLERANCE_PCT:
        state = "above"
    elif projected < 100 - PACE_TOLERANCE_PCT:
        state = "below"
    else:
        state = "on"
    out = {**out, "pace_state": state, "projected_pct": _round1(projected)}
    if state != "above":
        return out
    reaches = now - elapsed + elapsed * 100 / used_pct
    return {**out,
            "reaches_100_epoch": (reaches // 60) * 60.0,
            "reaches_100_in_s": max(0, math.ceil(reaches - now))}


def tight_epoch(pace_state, recent_eta_epoch, resets_at_epoch):
    """Time the RECENT burn would hit 100%, when that disagrees with the average.

    Returns recent_eta_epoch only when the window average says on or below
    pace yet the recent rate reaches 100% before the reset; else None. The
    'above' state already reports its own 100% time, so it never warns twice.
    """
    if pace_state not in ("on", "below"):
        return None
    if not (_num(recent_eta_epoch) and _num(resets_at_epoch)):
        return None
    return recent_eta_epoch if recent_eta_epoch < resets_at_epoch else None
