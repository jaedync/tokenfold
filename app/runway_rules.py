"""Pure /api/runway rules (docs/ux-2026-10/BUILD.md). No I/O, no routes.

Every input is a normalized runway window dict: id, kind, pct, resets_at,
pace (the app/usage_pace.py block or None), tight_epoch, stale. Nothing here
projects anything new: the 100% times come from the pace block or from
trend[bucket].tight_epoch, the one forecast the dashboard shows.
"""

import math

# A window that resets within this long of when it would fill (or of now,
# for a short window) costs at most this much waiting: "waitable".
WAITABLE_S = 3600
# Windows up to a day are "short" (Claude 5 h, OpenCode Go rolling 5 h);
# they reset too soon to be the hero, so they never bind.
SHORT_MAX_S = 86400
# Same noise floor as app/ha.py _implied_limit and the provider estimates:
# below 5% the integer-quantized meter makes $/point meaningless.
MIN_PCT_FOR_RATE = 5.0
LONG_KINDS = ("long", "month")
# Owner decision (2026-10-15): a long window projected at this percent or
# more counts as at risk, so it may take the hero from the main limit.
AT_RISK_PROJECTED_PCT = 90.0
# The main limit of each scope, in order: the hero when nothing is at risk.
CALM_DEFAULTS = ("claude:seven_day", "claude:month")


def _num(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def kind_for(window_s):
    """'short' for windows of a day or less, else 'long'."""
    return "short" if _num(window_s) and window_s <= SHORT_MAX_S else "long"


def _projected(window):
    pace = window.get("pace") or {}
    return pace.get("projected_pct")


def _score(window):
    """projected_pct when the pace has one, else the used pct (BUILD.md
    fallback); None when neither is known."""
    projected = _projected(window)
    if _num(projected):
        return projected
    return window.get("pct") if _num(window.get("pct")) else None


def at_risk(window):
    """A long window that runs out, is TIGHT, or projects >= 90%."""
    projected = _projected(window)
    return (verdict(window) in ("runs_out", "tight")
            or (_num(projected) and projected >= AT_RISK_PROJECTED_PCT))


def fill_epoch(window):
    """The earliest 100% time (pace or TIGHT) before the reset, else None."""
    resets_at = window.get("resets_at")
    pace = window.get("pace") or {}
    times = [t for t in (pace.get("reaches_100_epoch"), window.get("tight_epoch"))
             if _num(t) and (not _num(resets_at) or t < resets_at)]
    return min(times) if times else None


def binding(windows):
    """id of the long window most likely to stop work, or None.

    When no long window is at risk, the main limit binds (claude:seven_day,
    or the month meter in work scope): early in a window the other
    providers' projections would otherwise take the hero from a limit that
    is merely too early to project (owner decision, 2026-10-15).

    Otherwise the window that fills first binds; without one that fills
    before its reset, the highest score. A high projected percent early in
    a window can still fill later than a fuller window (2026-10-26: Codex
    267% in 53 h against Claude 7-day 124% in 22 h). A waitable fill (it
    resets within the hour; runway.py sets the flag) costs a short wait, so
    it ranks after a fill that stops work longer. A stale window binds
    only when every candidate is stale: an old snapshot must not keep the
    hero after a fresh window exists.
    """
    long_windows = [w for w in windows if w.get("kind") in LONG_KINDS]
    if not any(at_risk(w) for w in long_windows):
        ids = {w.get("id") for w in long_windows}
        calm = next((d for d in CALM_DEFAULTS if d in ids), None)
        if calm is not None:
            return calm
    scored = [(w, _score(w)) for w in long_windows]
    scored = [(w, s) for w, s in scored if s is not None]
    fresh = [(w, s) for w, s in scored if not w.get("stale")]
    pool = fresh or scored
    if not pool:
        return None
    filling = [(w, fill_epoch(w)) for w, _ in pool]
    filling = [(w, t) for w, t in filling if t is not None]
    if filling:
        return min(filling, key=lambda wt: (bool(wt[0].get("waitable")), wt[1]))[0]["id"]
    return max(pool, key=lambda ws: ws[1])[0]["id"]


def full_epoch(window):
    """When the window reaches 100%: the pace time (above pace), else the
    TIGHT time (recent burn), else None."""
    pace = window.get("pace") or {}
    reaches = pace.get("reaches_100_epoch")
    if _num(reaches):
        return reaches
    tight = window.get("tight_epoch")
    return tight if _num(tight) else None


def waitable(window, now):
    """True when waiting for the reset is cheap.

    Short window: it resets within WAITABLE_S. Long window: it fills, but
    its reset comes within WAITABLE_S of that 100% time. (BUILD.md says
    "reset sooner than its 100% time"; every pace or TIGHT 100% time is
    before the reset by construction, so the hour of slack is what makes
    the rule able to fire.)
    """
    resets_at = window.get("resets_at")
    if not (_num(resets_at) and _num(now)):
        return False
    if window.get("kind") == "short":
        return 0 <= resets_at - now <= WAITABLE_S
    full = full_epoch(window)
    return full is not None and resets_at - full <= WAITABLE_S


def verdict(window):
    """'runs_out' | 'tight' | 'too_early' | 'clear', or None without pace."""
    pace = window.get("pace")
    if not pace:
        return None
    reaches = pace.get("reaches_100_epoch")
    resets_at = window.get("resets_at")
    if _num(reaches) and _num(resets_at) and reaches < resets_at:
        return "runs_out"
    if _num(window.get("tight_epoch")):
        return "tight"
    if pace.get("pace_state") == "too_early":
        return "too_early"
    return "clear"


def usd_per_point(usd_spent, pct):
    """Dollars of API-equivalent spend per percentage point used."""
    if not (_num(usd_spent) and _num(pct)) or usd_spent <= 0:
        return None
    if pct < MIN_PCT_FOR_RATE:
        return None
    return usd_spent / pct


def usd_left(per_point, pct):
    """Value of the points not yet used, at the window's own rate."""
    if not (_num(per_point) and _num(pct)):
        return None
    return per_point * max(0.0, 100.0 - pct)
