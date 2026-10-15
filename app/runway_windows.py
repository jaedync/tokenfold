"""Normalize the /api/rate-limits payload into /api/runway windows[]. No I/O.

Inputs are the blocks _rate_limits_payload already built (Claude buckets with
pace, trend and window dollars; provider windows; the enterprise monthly
budget), so the runway never repeats their SQL and never forecasts on its own.
"""

import math
from datetime import datetime, timezone

from .limit_windows import _iso_to_epoch
from .quota_freshness import QUOTA_STALE_AFTER_S
from .runway_rules import kind_for, usd_left, usd_per_point
from .usage_pace import FIVE_HOUR_S, WEEK_S

PACE_KEYS = ("pace_state", "expected_pct", "projected_pct",
             "reaches_100_epoch", "elapsed_fraction")
PROVIDER_LABELS = {"codex": "Codex", "opencode-go": "OpenCode Go"}
CLAUDE_LABELS = {"seven_day": "7-day", "five_hour": "5-hour"}
# monthly_budget speaks under/on/over; the runway speaks usage_pace's words.
MONTH_PACE = {"under": "below", "on": "on", "over": "above"}


def _num(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _floor_minute(epoch):
    return (epoch // 60) * 60.0 if _num(epoch) else None


def pace_subset(pace):
    return {k: pace.get(k) for k in PACE_KEYS} if pace else None


def _freshness(observed, resets_at, now):
    """(stale, age_s). Stale: no observation, older than the quota freshness
    limit, or a window that has already reset."""
    age = now - observed if _num(observed) else None
    stale = (age is None or age > QUOTA_STALE_AFTER_S
             or not _num(resets_at) or resets_at <= now)
    return stale, (max(0, round(age)) if age is not None else None)


def _window(*, id_, provider, key, label, kind, window_s, pct, resets_at,
            window_start, observed, now, pace, tight=None, burn=None,
            usd_spent=None, per_point=None, left=None):
    stale, age = _freshness(observed, resets_at, now)
    return {
        "id": id_, "provider": provider, "key": key, "label": label,
        "kind": kind, "window_s": window_s, "pct": pct,
        "resets_at": resets_at, "window_start": window_start,
        "observed_at": observed, "stale": stale, "age_s": age,
        "pace": pace, "tight_epoch": tight, "burn_pct_per_h": burn,
        "usd_spent": usd_spent, "usd_per_point": per_point, "usd_left": left,
    }


def _claude_dollars(oauth, bucket):
    """(window_start, usd_spent) for one bucket, from the blocks
    _rate_limits_payload attaches only when the observation is current."""
    key = bucket["key"]
    block = {"seven_day": oauth.get("limit_window"),
             "five_hour": oauth.get("five_hour_window")}.get(key)
    if block:
        return block.get("start_epoch"), block.get("cost")
    if key.startswith("scoped:") and "window_cost" in bucket:
        return bucket.get("window_start_epoch"), bucket.get("window_cost")
    return None, None


def _burn(trend):
    if not trend:
        return None
    return {"h1": trend.get("burn_1h_pct_per_hr"),
            "h6": trend.get("burn_6h_pct_per_hr")}


def claude_windows(oauth, now):
    """One window per Claude bucket (seven_day, five_hour, scoped:*)."""
    if not oauth:
        return []
    observed = oauth.get("updated_at_epoch")
    trends = oauth.get("trend") or {}
    out = []
    for b in oauth.get("buckets") or []:
        key = b.get("key") or ""
        window_s = FIVE_HOUR_S if key == "five_hour" else WEEK_S
        name = CLAUDE_LABELS.get(key) or f"{b.get('label') or key} 7-day"
        start, spent = _claude_dollars(oauth, b)
        rate = usd_per_point(spent, b.get("pct"))
        trend = trends.get(key)
        out.append(_window(
            id_=f"claude:{key}", provider="claude", key=key,
            label=f"Claude · {name}", kind=kind_for(window_s),
            window_s=window_s, pct=b.get("pct"),
            resets_at=_iso_to_epoch(b.get("resets_at")),
            window_start=_floor_minute(start), observed=observed, now=now,
            pace=pace_subset(b.get("pace")),
            tight=(trend or {}).get("tight_epoch"), burn=_burn(trend),
            usd_spent=spent, per_point=rate, left=usd_left(rate, b.get("pct"))))
    return out


def provider_windows(providers, now):
    """Codex and OpenCode Go windows. Provider windows have no granted
    resets, so the nominal start (reset - length) is their real start."""
    out = []
    for provider, label in PROVIDER_LABELS.items():
        block = (providers or {}).get(provider) or {}
        observed = block.get("updated_at_epoch")
        for w in block.get("windows") or []:
            resets_at = _iso_to_epoch(w.get("resets_at"))
            window_s = w.get("window_seconds")
            start = w.get("window_start_epoch")
            if start is None and _num(resets_at) and _num(window_s):
                start = resets_at - window_s
            spent = w.get("window_cost")
            rate = usd_per_point(spent, w.get("pct"))
            out.append(_window(
                id_=f"{provider}:{w.get('key')}", provider=provider,
                key=w.get("key"), label=f"{label} · {w.get('label')}",
                kind=kind_for(window_s), window_s=window_s, pct=w.get("pct"),
                resets_at=resets_at, window_start=_floor_minute(start),
                observed=observed, now=now, pace=pace_subset(w.get("pace")),
                usd_spent=spent, per_point=rate,
                left=usd_left(rate, w.get("pct"))))
    return out


def _month_start(now):
    d = datetime.fromtimestamp(now, tz=timezone.utc)
    return datetime(d.year, d.month, 1, tzinfo=timezone.utc).timestamp()


def business_epoch_after(now, business_s):
    """The epoch at which business_s of Mon-Fri UTC time has passed after
    now (the basis app/monthly_budget.py paces on)."""
    t, left = now, business_s
    for _ in range(400):  # a bound, not a tuning value: > a year of days
        d = datetime.fromtimestamp(t, tz=timezone.utc)
        day_end = datetime(d.year, d.month, d.day, tzinfo=timezone.utc
                           ).timestamp() + 86400
        if d.weekday() < 5:
            if t + left <= day_end:
                return t + left
            left -= day_end - t
        t = day_end
    return None


def _month_reaches_100(block, now):
    """When the business-day pace of the monthly budget block hits the
    budget; only when that block says 'over' (as usage_pace does 'above')."""
    if block.get("pace") != "over":
        return None
    budget, spent = block["budget_usd"], block["mtd_cost"]
    if spent >= budget:
        return _floor_minute(now)
    elapsed_s = block["elapsed_fraction"] * block["business_days"] * 86400
    if elapsed_s <= 0 or spent <= 0:
        return None
    return _floor_minute(business_epoch_after(
        now, (budget - spent) * elapsed_s / spent))


def month_window(block, now):
    """The enterprise monthly billing meter as one kind='month' window."""
    budget, spent = block["budget_usd"], block["mtd_cost"]
    start, end = _month_start(now), float(block["month_end_epoch"])
    pct = spent / budget * 100 if budget > 0 else None
    projected = block.get("projected_eom_usd")
    pace = {
        "pace_state": MONTH_PACE.get(block.get("pace"), "too_early"),
        "expected_pct": block["expected_usd"] / budget * 100 if budget else None,
        "projected_pct": (projected / budget * 100
                          if projected is not None and budget else None),
        "reaches_100_epoch": _month_reaches_100(block, now),
        "elapsed_fraction": block["elapsed_fraction"],
    }
    # A fresh meter reading is the observation; an estimate is live events.
    observed = block.get("meter_updated_epoch") if block.get(
        "source") == "meter" else now
    w = _window(
        id_="claude:month", provider="claude", key="month",
        label="Claude · Monthly budget", kind="month", window_s=end - start,
        pct=pct, resets_at=end, window_start=start, observed=observed,
        now=now, pace=pace, usd_spent=spent, per_point=budget / 100,
        left=max(0.0, budget - spent))
    # Month staleness is the meter's age, not the quota poll's: the event
    # estimate is always current.
    return {**w, "stale": False}


def budget_block(block):
    """{budget_usd, spent_usd, projected_usd, pace_state, per_workday_left_usd}."""
    budget, spent = block["budget_usd"], block["mtd_cost"]
    days_left = block["business_days"] * (1 - block["elapsed_fraction"])
    return {
        "budget_usd": budget,
        "spent_usd": spent,
        "projected_usd": block.get("projected_eom_usd"),
        "pace_state": MONTH_PACE.get(block.get("pace"), "too_early"),
        "per_workday_left_usd": (max(0.0, budget - spent) / days_left
                                 if days_left > 0 else None),
    }
