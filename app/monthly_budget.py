"""Enterprise-only monthly $ budget: storage + pacing math for the
weekly_budget.monthly_budget block served by /api/rate-limits, and the
GET/POST /api/enterprise-budget setting endpoints.

METER-FIRST since the extra_usage capture (app/extra_usage.py): a fresh
capture supplies both MTD spend and the budget from Anthropic's own billing
meter. The meta scalar (key 'enterprise_monthly_budget_usd') survives as the
fallback when the meter is stale or carries no limit — no env var, no
auto-seeding. Month boundaries are UTC calendar months, matching
spend_history.monthly_costs exactly. Estimated MTD cost reuses
cost_windows.compute_window_cost (same dedupe/era/geo pricing as the
monthly spend chart) rather than duplicating pricing logic.
"""

import math
import sqlite3
import time
from calendar import monthrange
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

import app.config as config
from .auth import require_dashboard_auth
from .cost_windows import compute_window_cost
from .db import get_conn
from .models import EnterpriseBudgetRequest

router = APIRouter()

META_KEY = "enterprise_monthly_budget_usd"
BUDGET_MAX = 1_000_000
EARLY_MONTH_FRACTION = 0.05
PACE_DEADBAND = 0.10  # +/-10%, same as limit_trends.bucket_trend


def log(msg, *args):
    formatted = msg % args if args else msg
    print(f"[monthly_budget] {formatted}", flush=True)


def get_budget(conn: sqlite3.Connection) -> Optional[float]:
    """Read the stored enterprise monthly budget. Tolerates a missing row or
    a garbage stored value (never raises) — returns None in both cases."""
    row = conn.execute(
        "SELECT value FROM meta WHERE key=?", (META_KEY,)).fetchone()
    if row is None or row["value"] is None:
        return None
    try:
        value = float(row["value"])
    except (TypeError, ValueError):
        log("garbage meta value %r for %s", row["value"], META_KEY)
        return None
    if not math.isfinite(value):
        log("non-finite meta value %r for %s", value, META_KEY)
        return None
    return value


def set_budget(conn: sqlite3.Connection, value: Optional[float]) -> None:
    """Validate and store the budget; None clears the setting.

    Raises ValueError with a clear message on invalid input.
    """
    if value is None:
        conn.execute("DELETE FROM meta WHERE key=?", (META_KEY,))
        conn.commit()
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("budget_usd must be a number")
    if not math.isfinite(value):
        raise ValueError("budget_usd must be finite")
    if value <= 0:
        raise ValueError("budget_usd must be > 0")
    if value > BUDGET_MAX:
        raise ValueError(f"budget_usd must be <= {BUDGET_MAX}")

    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (META_KEY, str(float(value))))
    conn.commit()


def _month_bounds_utc(now_epoch: float):
    """Return (month_start_epoch, month_end_epoch, 'YYYY-MM', days_in_month)
    for the UTC calendar month containing now_epoch — identical math to
    spend_history.monthly_costs."""
    d = datetime.fromtimestamp(now_epoch, tz=timezone.utc)
    y, m = d.year, d.month
    start = datetime(y, m, 1, tzinfo=timezone.utc)
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    end = datetime(ny, nm, 1, tzinfo=timezone.utc)
    _, days_in_month = monthrange(y, m)
    return start.timestamp(), end.timestamp(), "%04d-%02d" % (y, m), days_in_month


def _business_elapsed_utc(now_epoch: float):
    """Return (elapsed_business_fraction, business_days_in_month) for the UTC
    calendar month containing now_epoch.

    Business days are Mon-Fri UTC. Each contributes 86400s to the
    denominator; weekends contribute nothing to either side, so the fraction
    stands still across Sat/Sun. Enterprise usage is work usage — a calendar
    basis under-states pace during the work week and over-projects across
    weekends. Every real month has >= 20 business days, so the denominator
    is never zero."""
    d = datetime.fromtimestamp(now_epoch, tz=timezone.utc)
    y, m = d.year, d.month
    _, days_in_month = monthrange(y, m)
    business_days = 0
    elapsed = 0.0
    for day in range(1, days_in_month + 1):
        day_start = datetime(y, m, day, tzinfo=timezone.utc)
        if day_start.weekday() >= 5:  # Saturday/Sunday
            continue
        business_days += 1
        overlap = now_epoch - day_start.timestamp()
        if overlap > 0:
            elapsed += min(overlap, 86400.0)
    return elapsed / (business_days * 86400.0), business_days


def monthly_budget_block(conn: sqlite3.Connection,
                         now: Optional[float] = None) -> Optional[dict]:
    """Enterprise-scope monthly pacing block, or None when there is neither
    a fresh meter limit nor a stored budget.

    METER-FIRST: a fresh extra_usage capture (Anthropic's own billing meter,
    see app/extra_usage.py) supplies both the MTD spend and — when present —
    the budget itself, so nothing needs hand-entering; the event-cost
    estimate rides along as measured_mtd_usd and the gap as
    unaccounted_mtd_usd (claude.ai web etc.). A stale/absent meter falls
    back to the stored budget scalar + estimate (source='estimate').

    Month boundaries are UTC calendar months. The meter's billing cycle is
    assumed calendar-aligned; if a used_credits rollover is ever observed
    mid-month the anchor can be inferred from extra_usage_readings.

    Pacing basis is BUSINESS days (Mon-Fri UTC, _business_elapsed_utc):
    elapsed_fraction / expected_usd / projected_eom_usd all divide by
    business time, so a month is "spent" over its ~21-23 workdays and the
    pace verdict does not drift across weekends.

    elapsed_fraction < 0.05 (first ~1.2 business days of the month)
    suppresses pace/projected_eom_usd (too little signal to project), but
    expected_usd is still returned.
    """
    now = time.time() if now is None else now

    from .extra_usage import latest_meter, meter_is_fresh
    meter = latest_meter(conn)
    use_meter = meter_is_fresh(meter, now)

    budget = None
    budget_from_meter = False
    if use_meter and meter["limit_usd"]:
        budget = meter["limit_usd"]
        budget_from_meter = True
    if budget is None:
        budget = get_budget(conn)
    if budget is None:
        return None

    month_start, month_end, month_label, _days_in_month = _month_bounds_utc(now)
    # Pace on BUSINESS days (Mon-Fri UTC), not calendar seconds: enterprise
    # spend accrues on workdays, so weekends neither raise expected_usd nor
    # dilute the projection. business_days also drives the dashboard's
    # per-day bar ticks.
    elapsed_fraction, business_days = _business_elapsed_utc(now)

    measured_mtd = compute_window_cost(
        conn, month_start, now, scope="enterprise")
    mtd_cost = meter["used_usd"] if use_meter else measured_mtd

    expected_usd = elapsed_fraction * budget

    pace = None
    projected_eom_usd = None
    if elapsed_fraction >= EARLY_MONTH_FRACTION:
        projected_eom_usd = mtd_cost / elapsed_fraction
        ratio = projected_eom_usd / budget
        if ratio < 1.0 - PACE_DEADBAND:
            pace = "under"
        elif ratio > 1.0 + PACE_DEADBAND:
            pace = "over"
        else:
            pace = "on"

    block = {
        "source": "meter" if use_meter else "estimate",
        # Distinct from source: a fresh meter can lack a limit, in which case
        # the stored scalar is still the budget and must stay editable.
        "budget_from_meter": budget_from_meter,
        "budget_usd": round(budget, 2),
        "month": month_label,
        "month_end_epoch": int(month_end),
        "business_days": business_days,
        "mtd_cost": round(mtd_cost, 2),
        "elapsed_fraction": round(elapsed_fraction, 4),
        "expected_usd": round(expected_usd, 2),
        "projected_eom_usd": (round(projected_eom_usd, 2)
                              if projected_eom_usd is not None else None),
        "pace": pace,
    }
    if use_meter:
        block["measured_mtd_usd"] = round(measured_mtd, 2)
        block["unaccounted_mtd_usd"] = round(mtd_cost - measured_mtd, 2)
        block["meter_updated_epoch"] = meter["fetched_epoch"]
        block["meter_machine"] = meter["machine"]
    return block


# ── HTTP surface ────────────────────────────────────────────────────────────
# Dashboard Basic-auth — this is an enterprise-scope SETTING, not a
# personal-only write, so it is intentionally not scope-restricted.

def _require_writable():
    """Fail-closed write policy (same as the ingest-key reveal): writes
    are human dashboard actions, so an open (password-less) instance must
    still reject them rather than silently accept unauthenticated writes."""
    if not config.DASHBOARD_PASSWORD:
        raise HTTPException(status_code=403,
                            detail="budget is write-disabled on open dashboards")


@router.get("/api/enterprise-budget",
            dependencies=[Depends(require_dashboard_auth)])
async def get_enterprise_budget():
    conn = get_conn()
    return {"budget_usd": get_budget(conn)}


@router.post("/api/enterprise-budget",
             dependencies=[Depends(require_dashboard_auth)])
async def post_enterprise_budget(req: EnterpriseBudgetRequest):
    _require_writable()
    conn = get_conn()
    try:
        set_budget(conn, req.budget_usd)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    from .aggregator import trigger_eager_rebuild
    trigger_eager_rebuild()

    return {"budget_usd": get_budget(conn)}
