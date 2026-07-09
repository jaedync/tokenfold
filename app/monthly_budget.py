"""Enterprise-only monthly $ budget: storage + pacing math for the
weekly_budget.monthly_budget block served by /api/rate-limits, and the
GET/POST /api/enterprise-budget setting endpoints.

Budget is a single scalar in meta (key 'enterprise_monthly_budget_usd') —
no env var, no auto-seeding. Month boundaries are UTC calendar months,
matching spend_history.monthly_costs exactly. MTD cost reuses
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


def monthly_budget_block(conn: sqlite3.Connection,
                         now: Optional[float] = None) -> Optional[dict]:
    """Enterprise-scope monthly pacing block, or None when no budget is set.

    See task brief for the exact JSON shape. elapsed_fraction < 0.05 (first
    ~1.2 days of the month) suppresses pace/projected_eom_usd (too little
    signal to project), but expected_usd is still returned.
    """
    budget = get_budget(conn)
    if budget is None:
        return None

    now = time.time() if now is None else now
    month_start, month_end, month_label, days_in_month = _month_bounds_utc(now)
    total_seconds = days_in_month * 86400.0
    elapsed_seconds = max(0.0, min(now, month_end) - month_start)
    elapsed_fraction = elapsed_seconds / total_seconds

    mtd_cost = compute_window_cost(conn, month_start, now, scope="enterprise")

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

    return {
        "budget_usd": round(budget, 2),
        "month": month_label,
        "month_end_epoch": int(month_end),
        "mtd_cost": round(mtd_cost, 2),
        "elapsed_fraction": round(elapsed_fraction, 4),
        "expected_usd": round(expected_usd, 2),
        "projected_eom_usd": (round(projected_eom_usd, 2)
                              if projected_eom_usd is not None else None),
        "pace": pace,
    }


# ── HTTP surface ────────────────────────────────────────────────────────────
# Same auth dependency as the billing-readings endpoints (dashboard/readings
# Basic-auth) — this is an enterprise-scope SETTING, not a personal-only
# write, so it is intentionally not scope-restricted.

def _require_writable():
    """Same fail-closed policy as billing_readings._require_writable: writes
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
