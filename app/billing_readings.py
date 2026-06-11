"""Enterprise billing readings: historize the Claude org-page MTD figure.

A single reading proves little, but the delta between consecutive readings vs
our measured cost over the exact same window quantifies measurement coverage
(sidecar calls, non-Claude-Code usage, list-vs-billed pricing) going forward.

Writes are human actions from the dashboard: HTTP Basic auth, and FAIL-CLOSED
on instances without DASHBOARD_PASSWORD (require_dashboard_auth is open there
by design, so these routes re-check — same policy as the ingest-key reveal).
"""

import math
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

import app.config as config
from .auth import require_dashboard_auth
from .cost_windows import compute_window_cost
from .db import get_conn
from .models import BillingReadingRequest

router = APIRouter()

NOTE_MAX = 256
AMOUNT_MAX = 10**7  # an MTD figure beyond $10M is a typo, not a bill


def _require_writable():
    if not config.DASHBOARD_PASSWORD:
        raise HTTPException(status_code=403,
                            detail="readings are write-disabled on open dashboards")


def _utc_now():
    return datetime.now(timezone.utc)


@router.post("/api/billing-readings",
             dependencies=[Depends(require_dashboard_auth)])
async def record_reading(req: BillingReadingRequest):
    _require_writable()
    amt = req.amount_usd
    if not (math.isfinite(amt) and 0 <= amt < AMOUNT_MAX):
        raise HTTPException(status_code=400, detail="amount_usd out of range")
    note = (req.note or "")[:NOTE_MAX] or None

    now = _utc_now()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    conn = get_conn()
    # Frozen "what we knew then" snapshot — same window the month hero shows.
    measured = compute_window_cost(
        conn, month_start.timestamp(), now.timestamp() + 1, "enterprise")

    cur = conn.execute(
        "INSERT INTO billing_readings(scope, amount_usd, measured_usd, month, "
        "recorded_at, recorded_epoch, note) VALUES('enterprise',?,?,?,?,?,?)",
        (round(amt, 2), round(measured, 2), now.strftime("%Y-%m"),
         now.isoformat(), now.timestamp(), note))
    conn.commit()

    from .aggregator import trigger_eager_rebuild
    trigger_eager_rebuild()

    return {
        "id": cur.lastrowid,
        "amount_usd": round(amt, 2),
        "measured_usd": round(measured, 2),
        "month": now.strftime("%Y-%m"),
        "recorded_at": now.isoformat(),
        "note": note,
    }


@router.delete("/api/billing-readings/{reading_id}",
               dependencies=[Depends(require_dashboard_auth)])
async def delete_reading(reading_id: int):
    _require_writable()
    conn = get_conn()
    cur = conn.execute("DELETE FROM billing_readings WHERE id=?", (reading_id,))
    conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="no such reading")

    from .aggregator import trigger_eager_rebuild
    trigger_eager_rebuild()
    return {"deleted": reading_id}


def build_readings_payload(conn, scope: str, limit: int = 50) -> list[dict]:
    """Newest-first readings for the dashboard payload (enterprise only).

    Each row whose predecessor is in the SAME UTC month carries
    delta_official / delta_measured (recomputed live over [t_prev, t_cur], so
    backfills keep sharpening history) and coverage_pct (None when Δofficial
    is ~0 — a ratio against nothing is noise). The newest current-month row
    also carries measured_since: our live cost since that reading; its
    official counterpart is unknowable until the next reading.
    """
    if scope != "enterprise":
        return []
    rows = [dict(r) for r in conn.execute(
        "SELECT id, amount_usd, measured_usd, month, recorded_at, "
        "recorded_epoch, note FROM billing_readings WHERE scope='enterprise' "
        "ORDER BY recorded_epoch DESC LIMIT ?", (limit,))]

    now = _utc_now()
    out = []
    for i, r in enumerate(rows):
        item = {
            "id": r["id"],
            "amount_usd": round(r["amount_usd"], 2),
            "measured_usd": (round(r["measured_usd"], 2)
                             if r["measured_usd"] is not None else None),
            "month": r["month"],
            "recorded_at": r["recorded_at"],
            "recorded_epoch": r["recorded_epoch"],
            "note": r["note"],
            "delta_official": None,
            "delta_measured": None,
            "coverage_pct": None,
        }
        prev = rows[i + 1] if i + 1 < len(rows) else None
        if prev and prev["month"] == r["month"]:
            d_off = r["amount_usd"] - prev["amount_usd"]
            d_meas = compute_window_cost(
                conn, prev["recorded_epoch"], r["recorded_epoch"], "enterprise")
            item["delta_official"] = round(d_off, 2)
            item["delta_measured"] = round(d_meas, 2)
            if d_off > 0.005:
                item["coverage_pct"] = round(100.0 * d_meas / d_off, 1)
        if i == 0 and r["month"] == now.strftime("%Y-%m"):
            item["measured_since"] = round(compute_window_cost(
                conn, r["recorded_epoch"], now.timestamp() + 1, "enterprise"), 2)
        out.append(item)
    return out
