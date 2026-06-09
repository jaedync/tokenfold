"""GET /api/ha — flat, scrubbed metrics for Home Assistant REST sensors.

Designed for HA's `rest:` platform: one HTTP fetch, many sensors via
value_template. All timestamps truncated to minute precision so the
microsecond offsets of OAuth resets_at cannot fingerprint the account.
"""

import json
import sqlite3
import time
from datetime import datetime, timezone

import app.config as config
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from .aggregator import build_dashboard_data
from .cost_windows import compute_window_cost
from .db import get_conn

router = APIRouter()

FIVE_HOURS_S = 5 * 3600
SEVEN_DAYS_S = 7 * 24 * 3600
MIN_PCT_FOR_IMPLIED_LIMIT = 5.0  # below this, API rounding noise swamps signal


def _truncate_to_minute(iso_str: str) -> tuple[str, float] | tuple[None, None]:
    """Parse an ISO-8601 timestamp, truncate to whole UTC minute.

    Returns (iso_string_with_00_seconds_and_no_microseconds, epoch_float).
    Returns (None, None) if the input is empty or malformed.
    """
    if not iso_str:
        return None, None
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None, None
    dt = dt.astimezone(timezone.utc).replace(second=0, microsecond=0)
    return dt.isoformat(), dt.timestamp()


def _implied_limit(spend_usd: float, pct_used: float | None) -> float | None:
    """spend / (pct/100), or None when utilization is too low to be trustworthy."""
    if pct_used is None or pct_used < MIN_PCT_FOR_IMPLIED_LIMIT:
        return None
    return round(spend_usd / (pct_used / 100.0), 2)


def _window_block(
    conn: sqlite3.Connection,
    raw_resets_at: str,
    pct_used: float | None,
    window_seconds: int,
    now_epoch: float,
    scope: str = "personal",
) -> dict | None:
    """Build a five_hour / weekly sub-object, or None if resets_at is missing.

    These windows reflect the personal Max budget utilization reported by the
    OAuth API, so spend must be PERSONAL-scoped for implied_limit arithmetic
    to be coherent: implied_limit = personal_spend / (personal_pct / 100).
    """
    resets_iso, resets_epoch = _truncate_to_minute(raw_resets_at)
    if resets_iso is None:
        return None
    start_epoch = resets_epoch - window_seconds
    spend = round(compute_window_cost(conn, start_epoch, resets_epoch, scope=scope), 2)
    return {
        "pct_used": pct_used if pct_used is not None else 0.0,
        "spend_usd": spend,
        "implied_limit_usd": _implied_limit(spend, pct_used),
        "resets_at": resets_iso,
        "resets_in_s": max(0, int(resets_epoch - now_epoch)),
    }


@router.get("/api/ha")
async def ha_metrics():
    """Flat metrics feed for Home Assistant REST sensors."""
    conn = get_conn()
    now_epoch = time.time()

    # Read the scope lock fresh so tests can monkeypatch app.config.LOCKED_SCOPE.
    locked_scope = config.LOCKED_SCOPE

    # Cost totals — enterprise-scoped (fail-closed default); always populated.
    # NOTE: cost_today_usd and cost_total_usd are enterprise-scoped.
    dash = build_dashboard_data()
    cost_today = round((dash.get("today") or {}).get("cost", 0.0) or 0.0, 2)
    cost_total = round(dash.get("total_cost", 0.0) or 0.0, 2)

    five_hour_block = None
    weekly_block = None
    updated_at_epoch = None

    if locked_scope != "enterprise":
        # OAuth usage reflects the PERSONAL Max budget — only emit when not
        # enterprise-locked so personal account activity can't surface on a
        # compliance instance. Skip the meta read entirely so a stale row from
        # before the lock can't leak through.
        row = conn.execute(
            "SELECT value FROM meta WHERE key='oauth_usage'"
        ).fetchone()

        if row:
            try:
                stored = json.loads(row["value"])
            except (ValueError, TypeError):
                stored = None

            if stored:
                usage = stored.get("data") or {}
                fh = usage.get("five_hour") or {}
                sd = usage.get("seven_day") or {}

                # Window spend is personal-scoped: these windows track the personal
                # Max utilization, so spend must match (personal) for
                # implied_limit = spend / (pct/100) to be coherent.
                five_hour_block = _window_block(
                    conn,
                    fh.get("resets_at", ""),
                    fh.get("utilization"),
                    FIVE_HOURS_S,
                    now_epoch,
                    scope="personal",
                )
                weekly_block = _window_block(
                    conn,
                    sd.get("resets_at", ""),
                    sd.get("utilization"),
                    SEVEN_DAYS_S,
                    now_epoch,
                    scope="personal",
                )

                updated_at_iso = stored.get("updated_at", "")
                if updated_at_iso:
                    try:
                        ua = datetime.fromisoformat(updated_at_iso).timestamp()
                        # Round to nearest 10 seconds for symmetry with resets_at.
                        updated_at_epoch = int(round(ua / 10.0) * 10)
                    except (ValueError, TypeError):
                        pass

    return JSONResponse(content={
        "cost_today_usd": cost_today,
        "cost_total_usd": cost_total,
        "five_hour": five_hour_block,
        "weekly": weekly_block,
        "updated_at_epoch": updated_at_epoch,
    })
