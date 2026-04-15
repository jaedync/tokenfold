"""GET /api/stats — returns dashboard JSON blob.
GET /api/rate-limits — returns rate limit / usage data.
"""

import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from .aggregator import build_dashboard_data, get_cache_version
from .config import IDLE_THRESHOLD_S, TZ_NAME
from .db import get_conn
from .pricing import compute_cost, display_model


def _scrub_to_minute(iso_str: str) -> str:
    """Truncate an ISO-8601 timestamp to whole-minute UTC precision.

    Removes subsecond and second precision so a per-account microsecond
    offset can't fingerprint the account across responses.

    Returns the input unchanged if empty or unparseable.
    """
    if not iso_str:
        return iso_str
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return iso_str
    return (
        dt.astimezone(timezone.utc).replace(second=0, microsecond=0).isoformat()
    )


router = APIRouter()
TZ = ZoneInfo(TZ_NAME)


@router.get("/api/stats/version")
async def stats_version():
    return {"version": get_cache_version()}


@router.get("/api/stats")
async def stats():
    data = build_dashboard_data()
    return JSONResponse(content=data)


@router.get("/api/rate-limits")
async def rate_limits():
    """Return rate limit data from OAuth usage stored in meta table."""
    conn = get_conn()
    oauth_row = conn.execute(
        "SELECT value FROM meta WHERE key='oauth_usage'"
    ).fetchone()

    if not oauth_row:
        return JSONResponse(content={"weekly_budget": None})

    try:
        stored = json.loads(oauth_row["value"])
    except (ValueError, KeyError):
        return JSONResponse(content={"weekly_budget": None})

    usage = stored.get("data", {})
    updated_at = stored.get("updated_at", "")

    seven_day = usage.get("seven_day") or {}
    five_hour = usage.get("five_hour") or {}
    seven_day_sonnet = usage.get("seven_day_sonnet") or {}
    seven_day_opus = usage.get("seven_day_opus") or {}
    extra = usage.get("extra_usage") or {}

    resets_at_iso = seven_day.get("resets_at", "")
    weekly_budget = {
        "source": "oauth",
        "weekly_pct": seven_day.get("utilization", 0),
        "weekly_resets_at": _scrub_to_minute(resets_at_iso),
        "five_hour_pct": five_hour.get("utilization", 0),
        "five_hour_resets_at": _scrub_to_minute(five_hour.get("resets_at", "")),
        "sonnet_pct": seven_day_sonnet.get("utilization", 0) if seven_day_sonnet else None,
        "opus_pct": seven_day_opus.get("utilization", 0) if seven_day_opus else None,
        "extra_usage": {
            "enabled": extra.get("is_enabled", False),
            "monthly_limit_cents": extra.get("monthly_limit", 0),
            "used_cents": extra.get("used_credits", 0),
            "pct": extra.get("utilization", 0),
        } if extra else None,
        "updated_at": updated_at,
    }

    # Add epoch for staleness indicator
    if updated_at:
        try:
            weekly_budget["updated_at_epoch"] = datetime.fromisoformat(updated_at).timestamp()
        except (ValueError, TypeError):
            pass

    # Precise weekly window stats (cost + active time)
    if resets_at_iso:
        try:
            reset_dt = datetime.fromisoformat(resets_at_iso.replace("Z", "+00:00"))
            reset_epoch = reset_dt.timestamp()
            week_start_epoch = reset_epoch - 7 * 24 * 3600

            # Cost: deduplicate by request_id, filter by epoch window
            week_cost = 0.0
            for r in conn.execute(
                "SELECT model, SUM(inp) as inp, SUM(out) as out, "
                "SUM(cc) as cc, SUM(cr) as cr "
                "FROM ("
                "  SELECT model, request_id, "
                "  MAX(input_tokens) as inp, MAX(output_tokens) as out, "
                "  MAX(cache_creation_tokens) as cc, MAX(cache_read_tokens) as cr "
                "  FROM events WHERE type='assistant' AND model IS NOT NULL "
                "  AND model != '<synthetic>' AND request_id IS NOT NULL "
                "  AND ts_epoch>=? AND ts_epoch<? "
                "  GROUP BY model, request_id"
                ") GROUP BY model",
                (week_start_epoch, reset_epoch),
            ):
                dm = display_model(r["model"])
                week_cost += compute_cost(
                    dm, r["inp"] or 0, r["out"] or 0,
                    r["cc"] or 0, r["cr"] or 0)

            # Active time: sum gaps within window
            week_active_s = 0.0
            prev_evt = None
            for e in conn.execute(
                "SELECT session_id, ts_epoch, type, is_sidechain, "
                "has_tool_use, has_tool_result, agent_id "
                "FROM events "
                "WHERE ts_epoch>=? AND ts_epoch<? "
                "AND type IN ('user','assistant') "
                "AND is_sidechain=0 AND agent_id IS NULL "
                "ORDER BY session_id, ts_epoch",
                (week_start_epoch, reset_epoch),
            ):
                if (prev_evt and
                        prev_evt["session_id"] == e["session_id"]):
                    gap = e["ts_epoch"] - prev_evt["ts_epoch"]
                    if 0 < gap < IDLE_THRESHOLD_S:
                        week_active_s += gap
                prev_evt = e

            weekly_budget["week_cost"] = round(week_cost, 2)
            weekly_budget["week_active_s"] = round(week_active_s)
            weekly_budget["week_start_epoch"] = week_start_epoch

            # Hourly cost breakdown for pace chart
            hourly_costs = []
            for r in conn.execute(
                "SELECT CAST((first_ts - ?) / 3600 AS INTEGER) as h, "
                "model, SUM(inp) as inp, SUM(outp) as outp, "
                "SUM(cc) as cc, SUM(cr) as cr "
                "FROM ("
                "  SELECT MIN(ts_epoch) as first_ts, model, request_id, "
                "  MAX(input_tokens) as inp, MAX(output_tokens) as outp, "
                "  MAX(cache_creation_tokens) as cc, MAX(cache_read_tokens) as cr "
                "  FROM events WHERE type='assistant' AND model IS NOT NULL "
                "  AND model != '<synthetic>' AND request_id IS NOT NULL "
                "  AND ts_epoch>=? AND ts_epoch<? "
                "  GROUP BY model, request_id"
                ") GROUP BY h, model",
                (week_start_epoch, week_start_epoch, reset_epoch),
            ):
                dm = display_model(r["model"])
                c = compute_cost(
                    dm, r["inp"] or 0, r["outp"] or 0,
                    r["cc"] or 0, r["cr"] or 0)
                if c > 0:
                    # Merge into hourly bucket
                    h_idx = r["h"]
                    # Find or create entry
                    found = False
                    for hc in hourly_costs:
                        if hc["h"] == h_idx:
                            hc["c"] = round(hc["c"] + c, 4)
                            found = True
                            break
                    if not found:
                        hourly_costs.append({"h": h_idx, "c": round(c, 4)})

            weekly_budget["hourly_costs"] = hourly_costs
        except (ValueError, TypeError, OSError):
            pass

    return JSONResponse(content={"weekly_budget": weekly_budget})
