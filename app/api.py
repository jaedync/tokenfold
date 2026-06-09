"""GET /api/stats — returns dashboard JSON blob.
GET /api/rate-limits — returns enterprise-only weekly spend (rolling 7-day window).
"""

import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from .aggregator import build_dashboard_data, get_cache_version
from .config import ENTERPRISE_PRED, IDLE_THRESHOLD_S
from .cost_windows import compute_window_cost
from .db import get_conn
from .pricing import compute_cost, display_model


router = APIRouter()


@router.get("/api/stats/version")
async def stats_version():
    return {"version": get_cache_version()}


@router.get("/api/stats")
async def stats():
    data = build_dashboard_data()
    return JSONResponse(content=data)


@router.get("/api/rate-limits")
async def rate_limits():
    """Return enterprise-only weekly spend over a rolling 7-day window.

    Completely decoupled from the personal Max account's OAuth usage row.
    Personal consumer-account gauge fields are intentionally absent —
    this is a compliance-facing, enterprise-only view.
    """
    conn = get_conn()
    now = time.time()
    week_start_epoch = now - 7 * 24 * 3600
    window_end = now

    # Cost: delegate to compute_window_cost which deduplicates by request_id
    # and correctly applies fast/geo pricing modifiers (speed, inference_geo).
    week_cost = compute_window_cost(conn, week_start_epoch, window_end)

    # Active time: sum gaps within rolling window
    week_active_s = 0.0
    prev_evt = None
    for e in conn.execute(
        "SELECT session_id, ts_epoch, type, is_sidechain, "
        "has_tool_use, has_tool_result, agent_id "
        "FROM events "
        "WHERE ts_epoch>=? AND ts_epoch<? "
        "AND type IN ('user','assistant') "
        "AND is_sidechain=0 AND agent_id IS NULL "
        f"AND {ENTERPRISE_PRED} "
        "ORDER BY session_id, ts_epoch",
        (week_start_epoch, window_end),
    ):
        if prev_evt and prev_evt["session_id"] == e["session_id"]:
            gap = e["ts_epoch"] - prev_evt["ts_epoch"]
            if 0 < gap < IDLE_THRESHOLD_S:
                week_active_s += gap
        prev_evt = e

    # Hourly cost breakdown for pace chart — mirrors aggregator._build_hourly's
    # cost query with speed + inference_geo so fast/geo events are correctly priced.
    hourly_costs = []
    for r in conn.execute(
        "SELECT CAST((first_ts - ?) / 3600 AS INTEGER) as h, "
        "model, speed, inference_geo, "
        "SUM(inp) as inp, SUM(outp) as outp, "
        "SUM(cc) as cc, SUM(cr) as cr "
        "FROM ("
        "  SELECT MIN(ts_epoch) as first_ts, model, request_id, "
        "  MAX(input_tokens) as inp, MAX(output_tokens) as outp, "
        "  MAX(cache_creation_tokens) as cc, MAX(cache_read_tokens) as cr, "
        "  MAX(speed) as speed, MAX(inference_geo) as inference_geo "
        "  FROM events WHERE type='assistant' AND model IS NOT NULL "
        "  AND model != '<synthetic>' AND request_id IS NOT NULL "
        f"  AND {ENTERPRISE_PRED} "
        "  AND ts_epoch>=? AND ts_epoch<? "
        "  GROUP BY model, request_id"
        ") GROUP BY h, model, speed, inference_geo",
        (week_start_epoch, week_start_epoch, window_end),
    ):
        dm = display_model(r["model"])
        c = compute_cost(
            dm, r["inp"] or 0, r["outp"] or 0,
            r["cc"] or 0, r["cr"] or 0,
            r["speed"], r["inference_geo"])
        if c > 0:
            h_idx = r["h"]
            found = False
            for hc in hourly_costs:
                if hc["h"] == h_idx:
                    hc["c"] = round(hc["c"] + c, 4)
                    found = True
                    break
            if not found:
                hourly_costs.append({"h": h_idx, "c": round(c, 4)})

    weekly_budget = {
        "source": "events",
        "window": "rolling_7d",
        "week_cost": round(week_cost, 2),
        "week_active_s": round(week_active_s),
        "hourly_costs": hourly_costs,
        "updated_at_epoch": now,
    }

    return JSONResponse(content={"weekly_budget": weekly_budget})
