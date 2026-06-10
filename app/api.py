"""GET /api/stats — returns dashboard JSON blob.
GET /api/rate-limits — returns scope-filtered weekly spend (rolling 7-day window).
Personal scope additionally returns oauth gauge fields when available.
"""

import json
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from .aggregator import build_dashboard_data, get_cache_version
from .auth import require_dashboard_auth
from .config import IDLE_THRESHOLD_S
from .cost_windows import compute_window_cost
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


def _resolve_scope(requested):
    """Resolve scope, reading exception classes fresh from app.config each call.

    This guards against importlib.reload(app.config) invalidating the class
    references captured at module import time — reload creates new class objects
    so 'except InvalidScope' would fail to catch the freshly-raised exception.
    By importing from sys.modules['app.config'] at call time we always compare
    the same class objects that resolve_scope raises.
    """
    import sys
    cfg = sys.modules["app.config"]
    try:
        return cfg.resolve_scope(requested)
    except cfg.InvalidScope:
        raise HTTPException(status_code=400, detail=f"invalid scope: {requested!r}")
    except cfg.ScopeLocked:
        raise HTTPException(status_code=403, detail=f"instance is locked to scope {cfg.LOCKED_SCOPE!r}")


router = APIRouter()


@router.get("/api/stats/version", dependencies=[Depends(require_dashboard_auth)])
async def stats_version():
    return {"version": get_cache_version()}


@router.get("/api/stats", dependencies=[Depends(require_dashboard_auth)])
async def stats(scope: Optional[str] = Query(default=None)):
    effective = _resolve_scope(scope)
    data = build_dashboard_data(effective)
    return JSONResponse(content=data)


@router.get("/api/rate-limits", dependencies=[Depends(require_dashboard_auth)])
async def rate_limits(scope: Optional[str] = Query(default=None)):
    """Return scope-filtered weekly spend over a rolling 7-day window.

    Defaults to enterprise scope. Pass ?scope=personal for personal view.

    OAuth gauge contract (scope-gated):
    - personal scope on a non-enterprise-locked instance, with an oauth_usage
      meta row present -> weekly_budget.oauth carries the Max-subscription
      gauge fields (weekly_pct, five_hour_pct, per-model pcts, extra_usage, ...).
    - enterprise scope, enterprise-locked instance, or no meta row -> the
      'oauth' key is NEVER present (compliance-facing invariant).
    """
    effective = _resolve_scope(scope)
    import sys
    pred = sys.modules["app.config"].scope_predicate(effective)
    conn = get_conn()
    now = time.time()
    week_start_epoch = now - 7 * 24 * 3600
    window_end = now

    # Cost: delegate to compute_window_cost which deduplicates by request_id
    # and correctly applies fast/geo pricing modifiers (speed, inference_geo).
    week_cost = compute_window_cost(conn, week_start_epoch, window_end, scope=effective)

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
        f"AND {pred} "
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
        "SUM(cc) as cc, SUM(cr) as cr, SUM(c5m) as c5m, SUM(c1h) as c1h, "
        "SUM(ws) as ws "
        "FROM ("
        "  SELECT MIN(ts_epoch) as first_ts, model, request_id, "
        "  MAX(input_tokens) as inp, MAX(output_tokens) as outp, "
        "  MAX(cache_creation_tokens) as cc, MAX(cache_read_tokens) as cr, "
        "  MAX(cache_ephemeral_5m) as c5m, MAX(cache_ephemeral_1h) as c1h, "
        "  MAX(web_search_requests) as ws, "
        "  MAX(speed) as speed, MAX(inference_geo) as inference_geo "
        "  FROM events WHERE type='assistant' AND model IS NOT NULL "
        "  AND model != '<synthetic>' AND request_id IS NOT NULL "
        f"  AND {pred} "
        "  AND ts_epoch>=? AND ts_epoch<? "
        "  GROUP BY model, request_id"
        ") GROUP BY h, model, speed, inference_geo",
        (week_start_epoch, week_start_epoch, window_end),
    ):
        dm = display_model(r["model"])
        c = compute_cost(
            dm, r["inp"] or 0, r["outp"] or 0,
            r["cc"] or 0, r["cr"] or 0,
            r["speed"], r["inference_geo"],
            cw_5m=r["c5m"] or 0, cw_1h=r["c1h"] or 0,
            web_search=r["ws"] or 0)
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

    # ── Personal-scope OAuth gauge fields ──────────────────────────────────
    # Only attach when this is a personal request on an instance that is NOT
    # enterprise-locked.  Enterprise scope (or locked instance): NO oauth key.
    import sys
    _cfg = sys.modules["app.config"]
    if effective == "personal" and _cfg.LOCKED_SCOPE != "enterprise":
        oauth_row = conn.execute(
            "SELECT value FROM meta WHERE key='oauth_usage'"
        ).fetchone()
        if oauth_row:
            try:
                stored = json.loads(oauth_row["value"])
                usage = stored.get("data", {})
                updated_at = stored.get("updated_at", "")

                seven_day = usage.get("seven_day") or {}
                five_hour = usage.get("five_hour") or {}
                seven_day_sonnet = usage.get("seven_day_sonnet") or {}
                seven_day_opus = usage.get("seven_day_opus") or {}
                extra = usage.get("extra_usage") or {}

                oauth_block = {
                    "weekly_pct": seven_day.get("utilization", 0),
                    "weekly_resets_at": _scrub_to_minute(seven_day.get("resets_at", "")),
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
                if updated_at:
                    try:
                        oauth_block["updated_at_epoch"] = datetime.fromisoformat(
                            updated_at.replace("Z", "+00:00")
                        ).timestamp()
                    except (ValueError, TypeError):
                        pass
                weekly_budget["oauth"] = oauth_block
            except (ValueError, KeyError, TypeError):
                pass  # malformed row — no oauth key

    return JSONResponse(content={"weekly_budget": weekly_budget})
