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
from .pricing import compute_cost, display_model, effective_geo
from .usage_buckets import normalize_usage_buckets


def _iso_to_epoch(iso_str: Optional[str]) -> Optional[float]:
    """Parse an ISO-8601 string to epoch seconds; None on ANY failure.

    Also imported by app/limit_readings.py (module-level `from .api import
    ..._iso_to_epoch`) — that direction is acyclic (this module never imports
    limit_readings at module scope, only lazily inside the rate_limits route
    body below), so the helper lives here once rather than being duplicated
    (Fix 9).
    """
    if not isinstance(iso_str, str) or not iso_str:
        return None
    try:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def _active_seconds(conn, pred, start, end):
    """Sum intra-session gaps below IDLE_THRESHOLD_S over [start, end) for the
    given scope predicate. Shared by the rolling-7d week_active_s and the
    limit-window active_s so both windows accumulate active time identically.
    """
    total = 0.0
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
        (start, end),
    ):
        if prev_evt and prev_evt["session_id"] == e["session_id"]:
            gap = e["ts_epoch"] - prev_evt["ts_epoch"]
            if 0 < gap < IDLE_THRESHOLD_S:
                total += gap
        prev_evt = e
    return total


def _scrub_to_minute_or_none(iso_str: Optional[str]) -> Optional[str]:
    """Truncate an ISO-8601 timestamp to whole-minute UTC precision.

    Removes subsecond and second precision so a per-account microsecond
    offset can't fingerprint the account across responses.

    Fails CLOSED: returns None when the input is empty or unparseable, so a
    raw (full-precision, fingerprintable) value can never pass through to
    the response.
    """
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
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
      gauge fields (weekly_pct, five_hour_pct, extra_usage, ...) plus
      'buckets': the normalized bucket list from usage_buckets (limits[]
      primary, legacy dicts fallback) as [{key, label, pct, resets_at}] —
      per-model limits appear as 'scoped:<model>' keys. The main
      weekly/five_hour gauge fields derive from the SAME merged buckets, so
      a limits[]-only payload (legacy dicts nulled) still populates them.
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

    # Active time: sum gaps within rolling window (shared with limit_window).
    week_active_s = _active_seconds(conn, pred, week_start_epoch, window_end)

    # Hourly cost breakdown for pace chart — mirrors aggregator._build_hourly's
    # cost query with speed + inference_geo so fast/geo events are correctly priced.
    hourly_costs = []
    for r in conn.execute(
        "SELECT CAST((first_ts - ?) / 3600 AS INTEGER) as h, "
        "model, speed, inference_geo, MIN(first_ts) as min_ts, "
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
        # Era representative = the group's earliest event ts (data-derived, so
        # the same historical events never re-price as the sliding week window
        # moves; a group straddling the boundary prices at its start era —
        # accepted hour-scale approximation).
        c = compute_cost(
            dm, r["inp"] or 0, r["outp"] or 0,
            r["cc"] or 0, r["cr"] or 0,
            r["speed"],
            effective_geo(r["inference_geo"],
                          enterprise=(effective == "enterprise")),
            cw_5m=r["c5m"] or 0, cw_1h=r["c1h"] or 0,
            web_search=r["ws"] or 0,
            ts_epoch=r["min_ts"])
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

                extra = usage.get("extra_usage") or {}

                # Normalize ONCE (limits[] primary, legacy dicts fallback):
                # the main weekly/5h gauges AND the buckets list all read the
                # merged view, so a limits[]-only payload (legacy dicts
                # nulled, as prod already does per-model) still populates
                # every gauge. resets_at leaves the normalizer RAW — scrub
                # (fail-closed) at this boundary.
                normalized = normalize_usage_buckets(usage)
                by_key = {b["key"]: b for b in normalized}
                seven_day = by_key.get("seven_day") or {}
                five_hour = by_key.get("five_hour") or {}

                buckets = [
                    {
                        "key": b["key"],
                        "label": b["label"],
                        "pct": b["utilization"],
                        "resets_at": _scrub_to_minute_or_none(b["resets_at"]),
                    }
                    for b in normalized
                ]

                oauth_block = {
                    "weekly_pct": seven_day.get("utilization", 0),
                    "weekly_resets_at":
                        _scrub_to_minute_or_none(seven_day.get("resets_at")) or "",
                    "five_hour_pct": five_hour.get("utilization", 0),
                    "five_hour_resets_at":
                        _scrub_to_minute_or_none(five_hour.get("resets_at")) or "",
                    "buckets": buckets,
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

                # Sub-window burn / ETA / pace / series per bucket (D2).
                # Bucket-name-generic: every distinct bucket historized in the
                # last 7d gets a trend entry, so a future scoped bucket appears
                # with zero code change. Omitted entirely when nothing is
                # historized yet (response shape unchanged). Lazy import breaks
                # the api <- limit_trends <- limit_readings <- api cycle.
                #
                # Narrow try/except (Fix 6): this used to sit inside only the
                # broad except below — a bug in trend math would silently
                # delete EVERY gauge (weekly_pct, buckets, extra_usage, ...),
                # not just the trend entries. Scope the blast radius to the
                # 'trend' key alone.
                try:
                    from .limit_trends import bucket_trend, distinct_buckets
                    present = distinct_buckets(conn, now)
                    if present:
                        oauth_block["trend"] = {
                            b: bucket_trend(conn, b, now) for b in present
                        }
                except Exception as e:
                    print(f"[rate-limits] trend computation failed: {e}",
                          flush=True)

                # Consistent "budget left" inputs (D5): cost + active time over
                # the ACTUAL weekly limit window [weekly_resets_at - 7d, now],
                # so the template no longer divides rolling-7d cost by
                # limit-window pct. Omitted when seven_day.resets_at is
                # unparseable. start_epoch is minute-floored (limit timestamps
                # never leave the server at sub-minute precision).
                #
                # Narrow try/except (Fix 6): same reasoning as the trend block
                # above — a bug here must only drop 'limit_window', never the
                # whole oauth block.
                try:
                    weekly_resets_epoch = _iso_to_epoch(
                        seven_day.get("resets_at"))
                    if weekly_resets_epoch is not None:
                        lw_start = (((weekly_resets_epoch // 60) * 60.0)
                                    - 7 * 86400)
                        # F1: a GRANTED mid-window reset voids pre-grant
                        # usage — the window's spend starts at the latest
                        # detected reset, not at resets_at − 7d (which only
                        # moves on NATURAL rollovers). persistent_resets
                        # (not raw detect_resets): a stale client replay
                        # must not move the cost window (review M1) — its
                        # active-window guard still means these events are
                        # exactly the granted kind. Lazy import:
                        # limit_readings imports api at module level, so
                        # api must never import it back at module level
                        # (same cycle-break as limit_trends above).
                        # floor_reset_events keeps start_epoch
                        # minute-floored either way.
                        from .limit_readings import (floor_reset_events,
                                                     persistent_resets)
                        lr_rows = conn.execute(
                            "SELECT bucket, fetched_epoch, utilization, "
                            "resets_at_epoch FROM limit_readings "
                            "WHERE bucket='seven_day' AND fetched_epoch>=? "
                            "ORDER BY fetched_epoch ASC",
                            (lw_start,)).fetchall()
                        granted = floor_reset_events(
                            persistent_resets(lr_rows))
                        if granted:
                            lw_start = max(lw_start,
                                           granted[-1]["at_epoch"])
                        oauth_block["limit_window"] = {
                            "start_epoch": lw_start,
                            "cost": round(compute_window_cost(
                                conn, lw_start, now, scope=effective), 2),
                            "active_s": round(
                                _active_seconds(conn, pred, lw_start, now)),
                        }
                except Exception as e:
                    print(f"[rate-limits] limit_window computation failed: "
                          f"{e}", flush=True)

                weekly_budget["oauth"] = oauth_block
            except (ValueError, KeyError, TypeError):
                pass  # malformed row — no oauth key

    return JSONResponse(content={"weekly_budget": weekly_budget})
