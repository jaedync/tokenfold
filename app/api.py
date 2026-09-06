"""GET /api/stats — returns dashboard JSON blob.
GET /api/rate-limits — returns scope-filtered weekly spend (rolling 7-day window).
Personal scope additionally returns oauth gauge fields when available.
"""

import json
import math
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .aggregator import build_dashboard_data, get_cache_version
from .auth import require_dashboard_auth
from .config import IDLE_THRESHOLD_S
from .cost_windows import (ANTHROPIC_ONLY_SQL, compute_window_cost,
                          compute_window_cost_by_model)
from .db import read_conn
from .pricing import (REPORTED_COST_SUM_SQL, compute_cost, display_model,
                      display_model_for_row, effective_geo, reported_cost)
from .usage_buckets import normalize_usage_buckets
from .quota_freshness import quota_window_valid


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


def _bucket_window_start(conn, bucket_key, resets_epoch, window_s, observed=None):
    """Start epoch of the CURRENT limit window for one historized bucket.

    resets_at − window_s (minute-floored — limit timestamps never leave the
    server at sub-minute precision), pushed forward to the latest persistent
    GRANTED reset recorded for that bucket: a granted mid-window reset voids
    pre-grant usage, so spend anchored at resets_at − window would overcount
    (F1). corroborated_resets = persistent_resets (not raw detect_resets: a
    stale client replay must not move the cost window, review M1) plus
    account-level resets borrowed from sibling buckets — a grant zeroes
    every bucket at once, and a bucket whose own decrease is too small for
    detection still had its window cut (2026-07-09 incident). Lazy import —
    limit_readings imports api at module level, so api must never import it
    back at module scope (same cycle-break as the limit_trends import in
    rate_limits).
    """
    start = ((resets_epoch // 60) * 60.0) - window_s
    from .limit_readings import corroborated_resets, floor_reset_events
    granted = floor_reset_events(corroborated_resets(
        conn, bucket_key, start, until_epoch=observed))
    if granted:
        start = max(start, granted[-1]["at_epoch"])
    return start


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
def stats(scope: Optional[str] = Query(default=None)):
    effective = _resolve_scope(scope)
    data = build_dashboard_data(effective)
    return JSONResponse(content=data)


def _oauth_snapshot(usage, updated_at, source=None, observed_at_epoch=None):
    """Shared quota-only shape for first paint and enriched responses."""
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

    oauth = {
        "weekly_pct": seven_day.get("utilization", 0),
        "weekly_resets_at":
            _scrub_to_minute_or_none(seven_day.get("resets_at")) or "",
        "five_hour_pct": five_hour.get("utilization", 0),
        "five_hour_resets_at":
            _scrub_to_minute_or_none(five_hour.get("resets_at")) or "",
        "buckets": buckets,
        "extra_usage": {
            "enabled": extra.get("is_enabled", False),
            "monthly_limit_cents": extra.get("monthly_limit"),
            "used_cents": extra.get("used_credits"),
            "pct": extra.get("utilization"),
        } if extra else None,
        "updated_at": updated_at,
    }
    if source:
        oauth["source"] = source
    if updated_at:
        try:
            oauth["updated_at_epoch"] = datetime.fromisoformat(
                updated_at.replace("Z", "+00:00")
            ).timestamp()
        except (ValueError, TypeError):
            pass

    if (isinstance(observed_at_epoch, (float, int))
            and not isinstance(observed_at_epoch, bool)
            and math.isfinite(observed_at_epoch) and observed_at_epoch > 0):
        oauth["updated_at_epoch"] = observed_at_epoch
    return oauth, by_key


@router.get("/api/rate-limit-snapshots", dependencies=[Depends(require_dashboard_auth)])
def rate_limit_snapshots(scope: Optional[str] = Query(default=None)):
    """Quota-only first paint: meta reads, never transcript scans or network I/O."""
    effective = _resolve_scope(scope)
    from .provider_usage import provider_usage_block
    import sys
    with read_conn() as conn:
        budget = {"providers": provider_usage_block(
            effective, conn=conn, include_costs=False)}
        if effective == "personal" and sys.modules["app.config"].LOCKED_SCOPE != "enterprise":
            row = conn.execute("SELECT value FROM meta WHERE key='oauth_usage'").fetchone()
            if row:
                try:
                    stored = json.loads(row["value"])
                    budget["oauth"], _ = _oauth_snapshot(
                        stored.get("data", {}), stored.get("updated_at", ""), stored.get("source"),
                        stored.get("observed_at_epoch"))
                except (ValueError, KeyError, TypeError):
                    pass
        return JSONResponse(content={"weekly_budget": budget})


@router.get("/api/rate-limits", dependencies=[Depends(require_dashboard_auth)])
async def rate_limits(scope: Optional[str] = Query(default=None)):
    """Scope-filtered quota gauges enriched with window spend and pacing."""
    effective = _resolve_scope(scope)
    return await run_in_threadpool(_rate_limits_response, effective)


def _rate_limits_response(scope):
    with read_conn() as conn:
        return _build_rate_limits(scope, conn)


def _build_rate_limits(scope, conn):
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
    - personal scope may include 'providers' with fresh Codex/OpenCode Go
      windows and actual observed OpenCode Zen API-equivalent spend. Missing
      or stale feeds are omitted.

    Monthly budget contract (scope-gated, opposite direction):
    - enterprise scope, with a budget set via /api/enterprise-budget ->
      weekly_budget.monthly_budget carries the pacing block (see
      app/monthly_budget.py). No budget set -> key omitted entirely.
    - personal scope -> the 'monthly_budget' key is NEVER present, even if
      a budget is set (compliance-facing invariant).
    """
    effective = _resolve_scope(scope)
    import sys
    pred = sys.modules["app.config"].scope_predicate(effective)
    now = time.time()
    week_start_epoch = now - 7 * 24 * 3600
    window_end = now

    # Cost: delegate to compute_window_cost which deduplicates by request_id
    # and correctly applies fast/geo pricing modifiers (speed, inference_geo).
    # Claude-subscription spend only (anthropic_only): the Codex/OpenCode
    # gauges live in their own providers block; counting their events here
    # would inflate the Claude gauges' "spent · this window".
    week_cost = compute_window_cost(conn, week_start_epoch, window_end,
                                    scope=effective, anthropic_only=True)

    # Active time: sum gaps within rolling window (shared with limit_window).
    week_active_s = _active_seconds(conn, pred, week_start_epoch, window_end)

    # Hourly cost breakdown for pace chart — mirrors aggregator._build_hourly's
    # cost query with speed + inference_geo so fast/geo events are correctly
    # priced. Anthropic-only for the same reason as week_cost: the chart sits
    # inside the Claude weekly gauge card.
    hourly_costs = []
    for r in conn.execute(
        "SELECT CAST((first_ts - ?) / 3600 AS INTEGER) as h, "
        "model, provider, source_client, speed, inference_geo, MIN(first_ts) as min_ts, "
        "SUM(inp) as inp, SUM(outp) as outp, "
        "SUM(cc) as cc, SUM(cr) as cr, SUM(c5m) as c5m, SUM(c1h) as c1h, "
        "SUM(ws) as ws, SUM(reported_input) as reported_input, "
        "SUM(reported_output) as reported_output, SUM(reported_cache_read) as reported_cache_read, "
        "SUM(reported_cache_write) as reported_cache_write, "
        f"{REPORTED_COST_SUM_SQL} as reported_total "
        "FROM ("
        "  SELECT MIN(ts_epoch) as first_ts, model, provider, source_client, request_id, "
        "  MAX(input_tokens) as inp, MAX(output_tokens) as outp, "
        "  MAX(cache_creation_tokens) as cc, MAX(cache_read_tokens) as cr, "
        "  MAX(cache_ephemeral_5m) as c5m, MAX(cache_ephemeral_1h) as c1h, "
        "  MAX(web_search_requests) as ws, "
        "  MAX(reported_cost_input) as reported_input, MAX(reported_cost_output) as reported_output, "
        "  MAX(reported_cost_cache_read) as reported_cache_read, MAX(reported_cost_cache_write) as reported_cache_write, "
        "  MAX(reported_cost_total) as reported_total, "
        "  MAX(speed) as speed, MAX(inference_geo) as inference_geo "
        "  FROM events WHERE type='assistant' AND model IS NOT NULL "
        "  AND model != '<synthetic>' AND request_id IS NOT NULL "
        f"  AND {pred} "
        f"  AND{ANTHROPIC_ONLY_SQL}"
        "  AND ts_epoch>=? AND ts_epoch<? "
        "  GROUP BY model, provider, source_client, request_id"
        ") GROUP BY h, model, provider, source_client, speed, inference_geo",
        (week_start_epoch, week_start_epoch, window_end),
    ):
        dm = display_model_for_row(r["model"], r["provider"], r["source_client"])
        # Era representative = the group's earliest event ts (data-derived, so
        # the same historical events never re-price as the sliding week window
        # moves; a group straddling the boundary prices at its start era —
        # accepted hour-scale approximation).
        c = reported_cost(r)
        if c is None:
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

    # ── Enterprise-scope monthly $ budget pacing block ─────────────────────
    # Only attached when this is an enterprise request AND a budget is set
    # (monthly_budget_block returns None otherwise). Never attached for
    # personal scope — mirrors the oauth gating below in the opposite
    # direction (compliance-facing invariant on both sides).
    if effective == "enterprise":
        from .monthly_budget import monthly_budget_block
        block = monthly_budget_block(conn, now)
        if block is not None:
            weekly_budget["monthly_budget"] = block

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

                oauth_block, by_key = _oauth_snapshot(usage, updated_at, stored.get("source"),
                                                     stored.get("observed_at_epoch"))
                observed = oauth_block.get("updated_at_epoch")
                seven_day = by_key.get("seven_day") or {}
                five_hour = by_key.get("five_hour") or {}
                buckets = oauth_block["buckets"]

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
                        trends = {}
                        for key in present:
                            reset = _iso_to_epoch((by_key.get(key) or {}).get("resets_at"))
                            duration = 5 * 3600 if key == "five_hour" else 7 * 86400
                            if quota_window_valid(observed, now, reset, (reset or 0) - duration):
                                trends[key] = bucket_trend(conn, key, observed)
                        if trends:
                            oauth_block["trend"] = trends
                except Exception as e:
                    print(f"[rate-limits] trend computation failed: {e}",
                          flush=True)

                # Consistent "budget left" inputs (D5): cost + active time over
                # the ACTUAL weekly limit window ending at the observation,
                # so the template no longer divides rolling-7d cost by
                # limit-window pct. Omitted when seven_day.resets_at is
                # unparseable/stale/expired. start_epoch is minute-floored (limit timestamps
                # never leave the server at sub-minute precision).
                #
                # Narrow try/except (Fix 6): same reasoning as the trend block
                # above — a bug here must only drop 'limit_window', never the
                # whole oauth block.
                try:
                    weekly_resets_epoch = _iso_to_epoch(
                        seven_day.get("resets_at"))
                    if quota_window_valid(observed, now, weekly_resets_epoch,
                                          (weekly_resets_epoch or 0) - 7 * 86400):
                        # F1 window anchoring + granted-reset truncation now
                        # lives in _bucket_window_start (shared with the
                        # five_hour and scoped windows below).
                        lw_start = _bucket_window_start(
                            conn, "seven_day", weekly_resets_epoch, 7 * 86400, observed)
                        oauth_block["limit_window"] = {
                            "start_epoch": lw_start,
                            "observed_at_epoch": observed,
                            "end_epoch": observed,
                            "cost": round(compute_window_cost(
                                conn, lw_start, observed, scope=effective,
                                anthropic_only=True), 2),
                            "active_s": round(
                                _active_seconds(conn, pred, lw_start, observed)),
                        }
                except Exception as e:
                    print(f"[rate-limits] limit_window computation failed: "
                          f"{e}", flush=True)

                # Same window-anchored spend for the 5-hour gauge — the
                # dollar-budget projection needs cost over the CURRENT 5h
                # window, not a rolling figure. Skipped when resets_at has
                # already passed: the stored blob is stale (the poller lags
                # the boundary during idle), so its pct describes a window
                # that ENDED — pairing it with cost through `now` would mix
                # two windows (the client nulls its expected marker for the
                # same staleness, D6).
                try:
                    fh_resets_epoch = _iso_to_epoch(
                        five_hour.get("resets_at"))
                    if quota_window_valid(observed, now, fh_resets_epoch,
                                          (fh_resets_epoch or 0) - 5 * 3600):
                        fh_start = _bucket_window_start(
                            conn, "five_hour", fh_resets_epoch, 5 * 3600, observed)
                        oauth_block["five_hour_window"] = {
                            "start_epoch": fh_start,
                            "observed_at_epoch": observed,
                            "end_epoch": observed,
                            "cost": round(compute_window_cost(
                                conn, fh_start, observed, scope=effective,
                                anthropic_only=True), 2),
                        }
                except Exception as e:
                    print(f"[rate-limits] five_hour_window computation "
                          f"failed: {e}", flush=True)

                # Per-model dollar windows: each scoped:* bucket gains
                # window_cost = spend on THAT model family over the bucket's
                # current 7d window (weekly_scoped limits are always 7-day),
                # granted-reset truncated like limit_window. Family match on
                # display_model names via the slug's FIRST token: prod sends
                # bare family words today ('Fable' → scoped:fable), but a
                # versioned display_name ('Opus 4.8' → scoped:opus_4_8)
                # would make the raw slug miss every space-separated display
                # name (review MEDIUM) — the stem ('opus') matches the whole
                # family, which is exactly what a scoped limit governs.
                # anthropic_only for the same reason as limit_window: a
                # scoped limit is a Claude-subscription window, so a Pi row
                # that ran the same family through OpenRouter (display name
                # 'OpenRouter / Fable 5.1' still matches 'fable') is billed
                # elsewhere and must not count.
                # Per-bucket try/except: one bad bucket must not strip the
                # others' costs (or anything else in the oauth block).
                for bkt in buckets:
                    bkt_key = bkt.get("key") or ""
                    if not bkt_key.startswith("scoped:"):
                        continue
                    try:
                        sb_resets_epoch = _iso_to_epoch(by_key[bkt_key].get("resets_at"))
                        if not quota_window_valid(observed, now, sb_resets_epoch,
                                                  (sb_resets_epoch or 0) - 7 * 86400):
                            continue  # unparseable or stale — no dollars
                        sb_start = _bucket_window_start(
                            conn, bkt_key, sb_resets_epoch, 7 * 86400, observed)
                        family = bkt_key.split(":", 1)[1].split("_")[0].lower()
                        if not family:
                            continue
                        sb_by_model = compute_window_cost_by_model(
                            conn, sb_start, observed, scope=effective,
                            anthropic_only=True)
                        bkt["window_cost"] = round(sum(
                            v for k, v in sb_by_model.items()
                            if family in k.lower()), 2)
                        bkt["window_start_epoch"] = sb_start
                        bkt["window_end_epoch"] = observed
                    except Exception as e:
                        print(f"[rate-limits] scoped window cost failed "
                              f"for {bkt_key}: {e}", flush=True)

                weekly_budget["oauth"] = oauth_block
            except (ValueError, KeyError, TypeError):
                pass  # malformed row — no oauth key

    # Pi's dotfleet extension reports Codex/OpenCode subscription windows
    # independently of transcript ingest, keyed by the reporting machine's
    # account class. Each scope sees only its own snapshots, so this block is
    # attached for both scopes and is empty (omitted) when nothing of that
    # class has reported.
    try:
        from .provider_usage import provider_usage_block
        providers = provider_usage_block(effective, now, conn=conn)
        if providers:
            weekly_budget["providers"] = providers
    except Exception as e:
        print(f"[rate-limits] provider usage failed: {e}", flush=True)

    return JSONResponse(content={"weekly_budget": weekly_budget})
