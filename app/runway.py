"""GET /api/runway, /api/runway/trajectory and /api/window-history (/next).

/api/runway reshapes the lean /api/rate-limits build (app/api.py
_rate_limits_payload) into one windows[] list with the BUILD.md rules
(app/runway_rules.py). It runs no SQL of its own, so the gauges, pace,
TIGHT and dollars always agree with the old dashboard.

Failure policy: one failing part (Claude, providers, month budget) is
logged and left out; the endpoint never answers 500 for it.
"""

import re
import sys
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from .api import _rate_limits_payload, _resolve_scope
from .auth import require_dashboard_auth
from .db import read_conn
from .month_trajectory import month_points
from .monthly_budget import monthly_budget_block
from .provider_usage import provider_usage_block
from .runway_history import claude_trajectory, window_history
from .runway_rules import binding, verdict, waitable
from .runway_windows import (budget_block, claude_windows, month_window,
                             provider_windows)

router = APIRouter()

_WINDOW_ID_RE = re.compile(r"^(claude|codex|opencode-go):[a-z0-9_:\-]{1,64}$")
_BUCKET_RE = re.compile(r"^[a-z0-9_:]{1,64}$")


def log(msg, *args):
    formatted = msg % args if args else msg
    print(f"[runway] {formatted}", flush=True)


def _part(name, build):
    try:
        return build()
    except Exception as e:
        log("%s failed: %r", name, e)
        return None


def build_runway(scope, conn):
    """The /api/runway body for one resolved scope."""
    payload = _part("rate-limits", lambda: _rate_limits_payload(
        scope, conn, lean=True)) or {}
    wb = payload.get("weekly_budget") or {}
    now = wb.get("updated_at_epoch") or time.time()
    month = wb.get("monthly_budget")
    windows = [
        *(_part("claude", lambda: claude_windows(wb.get("oauth"), now)) or []),
        *(_part("providers",
                lambda: provider_windows(wb.get("providers"), now)) or []),
    ]
    if month:
        mw = _part("month", lambda: month_window(month, now))
        windows = ([mw] if mw else []) + windows
    windows = [{**w, "waitable": waitable(w, now), "verdict": verdict(w)}
               for w in windows]
    return {
        "generated_at": now,
        "scope": scope,
        "binding": binding(windows),
        "windows": windows,
        "budget": (_part("budget", lambda: budget_block(month))
                   if month else None),
    }


@router.get("/api/runway", dependencies=[Depends(require_dashboard_auth)])
async def runway(scope: Optional[str] = Query(default=None)):
    effective = _resolve_scope(scope)
    return await run_in_threadpool(_runway_response, effective)


def _runway_response(scope):
    # One snapshot, as /api/rate-limits: trends read history up to the
    # meta observation time.
    with read_conn(snapshot=True) as conn:
        return build_runway(scope, conn)


def _personal_limits_allowed(effective):
    cfg = sys.modules["app.config"]  # fresh read: importlib.reload safety
    return effective == "personal" and cfg.LOCKED_SCOPE != "enterprise"


def _bounds_only(conn, effective, provider, now, window_id):
    """Provider windows keep no per-poll history: only their bounds are
    known, so points and resets stay empty. The month meter has its billing
    readings (app/month_trajectory.py)."""
    if provider == "claude":
        block = (monthly_budget_block(conn, now)
                 if effective == "enterprise" else None)
        windows = [month_window(block, now)] if block else []
        if block and window_id == "claude:month":
            w = windows[0]
            return {"points": month_points(conn, w["window_start"],
                                           block["month"],
                                           block["budget_usd"], now),
                    "resets": [], "start": w["window_start"],
                    "end": w["resets_at"]}
    else:
        windows = provider_windows(provider_usage_block(
            effective, now, conn=conn, include_costs=False), now)
    w = next((w for w in windows if w["id"] == window_id), None)
    if w is None:
        return None
    return {"points": [], "resets": [], "start": w["window_start"],
            "end": w["resets_at"]}


EMPTY_TRAJECTORY = {"points": [], "resets": [], "start": None, "end": None}


def _trajectory(effective, window_id):
    provider, key = window_id.split(":", 1)
    if (provider == "claude" and key != "month"
            and not _personal_limits_allowed(effective)):
        # Personal Max limit data: an enterprise view gets a neutral 404,
        # as /api/limit-history does.
        return None
    try:
        return _trajectory_body(effective, provider, key, window_id)
    except Exception as e:
        # The chart degrades to empty; the rest of the view keeps working.
        log("trajectory %s failed: %r", window_id, e)
        return dict(EMPTY_TRAJECTORY)


def _trajectory_body(effective, provider, key, window_id):
    now = time.time()
    with read_conn(snapshot=True) as conn:
        if provider == "claude" and key != "month":
            return claude_trajectory(conn, key, now)
        return _bounds_only(conn, effective, provider, now, window_id)


@router.get("/api/runway/trajectory",
            dependencies=[Depends(require_dashboard_auth)])
async def runway_trajectory(window: str = Query(...),
                            scope: Optional[str] = Query(default=None)):
    """{points: [[t, pct]], resets: [t], start, end} for the current window.

    resets are the 5 h window boundaries inside a long Claude window.
    """
    effective = _resolve_scope(scope)
    if not _WINDOW_ID_RE.fullmatch(window or ""):
        raise HTTPException(status_code=400, detail="invalid window")
    body = await run_in_threadpool(_trajectory, effective, window)
    if body is None:
        raise HTTPException(status_code=404, detail="not found")
    return body


@router.get("/api/window-history",
            dependencies=[Depends(require_dashboard_auth)])
async def runway_window_history(bucket: str = Query(default="seven_day"),
                                scope: Optional[str] = Query(default=None)):
    """Ended windows of one Claude bucket (personal Max data only)."""
    effective = _resolve_scope(scope)
    if not _personal_limits_allowed(effective):
        raise HTTPException(status_code=404, detail="not found")
    if not _BUCKET_RE.fullmatch(bucket or ""):
        raise HTTPException(status_code=400, detail="invalid bucket")

    def build():
        try:
            with read_conn() as conn:
                return window_history(conn, effective, bucket)
        except Exception as e:
            log("window-history %s failed: %r", bucket, e)
            return {"windows": []}
    return await run_in_threadpool(build)
