"""GET /api/limit-rate: the exchange rate between list-price dollars and
usage-limit points (Exchange rate panel of /next).

The fit scans about 45 days of requests (1 to 3 s on production data), so
it never runs on the request path: a background thread recomputes at most
once per RECOMPUTE_S and the route serves the cached result. The cache
lives in memory and in meta (key ``limit_rate_cache``) so a restart serves
the last fit at once. A request waits at most WAIT_S for a running fit,
then gets the stale fit or an all-null payload whose fit_quality says why.

Only the personal subscription has quota readings, so other scopes get the
null payload (never a 404: the panel just shows "not available").
"""

import copy
import json
import logging
import sys
import threading
import time
from typing import Optional

from fastapi import APIRouter, Depends, Query

from .api import _resolve_scope
from .auth import require_dashboard_auth
from .db import get_conn, read_conn, write_txn
from .limit_peak import PEAK_TZ, peak_state
from .limit_rate_analysis import analyze
from .limit_rate_data import load_granted, load_readings, load_requests
from .quota_history import active_history_source

logger = logging.getLogger(__name__)
router = APIRouter()

CACHE_KEY = "limit_rate_cache"
CACHE_VERSION = 1
RECOMPUTE_S = 3600.0
WAIT_S = 2.0
RETRY_AFTER_ERROR_S = 600.0
LOOKBACK_S = 45 * 86400
BUCKETS = ("seven_day", "five_hour")

_lock = threading.Lock()
_state: dict = {}


def reset_state() -> None:
    """Forget the in-memory cache (tests; a fresh process starts so)."""
    with _lock:
        _state.clear()
        _state.update(entry=None, loaded=False, thread=None,
                      error_at=0.0, error=None)


reset_state()


def compute_payload(now: Optional[float] = None) -> dict:
    """Run the full fit on its own read-only connection (never the shared
    ingest connection: a 1-3 s scan there would stall every other query)."""
    now = time.time() if now is None else now
    since = now - LOOKBACK_S
    with read_conn() as conn:
        source = active_history_source(conn)
        readings = {b: load_readings(conn, b, since, source, until=now) for b in BUCKETS}
        granted = load_granted(conn, since, source, until=now)
        firsts = [r[0][0] for r in readings.values() if r]
        requests = load_requests(conn, min(firsts), now) if firsts else []
    return analyze(readings, requests, granted, now)


def _load_meta() -> Optional[dict]:
    try:
        with read_conn() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key=?", (CACHE_KEY,)).fetchone()
        if row is None:
            return None
        entry = json.loads(row[0])
        if entry.get("version") != CACHE_VERSION or "payload" not in entry:
            return None
        return entry
    except Exception:
        logger.exception("limit-rate: meta cache unreadable; recomputing")
        return None


def _store(entry: dict) -> None:
    try:
        with write_txn(get_conn()) as conn:
            conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                         (CACHE_KEY, json.dumps(entry)))
    except Exception:
        # The in-memory copy still serves; only the restart warm start is lost.
        logger.exception("limit-rate: meta cache write failed")


def _refresh() -> None:
    try:
        payload = compute_payload()
    except Exception as exc:
        logger.exception("limit-rate fit failed; retry in %.0f s", RETRY_AFTER_ERROR_S)
        with _lock:
            _state.update(error_at=time.time(), error=type(exc).__name__)
        return
    entry = {"version": CACHE_VERSION, "computed_at": time.time(), "payload": payload}
    with _lock:
        _state.update(entry=entry, error=None)
    _store(entry)


def _ensure_refresh(now: float) -> Optional[threading.Thread]:
    """Start one background fit unless one runs or a failure is recent."""
    with _lock:
        running = _state["thread"]
        if running is not None and running.is_alive():
            return running
        if now - _state["error_at"] < RETRY_AFTER_ERROR_S:
            return None
        t = threading.Thread(target=_refresh, name="limit-rate-fit", daemon=True)
        _state["thread"] = t
        t.start()
        return t


def prewarm() -> Optional[threading.Thread]:
    """Start the fit at app startup unless meta holds a fresh one.

    A deploy restarts the process; without this the first visit waits
    WAIT_S and may still read "computing". Never blocks and never raises:
    startup must not depend on the fit.
    """
    try:
        if sys.modules["app.config"].LOCKED_SCOPE == "enterprise":
            return None
        now = time.time()
        entry = _entry(now)
        if entry is not None and now - entry["computed_at"] < RECOMPUTE_S:
            return None
        return _ensure_refresh(now)
    except Exception:
        logger.exception("limit-rate: startup prewarm failed; the first request fits")
        return None


def join_refresh(timeout: float) -> None:
    """Wait for a running fit (tests)."""
    t = _state.get("thread")
    if t is not None:
        t.join(timeout)


def _entry(now: float) -> Optional[dict]:
    with _lock:
        loaded = _state["loaded"]
    if not loaded:
        meta = _load_meta()
        with _lock:
            _state["loaded"] = True
            if _state["entry"] is None:
                _state["entry"] = meta
    with _lock:
        return _state["entry"]


def empty_payload(status: str, note: str, now: float) -> dict:
    band = {"value": None, "lo": None, "hi": None}
    return {
        "generated_at": now, "period_start": None,
        "fit_quality": {"r2": None, "intervals": 0, "status": status, "note": note},
        "usd_per_point": {"seven_day": dict(band), "five_hour": dict(band)},
        "steps": [], "families": [],
        "peak": {"active": False, "factor": None, "until": None, "tz": PEAK_TZ,
                 "hours": [], "next_start": None},
        "cache_read_weight": dict(band, family=None),
    }


def _serve(entry: dict, now: float) -> dict:
    out = copy.deepcopy(entry["payload"])
    peak = out.get("peak") or {}
    out["peak"] = {"active": False, "factor": peak.get("factor"), "until": None,
                   "tz": peak.get("tz", PEAK_TZ), "hours": peak.get("hours") or [],
                   "next_start": None}
    try:
        out["peak"].update(peak_state(peak, now))
    except Exception:
        logger.exception("limit-rate: peak state failed")
    age = now - entry["computed_at"]
    quality = out["fit_quality"]
    if age > RECOMPUTE_S * 1.5 and quality.get("status") == "ok":
        note = f"fit computed {age / 60:.0f} min ago; a new fit is running or failed"
        quality.update(status="stale", note="; ".join(x for x in (quality.get("note"), note) if x))
    return out


@router.get("/api/limit-rate", dependencies=[Depends(require_dashboard_auth)])
def limit_rate(scope: Optional[str] = Query(default=None)):
    effective = _resolve_scope(scope)
    now = time.time()
    if effective != "personal" or sys.modules["app.config"].LOCKED_SCOPE == "enterprise":
        return empty_payload("not_available_for_scope",
                             "usage limits are tracked for the personal subscription only", now)
    entry = _entry(now)
    if entry is None or now - entry["computed_at"] >= RECOMPUTE_S:
        thread = _ensure_refresh(now)
        if thread is not None:
            thread.join(WAIT_S)
        entry = _entry(now)
    if entry is not None:
        return _serve(entry, now)
    with _lock:
        failed = _state["error"] is not None
    if failed:
        return empty_payload("error", "the fit failed; it retries in a few minutes", now)
    return empty_payload("computing", "the first fit is running; try again in a few seconds", now)
