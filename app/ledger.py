"""GET /api/ledger: sessions, their subagent runs and daily spend by model.

Dollars are API-equivalent list price (app/cost_windows.py semantics). The
session list covers sessions with billable rows in [from, to); each session's
cost, runs and activity are clipped to that window. `daily` comes from the
daily_summary rollup, whole local days, so its edge days can hold spend from
outside [from, to). A Meridian twin's own calls (no Pi counterpart) bill in
`daily` but are not a session, as on the old dashboard card.

One failing part degrades to null (daily, or sessions + totals); the route
never 500s for one bad part.
"""

import json
import logging
import math
import time
from collections import Counter
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from starlette.concurrency import run_in_threadpool

from . import agent_state, db
from .aggregator import TZ, _conditional_model_names, _summary_model_identity, get_cache_version
from .api import _resolve_scope
from .auth import require_dashboard_auth
from .config import scope_predicate
from .db import read_conn
from .ledger_build import build_sessions
from .meter_gap import all_meter_gaps
from .read_cache import STALE_HEADER, ReadCache

logger = logging.getLogger(__name__)
router = APIRouter()

DEFAULT_DAYS = 30
# 92 days of sessions is ~3 s cold on the 2026-10-15 production copy; longer
# ranges belong to the History view's rollups, not this per-session read.
MAX_SPAN_S = 92 * 86400
# A value up to 15 min past its ttl may be served while it rebuilds; older
# (nobody polled for that long) waits for the build.
STALE_S = 900
# Polled every 2 min: past the ttl, or after an ingest (the cache version),
# a poll gets the cached body at once, marked stale, and one background build
# refreshes it (read_cache stale_s and version).
_cache = ReadCache(ttl=30, max_entries=16, stale_s=STALE_S, version=get_cache_version)


def parse_range(from_: Optional[str], to: Optional[str], now: float) -> tuple[float, float]:
    """Validated [start, end). Default: local midnight 29 days back to now."""
    def num(value, name):
        try:
            out = float(value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"{name} must be epoch seconds")
        if not math.isfinite(out) or out < 0:
            raise HTTPException(status_code=400, detail=f"{name} must be epoch seconds")
        return out

    end = now if to is None else num(to, "to")
    if from_ is None:
        midnight = datetime.fromtimestamp(end, TZ).replace(hour=0, minute=0, second=0,
                                                           microsecond=0)
        start = (midnight - timedelta(days=DEFAULT_DAYS - 1)).timestamp()
    else:
        start = num(from_, "from")
    if start >= end:
        raise HTTPException(status_code=400, detail="from must be before to")
    if end - start > MAX_SPAN_S:
        raise HTTPException(status_code=400, detail="range is longer than 92 days")
    return start, end


def _local_day(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, TZ).strftime("%Y-%m-%d")


def daily_by_model(conn, pred: str, start: float, end: float) -> list[dict]:
    """[{day, by_model: {identity: usd}}] for every local day in the range."""
    first, last = _local_day(start), _local_day(max(start, end - 1e-6))
    days: dict = {}
    d = datetime.strptime(first, "%Y-%m-%d")
    while d.strftime("%Y-%m-%d") <= last:
        days[d.strftime("%Y-%m-%d")] = Counter()
        d += timedelta(days=1)
    for row in conn.execute(
            f"SELECT day, model_json FROM daily_summary WHERE day >= ? AND day <= ? AND {pred}",
            (first, last)):
        for name, md in json.loads(row["model_json"] or "{}").items():
            if isinstance(md, dict) and md.get("cost"):
                days[row["day"]][_summary_model_identity(name)] += md["cost"]
    return [{"day": day, "by_model": dict(models)} for day, models in days.items()]


def live_session_ids() -> frozenset:
    """Sessions the hook relay reports now. Pi hooks use the raw Pi id, which
    is the tail of the stored `pi:<ns>:<raw>` id; both forms are returned.

    Call it on the event loop only: snapshot() prunes the shared session
    dict, which event-loop code iterates. The async routes take it before
    they hand the build to a worker thread.
    """
    try:
        return frozenset(agent_state.snapshot()["sessions"])
    except Exception:
        logger.exception("agent state snapshot failed; live flags are false")
        return frozenset()


def _is_live(sid: str, live: set) -> bool:
    return sid in live or (sid.startswith("pi:") and sid[44:] in live)


def public_run(run: dict, names: dict) -> dict:
    out = {k: v for k, v in run.items() if not k.startswith("_")}
    out["model"] = names.get(run["model"])
    return out


TOKEN_KEYS = ("input", "output", "cache_read", "cache_write")


def public_session(s: dict, names: dict, live: set) -> dict:
    out = {k: v for k, v in s.items() if not k.startswith("_")}
    out["model"] = names.get(s["model"])
    # Per-session breakdowns for the browser's pivots: the orchestrator's
    # dollars by model (runs carry their own), tool calls of the session and
    # its Pi children, and tokens of every thread.
    out["orch_models"] = {names[i]: usd for i, usd in s["_orch_models"].items()}
    out["tool_calls"] = dict(s["_tool_calls"])
    out["tokens"] = {k: s["_tokens"].get(k, 0) for k in TOKEN_KEYS}
    out["runs"] = [public_run(r, names) for r in s["runs"]]
    out["live"] = _is_live(s["id"], live)
    return out


def totals(sessions: list[dict]) -> dict:
    runs = [r for s in sessions for r in s["runs"]]
    return {"cost_usd": sum(s["cost_usd"] for s in sessions),
            "sessions": len(sessions), "runs": len(runs),
            "active_s": sum(s["active_s"] for s in sessions),
            "agent_active_s": sum(r["active_s"] for r in runs),
            "prompts": sum(s["prompts"] for s in sessions)}


def model_names(daily, sessions) -> dict:
    """identity -> display name; providers shown only where names collide."""
    identities = {i for d in daily or () for i in d["by_model"]}
    for s in sessions or ():
        identities.update(s["_models"])
        identities.update(r["model"] for r in s["runs"] if r["model"])
    return _conditional_model_names({i for i in identities if i})


def _part(name: str, fn):
    try:
        return fn()
    except Exception:
        logger.exception("ledger part %s failed; served as null", name)
        return None


def with_live(body: dict, live) -> dict:
    """A copy of a (cached) ledger body with the live flags of now."""
    if body.get("sessions") is None:
        return body
    return {**body, "sessions": [{**s, "live": _is_live(s["id"], live)}
                                 for s in body["sessions"]]}


def build_ledger(conn, scope: str, start: float, end: float,
                 live=frozenset()) -> dict:
    pred = scope_predicate(scope)
    daily = _part("daily", lambda: daily_by_model(conn, pred, start, end))
    sessions = _part("sessions", lambda: build_sessions(conn, pred, scope, start, end))
    names = model_names(daily, sessions)
    public = None if sessions is None else [public_session(s, names, live) for s in sessions]
    extra = {}
    if scope == "enterprise":
        # Only the work account has a billing meter to compare with.
        extra["meter_gaps"] = _part("meter_gaps", lambda: all_meter_gaps(conn, TZ, time.time()))
    return {
        "from": start, "to": end,
        # display name -> provider display name, for a provider-grouped legend
        "model_providers": {name: identity[0] for identity, name in names.items()},
        "daily": None if daily is None else [
            {"day": d["day"], "by_model": {names[i]: usd for i, usd in d["by_model"].items()}}
            for d in daily],
        "sessions": public,
        "totals": None if public is None else totals(public),
        **extra,
    }


@router.get("/api/ledger", dependencies=[Depends(require_dashboard_auth)])
async def ledger(response: Response, scope: Optional[str] = Query(default=None),
                 from_: Optional[str] = Query(default=None, alias="from"),
                 to: Optional[str] = Query(default=None)):
    effective = _resolve_scope(scope)
    start, end = parse_range(from_, to, time.time())
    live = live_session_ids()  # on the event loop (see live_session_ids)
    body, stale = await run_in_threadpool(cached_ledger_state, effective, start, end, to is not None)
    if stale:
        response.headers[STALE_HEADER] = "1"
    return with_live(body, live)


def cached_ledger_state(scope: str, start: float, end: float, closed: bool) -> tuple[dict, bool]:
    """(body, stale): see ReadCache.lookup."""
    # An open-ended range keys on "now" so repeat polls share one build.
    key = (db.DB_PATH, scope, start, end if closed else None)

    def build():
        with read_conn() as conn:
            return build_ledger(conn, scope, start, end)
    return _cache.lookup(key, build)


def cached_ledger(scope: str, start: float, end: float, closed: bool) -> dict:
    return cached_ledger_state(scope, start, end, closed)[0]


def warm(scope: str) -> None:
    """Build the default 30-day ledger (startup prewarm, app/prewarm.py).

    The closed days' billable scans (ledger_days) and the Pi lineage links
    stay cached after it, so the first visit after a restart reads them.
    """
    start, end = parse_range(None, None, time.time())
    cached_ledger(scope, start, end, False)


@router.get("/api/sessions/{session_id:path}", dependencies=[Depends(require_dashboard_auth)])
async def session(session_id: str, scope: Optional[str] = Query(default=None)):
    from .session_detail import build_detail
    effective = _resolve_scope(scope)
    if not session_id or len(session_id) > 1024:
        raise HTTPException(status_code=404, detail="session not found")

    live = live_session_ids()  # on the event loop (see live_session_ids)

    def build():
        with read_conn() as conn:
            return build_detail(conn, effective, session_id, live)
    out = await run_in_threadpool(build)
    if out is None:
        raise HTTPException(status_code=404, detail="session not found")
    return out
