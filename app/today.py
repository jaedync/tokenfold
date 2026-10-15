"""GET /api/today: today's spend against a typical day (/next, every 60 s).

Today's totals come from today's daily_summary rows, which ingest re-rolls
synchronously on every batch, so they are as fresh as the data. The
typical figures are medians over the last 28 complete local days (days
before the scope's first data day are left out, so a new install is not
compared with empty days):

- typical_day_usd: median daily_summary cost.
- typical_by_now_usd: median cumulative spend at this local time of day
  (app/today_typical.py; null until its background profile is built).

Money is API-equivalent at list price, as everywhere in Tokenfold.
"""
import json
import logging
import statistics
import sys
import time
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Query
from starlette.concurrency import run_in_threadpool

from . import agent_state, now_machines, today_typical
from .aggregator import _merge_summary_rows, _summary_model_identity, canonical_machine
from .api import _resolve_scope
from .auth import require_dashboard_auth
from .config import TZ_NAME
from .db import read_conn
from .now import _safe, scoped_sessions
from .pricing import _PROVIDER_DISPLAY

log = logging.getLogger(__name__)
router = APIRouter()

_PROVIDER_IDS = {v.lower(): k for k, v in _PROVIDER_DISPLAY.items()}


def provider_id(display: str) -> str:
    """'Anthropic' -> 'anthropic', 'OpenCode Go' -> 'opencode-go'."""
    low = display.strip().lower()
    return _PROVIDER_IDS.get(low, low.replace(" ", "-"))


def by_model(model_json: str) -> list[dict]:
    """daily_summary model_json -> one row per (provider, model), usd desc.

    Claude Code keys are bare names ("Opus 5.5"), Pi keys are
    "Provider / Model"; both Anthropic forms merge into one row."""
    merged: dict[tuple[str, str], dict] = {}
    for key, md in json.loads(model_json or "{}").items():
        provider, model = _summary_model_identity(key)
        row = merged.setdefault((provider_id(provider), model), {"usd": 0.0, "tokens": 0})
        row["usd"] += md.get("cost", 0.0) or 0.0
        row["tokens"] += sum(md.get(k, 0) or 0 for k in
                             ("input", "output", "cache_write", "cache_read"))
    rows = [{"model": model, "family": agent_state._family(model),
             "provider": provider, "usd": v["usd"], "tokens": v["tokens"]}
            for (provider, model), v in merged.items()]
    return sorted(rows, key=lambda r: (-r["usd"], r["model"]))


def by_machine(machine_json: str, working: dict[str, int]) -> list[dict]:
    """Canonical machines with spend today, plus any with live work."""
    usd: dict[str, float] = {}
    for raw, mv in json.loads(machine_json or "{}").items():
        name = canonical_machine(raw)
        if name:
            usd[name] = usd.get(name, 0.0) + (mv.get("cost", 0.0) or 0.0)
    names = set(usd) | {m for m, n in working.items() if n}
    rows = [{"machine": m, "usd": usd.get(m, 0.0), "working": working.get(m, 0)}
            for m in names]
    return sorted(rows, key=lambda r: (-r["usd"], r["machine"]))


def _daily_costs(conn, pred: str, days: list[str]) -> list[float]:
    if not days:
        return []
    marks = ",".join("?" for _ in days)
    found = {r[0]: r[1] or 0.0 for r in conn.execute(
        f"SELECT day, SUM(cost) FROM daily_summary WHERE day IN ({marks}) AND {pred} "
        "GROUP BY day", days)}
    return [found.get(d, 0.0) for d in days]


def _working_by_machine(conn, scope, snap, now, today) -> dict[str, int]:
    known = now_machines.scope_machines(conn, scope, today)
    counts: dict[str, int] = {}
    for row in scoped_sessions(conn, scope, snap, now, known):
        if row["state"] == "working" and row["machine"]:
            counts[row["machine"]] = counts.get(row["machine"], 0) + 1
    return counts


def build_today(scope: str, snap: dict, now: float) -> dict:
    tz = ZoneInfo(TZ_NAME)
    local = datetime.fromtimestamp(now, tz)
    today = local.strftime("%Y-%m-%d")
    pred = sys.modules["app.config"].scope_predicate(scope)
    with read_conn() as conn:
        rows = conn.execute(f"SELECT * FROM daily_summary WHERE day = ? AND {pred}",
                            (today,)).fetchall()
        first = conn.execute(f"SELECT MIN(day) FROM daily_summary WHERE {pred}").fetchone()[0]
        days = today_typical.history_days(today, first)
        costs = _daily_costs(conn, pred, days)
        working = _safe("live working", lambda: _working_by_machine(
            conn, scope, snap, now, today), {})
    seconds = local.hour * 3600 + local.minute * 60 + local.second + local.microsecond / 1e6
    typical_now = _safe("typical by now", lambda: today_typical.typical_by_now(
        scope, days, seconds, tz), None)
    merged = _merge_summary_rows(rows) if rows else None
    return {
        "day": today,
        "spend_usd": merged["cost"] if merged else 0.0,
        "typical_by_now_usd": typical_now,
        "typical_day_usd": statistics.median(costs) if costs else None,
        "prompts": merged["human_prompts"] if merged else 0,
        "active_s": merged["active_s"] if merged else 0.0,
        "by_model": _safe("by model", lambda: by_model(merged["model_json"]), []) if merged else [],
        "by_machine": _safe("by machine", lambda: by_machine(
            merged["machine_json"] if merged else "{}", working), []),
    }


@router.get("/api/today", dependencies=[Depends(require_dashboard_auth)])
async def get_today(scope: Optional[str] = Query(default=None)):
    effective = _resolve_scope(scope)
    snap = agent_state.snapshot()          # on the loop; see now.get_now
    return await run_in_threadpool(build_today, effective, snap, time.time())
