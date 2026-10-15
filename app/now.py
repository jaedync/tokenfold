"""GET /api/now: the live fleet, polled every 5 s by the /next Runway view.

Sessions come from agent_state (in memory, fed by /api/notify hook events
from Claude Code relays and the Pi dotfleet extension) and are joined to
events for cost, model, title and client (app/now_sessions.py). Each joined
part fails alone: a broken part logs and leaves its fields null.

Scope: agent_state is fleet-wide and carries no account. A session is in
the requested scope when its own events say so. A session with no events
yet (Claude Code pushes every 5 min; Pi pushes at turn end) cannot be
classified; it shows only when its machine has usage in this scope in the
last 30 days, else it is hidden (fail closed). Meridian's SDK copies of Pi
sessions (pi_mirror_sessions) are hidden like everywhere else.
"""
import logging
import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from starlette.concurrency import run_in_threadpool

from . import agent_state, now_machines, now_sessions
from .aggregator import canonical_machine
from .api import _resolve_scope
from .auth import require_dashboard_auth
from .binding_rate import binding_rate
from .config import TZ_NAME
from .cost_windows import compute_window_cost
from .db import read_conn
from .pricing import display_model, display_model_for_row

log = logging.getLogger(__name__)
router = APIRouter()

ONLINE_WINDOW_S = 900              # a machine with an event this recent is online
BURN_MIN_WINDOW_S = 300            # shorter spans extrapolate to absurd rates
_STATE_ORDER = {"waiting": 0, "working": 1, "idle": 2}


def burn_rate(cost_recent: float, first_ts: float, now: float) -> float:
    """$/h over the last 30 min, or over the session's own age when younger
    (floored at 5 min)."""
    window = max(BURN_MIN_WINDOW_S, min(now_sessions.BURN_WINDOW_S, now - first_ts))
    return cost_recent / window * 3600


def model_display(raw: str | None, provider: str | None = None,
                  client: str | None = None) -> str | None:
    """Display name without the provider prefix. Pi hooks send
    "provider/model-id"; Claude Code hooks and events send bare ids."""
    if not raw:
        return None
    if client is None and "/" in raw:
        provider, raw = raw.split("/", 1)
        client = "pi-agent"
    name = (display_model_for_row(raw, provider, client) if provider
            else display_model(raw))
    return name.split(" / ", 1)[-1]


def infer_client(raw_model: str | None) -> str | None:
    """Client for a session with no events yet, from the hook's model form."""
    if not raw_model:
        return None
    if "/" in raw_model:
        return "pi-agent"
    return "claude-code" if raw_model.startswith("claude") else None


def _safe(label, fn, default):
    try:
        return fn()
    except Exception:
        log.exception("/api/now: %s failed", label)
        return default


def _merge_rollups(rolls: list[dict]) -> dict | None:
    """One agent_state session can match several events ids (rare: a Pi id
    seen under two namespaces). Sum them; newest model wins."""
    if not rolls:
        return None
    newest = max(rolls, key=lambda r: r["last_ts"])
    return {**newest,
            "cost": sum(r["cost"] for r in rolls),
            "cost_recent": sum(r["cost_recent"] for r in rolls),
            "anthropic_recent": sum(r["anthropic_recent"] for r in rolls),
            "first_ts": min(r["first_ts"] for r in rolls),
            "enterprise": any(r["enterprise"] for r in rolls)}


def _session_row(sid, s, agg, title, rate, now):
    raw = s.get("model") or ""
    if agg:
        burn = burn_rate(agg["cost_recent"], agg["first_ts"], now)
        model = (model_display(raw) if raw else
                 model_display(agg["model"], agg["provider"], agg["client"]))
        share = None
        if rate and rate.get("usd_per_point"):
            share = burn_rate(agg["anthropic_recent"], agg["first_ts"], now) / rate["usd_per_point"]
    else:
        burn = share = None
        model = model_display(raw)
    return {
        "id": sid,
        "machine": canonical_machine(s.get("machine") or "") or None,
        "project": s.get("project") or None,
        "title": title,
        "state": s["state"] if s["state"] in _STATE_ORDER else "idle",
        "state_since": s.get("state_since"),
        "waiting_reason": s.get("waiting_reason"),
        "model": model,
        "family": agent_state._family(raw or (agg or {}).get("model") or ""),
        "client": agg["client"] if agg else infer_client(raw),
        "subagents": s.get("fanout", 0),
        "cost_usd": agg["cost"] if agg else None,
        "burn_usd_per_h": burn,
        "share_of_binding_pct_per_h": share,
    }


def _in_scope(agg, machine, scope, known_machines) -> bool:
    if agg is not None:
        return agg["enterprise"] == (scope == "enterprise")
    return machine in known_machines


def _sort_key(row):
    order = _STATE_ORDER[row["state"]]
    since = row["state_since"] or 0
    if row["state"] == "waiting":
        return (order, since)                    # longest wait first
    if row["state"] == "working":
        return (order, -(row["burn_usd_per_h"] or 0), since)
    return (order, -since)                       # most recently idle first


def _fleet_burn(conn, scope, now) -> dict:
    """Scope-wide $/h over the last 30 min and 1 h, every provider."""
    m30 = _safe("burn m30", lambda: compute_window_cost(conn, now - 1800, now, scope) * 2, None)
    h1 = _safe("burn h1", lambda: compute_window_cost(conn, now - 3600, now, scope), None)
    return {"m30": m30, "h1": h1}


def _machines(known, recent, directory, rows, now) -> tuple[list[dict], set]:
    last_event = dict((directory or {}).get("last_event") or {})
    for name, ts in recent.items():
        last_event[name] = max(ts, last_event.get(name, 0))
    ingest = (directory or {}).get("last_ingest") or {}
    names = set(known) | {r["machine"] for r in rows if r["machine"]}
    out = []
    for name in names:
        mine = [r for r in rows if r["machine"] == name]
        out.append({"name": name, "last_event": last_event.get(name),
                    "last_ingest": ingest.get(name),
                    "working": sum(r["state"] == "working" for r in mine),
                    "waiting": sum(r["state"] == "waiting" for r in mine)})
    online = {m["name"] for m in out
              if (m["last_event"] or 0) >= now - ONLINE_WINDOW_S}
    online |= {r["machine"] for r in rows if r["machine"]}
    out.sort(key=lambda m: (m["name"] not in online, -(m["last_event"] or 0), m["name"]))
    return out, online


def build_now(scope: str, snap: dict, now: float) -> dict:
    live = {sid: s for sid, s in snap["sessions"].items() if not s.get("ending")}
    today = datetime.fromtimestamp(now, _tz()).strftime("%Y-%m-%d")
    with read_conn() as conn:
        ids = _safe("pi id lookup", lambda: now_sessions.resolve_event_sessions(conn, live, now),
                    {sid: [sid] for sid in live})
        twins = _safe("twin check", lambda: now_sessions.mirror_sessions(conn, live), set())
        all_ids = [i for lst in ids.values() for i in lst]
        rollups = _safe("session cost", lambda: now_sessions.session_rollups(
            conn, all_ids, now, scope), {})
        titles = _safe("titles", lambda: now_sessions.session_titles(conn, all_ids), {})
        known = _safe("scope machines", lambda: now_machines.scope_machines(conn, scope, today), set())
        recent = _safe("recent events", lambda: now_machines.last_events(
            conn, scope, now - now_machines.RECENT_WINDOW_S), {})
        burn = _fleet_burn(conn, scope, now)
    directory = _safe("machine directory", lambda: now_machines.directory(scope), None)
    rate = _safe("binding rate", lambda: binding_rate(scope), None)

    rows = []
    for sid, s in live.items():
        if sid in twins:
            continue
        agg = _merge_rollups([rollups[i] for i in ids.get(sid, [sid]) if i in rollups])
        machine = canonical_machine(s.get("machine") or "")
        if not _in_scope(agg, machine, scope, known):
            continue
        title = next((titles[i] for i in ids.get(sid, [sid]) if i in titles), None)
        rows.append(_session_row(sid, s, agg, title, rate, now))
    rows.sort(key=_sort_key)

    machines, online = _machines(known, recent, directory, rows, now)
    newest = max((m["last_event"] for m in machines if m["last_event"]), default=None)
    return {
        "generated_at": now,
        "scope": scope,
        "live": {
            "working": sum(r["state"] == "working" for r in rows),
            "waiting": sum(r["state"] == "waiting" for r in rows),
            "idle": sum(r["state"] == "idle" for r in rows),
            "subagents": sum(r["subagents"] for r in rows),
            "machines_online": len(online),
            "machines_total": len(machines),
            "newest_data_age_s": round(now - newest, 1) if newest else None,
            "burn_usd_per_h": burn,
        },
        "sessions": rows,
        "machines": machines,
    }


def _tz():
    from zoneinfo import ZoneInfo
    return ZoneInfo(TZ_NAME)


@router.get("/api/now", dependencies=[Depends(require_dashboard_auth)])
async def get_now(scope: Optional[str] = Query(default=None)):
    effective = _resolve_scope(scope)
    # Snapshot on the event loop: agent_state is mutated only there, so the
    # threadpool join below never sees a dict change size mid-iteration.
    snap = agent_state.snapshot()
    return await run_in_threadpool(build_now, effective, snap, time.time())
