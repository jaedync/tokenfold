"""Fleet agent-state: per-session working | waiting | idle, in memory.

Fed by /api/notify events from the dotfleet relay hooks on every fleet
machine (UserPromptSubmit -> working, permission/question/attention ->
waiting, stop -> idle). Consumed two ways:

  1. notify.py's notification policy (dedup per waiting spell, presence
     damping, aggregation) reads and mutates this store.
  2. Ambient displays (the LED cube's cubestatusd) poll
     GET /api/agent-state for the fleet-wide snapshot.

In-memory by design: the state is ephemeral and self-healing. A session
that stops reporting decays to gone after AGENT_STATE_TTL_S, so a killed
terminal can never strand a "waiting" beacon. A server restart simply
forgets sessions until their next event. Single uvicorn worker, so no
cross-process concerns; FastAPI runs handlers on one event loop, and all
mutation happens synchronously within a request.
"""
import time

from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse

from .config import (AGENT_STATE_SUBAGENT_TTL_S, AGENT_STATE_TTL_S,
                     AGENT_STATE_WAITING_TTL_S)

router = APIRouter()

# sid -> {machine, project, state, ts, waiting_notified}
_sessions: dict[str, dict] = {}
_last_working_ts: float = 0.0


def _ttl_for(state: str) -> float:
    return AGENT_STATE_WAITING_TTL_S if state == "waiting" else AGENT_STATE_TTL_S


def _prune(now: float) -> None:
    stale = [sid for sid, s in _sessions.items()
             if now - s["ts"] > _ttl_for(s.get("state", ""))]
    for sid in stale:
        del _sessions[sid]


def remove(session_id: str) -> bool:
    """Explicit goodbye (SessionEnd hook): forget the session immediately.
    Returns whether it existed."""
    return _sessions.pop(session_id, None) is not None


def update(session_id: str, machine: str, project: str, state: str,
           now: float | None = None, event_ts: float | None = None,
           fleet_rev: str | None = None) -> str | None:
    """Record a state transition. Returns the session's previous state.

    A working event clears the waiting_notified flag: the next waiting
    spell is a fresh one and may notify again.

    event_ts is the CLIENT's clock at hook time. Hooks post from
    independent processes with retry/backoff, so arrival order is not
    send order: a sub-2s turn's stop can arrive before its delayed
    working retry, which used to strand the session at "working" for the
    whole TTL. Events of one session share one machine clock, so a
    transition older than the last applied one is stale and is
    discarded. Events without event_ts (legacy/codex) keep arrival order.
    """
    global _last_working_ts
    now = time.time() if now is None else now
    _prune(now)
    s = _sessions.get(session_id) or {"waiting_notified": False}
    prev = s.get("state")
    if state != prev:
        # journald is the event history; "why was the cube X at HH:MM" must
        # always be answerable after the fact.
        print("[agent-state] {} {}: {} -> {} (machine={} project={})".format(
            time.strftime("%H:%M:%S", time.localtime(now)), session_id[:12],
            prev or "new", state, machine, project), flush=True)
    if (event_ts is not None and s.get("event_ts") is not None
            and event_ts < s["event_ts"]):
        return prev                    # out-of-order delivery: stale, discard
    if event_ts is not None:
        s["event_ts"] = event_ts
    if fleet_rev:
        s["fleet_rev"] = fleet_rev
    s["machine"] = machine or s.get("machine", "")
    s["project"] = project or s.get("project", "")
    s["state"] = state
    s["ts"] = now
    if state == "working":
        s["waiting_notified"] = False
        _last_working_ts = now
        # Keep-alive: a working heartbeat refreshes this session's live
        # subagents so long background fan-outs keep their motes lit. A
        # subagent removed by SubagentStop is already gone from the dict,
        # so this never resurrects a finished one.
        if s.get("subagents"):
            s["subagents"] = {aid: now for aid in s["subagents"]}
    _sessions[session_id] = s
    return prev


# --- Fan-out (subagent) tracking ------------------------------------------
# Each session record carries subagents: {agent_id: last_seen_ts}. A start
# adds a mote, a stop removes it; snapshot counts only motes fresher than
# the subagent TTL, so a missed stop self-heals once the parent goes quiet.
def _active_subagents(s: dict, now: float) -> dict:
    subs = s.get("subagents") or {}
    return {aid: ts for aid, ts in subs.items()
            if now - ts <= AGENT_STATE_SUBAGENT_TTL_S}


def add_subagent(session_id: str, agent_id: str, machine: str = "",
                 project: str = "", agent_type: str = "",
                 now: float | None = None, fleet_rev: str | None = None) -> int:
    """Record a subagent spawn under its PARENT session. Returns the new
    fan-out width. If the parent has not reported yet (or aged out),
    create a minimal working record: spawning a subagent means the parent
    is working."""
    now = time.time() if now is None else now
    _prune(now)
    s = _sessions.get(session_id)
    if s is None:
        s = {"waiting_notified": False, "state": "working",
             "machine": machine or "", "project": project or ""}
        print("[agent-state] {} {}: new -> working (subagent spawn)".format(
            time.strftime("%H:%M:%S", time.localtime(now)), session_id[:12]),
            flush=True)
    subs = dict(s.get("subagents") or {})
    subs[agent_id] = now
    s["subagents"] = subs
    s["ts"] = now                     # fan-out activity keeps the parent alive
    if machine:
        s["machine"] = machine
    if project:
        s["project"] = project
    if fleet_rev:
        s["fleet_rev"] = fleet_rev
    _sessions[session_id] = s
    return len(subs)


def remove_subagent(session_id: str, agent_id: str,
                    now: float | None = None) -> int:
    """Record a subagent sunset. Returns the remaining fan-out width (0 if
    the parent is unknown)."""
    now = time.time() if now is None else now
    s = _sessions.get(session_id)
    if not s:
        return 0
    subs = dict(s.get("subagents") or {})
    subs.pop(agent_id, None)
    s["subagents"] = subs
    s["ts"] = now
    _sessions[session_id] = s
    return len(subs)


def get_session(session_id: str) -> dict | None:
    return _sessions.get(session_id)


def seconds_since_working(now: float | None = None) -> float | None:
    """Age of the most recent working event across the whole fleet, or
    None if no working event has been seen since startup. This is the
    presence signal: a recent prompt means someone is at a keyboard."""
    if not _last_working_ts:
        return None
    now = time.time() if now is None else now
    return now - _last_working_ts


def waiting_sessions(now: float | None = None) -> dict[str, dict]:
    now = time.time() if now is None else now
    _prune(now)
    return {sid: s for sid, s in _sessions.items() if s["state"] == "waiting"}


def mark_waiting_notified(session_id: str) -> None:
    if session_id in _sessions:
        _sessions[session_id]["waiting_notified"] = True


def snapshot(now: float | None = None) -> dict:
    now = time.time() if now is None else now
    _prune(now)
    sessions = {}
    counts = {"working": 0, "waiting": 0, "ready": 0, "idle": 0}
    total_fanout = 0
    for sid, s in _sessions.items():
        counts[s["state"]] = counts.get(s["state"], 0) + 1
        active = _active_subagents(s, now)
        total_fanout += len(active)
        sessions[sid] = {
            "machine": s["machine"],
            "project": s["project"],
            "state": s["state"],
            "age_s": round(now - s["ts"], 1),
            "fleet_rev": s.get("fleet_rev"),
            "fanout": len(active),
            # agent ids let an ambient display place one stable mote per
            # subagent; sorted for deterministic rendering.
            "subagents": sorted(active),
        }
    # machine -> latest observed fleet rev: the authoritative drift check.
    fleet_revs = {}
    for s in sorted(_sessions.values(), key=lambda s: s["ts"]):
        if s.get("fleet_rev"):
            fleet_revs[s.get("machine", "")] = s["fleet_rev"]
    return {
        "sessions": sessions,
        "summary": counts,
        "fleet_revs": fleet_revs,
        "total_fanout": total_fanout,
        "any_waiting": counts.get("waiting", 0) > 0,
        "any_working": counts.get("working", 0) > 0,
        "any_ready": counts.get("ready", 0) > 0,
    }


def reset() -> None:
    """Test support: forget everything."""
    global _last_working_ts
    _sessions.clear()
    _last_working_ts = 0.0


@router.get("/api/agent-state")
async def get_agent_state(authorization: str | None = Header(default=None)):
    # Same bearer token the event writers use: the fleet's ingest-only
    # token, which cubestatusd also holds. Never unauthenticated.
    from .notify import _check_auth

    if not _check_auth(authorization):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return snapshot()
