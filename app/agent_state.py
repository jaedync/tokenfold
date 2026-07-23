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

from .config import (AGENT_STATE_MOTE_MIN_VISIBLE_S, AGENT_STATE_MOTE_SUNSET_S,
                     AGENT_STATE_SUBAGENT_TTL_S, AGENT_STATE_TTL_S,
                     AGENT_STATE_WAITING_TTL_S, AGENT_STATE_WORKING_FRESH_S)

router = APIRouter()

# sid -> {machine, project, state, ts, waiting_notified, model, subagents}
_sessions: dict[str, dict] = {}
_last_working_ts: float = 0.0


def _family(model: str) -> str:
    """Classify a raw model id into a cube color family (policy lives here;
    the cube maps family -> palette)."""
    m = (model or "").lower()
    if "opus" in m:
        return "opus"
    if "sonnet" in m:
        return "sonnet"
    if "fable" in m:
        return "fable"
    if "haiku" in m:
        return "haiku"
    if "gpt" in m or "codex" in m:
        return "codex"
    return "unknown"


def _ttl_for(state: str) -> float:
    return AGENT_STATE_WAITING_TTL_S if state == "waiting" else AGENT_STATE_TTL_S


def _mote_phase(m: dict, now: float) -> str:
    """'live' | 'sunsetting' | 'gone' for a mote record.

    A mote is live while it has no stop_ts and stays fresh; sunsetting once it
    stops (but still inside its min-visible + sunset window, so an ambient
    display polling every ~2s can see and flash even a 1s subagent); gone after
    that. A stop that never arrives fades the same way past the subagent TTL,
    so a missed SubagentStop pops nothing - it sunsets."""
    stop = m.get("stop_ts")
    if stop is None:
        # A missed SubagentStop: once the mote ages past the TTL, treat it as
        # sunsetting from its last-seen so it fades rather than pops.
        if now - m["ts"] > AGENT_STATE_SUBAGENT_TTL_S:
            return "sunsetting" if now - m["ts"] <= (
                AGENT_STATE_SUBAGENT_TTL_S + AGENT_STATE_MOTE_SUNSET_S) else "gone"
        return "live"
    visible = (now - stop <= AGENT_STATE_MOTE_SUNSET_S
               or now - m["spawn_ts"] <= AGENT_STATE_MOTE_MIN_VISIBLE_S)
    return "sunsetting" if visible else "gone"


def _effective_stop(m: dict, now: float) -> float:
    """The stop timestamp to report, synthesizing one for a TTL-lapsed mote."""
    return m["stop_ts"] if m.get("stop_ts") is not None else m["ts"]


def _prune(now: float) -> None:
    stale = []
    for sid, s in _sessions.items():
        if s.get("ending_ts") is not None:
            # A sunsetting session fades its dot then drops after the window.
            if now - s["ending_ts"] > AGENT_STATE_MOTE_SUNSET_S:
                stale.append(sid)
            continue
        if now - s["ts"] > _ttl_for(s.get("state", "")):
            stale.append(sid)
            continue
        subs = s.get("subagents")
        if subs:
            s["subagents"] = _drawable_subagents(s, now)
    for sid in stale:
        del _sessions[sid]


def remove(session_id: str) -> bool:
    """Explicit goodbye (SessionEnd hook): forget the session immediately.
    Returns whether it existed."""
    return _sessions.pop(session_id, None) is not None


def sunset_session(session_id: str, now: float | None = None) -> bool:
    """SessionEnd: fade the session dot out over the sunset window instead of
    deleting it instantly, then let _prune drop it. Terminal: further events
    for this id are ignored (update() guards on ending_ts)."""
    now = time.time() if now is None else now
    s = _sessions.get(session_id)
    if not s:
        return False
    s["ending_ts"] = now
    _sessions[session_id] = s
    return True


def update(session_id: str, machine: str, project: str, state: str,
           now: float | None = None, event_ts: float | None = None,
           fleet_rev: str | None = None, model: str | None = None) -> str | None:
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
    if s.get("ending_ts") is not None:
        return prev            # terminal: SessionEnd already fired
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
    if model:
        s["model"] = model
    s["state"] = state
    s["ts"] = now
    if state == "working":
        s["working_ts"] = now
        s["waiting_notified"] = False
        _last_working_ts = now
        # Keep-alive: a working heartbeat refreshes this session's live
        # subagents so long background fan-outs keep their motes lit. A
        # subagent removed by SubagentStop is already gone from the dict,
        # so this never resurrects a finished one. Only the last-seen ts is
        # bumped; spawn_ts/model/stop_ts on each mote are preserved.
        if s.get("subagents"):
            s["subagents"] = {aid: dict(m, ts=now)
                              for aid, m in s["subagents"].items()}
    _sessions[session_id] = s
    return prev


# --- Fan-out (subagent) tracking ------------------------------------------
# Each session record carries subagents: {agent_id: {spawn_ts, ts, model,
# stop_ts}}. A start adds a mote, a stop removes it; snapshot counts only motes
# fresher than the subagent TTL, so a missed stop self-heals once the parent
# goes quiet. spawn_ts anchors the mote's displayed age; ts is the last-seen
# heartbeat used for TTL freshness.
def _active_subagents(s: dict, now: float) -> dict:
    """agent_id -> record for LIVE motes only (drives fan-out and the
    keep-alive refresh). A sunsetting mote is leaving, not live fan-out."""
    subs = s.get("subagents") or {}
    return {aid: m for aid, m in subs.items() if _mote_phase(m, now) == "live"}


def _drawable_subagents(s: dict, now: float) -> dict:
    """agent_id -> record for motes the cube should still draw (live +
    sunsetting). Fully-gone motes are dropped."""
    subs = s.get("subagents") or {}
    return {aid: m for aid, m in subs.items() if _mote_phase(m, now) != "gone"}


def add_subagent(session_id: str, agent_id: str, machine: str = "",
                 project: str = "", agent_type: str = "",
                 now: float | None = None, fleet_rev: str | None = None,
                 model: str | None = None) -> int:
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
    is_new_mote = agent_id not in subs
    if is_new_mote:
        # A genuinely new spawn anchors its own spawn_ts and model; stop_ts
        # stays None (Task 3 sets it to drive the mote's sunset fade).
        subs[agent_id] = {"spawn_ts": now, "ts": now,
                          "model": model or "", "stop_ts": None}
    elif subs[agent_id].get("stop_ts") is not None:
        # A re-spawn of an id that is still sunsetting. Keep-alive heartbeats
        # flow through update() (which preserves stop_ts on purpose), so an
        # add_subagent on an already-stopped id can only be a real
        # SubagentStart. Bring the mote back live - reset spawn_ts, clear
        # stop_ts, keep/refresh the model - and treat it as new so the parent's
        # working_ts is bumped and it rejoins fan-out and mood.
        subs[agent_id] = dict(subs[agent_id], spawn_ts=now, ts=now,
                              stop_ts=None,
                              model=model or subs[agent_id].get("model", ""))
        is_new_mote = True
    else:
        # Heartbeat refresh: bump last-seen only; keep spawn_ts/stop_ts, and
        # let a later-known model backfill an empty one.
        subs[agent_id] = dict(subs[agent_id], ts=now)
        if model:
            subs[agent_id]["model"] = model
    s["subagents"] = subs
    s["ts"] = now                     # fan-out activity keeps the parent alive
    if is_new_mote:
        # A genuinely new spawn is a work action and marks the parent working.
        # A heartbeat refresh of an existing mote must NOT bump working_ts, or a
        # handed-off parent (blocked in a synchronous fan-out) would never go
        # stale and the waiting_subagent mood could never fire.
        s["working_ts"] = now
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
    """Record a subagent stop by marking its mote sunsetting (stop_ts) instead
    of popping it, so a very short-lived subagent still lingers a min-visible +
    sunset window for the cube to flash. Returns the remaining LIVE fan-out
    width (0 if the parent is unknown)."""
    now = time.time() if now is None else now
    s = _sessions.get(session_id)
    if not s:
        return 0
    subs = dict(s.get("subagents") or {})
    if agent_id in subs:
        subs[agent_id] = dict(subs[agent_id], stop_ts=now)
    s["subagents"] = subs
    s["ts"] = now
    _sessions[session_id] = s
    return len(_active_subagents(s, now))


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


def _fleet_mood(now: float) -> str:
    """The single fleet-wide mood for the cube's bottom volume, top wins:
    needs_you (any session blocked on the human) > working (any session with a
    fresh working heartbeat) > waiting_subagent (live fan-out but nobody
    actively grinding) > idle. Active beats delegated on purpose; the roster's
    top motes already show the children working."""
    # A sunsetting (ending) session is leaving and drives no mood.
    vals = [s for s in _sessions.values() if s.get("ending_ts") is None]
    if any(s.get("state") == "waiting" for s in vals):
        return "needs_you"
    if any(s.get("state") == "working"
           and now - s.get("working_ts", 0.0) <= AGENT_STATE_WORKING_FRESH_S
           for s in vals):
        return "working"
    if any(_active_subagents(s, now) for s in vals):
        return "waiting_subagent"
    return "idle"


def snapshot(now: float | None = None) -> dict:
    now = time.time() if now is None else now
    _prune(now)
    sessions = {}
    agents = []
    counts = {"working": 0, "waiting": 0, "ready": 0, "idle": 0}
    total_fanout = 0
    for sid in sorted(_sessions):
        s = _sessions[sid]
        ending = s.get("ending_ts")
        active = _active_subagents(s, now)
        drawable = _drawable_subagents(s, now)
        # A sunsetting session is leaving: it drives no fleet-wide fan-out
        # (consistent with _fleet_mood excluding ending sessions) and no
        # longer counts toward the state summary. Its per-session fanout below
        # still reports its own live motes.
        if ending is None:
            total_fanout += len(active)
            counts[s["state"]] = counts.get(s["state"], 0) + 1
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
        # Flat drawable roster: the session dot, then one mote per drawable
        # child (live or sunsetting). Each entry carries model + family so the
        # cube colors agents by model.
        dot = {"id": sid, "kind": "session",
               "age_s": round(now - s["ts"], 1),
               "model": s.get("model", ""),
               "family": _family(s.get("model", ""))}
        if ending is not None:
            # Fade the dot: report it sunsetting with seconds since SessionEnd.
            dot["state"] = "sunsetting"
            dot["stop_age_s"] = round(now - ending, 1)
        else:
            dot["state"] = s["state"]
        agents.append(dot)
        for aid in sorted(drawable):
            m = drawable[aid]
            # Mote age is measured from spawn_ts (not the last heartbeat), so a
            # keep-alive refresh never rewinds a long-lived mote's displayed age.
            entry = {"id": sid + ":" + aid, "kind": "subagent",
                     "age_s": round(now - m["spawn_ts"], 1),
                     "model": m.get("model", ""),
                     "family": _family(m.get("model", ""))}
            if _mote_phase(m, now) == "sunsetting":
                entry["state"] = "sunsetting"
                entry["stop_age_s"] = round(now - _effective_stop(m, now), 1)
            else:
                entry["state"] = "working"
            agents.append(entry)
    # machine -> latest observed fleet rev: the authoritative drift check.
    fleet_revs = {}
    for s in sorted(_sessions.values(), key=lambda s: s["ts"]):
        if s.get("fleet_rev"):
            fleet_revs[s.get("machine", "")] = s["fleet_rev"]
    return {
        "sessions": sessions,
        "agents": agents,
        "mood": _fleet_mood(now),
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
