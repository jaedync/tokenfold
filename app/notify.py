"""Notification relay — forwards Claude Code hook events to Home Assistant.

Codex support is basic: the hook only sends a bare stop event (no model/usage
data), so notifications show "Response complete" without cost or model info.
"""

import asyncio
import hmac
import logging
import secrets
import sys
import time

import httpx
from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse

from . import agent_events, agent_state
from .config import (
    AGENT_PRESENCE_DAMPING_S,
    HA_DEVICES,
    HA_TOKEN,
    HA_URL,
    NOTIFY_TOKEN,
    RECEIPT_QUIET_S,
    STATS_API_KEY,
)
from .db import get_conn, write_txn
from .pricing import compute_cost, display_model

router = APIRouter()
log = logging.getLogger(__name__)

_notify_token: str = NOTIFY_TOKEN


def init_notify_token():
    """Resolve the notify token: env var > DB > auto-generate.

    Called during app lifespan after DB is ready.
    """
    global _notify_token

    if NOTIFY_TOKEN:
        _notify_token = NOTIFY_TOKEN
        print("[notify] Token set via NOTIFY_TOKEN env var", flush=True)
        return

    conn = get_conn()
    row = conn.execute("SELECT value FROM meta WHERE key='notify_token'").fetchone()
    if row:
        _notify_token = row["value"]
        print(f"[notify] Token (from DB): {_notify_token}", flush=True)
        return

    _notify_token = secrets.token_urlsafe(24)
    with write_txn(conn) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
            ("notify_token", _notify_token),
        )
    print(f"[notify] Generated token: {_notify_token}", flush=True)


def _check_auth(authorization: str | None) -> bool:
    if not authorization or not authorization.startswith("Bearer "):
        return False
    token = authorization[7:]
    # Constant-time compare; keep the fail-closed guards so an empty configured
    # token can never match an empty/absent bearer token.
    if _notify_token and hmac.compare_digest(token or "", _notify_token or ""):
        return True
    if STATS_API_KEY and hmac.compare_digest(token or "", STATS_API_KEY or ""):
        return True
    return False


def _format_duration(seconds: int) -> str:
    if seconds >= 3600:
        return f"{seconds // 3600}h {seconds % 3600 // 60}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds}s"


def _cost_from_usage(entry: dict) -> float:
    """Compute cost for a single usage entry using pricing.compute_cost()."""
    model_id = entry.get("model") or ""
    dname = display_model(model_id)

    inp = entry.get("input_tokens", 0)
    out = entry.get("output_tokens", 0)
    cr = entry.get("cache_read_input_tokens", 0)

    # Granular cache_creation breakdown (ephemeral_5m + ephemeral_1h)
    cache = entry.get("cache_creation") or {}
    cw = (
        cache.get("ephemeral_5m_input_tokens", 0)
        + cache.get("ephemeral_1h_input_tokens", 0)
    )
    # Fallback: flat cache_creation_input_tokens field
    if not cache:
        cw = entry.get("cache_creation_input_tokens", 0)

    # Web-search fee ($10/1k requests) — entries are transcript-usage-shaped.
    stu = entry.get("server_tool_use") or {}
    ws = stu.get("web_search_requests", 0) if isinstance(stu, dict) else 0
    if not (isinstance(ws, int) and ws > 0):
        ws = 0

    # Deliberately no ts_epoch: notifications price events that just happened,
    # so wall-clock "now" is by definition the correct pricing era.
    return compute_cost(dname, inp, out, cw, cr, web_search=ws)


def _build_ha_payload(data: dict) -> dict:
    """Build title/message for a Claude Code notification."""
    project = data.get("project", "unknown")
    event = data.get("event", "")

    if event == "permission":
        return {"title": f"Permission needed ({project})", "message": ""}
    if event == "question":
        return {"title": f"Answer needed ({project})", "message": ""}
    if event == "attention":
        return {"title": f"Claude Code ({project})", "message": ""}
    if event == "stop":
        title = f"Response complete ({project})"
        parts = []

        duration_s = data.get("duration_s")
        if duration_s is not None and duration_s >= 0:
            line1 = _format_duration(int(duration_s))
            tool_count = data.get("tool_count", 0)
            if tool_count and tool_count > 0:
                line1 += f", {tool_count} tools"
            parts.append(line1)

        models = data.get("models") or ([data["model"]] if data.get("model") else [])
        usage_list = data.get("usage") or []
        total_cost = sum(_cost_from_usage(u) for u in usage_list)

        line2_parts = []
        display_models = list(dict.fromkeys(display_model(m) for m in models if m))
        if display_models:
            line2_parts.append(" + ".join(display_models))
        if total_cost >= 0.005:
            line2_parts.append(f"${total_cost:.2f}")
        if line2_parts:
            parts.append(", ".join(line2_parts))

        return {"title": title, "message": "\n".join(parts)}

    return {"title": f"Claude Code ({project})", "message": ""}


async def _relay_to_ha(payload: dict, devices: list | None = None):
    """Send notification to Home Assistant devices. No-op if HA is not configured."""
    if not HA_URL or not HA_TOKEN:
        return []

    targets = devices or HA_DEVICES
    if not targets:
        return []

    if "message" not in payload:
        payload["message"] = ""

    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }

    errors = []
    async with httpx.AsyncClient(timeout=8) as client:
        for device in targets:
            try:
                r = await client.post(
                    f"{HA_URL}/api/services/notify/{device}",
                    headers=headers,
                    json=payload,
                )
                r.raise_for_status()
            except Exception as e:
                errors.append(f"{device}: {e}")
                log.warning("Failed to notify %s: %s", device, e)

    return errors


# Events that mean "Claude is blocked on the human".
WAITING_EVENTS = {"permission", "question", "attention"}


def _waiting_push_decision(session_id: str) -> str | None:
    """Policy gate for a waiting push. Returns a suppression reason, or
    None when the push should go out.

    - Dedup by construction: one push per waiting spell; the flag clears
      on the session's next working event.
    - Presence damping: a user prompt anywhere in the fleet within
      AGENT_PRESENCE_DAMPING_S means someone is at a keyboard; the
      ambient display carries the beacon instead. Known trade-off: a
      damped push is not retried later, so a prompt the user genuinely
      missed stays silent until the cube/beacon surfaces it.
    """
    sess = agent_state.get_session(session_id)
    if sess and sess.get("waiting_notified"):
        return "duplicate"
    since = agent_state.seconds_since_working()
    if AGENT_PRESENCE_DAMPING_S > 0 and since is not None and since < AGENT_PRESENCE_DAMPING_S:
        return "presence"
    return None


def _aggregate_waiting_payload(ha_payload: dict) -> dict:
    """N sessions waiting at once -> one combined push, not N pushes."""
    waiting = agent_state.waiting_sessions()
    if len(waiting) <= 1:
        return ha_payload
    where = ", ".join(
        f"{s['project']}@{s['machine']}" if s["machine"] else s["project"]
        for s in waiting.values()
    )
    return {"title": f"{len(waiting)} sessions waiting", "message": where}


# --- Delayed stop receipts (quiet window) ---------------------------------
# session_id -> {payload, devices, stop_seq, ts, task}. An interactive stop
# stores its built HA receipt here and schedules a flush RECEIPT_QUIET_S later.
# The flush pushes only if the receipt is still the session's latest, the
# session stayed quiet (a working signal cancels it at ingest), and the session
# has no live subagent motes. In-memory is fine: tokenfold is a single instance
# and a receipt lost on restart is harmless.
_pending_receipts: dict[str, dict] = {}
_stop_seq: int = 0


def _schedule_flush(session_id: str, stop_seq: int):
    """Schedule the quiet-window flush on the running loop. Returns the task, or
    None when there is no running loop (unit tests drive the flush directly)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    return loop.create_task(_deferred_flush(session_id, stop_seq))


async def _deferred_flush(session_id: str, stop_seq: int):
    """Sleep the quiet window, then flush. A replaced or cancelled receipt
    cancels this task, so a swallowed CancelledError just ends it quietly."""
    try:
        await asyncio.sleep(RECEIPT_QUIET_S)
    except asyncio.CancelledError:
        return
    await _flush_receipt(session_id, stop_seq)


async def _flush_receipt(session_id: str, stop_seq: int):
    """Push the pending receipt iff it is still the session's latest and the
    session has no live subagent motes. A working signal since the stop has
    already cancelled the receipt at ingest, so its mere presence here means the
    session stayed quiet. Consumes the receipt either way."""
    pending = _pending_receipts.get(session_id)
    if pending is None or pending["stop_seq"] != stop_seq:
        return                              # superseded by a newer stop, or gone
    if agent_state.has_live_subagents(session_id):
        _pending_receipts.pop(session_id, None)   # handoff, not a turn end
        return
    _pending_receipts.pop(session_id, None)
    await _relay_to_ha(pending["payload"], pending["devices"])


def _store_pending_receipt(session_id: str, payload: dict,
                           devices: list | None) -> None:
    """Store (replacing any prior) the session's pending receipt and arm its
    quiet-window flush. A replaced receipt's timer is cancelled: only the latest
    stop can push."""
    global _stop_seq
    _cancel_pending_receipt(session_id)     # the prior timer loses
    _stop_seq += 1
    seq = _stop_seq
    task = _schedule_flush(session_id, seq)
    _pending_receipts[session_id] = {
        "payload": payload, "devices": devices,
        "stop_seq": seq, "ts": time.time(), "task": task,
    }


def _cancel_pending_receipt(session_id: str) -> None:
    """Drop the session's pending receipt and cancel its flush timer. Called by
    every working signal (user prompt, tool_activity, subagent_start): fresh
    work means the turn did not really end, so the receipt is void."""
    pending = _pending_receipts.pop(session_id, None)
    if pending is not None and pending.get("task") is not None:
        pending["task"].cancel()


def _reset_pending() -> None:
    """Test support: cancel all pending flush timers and forget every receipt."""
    for pending in list(_pending_receipts.values()):
        task = pending.get("task")
        if task is not None:
            task.cancel()
    _pending_receipts.clear()


# Hard cap on per-tool SSE events fanned out from a single tool_activity ingest.
# A new relay posts count=1 tools={name:1} (one event); an old relay can coalesce
# a burst of ~30 tools into one batch, and expanding that 1:1 would blast that
# many cube ripples at once. Overflow past the cap is silently dropped, taking
# the first _MAX_TOOL_EVENTS in tally order.
_MAX_TOOL_EVENTS = 8


def _publish_tool_events(session_id: str, data: dict) -> None:
    """Fan a tool_activity ingest out to the SSE stream as one event per
    individual tool call. tools is a {name: n} tally expanded in insertion
    order, each name n times. A missing or empty tally (legacy relay) falls back
    to max(1, count) events of last_tool (or ""). Capped at _MAX_TOOL_EVENTS.

    State-only: this only puts events onto in-memory SSE queues; it never
    touches the HA push path, preserving the tool_activity early-return
    invariant."""
    tools = data.get("tools")
    if isinstance(tools, dict) and tools:
        published = 0
        for name, n in tools.items():
            try:
                n = int(n)
            except (TypeError, ValueError):
                n = 0
            for _ in range(max(0, n)):
                if published >= _MAX_TOOL_EVENTS:
                    return
                agent_events.publish(session_id, name)
                published += 1
        return
    # Legacy relay with no tally: one event per counted call of last_tool.
    try:
        count = int(data.get("count", 1))
    except (TypeError, ValueError):
        count = 1
    last_tool = data.get("last_tool") or ""
    for _ in range(min(max(1, count), _MAX_TOOL_EVENTS)):
        agent_events.publish(session_id, last_tool)


@router.post("/api/notify")
async def notify(request: Request, authorization: str | None = Header(default=None)):
    if not _check_auth(authorization):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    data = await request.json()

    event = data.get("event")
    session_id = None
    if event is not None:
        machine = data.get("machine", "")
        project = data.get("project", "unknown")
        # Codex and legacy clients send no session_id; key them stably by
        # machine+project so their stop events still resolve a state.
        session_id = data.get("session_id") or f"{machine}:{project}"
        try:
            client_ts = float(data.get("client_ts"))
        except (TypeError, ValueError):
            client_ts = None
        fleet_rev = data.get("fleet_rev") or None

        if event == "working":
            # Pure state transition: never a push. This is what clears a
            # waiting spell and feeds the presence signal. model (relay Task 1)
            # colors the session dot by model on the cube.
            agent_state.update(session_id, machine, project, "working",
                               event_ts=client_ts, fleet_rev=fleet_rev,
                               model=data.get("model"))
            # A user prompt means the turn resumed: void any pending receipt.
            _cancel_pending_receipt(session_id)
            return {"ok": True, "state": "working"}

        if event == "session_start":
            # SessionStart: register an idle session so its dot (and model
            # color) appear before the first prompt. State-only, never a push.
            agent_state.session_start(session_id, machine, project,
                                      event_ts=client_ts, fleet_rev=fleet_rev,
                                      model=data.get("model"))
            return {"ok": True, "state": "session_start"}

        if event == "tool_activity":
            # PostToolUse batch (the new working heartbeat): accumulate the
            # cumulative tool ticker and last-tool spark. State-only.
            agent_state.tool_activity(
                session_id, machine, project,
                count=data.get("count", 1), last_tool=data.get("last_tool"),
                event_ts=client_ts, fleet_rev=fleet_rev, model=data.get("model"))
            # Per-call SSE fan-out (v2.3): expand the batch into individual
            # tool_call events for realtime cube ripples. State-only: publishes
            # to in-memory SSE queues, never the HA push, so the early return
            # below (the binding tool_activity invariant) stays intact.
            _publish_tool_events(session_id, data)
            # Fresh tool progress means the turn is still running: void the receipt.
            _cancel_pending_receipt(session_id)
            return {"ok": True, "state": "tool_activity"}

        if event in ("tool_trouble", "stop_failure"):
            # A tool failure: overlay only, never a phone buzz and never a
            # state change. Both names route to the same trouble overlay.
            agent_state.tool_trouble(session_id)
            return {"ok": True, "state": event}

        if event == "compact_start":
            agent_state.compact_start(session_id)
            return {"ok": True, "state": "compact_start"}

        if event == "compact_end":
            agent_state.compact_end(session_id)
            return {"ok": True, "state": "compact_end"}

        if event == "subagent_start":
            # Fan-out mote on. session_id is the PARENT (the relay recovered
            # it from the transcript path). State-only, never a push. model is
            # the subagent's own model, so its mote gets its own color.
            n = agent_state.add_subagent(
                session_id, data.get("agent_id", ""), machine=machine,
                project=project, agent_type=data.get("agent_type", ""),
                fleet_rev=fleet_rev, model=data.get("model"))
            # A new fan-out means the turn handed off, not ended: void the receipt.
            _cancel_pending_receipt(session_id)
            return {"ok": True, "state": "subagent_start", "fanout": n}

        if event == "subagent_stop":
            n = agent_state.remove_subagent(session_id, data.get("agent_id", ""))
            return {"ok": True, "state": "subagent_stop", "fanout": n}

        if event in WAITING_EVENTS:
            agent_state.update(session_id, machine, project, "waiting", event_ts=client_ts, fleet_rev=fleet_rev)
            reason = _waiting_push_decision(session_id)
            if reason:
                return {"ok": True, "suppressed": reason}

        if event == "gone":
            # SessionEnd: state-only, never a push. Sunset (fade) the dot over
            # the sunset window instead of deleting it instantly, so the cube
            # can play the session's death flash; _prune drops it after. This
            # ignores client_ts ordering on purpose: gone is terminal, and a
            # session id is never reused after SessionEnd.
            agent_state.sunset_session(session_id)
            return {"ok": True, "state": "gone"}

        if event == "idle":
            # idle_prompt: the session has sat unattended past the CLI's
            # idle threshold. True idle. Always state-only.
            agent_state.update(session_id, machine, project, "idle", event_ts=client_ts, fleet_rev=fleet_rev)
            return {"ok": True, "state": "idle"}

        if event == "stop":
            # Turn ended. The cube reads this as idle (binary attention: a
            # finished turn is not "come look", the response is in the
            # terminal). Automated/codex turns (state_only) also idle and never
            # push. The idle transition is immediate for both.
            agent_state.update(session_id, machine, project, "idle", event_ts=client_ts, fleet_rev=fleet_rev)
            if data.get("state_only"):
                return {"ok": True, "state": "idle"}
            # Interactive stop: never push "Response complete" now. The Stop hook
            # fires at the end of EVERY main-loop turn, including handoff turns,
            # so hold the built receipt and let the quiet-window flush decide.
            _store_pending_receipt(session_id, _build_ha_payload(data),
                                   data.get("devices"))
            # Best-effort ORBB light: go idle if no active sessions remain.
            try:
                from .light import signal_idle
                await signal_idle()
            except Exception:
                pass
            return {"ok": True, "state": "idle"}

    if "event" in data:
        ha_payload = _build_ha_payload(data)
        devices = data.get("devices")
        if event in WAITING_EVENTS:
            ha_payload = _aggregate_waiting_payload(ha_payload)
    else:
        ha_payload = dict(data)
        devices = ha_payload.pop("devices", None)

    errors = await _relay_to_ha(ha_payload, devices)

    if event in WAITING_EVENTS and not errors:
        # The push went out; close this waiting spell for every session it
        # covered (an aggregate push covers all currently-waiting ones).
        agent_state.mark_waiting_notified(session_id)
        for sid in agent_state.waiting_sessions():
            agent_state.mark_waiting_notified(sid)

    if errors:
        return JSONResponse({"ok": False, "errors": errors}, status_code=502)
    return {"ok": True}
