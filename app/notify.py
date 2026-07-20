"""Notification relay — forwards Claude Code hook events to Home Assistant.

Codex support is basic: the hook only sends a bare stop event (no model/usage
data), so notifications show "Response complete" without cost or model info.
"""

import hmac
import logging
import secrets
import sys

import httpx
from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse

from . import agent_state
from .config import (
    AGENT_PRESENCE_DAMPING_S,
    HA_DEVICES,
    HA_TOKEN,
    HA_URL,
    NOTIFY_TOKEN,
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

        if event == "working":
            # Pure state transition: never a push. This is what clears a
            # waiting spell and feeds the presence signal.
            agent_state.update(session_id, machine, project, "working", event_ts=client_ts)
            return {"ok": True, "state": "working"}

        if event in WAITING_EVENTS:
            agent_state.update(session_id, machine, project, "waiting", event_ts=client_ts)
            reason = _waiting_push_decision(session_id)
            if reason:
                return {"ok": True, "suppressed": reason}

        if event == "gone":
            # SessionEnd: state-only, never a push. Deletion ignores
            # client_ts ordering on purpose: gone is terminal, and a session
            # id is never reused after SessionEnd.
            agent_state.remove(session_id)
            return {"ok": True, "state": "gone"}

        if event == "stop":
            agent_state.update(session_id, machine, project, "idle", event_ts=client_ts)
            if data.get("state_only"):
                # Client reported a transition with no push-worthy payload
                # (e.g. an automated session's turn ended). State is now
                # recorded; policy says nothing goes to HA.
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

    # Best-effort: clean up stale ORBB sessions and go idle if none remain
    if data.get("event") == "stop":
        try:
            from .light import signal_idle
            await signal_idle()
        except Exception:
            pass

    if errors:
        return JSONResponse({"ok": False, "errors": errors}, status_code=502)
    return {"ok": True}
