"""ORBB activity light — changes color based on Claude Code activity state.

Tracks active sessions by ID. Light is "working" (blue) when any session is
active, "idle" (amber) only when all sessions have stopped. Stale sessions
auto-expire after ORBB_SESSION_TTL seconds.

Uses Home Assistant light/turn_on service with gentle transitions.
Entity: light.orbb (IKEA ZHA bulb, supports xy + color_temp modes).
"""

import logging
import time

import httpx
from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse

from .config import (
    HA_TOKEN,
    HA_URL,
    ORBB_ENTITY,
    ORBB_IDLE_COLOR,
    ORBB_SESSION_TTL,
    ORBB_TRANSITION,
    ORBB_WORKING_COLOR,
)
from .notify import _check_auth

router = APIRouter()
log = logging.getLogger(__name__)

# Active session tracking: session_id → last_seen_timestamp
# Note: in-memory dict — works with single-worker uvicorn only.
_active_sessions: dict[str, float] = {}


def _cleanup_stale():
    """Remove sessions that haven't reported in longer than ORBB_SESSION_TTL."""
    now = time.monotonic()
    expired = [sid for sid, ts in _active_sessions.items() if now - ts > ORBB_SESSION_TTL]
    for sid in expired:
        del _active_sessions[sid]
        log.info("ORBB session expired: %s", sid)


async def signal_idle():
    """Clean up stale sessions and set light to idle if no sessions remain.

    Called from notify.py as a best-effort fallback on stop events.
    """
    _cleanup_stale()
    if not _active_sessions:
        await _set_light_color(ORBB_IDLE_COLOR)


async def _set_light_color(rgb: list[int], transition: int = ORBB_TRANSITION):
    """Call HA light/turn_on to set ORBB color with transition."""
    if not HA_URL or not HA_TOKEN:
        log.debug("HA not configured, skipping light control")
        return

    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "entity_id": ORBB_ENTITY,
        "rgb_color": rgb,
        "transition": transition,
    }

    async with httpx.AsyncClient(timeout=8) as client:
        r = await client.post(
            f"{HA_URL}/api/services/light/turn_on",
            headers=headers,
            json=payload,
        )
        r.raise_for_status()
        log.info("ORBB set to rgb=%s transition=%ds", rgb, transition)


@router.post("/api/light")
async def light_state(request: Request, authorization: str | None = Header(default=None)):
    if not _check_auth(authorization):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    data = await request.json()
    state = data.get("state", "").lower()
    session_id = data.get("session_id", "anonymous")

    if state not in ("working", "idle"):
        return JSONResponse({"error": "state must be 'working' or 'idle'"}, status_code=400)

    _cleanup_stale()
    was_active = len(_active_sessions) > 0

    if state == "working":
        _active_sessions[session_id] = time.monotonic()
        is_active = True
    else:  # idle
        _active_sessions.pop(session_id, None)
        is_active = len(_active_sessions) > 0

    # Only change the light when transitioning between states
    changed = False
    try:
        if is_active and not was_active:
            await _set_light_color(ORBB_WORKING_COLOR)
            changed = True
        elif not is_active and was_active:
            await _set_light_color(ORBB_IDLE_COLOR)
            changed = True
    except Exception as e:
        log.warning("Failed to set ORBB light: %s", e)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)

    return {
        "ok": True,
        "state": state,
        "session_id": session_id,
        "active_sessions": len(_active_sessions),
        "light_changed": changed,
    }
