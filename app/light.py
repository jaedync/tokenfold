"""ORBB activity light — changes color based on Claude Code activity state.

Tracks active sessions by ID. Light is "working" when any session is
active, "idle" only when all sessions have stopped. Stale sessions
auto-expire after ORBB_SESSION_TTL seconds via a background watchdog.

Uses Home Assistant light/turn_on service with gentle transitions.
Entity: light.orbb (IKEA ZHA bulb, supports xy + color_temp modes).
"""

import asyncio
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
    ORBB_IDLE_KELVIN,
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

# Background watchdog task handle
_watchdog_task: asyncio.Task | None = None

_WATCHDOG_INTERVAL = 60  # seconds between stale-session sweeps


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
        await _set_light(rgb=ORBB_IDLE_COLOR, kelvin=ORBB_IDLE_KELVIN)


async def _set_light(
    *,
    rgb: list[int] | None = None,
    kelvin: int | None = None,
    transition: int = ORBB_TRANSITION,
):
    """Call HA light/turn_on with either rgb_color or color_temp_kelvin."""
    if not HA_URL or not HA_TOKEN:
        log.debug("HA not configured, skipping light control")
        return

    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }
    payload: dict = {
        "entity_id": ORBB_ENTITY,
        "transition": transition,
    }
    if rgb:
        payload["rgb_color"] = rgb
    elif kelvin:
        payload["color_temp_kelvin"] = kelvin

    async with httpx.AsyncClient(timeout=8) as client:
        r = await client.post(
            f"{HA_URL}/api/services/light/turn_on",
            headers=headers,
            json=payload,
        )
        r.raise_for_status()
        log.info("ORBB set to %s transition=%ds", payload, transition)


# ── Background watchdog ──────────────────────────────────────────────


async def _watchdog_loop():
    """Periodically clean stale sessions and transition light to idle."""
    while True:
        await asyncio.sleep(_WATCHDOG_INTERVAL)
        try:
            was_active = len(_active_sessions) > 0
            _cleanup_stale()
            is_active = len(_active_sessions) > 0
            if was_active and not is_active:
                await _set_light(rgb=ORBB_IDLE_COLOR, kelvin=ORBB_IDLE_KELVIN)
                log.info("ORBB watchdog: all sessions expired, set to idle")
        except Exception as e:
            log.warning("ORBB watchdog error: %s", e)


def start_watchdog():
    """Start the background watchdog task. Called during app lifespan."""
    global _watchdog_task
    _watchdog_task = asyncio.create_task(_watchdog_loop())
    log.info("ORBB watchdog started (interval=%ds, ttl=%ds)", _WATCHDOG_INTERVAL, ORBB_SESSION_TTL)


def stop_watchdog():
    """Cancel the background watchdog task. Called during app shutdown."""
    global _watchdog_task
    if _watchdog_task:
        _watchdog_task.cancel()
        _watchdog_task = None


async def init_light():
    """Set light to idle on startup so we always start from a known state."""
    try:
        await _set_light(rgb=ORBB_IDLE_COLOR, kelvin=ORBB_IDLE_KELVIN)
        log.info("ORBB initialized to idle on startup")
    except Exception as e:
        log.warning("ORBB init failed (HA may be unreachable): %s", e)


# ── Endpoints ────────────────────────────────────────────────────────


@router.get("/api/light")
async def light_status():
    now = time.monotonic()
    sessions = {
        sid: {"age_s": round(now - ts, 1), "ttl_remaining_s": round(ORBB_SESSION_TTL - (now - ts), 1)}
        for sid, ts in _active_sessions.items()
    }

    return {
        "state": "working" if _active_sessions else "idle",
        "active_sessions": len(_active_sessions),
        "sessions": sessions,
        "config": {
            "entity": ORBB_ENTITY,
            "working_color": ORBB_WORKING_COLOR,
            "idle_color": ORBB_IDLE_COLOR,
            "transition_s": ORBB_TRANSITION,
            "session_ttl_s": ORBB_SESSION_TTL,
        },
    }


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

    if state == "working":
        _active_sessions[session_id] = time.monotonic()
        is_active = True
    else:  # idle
        _active_sessions.pop(session_id, None)
        is_active = len(_active_sessions) > 0

    # Always reassert the correct color — acts as a heartbeat so the light
    # stays accurate even if HA drifted or someone changed it manually.
    try:
        if is_active:
            await _set_light(rgb=ORBB_WORKING_COLOR)
        else:
            await _set_light(rgb=ORBB_IDLE_COLOR, kelvin=ORBB_IDLE_KELVIN)
    except Exception as e:
        log.warning("Failed to set ORBB light: %s", e)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)

    return {
        "ok": True,
        "state": state,
        "session_id": session_id,
        "active_sessions": len(_active_sessions),
        "light_color": ORBB_WORKING_COLOR if is_active else {"kelvin": ORBB_IDLE_KELVIN},
    }
