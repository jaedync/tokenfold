"""Notification relay — forwards Claude Code hook events to Home Assistant.

Codex support is basic: the hook only sends a bare stop event (no model/usage
data), so notifications show "Response complete" without cost or model info.
"""

import logging
import secrets
import sys

import httpx
from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse

from .config import HA_DEVICES, HA_TOKEN, HA_URL, NOTIFY_TOKEN, STATS_API_KEY
from .db import get_conn
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
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
        ("notify_token", _notify_token),
    )
    conn.commit()
    print(f"[notify] Generated token: {_notify_token}", flush=True)


def _check_auth(authorization: str | None) -> bool:
    if not authorization or not authorization.startswith("Bearer "):
        return False
    token = authorization[7:]
    if _notify_token and token == _notify_token:
        return True
    if STATS_API_KEY and token == STATS_API_KEY:
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

    return compute_cost(dname, inp, out, cw, cr)


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


@router.post("/api/notify")
async def notify(request: Request, authorization: str | None = Header(default=None)):
    if not _check_auth(authorization):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    data = await request.json()

    if "event" in data:
        ha_payload = _build_ha_payload(data)
        devices = data.get("devices")
    else:
        ha_payload = dict(data)
        devices = ha_payload.pop("devices", None)

    errors = await _relay_to_ha(ha_payload, devices)

    if errors:
        return JSONResponse({"ok": False, "errors": errors}, status_code=502)
    return {"ok": True}
