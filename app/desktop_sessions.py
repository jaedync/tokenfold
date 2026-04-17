"""POST /api/desktop-metadata and the UPSERT logic for Claude Desktop session metadata."""

import json
import sqlite3
import time

from fastapi import APIRouter, Header, HTTPException

from .config import STATS_API_KEY
from .db import get_conn
from .models import (
    DesktopMetadataRequest,
    DesktopMetadataResponse,
)

router = APIRouter()


_JSON_FIELDS = {"enabled_mcp_tools", "remote_mcp_servers", "chrome_allowed_domains"}

_COLS = [
    "cli_session_id", "desktop_session_id", "source_machine",
    "title", "model", "effort", "permission_mode", "completed_turns",
    "is_archived", "cwd", "origin_cwd",
    "created_at_ms", "last_activity_at_ms",
    "enabled_mcp_tools", "remote_mcp_servers",
    "chrome_permission_mode", "chrome_allowed_domains",
    "updated_at_ms",
]

# COALESCE preserves existing value when the incoming field is NULL.
# Applied to every column except the primary key, last_activity_at_ms,
# and updated_at_ms (both are always set to the newer value).
_UPDATE_SET = ", ".join(
    f"{c} = COALESCE(excluded.{c}, desktop_sessions.{c})"
    for c in _COLS
    if c not in ("cli_session_id", "last_activity_at_ms", "updated_at_ms")
)
_UPDATE_SET += (
    ", last_activity_at_ms = excluded.last_activity_at_ms"
    ", updated_at_ms = excluded.updated_at_ms"
)

_UPSERT_SQL = f"""
INSERT INTO desktop_sessions ({", ".join(_COLS)})
VALUES ({", ".join("?" for _ in _COLS)})
ON CONFLICT(cli_session_id) DO UPDATE SET {_UPDATE_SET}
WHERE COALESCE(excluded.last_activity_at_ms, 0)
      >= COALESCE(desktop_sessions.last_activity_at_ms, 0)
"""


def _to_row(session: dict, machine: str, now_ms: int) -> tuple:
    """Turn a session dict into a tuple aligned with _COLS."""
    row = {c: session.get(c) for c in _COLS}
    row["source_machine"] = machine
    row["updated_at_ms"] = now_ms

    if row["is_archived"] is not None:
        row["is_archived"] = 1 if row["is_archived"] else 0

    for key in _JSON_FIELDS:
        if row[key] is not None:
            row[key] = json.dumps(row[key], separators=(",", ":"))

    return tuple(row[c] for c in _COLS)


def upsert_desktop_sessions(
    conn: sqlite3.Connection, machine: str, sessions: list
) -> dict:
    """Apply a batch of desktop session upserts. Returns counts."""
    now_ms = int(time.time() * 1000)
    inserted = updated = stale = 0

    for s in sessions:
        data = s if isinstance(s, dict) else s.model_dump()
        cli_id = data.get("cli_session_id")
        if not cli_id:
            continue

        prior = conn.execute(
            "SELECT last_activity_at_ms FROM desktop_sessions "
            "WHERE cli_session_id = ?",
            (cli_id,),
        ).fetchone()

        new_last = data.get("last_activity_at_ms") or 0
        prior_last = (prior[0] if prior else None) or 0

        if prior and new_last < prior_last:
            stale += 1
            continue

        conn.execute(_UPSERT_SQL, _to_row(data, machine, now_ms))

        if prior is None:
            inserted += 1
        else:
            updated += 1

    conn.commit()
    return {"inserted": inserted, "updated": updated, "ignored_stale": stale}


@router.post("/api/desktop-metadata", response_model=DesktopMetadataResponse)
async def ingest_desktop_metadata(
    req: DesktopMetadataRequest,
    x_api_key: str = Header(alias="X-API-Key"),
):
    if not STATS_API_KEY or x_api_key != STATS_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    conn = get_conn()
    result = upsert_desktop_sessions(
        conn, req.machine, [s.model_dump() for s in req.sessions]
    )
    return DesktopMetadataResponse(**result)
