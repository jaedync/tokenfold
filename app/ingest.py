"""POST /api/ingest - parse events, store, dedup.
POST /api/usage - store OAuth usage data from client.
"""

import json
import re
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Request

from .auth import require_api_key
from .config import TZ_NAME
from .db import get_conn
from .models import BackfillRequest, CursorState, IngestRequest, IngestResponse

router = APIRouter()
TZ = ZoneInfo(TZ_NAME)


def _parse_ts(ts_str: str) -> tuple[datetime, float, str] | None:
    """Parse ISO timestamp -> (datetime, epoch, day_str)."""
    if not ts_str:
        return None
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        epoch = ts.timestamp()
        day = ts.astimezone(TZ).strftime("%Y-%m-%d")
        return ts, epoch, day
    except (ValueError, TypeError):
        return None


def _is_human_prompt(rec: dict) -> bool:
    if rec.get("type") != "user" or rec.get("userType") != "external":
        return False
    content = rec.get("message", {}).get("content")
    if isinstance(content, str):
        return not content.startswith("<")
    if isinstance(content, list):
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            return False
        texts = [b.get("text", "") for b in content
                 if isinstance(b, dict) and b.get("type") == "text"]
        if texts:
            return not texts[0].startswith("<")
    return False


def _extract_event(rec: dict, machine: str, project_dir: str,
                   account: dict | None = None) -> dict | None:
    """Extract an events row from a raw JSONL record."""
    uuid = rec.get("uuid")
    rtype = rec.get("type", "")
    ts_str = rec.get("timestamp", "")
    parsed = _parse_ts(ts_str)
    if not parsed or not uuid:
        return None
    ts_dt, ts_epoch, day = parsed

    row = {
        "uuid": uuid,
        "type": rtype,
        "subtype": None,
        "timestamp": ts_str,
        "ts_epoch": ts_epoch,
        "day": day,
        "session_id": rec.get("sessionId"),
        "parent_uuid": rec.get("parentUuid"),
        "is_sidechain": 1 if rec.get("isSidechain") else 0,
        "user_type": rec.get("userType"),
        "cwd": rec.get("cwd"),
        "git_branch": rec.get("gitBranch"),
        "version": rec.get("version"),
        "slug": rec.get("slug"),
        "agent_id": rec.get("agentId"),
        "permission_mode": rec.get("permissionMode"),
        "source_machine": machine,
        "project_dir": project_dir,
        "account_email": (account or {}).get("account_email"),
        "org_name": (account or {}).get("org_name"),
        "plan": (account or {}).get("plan"),
        "rate_limit_tier": (account or {}).get("rate_limit_tier"),
        "org_type": (account or {}).get("org_type"),
        "org_uuid": (account or {}).get("org_uuid"),
        "model": None,
        "message_id": None,
        "request_id": rec.get("requestId"),
        "stop_reason": None,
        "api_error": None,
        "is_api_error": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
        "cache_ephemeral_5m": 0,
        "cache_ephemeral_1h": 0,
        "web_search_requests": 0,
        "web_fetch_requests": 0,
        "service_tier": None,
        "speed": None,
        "inference_geo": None,
        "has_text": 0,
        "has_thinking": 0,
        "has_tool_use": 0,
        "has_tool_result": 0,
        "has_image": 0,
        "is_human_prompt": 0,
        "text_length": 0,
        "thinking_length": 0,
        "level": None,
        "duration_ms": None,
        "error_status": None,
        "retry_attempt": None,
        "max_retries": None,
        "progress_type": None,
        "hook_event": None,
        "hook_name": None,
        "tool_use_id_ref": None,
        "file_op_type": None,
        "file_path": None,
        "queue_operation": None,
    }

    msg = rec.get("message", {})
    if not isinstance(msg, dict):
        msg = {}

    if rtype == "assistant":
        row["model"] = msg.get("model")
        row["message_id"] = msg.get("id")
        row["stop_reason"] = msg.get("stop_reason")
        usage = msg.get("usage", {})
        if isinstance(usage, dict):
            row["input_tokens"] = usage.get("input_tokens", 0)
            row["output_tokens"] = usage.get("output_tokens", 0)
            row["cache_creation_tokens"] = usage.get("cache_creation_input_tokens", 0)
            row["cache_read_tokens"] = usage.get("cache_read_input_tokens", 0)
            # Real transcript shape (verified against live Claude Code output):
            # usage.cache_creation = {ephemeral_5m_input_tokens, ephemeral_1h_input_tokens}.
            # The flat *_5m/*_1h keys are kept as a fallback for older payloads.
            _cc_obj = usage.get("cache_creation")
            if isinstance(_cc_obj, dict):
                row["cache_ephemeral_5m"] = _cc_obj.get("ephemeral_5m_input_tokens", 0)
                row["cache_ephemeral_1h"] = _cc_obj.get("ephemeral_1h_input_tokens", 0)
            else:
                row["cache_ephemeral_5m"] = usage.get("cache_creation_input_tokens_5m", 0)
                row["cache_ephemeral_1h"] = usage.get("cache_creation_input_tokens_1h", 0)
            # Server-tool usage: usage.server_tool_use = {web_search_requests,
            # web_fetch_requests}. Web search bills $10/1k requests on top of
            # token cost; fetch is free but tracked. Counts are untrusted —
            # a non-int would raise at the sqlite bind and 500 the batch.
            _stu = usage.get("server_tool_use")
            if isinstance(_stu, dict):
                _ws = _stu.get("web_search_requests")
                _wf = _stu.get("web_fetch_requests")
                if isinstance(_ws, int) and 0 <= _ws < 10**9:
                    row["web_search_requests"] = _ws
                if isinstance(_wf, int) and 0 <= _wf < 10**9:
                    row["web_fetch_requests"] = _wf
            # service_tier/speed/inference_geo come from untrusted transcript JSON:
            # coerce non-strings to NULL (a dict/list would raise at sqlite bind
            # and 500 the whole batch) and truncate to a sane length.
            # Realistic values are short tokens ('standard', 'priority', 'fast', 'us').
            _tier = usage.get("service_tier")
            row["service_tier"] = (_tier[:32] if isinstance(_tier, str) else None)
            _spd = usage.get("speed")
            row["speed"] = (_spd[:32] if isinstance(_spd, str) else None)
            _geo = usage.get("inference_geo")
            row["inference_geo"] = (_geo[:32] if isinstance(_geo, str) else None)

        content = msg.get("content", [])
        if isinstance(content, list):
            for blk in content:
                if not isinstance(blk, dict):
                    continue
                bt = blk.get("type", "")
                if bt == "text":
                    row["has_text"] = 1
                    row["text_length"] += len(blk.get("text", ""))
                elif bt == "thinking":
                    row["has_thinking"] = 1
                    row["thinking_length"] += len(blk.get("thinking", ""))
                elif bt == "tool_use":
                    row["has_tool_use"] = 1
                elif bt == "image":
                    row["has_image"] = 1

    elif rtype == "user":
        row["is_human_prompt"] = 1 if _is_human_prompt(rec) else 0
        content = msg.get("content", [])
        if isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict):
                    bt = blk.get("type", "")
                    if bt == "tool_result":
                        row["has_tool_result"] = 1
                    elif bt == "text":
                        row["has_text"] = 1
                        row["text_length"] += len(blk.get("text", ""))
                    elif bt == "image":
                        row["has_image"] = 1
        elif isinstance(content, str):
            row["has_text"] = 1
            row["text_length"] = len(content)

    elif rtype == "system":
        row["subtype"] = rec.get("subtype")
        row["level"] = rec.get("level")
        row["duration_ms"] = rec.get("duration_ms") or rec.get("durationMs")
        row["error_status"] = rec.get("errorStatus")
        row["retry_attempt"] = rec.get("retryAttempt")
        row["max_retries"] = rec.get("maxRetries")
        if rec.get("apiError"):
            row["api_error"] = str(rec["apiError"])
            row["is_api_error"] = 1

    elif rtype == "progress":
        row["progress_type"] = rec.get("progressType")
        row["hook_event"] = rec.get("hookEvent")
        row["hook_name"] = rec.get("hookName")
        row["tool_use_id_ref"] = rec.get("toolUseId")

    elif rtype in ("create", "update"):
        row["file_op_type"] = rtype
        row["file_path"] = rec.get("filePath")

    elif rtype == "queue-operation":
        row["queue_operation"] = rec.get("operation")

    elif rtype == "file-history-snapshot":
        pass  # Store with base fields only

    return row


def _extract_tool_uses(rec: dict, event_uuid: str, machine: str,
                       session_id: str | None, ts_str: str,
                       ts_epoch: float, day: str) -> list[dict]:
    """Extract tool_use blocks from an assistant event."""
    tools = []
    msg = rec.get("message", {})
    if not isinstance(msg, dict):
        return tools
    content = msg.get("content", [])
    if not isinstance(content, list):
        return tools
    for blk in content:
        if not isinstance(blk, dict) or blk.get("type") != "tool_use":
            continue
        tid = blk.get("id")
        if not tid:
            continue
        tools.append({
            "tool_use_id": tid,
            "event_uuid": event_uuid,
            "session_id": session_id,
            "source_machine": machine,
            "name": blk.get("name", "unknown"),
            "timestamp": ts_str,
            "ts_epoch": ts_epoch,
            "day": day,
            "result_event_uuid": None,
            "is_error": 0,
            "duration_ms": None,
        })
    return tools


EVENT_COLS = [
    "uuid", "type", "subtype", "timestamp", "ts_epoch", "day",
    "session_id", "parent_uuid", "is_sidechain", "user_type",
    "cwd", "git_branch", "version", "slug", "agent_id", "permission_mode",
    "source_machine", "project_dir",
    "account_email", "org_name", "plan", "rate_limit_tier", "org_type", "org_uuid",
    "model", "message_id", "request_id", "stop_reason", "api_error", "is_api_error",
    "input_tokens", "output_tokens", "cache_creation_tokens", "cache_read_tokens",
    "cache_ephemeral_5m", "cache_ephemeral_1h",
    "web_search_requests", "web_fetch_requests",
    "service_tier", "speed", "inference_geo",
    "has_text", "has_thinking", "has_tool_use", "has_tool_result", "has_image",
    "is_human_prompt", "text_length", "thinking_length",
    "level", "duration_ms", "error_status", "retry_attempt", "max_retries",
    "progress_type", "hook_event", "hook_name", "tool_use_id_ref",
    "file_op_type", "file_path", "queue_operation",
]

TOOL_COLS = [
    "tool_use_id", "event_uuid", "session_id", "source_machine",
    "name", "timestamp", "ts_epoch", "day",
    "result_event_uuid", "is_error", "duration_ms",
]

_EVENT_SQL = (
    f"INSERT OR IGNORE INTO events({','.join(EVENT_COLS)}) "
    f"VALUES({','.join('?' for _ in EVENT_COLS)})"
)
_TOOL_SQL = (
    f"INSERT OR IGNORE INTO tool_uses({','.join(TOOL_COLS)}) "
    f"VALUES({','.join('?' for _ in TOOL_COLS)})"
)


@router.post("/api/ingest", response_model=IngestResponse,
             dependencies=[Depends(require_api_key)])
async def ingest(req: IngestRequest):

    conn = get_conn()
    accepted = 0
    duplicates = 0
    touched_days: set[str] = set()

    event_rows = []
    tool_rows = []

    account = {
        "account_email": req.account_email,
        "org_name": req.org_name,
        "plan": req.plan,
        "rate_limit_tier": req.rate_limit_tier,
        "org_type": req.org_type,
        "org_uuid": req.org_uuid,
    }
    ai_titles: dict[str, str] = {}
    for raw in req.events:
        # ai-title records carry the session's AI-assigned name but no
        # uuid/timestamp — capture them here (last one in the batch wins).
        if raw.get("type") == "ai-title":
            sid, title = raw.get("sessionId"), raw.get("aiTitle")
            if isinstance(sid, str) and sid and isinstance(title, str) and title:
                ai_titles[sid] = title[:256]
            continue
        row = _extract_event(raw, req.machine, req.project_dir, account)
        if row is None:
            continue
        event_rows.append(tuple(row[c] for c in EVENT_COLS))
        touched_days.add(row["day"])

        # Extract tool uses from assistant events
        if row["type"] == "assistant" and row["has_tool_use"]:
            tools = _extract_tool_uses(
                raw, row["uuid"], req.machine,
                row["session_id"], row["timestamp"],
                row["ts_epoch"], row["day"],
            )
            for t in tools:
                tool_rows.append(tuple(t[c] for c in TOOL_COLS))

    # Batch insert
    try:
        cur = conn.cursor()
        for erow in event_rows:
            try:
                cur.execute(_EVENT_SQL, erow)
                if cur.rowcount > 0:
                    accepted += 1
                else:
                    duplicates += 1
            except sqlite3.IntegrityError:
                duplicates += 1

        for trow in tool_rows:
            try:
                cur.execute(_TOOL_SQL, trow)
            except sqlite3.IntegrityError:
                pass

        # Upsert AI session titles
        now = datetime.now(TZ).isoformat()
        for sid, title in ai_titles.items():
            cur.execute(
                "INSERT OR REPLACE INTO session_titles(session_id, title, updated_at) "
                "VALUES(?, ?, ?)", (sid, title, now))

        # Update sync cursor
        new_line_num = req.cursor.last_line_num + len(req.events)
        last_ts = None
        if req.events:
            last_ts = req.events[-1].get("timestamp")
        conn.execute(
            "INSERT OR REPLACE INTO sync_cursors(machine, project_dir, session_file, last_line_num, last_timestamp, updated_at) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (req.machine, req.project_dir, req.session_file, new_line_num, last_ts, now),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    # Recompute the summary for EVERY day this batch touched — a newly
    # connected machine pushes months of historical transcripts, and only
    # re-rolling "today" left those events invisible in the daily rollup
    # (stale daily table / heatmap / month counter) until a manual re-roll.
    if accepted > 0:
        from .summarizer import summarize_days
        from .aggregator import trigger_eager_rebuild
        today = datetime.now(TZ).strftime("%Y-%m-%d")
        summarize_days(sorted(touched_days | {today}))
        trigger_eager_rebuild()

    return IngestResponse(
        accepted=accepted,
        duplicates=duplicates,
        cursor=CursorState(last_line_num=req.cursor.last_line_num + len(req.events)),
    )


@router.post("/api/backfill", dependencies=[Depends(require_api_key)])
async def backfill(req: BackfillRequest):
    """Repair historical rows from a machine's local transcripts: set the
    cache-tier split and server-tool request counts on events where they are
    still unset (never clobbers real data) and upsert AI session titles that
    predate ai-title capture (an existing title wins — live ingest is fresher
    than a backfill). Re-rolls every day whose events changed so stored costs
    correct themselves."""
    conn = get_conn()
    cur = conn.cursor()
    updated_events = 0
    updated_server_tools = 0
    touched_days: set[str] = set()
    try:
        for uuid, pair in req.cache_tiers.items():
            if not (isinstance(pair, list) and len(pair) == 2):
                continue
            c5m, c1h = pair
            if not all(isinstance(x, int) and 0 <= x < 10**12 for x in (c5m, c1h)):
                continue
            if c5m == 0 and c1h == 0:
                continue
            row = cur.execute(
                "SELECT day FROM events WHERE uuid=? "
                "AND COALESCE(cache_ephemeral_5m,0)=0 "
                "AND COALESCE(cache_ephemeral_1h,0)=0", (uuid,)).fetchone()
            if row is None:
                continue
            cur.execute(
                "UPDATE events SET cache_ephemeral_5m=?, cache_ephemeral_1h=? "
                "WHERE uuid=?", (c5m, c1h, uuid))
            updated_events += 1
            touched_days.add(row["day"])

        # Server-tool request counts: same fill-only-unset contract as the
        # cache-tier split above.
        for uuid, pair in req.server_tools.items():
            if not (isinstance(pair, list) and len(pair) == 2):
                continue
            ws, wf = pair
            if not all(isinstance(x, int) and 0 <= x < 10**9 for x in (ws, wf)):
                continue
            if ws == 0 and wf == 0:
                continue
            row = cur.execute(
                "SELECT day FROM events WHERE uuid=? "
                "AND COALESCE(web_search_requests,0)=0 "
                "AND COALESCE(web_fetch_requests,0)=0", (uuid,)).fetchone()
            if row is None:
                continue
            cur.execute(
                "UPDATE events SET web_search_requests=?, web_fetch_requests=? "
                "WHERE uuid=?", (ws, wf, uuid))
            updated_server_tools += 1
            touched_days.add(row["day"])

        updated_titles = 0
        now = datetime.now(TZ).isoformat()
        for sid, title in req.titles.items():
            if not (isinstance(sid, str) and sid and isinstance(title, str) and title):
                continue
            cur.execute(
                "INSERT INTO session_titles(session_id, title, updated_at) "
                "VALUES(?, ?, ?) ON CONFLICT(session_id) DO NOTHING",
                (sid, title[:256], now))
            updated_titles += cur.rowcount if cur.rowcount > 0 else 0
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    # Explicit final-pass days (validated: strict YYYY-MM-DD only)
    valid_day = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    final_days = {d for d in req.reroll_days if valid_day.match(d)}
    days_to_roll = (touched_days if req.reroll else set()) | final_days
    if days_to_roll:
        from .summarizer import summarize_days
        summarize_days(sorted(days_to_roll))
    if days_to_roll or updated_titles:
        from .aggregator import trigger_eager_rebuild
        trigger_eager_rebuild()

    return {
        "updated_events": updated_events,
        "updated_server_tools": updated_server_tools,
        "updated_titles": updated_titles,
        "touched_days": sorted(touched_days),
    }


@router.post("/api/usage", dependencies=[Depends(require_api_key)])
async def store_usage(request: Request):
    """Store OAuth usage data pushed by the client."""

    body = await request.json()
    usage = body.get("usage")
    if not isinstance(usage, dict):
        raise HTTPException(status_code=400, detail="Missing usage data")

    conn = get_conn()
    now = datetime.now(ZoneInfo(TZ_NAME)).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        ("oauth_usage", json.dumps({"data": usage, "updated_at": now})),
    )
    conn.commit()

    # Rate limit data is now served from /api/rate-limits directly,
    # no need to rebuild the full dashboard cache for usage updates.

    return {"status": "ok", "updated_at": now}
