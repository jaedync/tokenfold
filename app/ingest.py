"""POST /api/ingest - parse events, store, dedup.
POST /api/usage - store OAuth usage data from client.
"""

import json
import logging
import re
import hashlib
import sqlite3
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Request

from .auth import require_api_key
from .config import TZ_NAME
from .db import get_conn, write_txn
from .models import (BackfillRequest, CursorState, IngestRequest, IngestResponse,
                     PiIngestRequest, ProviderUsageRequest)
from .sigheader import decode_header, split_signature

router = APIRouter()
TZ = ZoneInfo(TZ_NAME)
logger = logging.getLogger(__name__)


def _safe_count(value, cap=10**12) -> int:
    """Coerce an untrusted transcript token count to a safe int.

    sqlite3 raises InterfaceError on a dict/list bind (failing the whole
    batch) and silently STORES a str in the INTEGER column (poisoning every
    SUM over it) — same contract as the service_tier/speed coercion below.
    bool is an int subclass; True is not 1 token.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    if not 0 <= value < cap:
        return 0
    return value


# Bounds for the signature-header columns. The blob itself never reaches the
# DB; only the ~200-byte plaintext header does (4096 leaves generous headroom
# for a format change without letting a hostile payload store a novel).
MAX_SIG_HEADER_CHARS = 4096
MAX_SIG_BLOB_CHARS = 65536      # raw-signature fallback: bound the parse work
MAX_SIG_VERSION = 10**6
MAX_SIG_CIPHER_LEN = 10**9


def _sig_columns(sig_version, sig_header, sig_cipher_len) -> dict | None:
    """Validate a client-decoded signature header -> the five events columns.

    Returns None when the header is unusable, so the row simply keeps its
    NULLs (a backfill can fill them in later). Everything here comes from
    untrusted transcript JSON: a dict/list bound to sqlite raises and would
    500 the whole batch, and a str in an INTEGER column poisons every SUM
    over it, so each value is coerced, not trusted.
    """
    if not isinstance(sig_header, str) or not sig_header:
        return None
    if len(sig_header) > MAX_SIG_HEADER_CHARS:
        return None
    decoded = decode_header(sig_header)
    if not decoded["fields"]:
        return None  # nothing parsed out of it, not a header we understand
    return {
        "served_model": decoded["served_model"],
        "sig_version": _safe_count(sig_version, cap=MAX_SIG_VERSION),
        "sig_header": sig_header,
        "sig_cipher_len": _safe_count(sig_cipher_len, cap=MAX_SIG_CIPHER_LEN),
        "sig_fields": decoded["fields"],
    }


def _signature_fields(blk: dict) -> dict:
    """Signature columns for one thinking block, or {} when it carries none.

    Prefers the fields a current client already split out (the blob is dropped
    client-side: it is 7% of all uploaded bytes and the header is all the
    plaintext there is). Falls back to splitting a raw `signature` server-side
    so older clients, and any transcript replayed straight through, keep
    working. One decoder either way.
    """
    header = blk.get("sig_header")
    if isinstance(header, str) and header:
        return _sig_columns(blk.get("sig_version"), header,
                            blk.get("sig_cipher_len")) or {}

    raw = blk.get("signature")
    if not isinstance(raw, str) or not raw or len(raw) > MAX_SIG_BLOB_CHARS:
        return {}
    # A new client sends the "[N chars]" placeholder here; split_signature
    # returns no header for it, so it falls out as {} like any other garbage.
    version, header_b64, cipher_len = split_signature(raw)
    if header_b64 is None:
        return {}
    return _sig_columns(version, header_b64, cipher_len) or {}


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
        "served_model": None,
        "sig_version": None,
        "sig_header": None,
        "sig_cipher_len": None,
        "sig_fields": None,
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
        "source_client": "claude-code",
        "provider": None,
        "api": None,
        "usage_kind": None,
        "reasoning_tokens": 0,
        "reported_cost_input": None,
        "reported_cost_output": None,
        "reported_cost_cache_read": None,
        "reported_cost_cache_write": None,
        "reported_cost_total": None,
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
            row["input_tokens"] = _safe_count(usage.get("input_tokens", 0))
            row["output_tokens"] = _safe_count(usage.get("output_tokens", 0))
            row["cache_creation_tokens"] = _safe_count(
                usage.get("cache_creation_input_tokens", 0))
            row["cache_read_tokens"] = _safe_count(
                usage.get("cache_read_input_tokens", 0))
            # Real transcript shape (verified against live Claude Code output):
            # usage.cache_creation = {ephemeral_5m_input_tokens, ephemeral_1h_input_tokens}.
            # The flat *_5m/*_1h keys are kept as a fallback for older payloads.
            _cc_obj = usage.get("cache_creation")
            if isinstance(_cc_obj, dict):
                row["cache_ephemeral_5m"] = _safe_count(
                    _cc_obj.get("ephemeral_5m_input_tokens", 0))
                row["cache_ephemeral_1h"] = _safe_count(
                    _cc_obj.get("ephemeral_1h_input_tokens", 0))
            else:
                row["cache_ephemeral_5m"] = _safe_count(
                    usage.get("cache_creation_input_tokens_5m", 0))
                row["cache_ephemeral_1h"] = _safe_count(
                    usage.get("cache_creation_input_tokens_1h", 0))
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
                    # One thinking block per assistant record in Claude Code
                    # transcripts; if that ever changes, the first block with
                    # a readable header wins (the spec's tie-break).
                    if row["sig_header"] is None:
                        row.update(_signature_fields(blk))
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
            "source_client": "claude-code",
            "provider": None,
            "api": None,
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
    "served_model", "sig_version", "sig_header", "sig_cipher_len", "sig_fields",
    "has_text", "has_thinking", "has_tool_use", "has_tool_result", "has_image",
    "is_human_prompt", "text_length", "thinking_length",
    "level", "duration_ms", "error_status", "retry_attempt", "max_retries",
    "progress_type", "hook_event", "hook_name", "tool_use_id_ref",
    "file_op_type", "file_path", "queue_operation",
    "source_client", "provider", "api", "usage_kind", "reasoning_tokens",
    "reported_cost_input", "reported_cost_output",
    "reported_cost_cache_read", "reported_cost_cache_write", "reported_cost_total",
]

TOOL_COLS = [
    "tool_use_id", "event_uuid", "session_id", "source_machine",
    "name", "timestamp", "ts_epoch", "day",
    "result_event_uuid", "is_error", "duration_ms",
    "source_client", "provider", "api",
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
def ingest(req: IngestRequest):
    # Plain `def` (threadpool), NOT async: this handler blocks on WRITE_LOCK
    # and can run summarize_days for minutes on a historical re-push — as a
    # coroutine that would freeze the event loop for every client.

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

    # Batch insert — one serialized write transaction (see db.write_txn).
    with write_txn() as conn:
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

    # Recompute the summary for EVERY day this batch touched — a newly
    # connected machine pushes months of historical transcripts, and only
    # re-rolling "today" left those events invisible in the daily rollup
    # (stale daily table / heatmap / month counter) until a manual re-roll.
    if accepted > 0:
        from .summarizer import summarize_days
        from .aggregator import trigger_eager_rebuild
        today = datetime.now(TZ).strftime("%Y-%m-%d")
        try:
            summarize_days(sorted(touched_days | {today}))
            trigger_eager_rebuild()
        except Exception:
            # The events above are already durable — a rebuild failure must
            # not 500 this request (the client would retry, dedupe to
            # accepted=0, and never re-roll these days). The rollup
            # self-heals on the next batch or hourly sweep.
            logger.exception(
                "post-ingest summary rebuild failed (days=%s) — events "
                "stored, rollup will self-heal on next batch/sweep",
                sorted(touched_days))

    return IngestResponse(
        accepted=accepted,
        duplicates=duplicates,
        cursor=CursorState(last_line_num=req.cursor.last_line_num + len(req.events)),
    )


_PI_USAGE_KINDS = {"assistant", "compaction", "branch_summary", "tool_usage"}


def _pi_namespace(account_class: str, machine: str, session_file: str,
                  session_id: str) -> str:
    """Stable, scope-isolated namespace for native Pi identifiers."""
    raw = "\0".join((account_class, machine, session_file, session_id)).encode()
    return "pi:" + hashlib.sha256(raw).hexdigest()[:40]


def _pi_id(namespace: str, value: str) -> str:
    return f"{namespace}:{value}"


def _pi_account(account_class: str) -> dict[str, str]:
    """Stable synthetic identity keeps Pi work/personal rollups disjoint."""
    return {
        "account_email": f"pi-{account_class}@dotfleet.local",
        "org_name": "dotfleet",
        "plan": "enterprise" if account_class == "work" else "personal",
        "org_type": f"dotfleet_{account_class}",
    }


def _pi_event_row(event, machine: str, project_dir: str, session_file: str,
                  account_class: str):
    parsed = _parse_ts(event.timestamp)
    if parsed is None:
        raise HTTPException(status_code=422, detail="events.timestamp must be ISO-8601")
    ts_dt, ts_epoch, day = parsed
    ns = _pi_namespace(account_class, machine, session_file, event.session_id)
    usage_kind = event.kind if event.kind in _PI_USAGE_KINDS else None
    usage = event.usage
    # Every usage-bearing Pi record is a canonical assistant row. A stable
    # fallback request id is essential for cost/token aggregation and replay.
    req_id = None
    if usage_kind:
        req_id = _pi_id(ns, event.request_id or "event:" + event.event_id)
    elif event.request_id:
        req_id = _pi_id(ns, event.request_id)
    model = event.model if usage_kind else event.model
    if usage_kind and not model:
        model = "<unknown>"
    account = _pi_account(account_class)
    row = {c: None for c in EVENT_COLS}
    row.update({
        "uuid": _pi_id(ns, event.event_id),
        "type": "assistant" if usage_kind else ("user" if event.kind == "tool_result" else event.kind),
        "subtype": None,
        "timestamp": event.timestamp,
        "ts_epoch": ts_epoch,
        "day": day,
        "session_id": _pi_id(ns, event.session_id),
        "parent_uuid": (_pi_id(ns, event.parent_event_id)
                         if event.parent_event_id else None),
        "is_sidechain": int(event.is_sidechain),
        "user_type": None,
        "cwd": None,
        "git_branch": None,
        "version": None,
        "slug": None,
        "agent_id": (_pi_id(ns, event.agent_id) if event.agent_id else None),
        "permission_mode": None,
        "source_machine": machine,
        "project_dir": project_dir,
        "account_email": account["account_email"],
        "org_name": account["org_name"],
        "plan": account["plan"],
        "rate_limit_tier": None,
        "org_type": account["org_type"],
        "org_uuid": None,
        "model": model,
        "message_id": None,
        "request_id": req_id,
        "stop_reason": event.stop_reason,
        "api_error": None,
        "is_api_error": 0,
        "input_tokens": usage.input if usage else 0,
        "output_tokens": usage.output if usage else 0,
        "cache_creation_tokens": usage.cache_write if usage else 0,
        "cache_read_tokens": usage.cache_read if usage else 0,
        "cache_ephemeral_5m": 0,
        "cache_ephemeral_1h": 0,
        "web_search_requests": 0,
        "web_fetch_requests": 0,
        "service_tier": None,
        "speed": None,
        "inference_geo": None,
        "has_text": int(event.has_text),
        "has_thinking": int(event.has_thinking),
        "has_tool_use": int(event.has_tool_use or bool(event.tools)),
        "has_tool_result": int(event.has_tool_result or event.kind == "tool_result"),
        "has_image": int(event.has_image),
        "is_human_prompt": int(event.kind == "user" and not event.is_sidechain),
        "text_length": event.text_length,
        "thinking_length": event.thinking_length,
        "source_client": "pi-agent",
        "provider": event.provider,
        "api": event.api,
        "usage_kind": usage_kind,
        "reasoning_tokens": usage.reasoning if usage else 0,
        "reported_cost_input": usage.cost_input if usage else None,
        "reported_cost_output": usage.cost_output if usage else None,
        "reported_cost_cache_read": usage.cost_cache_read if usage else None,
        "reported_cost_cache_write": usage.cost_cache_write if usage else None,
        "reported_cost_total": usage.cost_total if usage else None,
    })
    tools = []
    for tool in event.tools:
        tools.append({
            "tool_use_id": _pi_id(ns, tool.tool_use_id),
            "event_uuid": row["uuid"],
            "session_id": row["session_id"],
            "source_machine": machine,
            "name": tool.name,
            "timestamp": event.timestamp,
            "ts_epoch": ts_epoch,
            "day": day,
            "result_event_uuid": None,
            "is_error": 0,
            "duration_ms": None,
            "source_client": "pi-agent",
            "provider": event.provider,
            "api": event.api,
        })
    return row, tools


def _store_pi_rows(req: PiIngestRequest, event_rows: list[tuple],
                   tool_rows: list[tuple]) -> tuple[int, int]:
    accepted = duplicates = 0
    with write_txn() as conn:
        cur = conn.cursor()
        for values in event_rows:
            try:
                cur.execute(_EVENT_SQL, values)
                accepted += int(cur.rowcount > 0)
                duplicates += int(cur.rowcount <= 0)
            except sqlite3.IntegrityError:
                duplicates += 1
        for values in tool_rows:
            try:
                cur.execute(_TOOL_SQL, values)
            except sqlite3.IntegrityError:
                pass
        now = datetime.now(TZ).isoformat()
        cursor_machine = f"pi:{req.account_class}:{req.machine}"
        last_ts = req.events[-1].timestamp if req.events else None
        conn.execute(
            "INSERT OR REPLACE INTO sync_cursors(machine, project_dir, session_file, "
            "last_line_num, last_timestamp, updated_at) VALUES(?,?,?,?,?,?)",
            (cursor_machine, req.project_dir, req.session_file,
             req.cursor.last_line_num + len(req.events), last_ts, now),
        )
    return accepted, duplicates


@router.post("/api/ingest/pi", response_model=IngestResponse,
             dependencies=[Depends(require_api_key)])
def ingest_pi(req: PiIngestRequest):
    """Ingest privacy-scrubbed normalized Pi Agent events."""
    event_rows, tool_rows, touched_days = [], [], set()
    for event in req.events:
        row, tools = _pi_event_row(event, req.machine, req.project_dir,
                                   req.session_file, req.account_class)
        event_rows.append(tuple(row[c] for c in EVENT_COLS))
        touched_days.add(row["day"])
        tool_rows.extend(tuple(tool[c] for c in TOOL_COLS) for tool in tools)
    if len(tool_rows) > 20000:
        raise HTTPException(status_code=422, detail="too many tools in batch")

    accepted, duplicates = _store_pi_rows(req, event_rows, tool_rows)
    if accepted > 0:
        from .summarizer import summarize_days
        from .aggregator import trigger_eager_rebuild
        today = datetime.now(TZ).strftime("%Y-%m-%d")
        try:
            summarize_days(sorted(touched_days | {today}))
            trigger_eager_rebuild()
        except Exception:
            logger.exception("post-Pi-ingest summary rebuild failed; events are durable")
    return IngestResponse(
        accepted=accepted,
        duplicates=duplicates,
        cursor=CursorState(last_line_num=req.cursor.last_line_num + len(req.events)),
    )


@router.post("/api/backfill", dependencies=[Depends(require_api_key)])
def backfill(req: BackfillRequest):
    """Repair historical rows from a machine's local transcripts: set the
    cache-tier split, server-tool request counts and thinking-signature header
    on events where they are still unset (never clobbers real data) and upsert
    AI session titles that predate ai-title capture (an existing title wins,
    live ingest is fresher than a backfill). Re-rolls every day whose events
    changed so stored costs correct themselves."""
    updated_events = 0
    updated_server_tools = 0
    updated_sig_headers = 0
    touched_days: set[str] = set()
    with write_txn() as conn:
        cur = conn.cursor()
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

        # Signature headers: same fill-only-unset contract. These days are
        # deliberately NOT added to touched_days: no rollup or stored cost
        # reads served_model (the dashboard chip and /api/served-models both
        # query events directly), so re-rolling them would be pure cost.
        for uuid, triple in req.sig_headers.items():
            if not (isinstance(triple, list) and len(triple) == 3):
                continue
            cols = _sig_columns(triple[0], triple[1], triple[2])
            if cols is None:
                continue
            row = cur.execute(
                "SELECT day FROM events WHERE uuid=? AND sig_header IS NULL",
                (uuid,)).fetchone()
            if row is None:
                continue
            cur.execute(
                "UPDATE events SET served_model=?, sig_version=?, sig_header=?, "
                "sig_cipher_len=?, sig_fields=? WHERE uuid=?",
                (cols["served_model"], cols["sig_version"], cols["sig_header"],
                 cols["sig_cipher_len"], cols["sig_fields"], uuid))
            updated_sig_headers += 1

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

    # Explicit final-pass days (validated: strict YYYY-MM-DD only)
    valid_day = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    final_days = {d for d in req.reroll_days if valid_day.match(d)}
    days_to_roll = (touched_days if req.reroll else set()) | final_days
    if days_to_roll:
        from .summarizer import summarize_days
        summarize_days(sorted(days_to_roll))
    if days_to_roll or updated_titles or updated_sig_headers:
        from .aggregator import trigger_eager_rebuild
        trigger_eager_rebuild()

    return {
        "updated_events": updated_events,
        "updated_server_tools": updated_server_tools,
        "updated_sig_headers": updated_sig_headers,
        "updated_titles": updated_titles,
        "touched_days": sorted(touched_days),
    }


@router.post("/api/provider-usage", dependencies=[Depends(require_api_key)])
def store_provider_limits(req: ProviderUsageRequest):
    """Merge metadata-only quota snapshots reported by the Pi extension."""
    from .provider_usage import store_provider_usage
    providers = store_provider_usage(req.machine, req.account_class, req.limits)
    return {"status": "ok", "providers": providers}


@router.post("/api/usage", dependencies=[Depends(require_api_key)])
async def store_usage(request: Request):
    """Store OAuth usage data pushed by the client."""

    body = await request.json()
    usage = body.get("usage")
    if not isinstance(usage, dict):
        raise HTTPException(status_code=400, detail="Missing usage data")

    # Account-stomp guard (2026-07-09 incident): every machine pushes usage
    # fetched with ITS OWN OAuth token, and a machine logged into an
    # enterprise account gets a payload whose limit buckets are all null
    # (enterprise accounts have no Max limits) — under the blind REPLACE
    # below, that stomped the personal snapshot (gauges zeroed, scoped
    # limits gone, extra_usage surfacing org numbers) until the server
    # poller healed it up to 10 minutes later. A payload that normalizes
    # to ZERO usable buckets carries nothing any consumer reads (the oauth
    # block is personal-only by compliance), so it never overwrites.
    from .usage_buckets import normalize_usage_buckets
    if not normalize_usage_buckets(usage):
        machine = body.get("machine") if isinstance(
            body.get("machine"), str) else "unknown"
        machine = machine[:128]  # bound what lands in meta/meter rows
        print(f"[ingest] /api/usage ignored: no usable limit buckets "
              f"(machine={machine!r} — enterprise-account push?)",
              flush=True)
        # Retain the one billing-grade number these payloads carry:
        # extra_usage is Anthropic's server-side metered spend for the org's
        # current billing cycle (monthly_limit / used_credits in US cents).
        # It lands in its OWN meta key so the personal snapshot invariant
        # above stays intact; an absent/empty block never clobbers a prior
        # capture (nothing worth overwriting with).
        extra = usage.get("extra_usage")
        captured = isinstance(extra, dict) and bool(extra)
        if captured:
            with write_txn() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                    ("oauth_usage_enterprise", json.dumps({
                        "machine": machine,
                        "extra_usage": extra,
                        "updated_at":
                            datetime.now(ZoneInfo(TZ_NAME)).isoformat(),
                    })),
                )
            from .extra_usage import record_meter_reading
            record_meter_reading(conn, machine, extra, time.time())
            from .aggregator import trigger_eager_rebuild
            trigger_eager_rebuild()
        return {"status": "ignored_no_limits", "updated_at": None,
                "captured_extra_usage": captured}

    now = datetime.now(ZoneInfo(TZ_NAME)).isoformat()
    with write_txn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("oauth_usage", json.dumps({"data": usage, "updated_at": now})),
        )

    # Historize per-bucket readings — but ONLY on instances not locked to
    # enterprise scope: the compliance invariant says locked instances must
    # never PERSIST personal Max limit history (test_enterprise_only). The
    # meta snapshot write above stays ungated (existing behavior). Read
    # LOCKED_SCOPE fresh via sys.modules for importlib.reload safety, like
    # api.py does. Bucket-level validation lives inside record_limit_readings
    # (invalid buckets skipped, valid ones recorded, never raises).
    import sys
    if sys.modules["app.config"].LOCKED_SCOPE != "enterprise":
        from .limit_readings import record_limit_readings
        record_limit_readings(conn, usage, time.time(), "client")

    # The monthly hero/billing meter also depends on this observation, not
    # only the separately polled quota gauges.
    from .aggregator import trigger_eager_rebuild
    trigger_eager_rebuild()

    return {"status": "ok", "updated_at": now}
