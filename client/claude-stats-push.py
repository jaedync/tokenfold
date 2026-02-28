#!/usr/bin/env python3
"""Push Claude Code session events to a Tokenfold server.

Zero external dependencies - stdlib only.
Designed to run every 5 minutes via cron (Linux) or launchd (macOS).
"""

import json
import os
import socket
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ── Config ──
SERVER_URL = os.environ.get("TOKENFOLD_URL", os.environ.get("CLAUDE_STATS_URL", ""))
API_KEY = os.environ.get("TOKENFOLD_API_KEY", os.environ.get("CLAUDE_STATS_API_KEY", ""))
MACHINE_NAME = os.environ.get("TOKENFOLD_MACHINE", os.environ.get("CLAUDE_STATS_MACHINE", socket.gethostname()))
CURSOR_FILE = Path(os.environ.get(
    "TOKENFOLD_CURSOR",
    os.environ.get("CLAUDE_STATS_CURSOR", str(Path.home() / ".tokenfold-cursor.json")),
))
CLAUDE_DIR = Path.home() / ".claude" / "projects"
CREDENTIALS_FILE = Path.home() / ".claude" / ".credentials.json"
BATCH_SIZE = 2000
VERBOSE = os.environ.get("TOKENFOLD_VERBOSE", os.environ.get("CLAUDE_STATS_VERBOSE", "0")) == "1"


def log(msg):
    if VERBOSE:
        print(f"[tokenfold] {msg}", file=sys.stderr)


def load_cursors() -> dict:
    if CURSOR_FILE.exists():
        try:
            return json.loads(CURSOR_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_cursors(cursors: dict):
    CURSOR_FILE.write_text(json.dumps(cursors, indent=2))


def strip_content(rec: dict) -> dict:
    """Strip large content from events, keeping only metadata and sizes."""
    rec = dict(rec)  # shallow copy
    msg = rec.get("message")
    if not isinstance(msg, dict):
        return rec

    msg = dict(msg)  # shallow copy
    rec["message"] = msg
    content = msg.get("content")

    if isinstance(content, list):
        stripped = []
        for blk in content:
            if not isinstance(blk, dict):
                stripped.append(blk)
                continue
            blk = dict(blk)
            bt = blk.get("type", "")
            if bt == "thinking":
                # Keep type + length, strip text
                blk["thinking"] = f"[{len(blk.get('thinking', ''))} chars]"
            elif bt == "text":
                text = blk.get("text", "")
                if len(text) > 500:
                    blk["text"] = text[:200] + f"... [{len(text)} chars total]"
            elif bt == "tool_use":
                # Keep id, name, type - strip input value if large
                inp = blk.get("input")
                if isinstance(inp, dict):
                    for k, v in inp.items():
                        if isinstance(v, str) and len(v) > 300:
                            inp[k] = f"[{len(v)} chars]"
                elif isinstance(inp, str) and len(inp) > 300:
                    blk["input"] = f"[{len(inp)} chars]"
            elif bt == "tool_result":
                # Keep type, tool_use_id, is_error - strip content
                result_content = blk.get("content")
                if isinstance(result_content, list):
                    stripped_result = []
                    for rb in result_content:
                        if isinstance(rb, dict):
                            rb = dict(rb)
                            if rb.get("type") == "text":
                                text = rb.get("text", "")
                                if len(text) > 300:
                                    rb["text"] = f"[{len(text)} chars]"
                            elif rb.get("type") == "image":
                                rb.pop("source", None)
                                rb["_stripped"] = True
                        stripped_result.append(rb)
                    blk["content"] = stripped_result
                elif isinstance(result_content, str) and len(result_content) > 300:
                    blk["content"] = f"[{len(result_content)} chars]"
            elif bt == "image":
                blk.pop("source", None)
                blk["_stripped"] = True
            stripped.append(blk)
        msg["content"] = stripped
    elif isinstance(content, str) and len(content) > 500:
        msg["content"] = content[:200] + f"... [{len(content)} chars total]"

    return rec


def find_session_files() -> list[tuple[str, Path]]:
    """Find all JSONL session files, returning (project_dir, path) tuples."""
    if not CLAUDE_DIR.exists():
        return []
    results = []
    for jsonl_path in sorted(CLAUDE_DIR.rglob("*.jsonl")):
        # project_dir is the first dir under projects/
        try:
            rel = jsonl_path.relative_to(CLAUDE_DIR)
            project_dir = rel.parts[0] if rel.parts else "unknown"
        except ValueError:
            project_dir = "unknown"
        results.append((project_dir, jsonl_path))
    return results


def push_batch(project_dir: str, session_file: str, cursor_line: int,
               events: list[dict]) -> dict | None:
    """POST a batch to the server. Returns response dict or None on failure."""
    payload = json.dumps({
        "machine": MACHINE_NAME,
        "project_dir": project_dir,
        "session_file": session_file,
        "cursor": {"last_line_num": cursor_line},
        "events": events,
    }).encode()

    req = urllib.request.Request(
        f"{SERVER_URL}/api/ingest",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": API_KEY,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        log(f"HTTP {e.code}: {e.read().decode()[:200]}")
        return None
    except Exception as e:
        log(f"Error: {e}")
        return None


def _get_oauth_token() -> str | None:
    """Read OAuth access token from Claude credentials, refreshing if expired."""
    if not CREDENTIALS_FILE.exists():
        log("No credentials file found")
        return None
    try:
        creds = json.loads(CREDENTIALS_FILE.read_text())
        oauth = creds.get("claudeAiOauth", {})
    except (json.JSONDecodeError, OSError) as e:
        log(f"Cannot read credentials: {e}")
        return None

    token = oauth.get("accessToken")
    if not token:
        log("No access token in credentials")
        return None

    expires_at = oauth.get("expiresAt", 0)
    now_ms = time.time() * 1000
    # Refresh if token expires within 5 minutes
    if expires_at - now_ms < 300_000:
        refresh_token = oauth.get("refreshToken")
        if not refresh_token:
            log("Token expired, no refresh token")
            return None
        log("Refreshing OAuth token")
        try:
            body = json.dumps({
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            }).encode()
            req = urllib.request.Request(
                "https://platform.claude.com/v1/oauth/token",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "anthropic-beta": "oauth-2025-04-20",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            new_token = data.get("access_token")
            new_refresh = data.get("refresh_token", refresh_token)
            new_expires = int(time.time() * 1000) + data.get("expires_in", 7200) * 1000
            # Update credentials file
            oauth["accessToken"] = new_token
            oauth["refreshToken"] = new_refresh
            oauth["expiresAt"] = new_expires
            creds["claudeAiOauth"] = oauth
            CREDENTIALS_FILE.write_text(json.dumps(creds, indent=2))
            log("Token refreshed successfully")
            return new_token
        except Exception as e:
            log(f"Token refresh failed: {e}")
            return None

    return token


def _fetch_usage(token: str) -> dict | None:
    """Fetch usage data from Anthropic OAuth API."""
    try:
        req = urllib.request.Request(
            "https://api.anthropic.com/api/oauth/usage",
            headers={
                "Authorization": f"Bearer {token}",
                "anthropic-beta": "oauth-2025-04-20",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception as e:
        log(f"Usage fetch failed: {e}")
        return None


def _push_usage(usage_data: dict):
    """POST usage data to the server."""
    payload = json.dumps({
        "machine": MACHINE_NAME,
        "usage": usage_data,
    }).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/api/usage",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": API_KEY,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            log(f"Usage pushed: {result}")
    except Exception as e:
        log(f"Usage push failed: {e}")


def main():
    if not SERVER_URL:
        print("TOKENFOLD_URL not set (e.g. https://your-server.example.com)", file=sys.stderr)
        sys.exit(1)
    if not API_KEY:
        print("TOKENFOLD_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    cursors = load_cursors()
    total_accepted = 0
    total_dupes = 0

    for project_dir, jsonl_path in find_session_files():
        cursor_key = str(jsonl_path)
        cursor_line = cursors.get(cursor_key, 0)

        # Read new lines
        try:
            with open(jsonl_path) as f:
                all_lines = f.readlines()
        except OSError as e:
            log(f"Cannot read {jsonl_path}: {e}")
            continue

        if cursor_line >= len(all_lines):
            continue  # No new lines

        new_lines = all_lines[cursor_line:]
        log(f"{jsonl_path.name}: {len(new_lines)} new lines (from line {cursor_line})")

        # Parse and strip
        events = []
        for line in new_lines:
            try:
                rec = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            events.append(strip_content(rec))

        if not events:
            cursors[cursor_key] = len(all_lines)
            continue

        # Send in batches
        session_file = jsonl_path.name
        for batch_start in range(0, len(events), BATCH_SIZE):
            batch = events[batch_start:batch_start + BATCH_SIZE]
            batch_cursor = cursor_line + batch_start
            result = push_batch(project_dir, session_file, batch_cursor, batch)
            if result:
                total_accepted += result.get("accepted", 0)
                total_dupes += result.get("duplicates", 0)
                log(f"  -> accepted={result['accepted']}, dupes={result['duplicates']}")
            else:
                log(f"  -> FAILED batch at line {batch_cursor}, will retry next run")
                break  # Stop processing this file, retry next run
        else:
            # All batches succeeded - update cursor
            cursors[cursor_key] = len(all_lines)

    save_cursors(cursors)
    if total_accepted or total_dupes:
        log(f"Done: {total_accepted} accepted, {total_dupes} duplicates")

    # Push OAuth usage data (best-effort, failures don't affect event sync)
    token = _get_oauth_token()
    if token:
        usage = _fetch_usage(token)
        if usage:
            _push_usage(usage)


if __name__ == "__main__":
    main()
