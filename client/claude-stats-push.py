#!/usr/bin/env python3
"""Push Claude Code session events to a Tokenfold server.

Zero external dependencies - stdlib only.
Designed to run every 5 minutes via cron (Linux) or launchd (macOS).
"""

import json
import os
import random
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
DESKTOP_DIR = Path.home() / "Library" / "Application Support" / "Claude" / "claude-code-sessions"
DESKTOP_CURSOR_KEY = "__desktop_last_activity_ms"
CREDENTIALS_FILE = Path.home() / ".claude" / ".credentials.json"
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", str(Path.home() / ".local" / "bin" / "claude"))
BATCH_SIZE = 2000
VERBOSE = os.environ.get("TOKENFOLD_VERBOSE", os.environ.get("CLAUDE_STATS_VERBOSE", "0")) == "1"


def log(msg):
    if VERBOSE:
        print(f"[tokenfold] {msg}", file=sys.stderr)


def err(msg):
    """Always print to stderr — not gated on VERBOSE."""
    print(f"[tokenfold] {msg}", file=sys.stderr)


def read_desktop_cursor(cursors: dict) -> int:
    val = cursors.get(DESKTOP_CURSOR_KEY, 0)
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def write_desktop_cursor(cursors: dict, value: int) -> None:
    cursors[DESKTOP_CURSOR_KEY] = int(value)


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


def desktop_dir() -> Path | None:
    """Return the Claude Desktop metadata root, or None if not on macOS or not present."""
    if sys.platform != "darwin":
        return None
    if not DESKTOP_DIR.exists():
        return None
    return DESKTOP_DIR


def extract_desktop_session(path: Path) -> dict | None:
    """Read one local_*.json file and normalize to the server schema.

    Returns None on read error, parse error, or missing cliSessionId.
    """
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    cli_id = raw.get("cliSessionId")
    if not cli_id:
        return None

    return {
        "cli_session_id": cli_id,
        "desktop_session_id": raw.get("sessionId"),
        "title": raw.get("title"),
        "model": raw.get("model"),
        "effort": raw.get("effort"),
        "permission_mode": raw.get("permissionMode"),
        "completed_turns": raw.get("completedTurns"),
        "is_archived": raw.get("isArchived"),
        "cwd": raw.get("cwd"),
        "origin_cwd": raw.get("originCwd"),
        "created_at_ms": raw.get("createdAt"),
        "last_activity_at_ms": raw.get("lastActivityAt"),
        "enabled_mcp_tools": raw.get("enabledMcpTools"),
        "remote_mcp_servers": raw.get("remoteMcpServersConfig"),
        "chrome_permission_mode": raw.get("chromePermissionMode"),
        "chrome_allowed_domains": raw.get("chromeAllowedDomains"),
    }


def find_desktop_sessions(root: Path, cursor_ms: int) -> list[dict]:
    """Scan `root` for local_*.json files updated after `cursor_ms`."""
    if not root or not root.exists():
        return []
    rows = []
    for path in root.rglob("local_*.json"):
        row = extract_desktop_session(path)
        if row is None:
            continue
        last = row.get("last_activity_at_ms") or 0
        if last > cursor_ms:
            rows.append(row)
    return rows


def push_desktop_sessions(sessions: list[dict]) -> int | None:
    """POST sessions to /api/desktop-metadata. Returns the new cursor value
    (max last_activity_at_ms across pushed sessions), or None on failure."""
    if not sessions:
        return None
    payload = json.dumps({
        "machine": MACHINE_NAME,
        "sessions": sessions,
    }).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/api/desktop-metadata",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": API_KEY,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            json.loads(resp.read())  # drain
    except urllib.error.HTTPError as e:
        err(f"desktop metadata HTTP {e.code}: {e.read().decode()[:200]}")
        return None
    except Exception as e:  # noqa: BLE001
        err(f"desktop metadata error: {e}")
        return None

    return max(s.get("last_activity_at_ms") or 0 for s in sessions)


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
        return None
    try:
        creds = json.loads(CREDENTIALS_FILE.read_text())
        oauth = creds.get("claudeAiOauth", {})
    except (json.JSONDecodeError, OSError) as e:
        err(f"Cannot read credentials: {e}")
        return None

    token = oauth.get("accessToken")
    if not token:
        return None

    expires_at = oauth.get("expiresAt", 0)
    now_ms = time.time() * 1000
    if expires_at - now_ms < 300_000:
        refresh_token = oauth.get("refreshToken")
        if not refresh_token:
            return None
        log("Refreshing OAuth token")
        try:
            body = json.dumps({
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
                "scope": "user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload",
            }).encode()
            req = urllib.request.Request(
                "https://platform.claude.com/v1/oauth/token",
                data=body,
                headers={
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            new_token = data.get("access_token")
            new_refresh = data.get("refresh_token", refresh_token)
            new_expires = int(time.time() * 1000) + data.get("expires_in", 7200) * 1000
            oauth["accessToken"] = new_token
            oauth["refreshToken"] = new_refresh
            oauth["expiresAt"] = new_expires
            creds["claudeAiOauth"] = oauth
            # Atomic write: tmp file + rename to avoid corruption on concurrent reads
            tmp = CREDENTIALS_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(creds, indent=2))
            tmp.rename(CREDENTIALS_FILE)
            log("Token refreshed successfully")
            return new_token
        except Exception as e:
            err(f"Token refresh failed: {e}")
            return None

    return token


def _get_claude_version() -> str:
    """Get installed claude CLI version for User-Agent header."""
    try:
        import subprocess
        result = subprocess.run(
            [CLAUDE_BIN, "--version"],
            capture_output=True, text=True, timeout=5,
            env={**os.environ, "CLAUDECODE": ""},
        )
        ver = result.stdout.strip().split()[0] if result.stdout else ""
        if ver and ver[0].isdigit():
            return ver
    except Exception:
        pass
    return "2.1.71"  # fallback


USAGE_BACKOFF_FILE = Path.home() / ".tokenfold-usage-backoff"
USAGE_BACKOFF_STEPS = [300, 900, 1800, 3600, 7200, 14400]  # 5m, 15m, 30m, 1h, 2h, 4h


def _usage_is_backed_off() -> bool:
    """Check if we're in a backoff period from a previous 429."""
    if not USAGE_BACKOFF_FILE.exists():
        return False
    try:
        data = json.loads(USAGE_BACKOFF_FILE.read_text())
        resume_at = data.get("resume_at", 0)
        if time.time() < resume_at:
            remaining = int(resume_at - time.time())
            log(f"Usage fetch backed off, {remaining}s remaining")
            return True
        return False
    except (json.JSONDecodeError, OSError):
        return False


def _usage_backoff_on_429():
    """Record a 429 and set the next backoff window."""
    failures = 0
    try:
        if USAGE_BACKOFF_FILE.exists():
            data = json.loads(USAGE_BACKOFF_FILE.read_text())
            failures = data.get("failures", 0)
    except (json.JSONDecodeError, OSError):
        pass
    failures += 1
    step = min(failures - 1, len(USAGE_BACKOFF_STEPS) - 1)
    wait = USAGE_BACKOFF_STEPS[step]
    resume_at = time.time() + wait
    USAGE_BACKOFF_FILE.write_text(json.dumps({"failures": failures, "resume_at": resume_at}))
    log(f"Usage 429'd, backing off {wait}s (failure #{failures})")


def _usage_backoff_clear():
    """Clear the backoff file on success."""
    try:
        USAGE_BACKOFF_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def _fetch_and_push_usage():
    """Fetch usage from Anthropic API and push to server (best-effort)."""
    if _usage_is_backed_off():
        return

    # Jitter: 0-90s random delay so requests don't land on exact cron intervals
    jitter = random.randint(0, 90)
    log(f"Usage fetch jitter: {jitter}s")
    time.sleep(jitter)

    token = _get_oauth_token()
    if not token:
        return
    try:
        req = urllib.request.Request(
            "https://api.anthropic.com/api/oauth/usage",
            headers={
                "Authorization": f"Bearer {token}",
                "anthropic-beta": "oauth-2025-04-20",
                "Content-Type": "application/json",
                "User-Agent": f"claude-code/{_get_claude_version()}",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            usage = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err(f"Usage fetch: HTTP {e.code}")
        if e.code == 429:
            _usage_backoff_on_429()
        return
    except Exception as e:
        err(f"Usage fetch: {e}")
        return

    payload = json.dumps({"machine": MACHINE_NAME, "usage": usage}).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/api/usage",
        data=payload,
        headers={"Content-Type": "application/json", "X-API-Key": API_KEY},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            log(f"Usage pushed: {result}")
            _usage_backoff_clear()
    except Exception as e:
        err(f"Usage push: {e}")


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

    # Desktop session metadata (macOS-only, no-op otherwise).
    root = desktop_dir()
    if root is not None:
        desktop_cursor = read_desktop_cursor(cursors)
        desktop_sessions = find_desktop_sessions(root, desktop_cursor)
        if desktop_sessions:
            log(f"desktop: pushing {len(desktop_sessions)} updated sessions")
            new_cursor = push_desktop_sessions(desktop_sessions)
            if new_cursor is not None:
                write_desktop_cursor(cursors, new_cursor)
                log(f"desktop: cursor advanced to {new_cursor}")

    save_cursors(cursors)
    if total_accepted or total_dupes:
        log(f"Done: {total_accepted} accepted, {total_dupes} duplicates")

    # Push OAuth usage data (best-effort, failures don't affect event sync)
    _fetch_and_push_usage()


if __name__ == "__main__":
    main()
