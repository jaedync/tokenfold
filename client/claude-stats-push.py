#!/usr/bin/env python3
"""Push Claude Code session events to a Tokenfold server.

Zero external dependencies - stdlib only.
Designed to run every 5 minutes via cron (Linux) or launchd (macOS).
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# fcntl is POSIX-only. Windows is not a target, but the module must still
# IMPORT there (e.g. for the desktop-metadata unit tests on any OS). Guard the
# import; acquire_singleton_lock() degrades to a no-op lock if it is absent.
try:
    import fcntl
except ImportError:  # pragma: no cover - Windows only
    fcntl = None

# ── Config ──
# Config-file fallbacks, mirroring the hook wrapper (tokenfold-usage-push.sh).
# The launchd watch daemon is launched by launchd with NO environment, so an
# env-only resolve exits 1 and KeepAlive crash-loops it. These files are the
# same ones the hook wrapper reads, so the daemon and hook paths agree.
URL_FILE = Path.home() / ".config" / "notify-relay-url"
API_KEY_FILE = Path.home() / ".config" / "tokenfold-api-key"      # dedicated ingest token (preferred)
NOTIFY_TOKEN_FILE = Path.home() / ".config" / "notify-relay-token"  # back-compat fallback


def _read_config_file(path: Path) -> str:
    """Return the whitespace-stripped contents of `path`, or "" if it is
    missing, unreadable, or empty. Never raises — callers fall through to the
    next source on any failure. NEVER logs the value (may be a secret)."""
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _resolve_url() -> str:
    """URL: env TOKENFOLD_URL -> env CLAUDE_STATS_URL -> ~/.config/notify-relay-url -> ""."""
    return (
        os.environ.get("TOKENFOLD_URL")
        or os.environ.get("CLAUDE_STATS_URL")
        or _read_config_file(URL_FILE)
    )


def _resolve_api_key() -> str:
    """API key: env TOKENFOLD_API_KEY -> env CLAUDE_STATS_API_KEY
    -> ~/.config/tokenfold-api-key -> ~/.config/notify-relay-token -> "".
    NEVER log/print the returned value."""
    return (
        os.environ.get("TOKENFOLD_API_KEY")
        or os.environ.get("CLAUDE_STATS_API_KEY")
        or _read_config_file(API_KEY_FILE)
        or _read_config_file(NOTIFY_TOKEN_FILE)
    )


# Resolved at module import (unchanged lifecycle): main()'s sys.exit(1) guards
# and the push functions all read these globals, so keeping resolution here
# preserves existing behavior. The file fallbacks only fire when env is absent.
SERVER_URL = _resolve_url()
API_KEY = _resolve_api_key()
MACHINE_NAME = os.environ.get("TOKENFOLD_MACHINE", os.environ.get("CLAUDE_STATS_MACHINE", socket.gethostname()))
CURSOR_FILE = Path(os.environ.get(
    "TOKENFOLD_CURSOR",
    os.environ.get("CLAUDE_STATS_CURSOR", str(Path.home() / ".tokenfold-cursor.json")),
))
CLAUDE_DIR = Path.home() / ".claude" / "projects"
DESKTOP_DIR = Path.home() / "Library" / "Application Support" / "Claude" / "claude-code-sessions"
DESKTOP_CURSOR_KEY = "__desktop_last_activity_ms"
STAT_CACHE_KEY = "__stat_cache"  # {path: [mtime_ns, size]} — skip unchanged files
USAGE_FETCH_STAMP = Path.home() / ".tokenfold-usage-stamp"
USAGE_FETCH_MIN_INTERVAL = 300  # don't hit Anthropic's usage API > once / 5 min
CREDENTIALS_FILE = Path.home() / ".claude" / ".credentials.json"
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", str(Path.home() / ".local" / "bin" / "claude"))
BATCH_SIZE = 2000
VERBOSE = os.environ.get("TOKENFOLD_VERBOSE", os.environ.get("CLAUDE_STATS_VERBOSE", "0")) == "1"

# ── Watch-mode config ──
# Single-instance advisory lock. Held for the whole read→push→save-cursors cycle
# in one-shot mode, and for the entire process lifetime in --watch mode, so a
# hook-fired one-shot push cannot race the resident daemon on the cursor file.
LOCK_FILE = Path.home() / ".tokenfold-push.lock"
HOT_POLL_S = 1.0          # stat the hot set this often (O(hot set), no opens)
HOT_WINDOW_S = 2 * 3600   # a session file is "hot" if touched within this window
RESCAN_S = 60             # full find_session_files() glob this often


def log(msg):
    if VERBOSE:
        print(f"[tokenfold] {msg}", file=sys.stderr)


def err(msg):
    """Always print to stderr — not gated on VERBOSE."""
    print(f"[tokenfold] {msg}", file=sys.stderr)


def _file_sig(path: Path):
    """A cheap change-signature for a transcript file: [mtime_ns, size].
    Captured BEFORE reading, so any append during/after the read leaves a
    newer signature next run -> the file is re-read and new lines caught."""
    try:
        st = path.stat()
        return [st.st_mtime_ns, st.st_size]
    except OSError:
        return None


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
    # Atomic write: a Stop push can run concurrently with an in-flight
    # PostToolUse push (Stop/SessionEnd are undebounced). A plain write_text
    # truncates then rewrites, so a concurrent load_cursors() can read a torn
    # file -> {} -> cursors reset to 0 -> entire history re-sent. tmp + replace
    # makes readers see only the complete old or complete new file.
    tmp = CURSOR_FILE.with_suffix(CURSOR_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(cursors, indent=2))
    os.replace(tmp, CURSOR_FILE)


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
        if e.code in (401, 403):
            err(f"auth rejected (HTTP {e.code}) — TOKENFOLD_API_KEY must match the server's STATS_API_KEY")
        else:
            err(f"desktop metadata HTTP {e.code}: {e.read().decode()[:200]}")
        return None
    except Exception as e:  # noqa: BLE001
        err(f"desktop metadata error: {e}")
        return None

    return max(s.get("last_activity_at_ms") or 0 for s in sessions)


def push_batch(project_dir: str, session_file: str, cursor_line: int,
               events: list[dict], account: dict) -> dict | None:
    """POST a batch to the server. Returns response dict or None on failure."""
    payload = json.dumps({
        "machine": MACHINE_NAME,
        "project_dir": project_dir,
        "session_file": session_file,
        "cursor": {"last_line_num": cursor_line},
        "events": events,
        **account,
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
        if e.code in (401, 403):
            err(f"auth rejected (HTTP {e.code}) — TOKENFOLD_API_KEY must match the server's STATS_API_KEY")
        else:
            log(f"HTTP {e.code}: {e.read().decode()[:200]}")
        return None
    except Exception as e:
        log(f"Error: {e}")
        return None




def read_account(claude_dir):
    """Local read of account identity. Sends NO secrets — only email / org / plan /
    rate-limit tier / org_type / org_uuid. Ported from claude-usage-telemetry's read_account()."""
    acct = {
        "account_email": None,
        "org_name": None,
        "plan": None,
        "rate_limit_tier": None,
        "org_type": None,
        "org_uuid": None,
    }
    cj = Path.home() / ".claude.json"
    try:
        root = json.loads(cj.read_text())
        oa = root.get("oauthAccount") if isinstance(root, dict) else None
        if isinstance(oa, dict):
            acct["account_email"] = oa.get("emailAddress")
            acct["org_name"] = oa.get("organizationName")
            acct["org_type"] = oa.get("organizationType")
            acct["org_uuid"] = oa.get("organizationUuid")
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        pass
    cred = claude_dir / ".credentials.json"
    try:
        blob = json.loads(cred.read_text())
        o = blob.get("claudeAiOauth") if isinstance(blob, dict) else None
        if not isinstance(o, dict):
            o = blob if isinstance(blob, dict) else None
        if isinstance(o, dict):
            acct["plan"] = o.get("subscriptionType")
            acct["rate_limit_tier"] = o.get("rateLimitTier")
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        pass
    return acct


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


def _usage_fetch_too_soon() -> bool:
    """True if we fetched OAuth usage within USAGE_FETCH_MIN_INTERVAL. The
    pusher fires on every hook; without this gate it would hit Anthropic's
    usage API on every fire. The gauge only changes slowly, so once per a few
    minutes is plenty."""
    try:
        last = json.loads(USAGE_FETCH_STAMP.read_text()).get("last", 0)
    except (json.JSONDecodeError, OSError):
        return False
    return (time.time() - last) < USAGE_FETCH_MIN_INTERVAL


def _usage_fetch_stamp():
    try:
        USAGE_FETCH_STAMP.write_text(json.dumps({"last": time.time()}))
    except OSError:
        pass


def _fetch_and_push_usage():
    """Fetch usage from Anthropic API and push to server (best-effort)."""
    if _usage_is_backed_off():
        return
    # Interval-gate: this runs on every hook fire, but the OAuth usage endpoint
    # only needs polling every few minutes. Replaces the old 0-90s jitter sleep
    # (a cron-era relic that made every push process linger up to 90s — even
    # enterprise machines with no token, which fetch nothing).
    if _usage_fetch_too_soon():
        return
    _usage_fetch_stamp()

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
        if e.code in (401, 403):
            err(f"auth rejected (HTTP {e.code}) — TOKENFOLD_API_KEY must match the server's STATS_API_KEY")
        else:
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


# ── Single-instance lock (Part A) ──

def acquire_singleton_lock():
    """Try to take the exclusive advisory lock. Returns an open file handle on
    success (KEEP it alive — closing it releases the lock), or None if another
    process holds it. On a platform without fcntl (Windows, not a target) the
    lock degrades to a no-op: we still return a handle so callers proceed.

    Concurrent hook-fired pushes race on the cursor tmp file (observed
    FileNotFoundError on os.replace); this LOCK_EX|LOCK_NB serializes them."""
    try:
        fh = open(LOCK_FILE, "w")
    except OSError as e:
        err(f"cannot open lock file {LOCK_FILE}: {e}")
        return None
    if fcntl is None:  # pragma: no cover - Windows only
        return fh
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def release_singleton_lock(fh) -> None:
    """Release + close the lock handle. Safe to call with None."""
    if fh is None:
        return
    try:
        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        fh.close()
    except OSError:
        pass


# ── Hot-set helpers (Part B) ──

def select_hot_set(files, now=None):
    """From (project_dir, path) tuples, return the paths whose mtime is within
    HOT_WINDOW_S of now — the files worth stat-polling every tick. O(n) stats,
    but only run on a full rescan (every RESCAN_S), never per HOT_POLL_S tick."""
    if now is None:
        now = time.time()
    hot = []
    for _project_dir, path in files:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if now - mtime <= HOT_WINDOW_S:
            hot.append(path)
    return hot


def snapshot_sigs(paths):
    """Map {str(path): _file_sig(path)} for the hot set — the change baseline."""
    sigs = {}
    for p in paths:
        sig = _file_sig(p)
        if sig is not None:
            sigs[str(p)] = sig
    return sigs


def hot_set_changed(paths, prev_sigs) -> bool:
    """True if any hot file's (mtime_ns, size) signature differs from prev_sigs
    (or is brand-new). Pure stat() — no file opens, no globbing."""
    for p in paths:
        sig = _file_sig(p)
        if sig is None:
            continue
        if prev_sigs.get(str(p)) != sig:
            return True
    return False


# ── Push cycle (shared by one-shot and watch) ──

def run_push_cycle(cursors: dict) -> dict:
    """Scan session files, push new events, advance cursors. Returns the
    (mutated) cursors dict; caller is responsible for save_cursors().

    Pure refactor of the classic one-shot body: identical behavior and log
    lines. The account read stays inside the cycle so a long-lived --watch
    process picks up plan/org changes across cycles."""
    account = read_account(Path.home() / ".claude")
    total_accepted = 0
    total_dupes = 0

    # stat cache: skip files unchanged since we last fully consumed them, so a
    # hook fire stats N files instead of reading every byte of all of them.
    stat_cache = cursors.get(STAT_CACHE_KEY)
    if not isinstance(stat_cache, dict):
        stat_cache = {}
    fresh_stats = {}

    for project_dir, jsonl_path in find_session_files():
        cursor_key = str(jsonl_path)
        cursor_line = cursors.get(cursor_key, 0)

        # Fast skip: we've fully consumed this file before AND its signature
        # is unchanged -> nothing new, don't open it. (sig captured pre-read.)
        sig = _file_sig(jsonl_path)
        if cursor_line and sig is not None and stat_cache.get(cursor_key) == sig:
            fresh_stats[cursor_key] = sig
            continue

        # Read new lines
        try:
            with open(jsonl_path) as f:
                all_lines = f.readlines()
        except OSError as e:
            log(f"Cannot read {jsonl_path}: {e}")
            continue

        if cursor_line >= len(all_lines):
            fresh_stats[cursor_key] = sig  # fully consumed, remember to skip
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
            fresh_stats[cursor_key] = sig
            continue

        # Send in batches
        session_file = jsonl_path.name
        for batch_start in range(0, len(events), BATCH_SIZE):
            batch = events[batch_start:batch_start + BATCH_SIZE]
            batch_cursor = cursor_line + batch_start
            result = push_batch(project_dir, session_file, batch_cursor, batch, account)
            if result:
                total_accepted += result.get("accepted", 0)
                total_dupes += result.get("duplicates", 0)
                log(f"  -> accepted={result['accepted']}, dupes={result['duplicates']}")
            else:
                log(f"  -> FAILED batch at line {batch_cursor}, will retry next run")
                break  # Stop processing this file, retry next run
        else:
            # All batches succeeded - update cursor + remember signature.
            # sig is the PRE-read value: if the file grew during the read we
            # re-read once next run (sig mismatch) rather than skip unread data.
            cursors[cursor_key] = len(all_lines)
            fresh_stats[cursor_key] = sig

    # Only keep signatures for files seen this run (prunes deleted transcripts).
    cursors[STAT_CACHE_KEY] = fresh_stats

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

    if total_accepted or total_dupes:
        log(f"Done: {total_accepted} accepted, {total_dupes} duplicates")

    return cursors


# ── Watch loop (Part B) ──

def _watch_sleep(seconds: float) -> None:
    """Indirection so tests can stub the tick sleep (and inject side effects)."""
    time.sleep(seconds)


def watch_loop(max_iterations=None) -> None:
    """Resident poll loop. Every HOT_POLL_S seconds, stat only the hot set
    (O(hot set), no opens, no globbing); on a signature change run a push cycle.
    Every RESCAN_S seconds, do a full glob rescan to discover brand-new session
    files and refresh the hot set, and always tick the usage fetch (its own 300s
    stamp keeps the external cadence unchanged).

    Cursor safety: cursors are held IN MEMORY across cycles and saved after each
    push cycle. A crashed cycle leaves the last-saved cursor on disk (which the
    server has already ingested, since we only advance a cursor after a
    successful batch), so cursors never wind forward past un-pushed data. The
    server dedups by UUID, so a replay of the in-memory tail is harmless.

    max_iterations bounds the loop for tests; None runs forever.
    """
    cursors = load_cursors()
    files = find_session_files()
    hot = select_hot_set(files)
    sigs = snapshot_sigs(hot)
    last_rescan = time.time()
    iteration = 0

    while max_iterations is None or iteration < max_iterations:
        iteration += 1
        try:
            now = time.time()
            do_rescan = (now - last_rescan) >= RESCAN_S
            did_cycle = False

            if do_rescan:
                last_rescan = now
                files = find_session_files()
                hot = select_hot_set(files, now=now)
                # A fresh glob may reveal new/changed files: compare against the
                # signatures we last recorded, then push if anything moved.
                if hot_set_changed(hot, sigs):
                    cursors = run_push_cycle(cursors)
                    save_cursors(cursors)
                    did_cycle = True
                sigs = snapshot_sigs(hot)
                # Usage meter freshness during idle: fire at least once per
                # rescan even when no events moved. Its 300s stamp gate throttles
                # the actual Anthropic call.
                _fetch_and_push_usage()
            elif hot_set_changed(hot, sigs):
                cursors = run_push_cycle(cursors)
                save_cursors(cursors)
                sigs = snapshot_sigs(hot)
                did_cycle = True
                _fetch_and_push_usage()

            if did_cycle:
                log("watch: push cycle complete")
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001 - a transient error must not kill the daemon
            err(f"watch loop error (continuing): {e}")

        _watch_sleep(HOT_POLL_S)


def main():
    if not SERVER_URL:
        print("TOKENFOLD_URL not set (e.g. https://your-server.example.com)", file=sys.stderr)
        sys.exit(1)
    if not API_KEY:
        print("TOKENFOLD_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    watch = "--watch" in sys.argv[1:]

    # Single-instance lock, held for the whole cycle (one-shot) or the whole
    # process lifetime (watch). A hook-fired push while the daemon runs, or a
    # second --watch instance, finds the lock held → logs one line and exits 0.
    lock = acquire_singleton_lock()
    if lock is None:
        err("another tokenfold push holds the lock — exiting (holder will pick up new events)")
        return

    try:
        if watch:
            err("watch: resident daemon started")
            try:
                watch_loop()
            except KeyboardInterrupt:
                err("watch: interrupted, exiting")
            return

        # One-shot: load cursors from disk, one cycle, save, then usage fetch.
        cursors = load_cursors()
        cursors = run_push_cycle(cursors)
        save_cursors(cursors)
        _fetch_and_push_usage()
    finally:
        release_singleton_lock(lock)


if __name__ == "__main__":
    main()
