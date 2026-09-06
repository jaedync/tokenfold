"""Server-side usage data management.

Fetches OAuth usage data directly from the Anthropic API on a 10-minute
interval, and keeps the OAuth token refreshed so it never expires.
"""

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path

import httpx

from .config import CLAUDE_CREDENTIALS_PATH
from .db import get_conn, write_txn
from .version import get_claude_code_version

_log = logging.getLogger("tokenfold.usage_fetcher")


def log(msg, *args):
    """Print with prefix (uvicorn swallows custom loggers by default)."""
    formatted = msg % args if args else msg
    print(f"[usage_fetcher] {formatted}", flush=True)

USAGE_FETCH_INTERVAL_S = 600  # fetch usage every 10 minutes
TOKEN_CHECK_INTERVAL_S = 3600  # check token freshness every hour
LIMIT_PRUNE_INTERVAL_S = 86400  # prune limit_readings at most once per day
OAUTH_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
USAGE_API_URL = "https://api.anthropic.com/api/oauth/usage"
ANTHROPIC_BETA = "oauth-2025-04-20"
OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
OAUTH_SCOPES = "user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"


def _claude_code_ua() -> str:
    """Match Claude Code's exact User-Agent: AP() returns 'claude-code/{VERSION}'."""
    return f"claude-code/{get_claude_code_version()}"

_task: asyncio.Task | None = None
_backoff_until: float = 0.0  # epoch time until which we skip fetches

# In-memory token cache — survives read-only file mounts
_cached_oauth: dict | None = None

# Track consecutive refresh failures to detect permanently revoked tokens
_consecutive_refresh_failures: int = 0
_MAX_REFRESH_FAILURES = 5  # after this many, back off to hourly retries
_refresh_retry_at: float = 0.0
_refresh_identity: str | None = None
_file_identity: str | None = None


def _credential_identity(oauth):
    # Identity only, never persisted or logged; include refresh-token rotation.
    if not oauth:
        return None
    value = json.dumps([oauth.get("accessToken"), oauth.get("refreshToken")])
    return hashlib.sha256(value.encode()).hexdigest()


def _adopt_oauth(oauth):
    global _cached_oauth, _refresh_identity, _refresh_retry_at
    global _consecutive_refresh_failures, _backoff_until
    identity = _credential_identity(oauth)
    if identity != _refresh_identity:
        _refresh_identity = identity
        _consecutive_refresh_failures = 0
        _refresh_retry_at = 0.0
        _backoff_until = 0.0
    _cached_oauth = dict(oauth)
    return _cached_oauth


def _refresh_failed():
    global _consecutive_refresh_failures, _refresh_retry_at
    _consecutive_refresh_failures += 1
    if _consecutive_refresh_failures >= _MAX_REFRESH_FAILURES:
        _refresh_retry_at = time.time() + TOKEN_CHECK_INTERVAL_S


def _read_credentials_file() -> tuple[dict | None, dict | None]:
    """Read the full credentials file and return (full_creds, oauth_section)."""
    path = Path(CLAUDE_CREDENTIALS_PATH)
    if not path.exists():
        return None, None
    try:
        creds = json.loads(path.read_text())
        return creds, creds.get("claudeAiOauth")
    except (json.JSONDecodeError, OSError) as e:
        log("Cannot read credentials file: %s", e)
        return None, None


def _write_credentials_file(creds: dict):
    """Write updated credentials back to the file (if writable)."""
    path = Path(CLAUDE_CREDENTIALS_PATH)
    try:
        path.write_text(json.dumps(creds, indent=2))
    except OSError as e:
        log("Credentials file not writable (expected with :ro mount): %s", e)


def _get_oauth() -> dict | None:
    """Get the best available OAuth credentials (memory > file)."""
    global _file_identity
    _, file_oauth = _read_credentials_file()
    file_identity = _credential_identity(file_oauth)
    changed = (_file_identity is not None and file_identity is not None
               and file_identity != _file_identity)
    if file_identity is not None:
        _file_identity = file_identity
    # An actually changed mounted credential is an explicit rotation, even
    # with an earlier expiry. An unchanged read-only file must not undo refresh.
    if file_oauth and (changed or not _cached_oauth or
                      file_oauth.get("expiresAt", 0) > _cached_oauth.get("expiresAt", 0)):
        return _adopt_oauth(file_oauth)
    if _cached_oauth:
        return _adopt_oauth(_cached_oauth)
    return None


def _get_access_token() -> str | None:
    """Get the current OAuth access token (memory cache > file)."""
    oauth = _get_oauth()
    if not oauth:
        return None
    return oauth.get("accessToken")


async def _refresh_token_if_needed(force: bool = False):
    """Check if the OAuth token is expiring soon and refresh it.

    Args:
        force: If True, refresh regardless of remaining time (e.g. after 401).
    """
    global _cached_oauth, _consecutive_refresh_failures, _refresh_retry_at
    oauth = _get_oauth()
    if not oauth:
        return

    token = oauth.get("accessToken")
    if not token:
        return

    expires_at = oauth.get("expiresAt", 0)
    now_ms = time.time() * 1000
    remaining_min = (expires_at - now_ms) / 1000 / 60

    if not force and remaining_min > 30:
        log("Token valid for %.0f more minutes", remaining_min)
        return

    # If we've hit too many consecutive failures, the refresh token is likely
    # revoked — only retry once per hour to avoid spamming the token endpoint.
    if time.time() < _refresh_retry_at:
        return

    refresh_token = oauth.get("refreshToken")
    if not refresh_token:
        log("Token expiring in %.0fm but no refresh token", remaining_min)
        return

    log("Token %s (%.0fm remaining), refreshing",
        "force-refresh" if force else "expiring", remaining_min)
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                OAUTH_TOKEN_URL,
                json={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": OAUTH_CLIENT_ID,
                    "scope": OAUTH_SCOPES,
                },
                headers={
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        new_token = data.get("access_token")
        if not new_token:
            log("Token refresh response missing access_token")
            _refresh_failed()
            return

        # Build a new dict atomically to avoid partial-update reads
        refreshed = {
            **oauth,
            "accessToken": new_token,
            "refreshToken": data.get("refresh_token", refresh_token),
            "expiresAt": int(time.time() * 1000) + data.get("expires_in", 7200) * 1000,
        }

        # Single atomic assignment — no partial state visible to other coroutines
        _adopt_oauth(refreshed)
        _consecutive_refresh_failures = 0
        _refresh_retry_at = 0.0

        # Best-effort write back to file (may be read-only mount)
        full_creds, _ = _read_credentials_file()
        if full_creds:
            full_creds["claudeAiOauth"] = refreshed
            _write_credentials_file(full_creds)

        with write_txn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("oauth_credentials", json.dumps(refreshed)),
            )

        log("Token refreshed, valid for %.0f more minutes",
                 (refreshed["expiresAt"] - time.time() * 1000) / 1000 / 60)
    except httpx.HTTPStatusError as e:
        _refresh_failed()
        status = e.response.status_code
        if status in (400, 401, 403):
            log("Token refresh failed: HTTP %s — refresh token may be "
                "revoked (%d/%d failures before backoff)",
                status, _consecutive_refresh_failures, _MAX_REFRESH_FAILURES)
        else:
            log("Token refresh failed: HTTP %s", status)
    except Exception as e:
        _refresh_failed()
        log("Token refresh failed: %s", e)


async def _fetch_usage():
    """Fetch usage data from Anthropic API and store it."""
    global _backoff_until

    from .claude_usage import managed_source_owns_usage
    from .db import read_conn
    with read_conn() as conn:
        if managed_source_owns_usage(conn):
            return
    token = _get_access_token()
    if time.time() < _backoff_until:
        return  # still in backoff, skip silently
    if not token:
        log("No OAuth token available, skipping usage fetch")
        return

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                USAGE_API_URL,
                headers={
                    "Authorization": f"Bearer {token}",
                    "anthropic-beta": ANTHROPIC_BETA,
                    "Content-Type": "application/json",
                    "User-Agent": _claude_code_ua(),
                },
            )
            if resp.status_code == 429:
                _backoff_until = time.time() + 1800  # back off 30 min
                log("Usage API rate-limited (429), backing off 30m")
                return
            if resp.status_code == 401:
                log("Usage API 401 (token stale), attempting refresh")
                await _refresh_token_if_needed(force=True)
                _backoff_until = time.time() + 60  # brief backoff after refresh
                return
            resp.raise_for_status()
            usage = resp.json()
    except httpx.HTTPStatusError as e:
        log("Usage fetch failed: HTTP %s", e.response.status_code)
        return
    except Exception as e:
        log("Usage fetch failed: %s", e)
        return

    _backoff_until = 0.0  # clear backoff on success

    from .claude_usage import store_snapshot
    if store_snapshot(usage, time.time(), "server") is None:
        return  # managed source took ownership while the network call ran

    from .aggregator import invalidate_cache
    invalidate_cache()

    log("Usage data fetched and stored")


# Module-level bookkeeping so the prune runs at most once per day across
# loop iterations (mirrors the last_token_check idiom, but module-scoped so
# tests can drive it directly).
_last_prune_epoch: float = 0.0


def _maybe_prune_limit_readings(now: float) -> bool:
    """Prune limit_readings at most once per LIMIT_PRUNE_INTERVAL_S.

    Placement here is compliance-safe: on enterprise-locked instances this
    fetcher never runs (should_run) AND the ingest history write is gated,
    so no limit_readings rows accumulate there to need pruning.
    """
    global _last_prune_epoch
    if now - _last_prune_epoch < LIMIT_PRUNE_INTERVAL_S:
        return False
    _last_prune_epoch = now
    try:
        from .limit_readings import prune_limit_readings
        prune_limit_readings(get_conn(), now_epoch=now)
    except Exception as e:
        log("Error pruning limit readings: %s", e)
    return True


async def _maintenance_loop():
    """Periodic: fetch usage data and keep the OAuth token alive."""
    await asyncio.sleep(10)  # brief startup delay

    last_token_check = 0.0

    while True:
        # Check token freshness before fetching (refresh first so fetch uses good token)
        now = time.time()
        if now - last_token_check >= TOKEN_CHECK_INTERVAL_S:
            try:
                await _refresh_token_if_needed()
            except Exception:
                log("Error in token refresh")
            last_token_check = now

        # Retention housekeeping (self-limits to once per day)
        _maybe_prune_limit_readings(now)

        # Fetch usage
        try:
            await _fetch_usage()
        except Exception:
            log("Error fetching usage")

        await asyncio.sleep(USAGE_FETCH_INTERVAL_S)


def _load_cached_credentials():
    """On startup, load DB-cached credentials if they're fresher than the file."""
    global _cached_oauth
    try:
        conn = get_conn()
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'oauth_credentials'"
        ).fetchone()
        if not row:
            return
        db_oauth = json.loads(row[0])
        _, file_oauth = _read_credentials_file()
        file_exp = (file_oauth or {}).get("expiresAt", 0)
        db_exp = db_oauth.get("expiresAt", 0)
        if db_exp > file_exp:
            _cached_oauth = db_oauth
            log("Loaded fresher token from DB cache (expires in %.0fm)",
                (db_exp - time.time() * 1000) / 1000 / 60)
    except Exception as e:
        log("Could not load cached credentials: %s", e)


def should_run() -> bool:
    """The fetcher refreshes the PERSONAL Max OAuth token; don't run it on an
    enterprise-locked compliance instance (personal data is out of scope there)."""
    import app.config as cfg
    return cfg.LOCKED_SCOPE != "enterprise"


def start():
    """Start the background usage fetch + token maintenance task."""
    global _task
    if _task is not None:
        return
    _load_cached_credentials()
    _task = asyncio.ensure_future(_maintenance_loop())
    log("Usage fetcher started (every %dm, token check every %dm)",
             USAGE_FETCH_INTERVAL_S // 60, TOKEN_CHECK_INTERVAL_S // 60)


def stop():
    """Cancel the background task."""
    global _task
    if _task is not None:
        _task.cancel()
        _task = None
