"""Server-side usage fallback.

Fetches OAuth usage from the Anthropic API every 10 minutes while no Meridian
source owns the quota. It reads the mounted Claude credentials and never
refreshes them: a refresh-token grant rotates the token server-side, and the
read-only mount cannot save the new pair, so a refresh here spends the token
that Claude Code and Meridian on the host need. (It did, in October 2026: two
token lines forked and the host's quota collector stopped.) Claude Code renews
the file when it runs.
"""

import asyncio
import json
import logging
import time
from pathlib import Path

import httpx

from .config import CLAUDE_CREDENTIALS_PATH
from .db import get_conn
from .version import get_claude_code_version

_log = logging.getLogger("tokenfold.usage_fetcher")


def log(msg, *args):
    """Print with prefix (uvicorn swallows custom loggers by default)."""
    formatted = msg % args if args else msg
    print(f"[usage_fetcher] {formatted}", flush=True)

USAGE_FETCH_INTERVAL_S = 600  # fetch usage every 10 minutes
LIMIT_PRUNE_INTERVAL_S = 86400  # prune limit_readings at most once per day
RATE_LIMIT_BACKOFF_S = 1800  # after a 429
UNAUTHORIZED_BACKOFF_S = 1800  # after a 401: wait for Claude Code to renew the file
USAGE_API_URL = "https://api.anthropic.com/api/oauth/usage"
ANTHROPIC_BETA = "oauth-2025-04-20"


def _claude_code_ua() -> str:
    """Match Claude Code's exact User-Agent: AP() returns 'claude-code/{VERSION}'."""
    return f"claude-code/{get_claude_code_version()}"

_task: asyncio.Task | None = None
_backoff_until: float = 0.0  # epoch time until which we skip fetches
# Last logged credential state ('missing' | 'expired' | 'ok'), so a skipped
# fetch logs once per change and not every 10 minutes.
_token_state: str | None = None


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


def _note_token_state(state, message):
    global _token_state
    if state != _token_state and message:
        log(message)
    _token_state = state


def _get_access_token() -> str | None:
    """The mounted file's access token while it is valid, else None."""
    _, oauth = _read_credentials_file()
    token = (oauth or {}).get("accessToken")
    if not token:
        _note_token_state("missing", "No OAuth token in the credentials file; skipping usage fetch")
        return None
    if (oauth.get("expiresAt") or 0) <= time.time() * 1000:
        _note_token_state("expired", "The OAuth token in the credentials file expired; "
                          "Claude Code renews it when it runs")
        return None
    _note_token_state("ok", None)
    return token


async def _fetch_usage():
    """Fetch usage data from Anthropic API and store it."""
    global _backoff_until

    from .claude_usage import managed_source_owns_usage
    from .db import read_conn
    with read_conn() as conn:
        if managed_source_owns_usage(conn):
            return
    if time.time() < _backoff_until:
        return  # still in backoff, skip silently
    token = _get_access_token()
    if not token:
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
                _backoff_until = time.time() + RATE_LIMIT_BACKOFF_S
                log("Usage API rate-limited (429), backing off %dm", RATE_LIMIT_BACKOFF_S // 60)
                return
            if resp.status_code == 401:
                # Never refresh here (see the module docstring).
                _backoff_until = time.time() + UNAUTHORIZED_BACKOFF_S
                log("Usage API 401: the credentials are stale; Claude Code renews them "
                    "when it runs. Backing off %dm", UNAUTHORIZED_BACKOFF_S // 60)
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
    """Periodic: prune old limit readings and fetch usage."""
    await asyncio.sleep(10)  # brief startup delay

    while True:
        # Retention housekeeping (self-limits to once per day)
        _maybe_prune_limit_readings(time.time())

        try:
            await _fetch_usage()
        except Exception:
            log("Error fetching usage")

        await asyncio.sleep(USAGE_FETCH_INTERVAL_S)


def should_run() -> bool:
    """The fetcher reads PERSONAL Max usage; don't run it on an
    enterprise-locked compliance instance (personal data is out of scope there)."""
    import app.config as cfg
    return cfg.LOCKED_SCOPE != "enterprise"


def start():
    """Start the background usage fetch task."""
    global _task
    if _task is not None:
        return
    _task = asyncio.ensure_future(_maintenance_loop())
    log("Usage fetcher started (every %dm; reads the credentials file, never refreshes it)",
        USAGE_FETCH_INTERVAL_S // 60)


def stop():
    """Cancel the background task."""
    global _task
    if _task is not None:
        _task.cancel()
        _task = None
