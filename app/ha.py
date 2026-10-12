"""GET /api/ha — flat, scrubbed metrics for Home Assistant REST sensors.

Designed for HA's `rest:` platform: one HTTP fetch, many sensors via
value_template. All timestamps truncated to minute precision so the
microsecond offsets of OAuth resets_at cannot fingerprint the account.
"""

import json
import math
import sqlite3
import time
from datetime import datetime, timezone

import app.config as config
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from .auth import require_read_key

from .aggregator import build_dashboard_data
from .api import _bucket_window_start, _iso_to_epoch
from .cost_windows import compute_window_cost
from .db import read_conn
from .usage_buckets import normalize_usage_buckets

router = APIRouter()

FIVE_HOURS_S = 5 * 3600
SEVEN_DAYS_S = 7 * 24 * 3600
MIN_PCT_FOR_IMPLIED_LIMIT = 5.0  # below this, API rounding noise swamps signal


def _truncate_to_minute(iso_str: str) -> tuple[str, float] | tuple[None, None]:
    """Parse an ISO-8601 timestamp, truncate to whole UTC minute.

    Returns (iso_string_with_00_seconds_and_no_microseconds, epoch_float).
    Returns (None, None) if the input is empty or malformed.
    """
    if not iso_str:
        return None, None
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None, None
    dt = dt.astimezone(timezone.utc).replace(second=0, microsecond=0)
    return dt.isoformat(), dt.timestamp()


def _model_buckets(buckets: list, now_epoch: float) -> dict | None:
    """Per-model utilization from the scoped entries of the normalized
    bucket list (the caller runs normalize_usage_buckets exactly once).

    {<slug>: {pct_used, resets_at, resets_in_s}} — resets_at minute-truncated
    (anti-fingerprinting, hard invariant). No implied_limit_usd here: scoped
    utilization can't be paired with a per-model spend split cheaply, and a
    wrong dollar figure is worse than none. Returns None when the payload has
    no scoped buckets.
    """
    out = {}
    for b in buckets:
        if not b["key"].startswith("scoped:"):
            continue
        resets_iso, resets_epoch = _truncate_to_minute(b["resets_at"] or "")
        out[b["key"][len("scoped:"):]] = {
            "pct_used": b["utilization"],
            "resets_at": resets_iso,
            "resets_in_s": (max(0, int(resets_epoch - now_epoch))
                            if resets_epoch is not None else None),
        }
    return out or None


def _implied_limit(spend_usd: float, pct_used: float | None) -> float | None:
    """spend / (pct/100), or None when utilization is too low to be trustworthy."""
    if pct_used is None or pct_used < MIN_PCT_FOR_IMPLIED_LIMIT:
        return None
    return round(spend_usd / (pct_used / 100.0), 2)


def _observed_epoch(stored: dict) -> float | None:
    """When the stored utilization was observed, with the same precedence as
    /api/rate-limits (_oauth_snapshot): the managed observation clock first,
    else updated_at. None when neither parses.
    """
    obs = stored.get("observed_at_epoch")
    if (isinstance(obs, (float, int)) and not isinstance(obs, bool)
            and math.isfinite(obs) and obs > 0):
        return float(obs)
    return _iso_to_epoch(stored.get("updated_at"))


def _window_dollars(conn, bucket_key, resets_epoch, window_seconds, now_epoch,
                    observed, source, scope, pct_used) -> dict | None:
    """spend_usd / implied_limit_usd / window_start for one window, or None
    when the percent cannot be paired with spend.

    pct_used describes the observation, so the spend paired with it ends
    there. An observation that is missing, in the future, or not after the
    window start has no such pairing, and spend up to another time would
    give implied_limit_usd no meaning. The end is capped at resets_at so an
    ended window never borrows the next one's spend.
    """
    if observed is None or observed > now_epoch:
        return None
    start_epoch = _bucket_window_start(
        conn, bucket_key, resets_epoch, window_seconds, observed, source=source)
    if observed <= start_epoch:
        return None
    end_epoch = min(observed, resets_epoch)
    spend = round(compute_window_cost(conn, start_epoch, end_epoch, scope=scope,
                                      anthropic_only=True), 2)
    # Minute-floored again here, not trusted from the helper: window_start
    # leaves the server, and a grant time is a poll time (sub-minute).
    start_iso = datetime.fromtimestamp(
        (start_epoch // 60) * 60, tz=timezone.utc).isoformat()
    return {
        "spend_usd": spend,
        "implied_limit_usd": _implied_limit(spend, pct_used),
        "window_start": start_iso,
    }


def _window_block(
    conn: sqlite3.Connection,
    bucket_key: str,
    raw_resets_at: str,
    pct_used: float | None,
    window_seconds: int,
    now_epoch: float,
    observed: float | None,
    source: str | None,
    scope: str = "personal",
) -> dict | None:
    """Build a five_hour / weekly sub-object, or None if resets_at is missing.

    These windows reflect the personal Max budget utilization reported by the
    OAuth API, so spend must be PERSONAL-scoped for implied_limit arithmetic
    to be coherent: implied_limit = personal_spend / (personal_pct / 100).

    The span is the one /api/rate-limits uses for limit_window and
    five_hour_window: from resets_at - window, or from a later persistent
    granted reset (a grant voids pre-grant usage, so counting it inflated
    implied_limit roughly 9x after the 2026-10-12 grant), to the
    observation. Claude-subscription spend only (anthropic_only): Codex and
    OpenCode Pi rows do not consume this limit. window_start tells clients
    which part of the window spend_usd covers.

    Unlike /api/rate-limits, there is no quota_window_valid gate: HA records
    pct_used history, so the quota fields stay even when the snapshot is
    stale or its window has ended. Only the dollar fields need a pairing,
    and _window_dollars decides that. When it cannot pair them, or when it
    raises, spend_usd, implied_limit_usd and window_start are None (keys
    kept; scripts/ha-smoke.sh checks them). Never resets_at - window as a
    fallback: that silently brings back the pre-grant overcount.
    """
    resets_iso, resets_epoch = _truncate_to_minute(raw_resets_at)
    if resets_iso is None:
        return None
    block = {
        "pct_used": pct_used if pct_used is not None else 0.0,
        "spend_usd": None,
        "implied_limit_usd": None,
        "resets_at": resets_iso,
        "resets_in_s": max(0, int(resets_epoch - now_epoch)),
        "window_start": None,
    }
    try:
        dollars = _window_dollars(conn, bucket_key, resets_epoch, window_seconds,
                                  now_epoch, observed, source, scope, pct_used)
    except Exception as e:
        # Narrow on purpose (same as rate_limits' Fix 6): a reset-detection or
        # pricing bug drops this window's dollars, never the whole feed.
        print(f"[ha] {bucket_key} window dollars failed: {e!r}", flush=True)
        dollars = None
    return {**block, **dollars} if dollars else block


def _oauth_blocks(conn, now_epoch: float) -> tuple:
    """(five_hour, weekly, model_buckets, updated_at_epoch) from the stored
    OAuth snapshot. The caller passes one snapshot connection, so the meta
    row and the limit history it is paired with are one consistent state.
    """
    row = conn.execute(
        "SELECT value FROM meta WHERE key='oauth_usage'"
    ).fetchone()
    if not row:
        return None, None, None, None
    try:
        stored = json.loads(row["value"])
    except (ValueError, TypeError):
        stored = None
    if not stored:
        return None, None, None, None

    usage = stored.get("data") or {}
    # Normalize ONCE (limits[] primary, legacy dicts fallback) so
    # a limits[]-only payload (legacy five_hour/seven_day nulled)
    # still feeds the window blocks. resets_at stays RAW here —
    # _truncate_to_minute below enforces minute precision.
    buckets = normalize_usage_buckets(usage)
    by_key = {b["key"]: b for b in buckets}
    fh = by_key.get("five_hour") or {}
    sd = by_key.get("seven_day") or {}
    # Same observation clock and history source as /api/rate-limits, so
    # both surfaces anchor the window on the same granted reset.
    observed = _observed_epoch(stored)
    history_source = stored.get("source")

    # Window spend is personal-scoped: these windows track the personal
    # Max utilization, so spend must match (personal) for
    # implied_limit = spend / (pct/100) to be coherent.
    five_hour_block = _window_block(
        conn, "five_hour", fh.get("resets_at") or "", fh.get("utilization"),
        FIVE_HOURS_S, now_epoch, observed, history_source, scope="personal")
    weekly_block = _window_block(
        conn, "seven_day", sd.get("resets_at") or "", sd.get("utilization"),
        SEVEN_DAYS_S, now_epoch, observed, history_source, scope="personal")
    # Per-model scoped buckets — same enterprise gate as the window blocks
    # (the caller only reads this snapshot when not enterprise-locked).
    model_buckets = _model_buckets(buckets, now_epoch)

    updated_at_epoch = None
    updated_at_iso = stored.get("updated_at", "")
    if updated_at_iso:
        try:
            ua = datetime.fromisoformat(updated_at_iso).timestamp()
            # Round to nearest 10 seconds for symmetry with resets_at.
            updated_at_epoch = int(round(ua / 10.0) * 10)
        except (ValueError, TypeError):
            pass
    return five_hour_block, weekly_block, model_buckets, updated_at_epoch


@router.get("/api/ha", dependencies=[Depends(require_read_key)])
async def ha_metrics():
    """Flat metrics feed for Home Assistant REST sensors."""
    now_epoch = time.time()

    # Read the scope lock fresh so tests can monkeypatch app.config.LOCKED_SCOPE.
    locked_scope = config.LOCKED_SCOPE

    # The ENTIRE /api/ha feed is PERSONAL-scoped: Home Assistant is the personal
    # home surface, so enterprise (work) usage must never flow to it. Cost totals
    # are personal — enterprise excluded — matching the personal-scoped windows below.
    dash = build_dashboard_data("personal")
    cost_today = round((dash.get("today") or {}).get("cost", 0.0) or 0.0, 2)
    cost_total = round(dash.get("total_cost", 0.0) or 0.0, 2)

    five_hour_block = weekly_block = model_buckets = updated_at_epoch = None
    if locked_scope != "enterprise":
        # OAuth usage reflects the PERSONAL Max budget — only emit when not
        # enterprise-locked so personal account activity can't surface on a
        # compliance instance. Skip the meta read entirely so a stale row from
        # before the lock can't leak through. One snapshot (as rate_limits):
        # a compacted history tail moving between the meta read and the
        # reset detection cannot mix two states.
        with read_conn(snapshot=True) as conn:
            five_hour_block, weekly_block, model_buckets, updated_at_epoch = (
                _oauth_blocks(conn, now_epoch))

    return JSONResponse(content={
        "cost_today_usd": cost_today,
        "cost_total_usd": cost_total,
        "five_hour": five_hour_block,
        "weekly": weekly_block,
        "model_buckets": model_buckets,
        "updated_at_epoch": updated_at_epoch,
    })
