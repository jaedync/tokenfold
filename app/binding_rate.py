"""Dollars per limit point of the binding Claude window. No routes.

/api/now turns a session's burn ($/h) into the share of the binding window
it eats per hour (points/h = $/h / $/point). $/point here is the simple
ratio "API-equivalent spend in the current window / percent used", the same
inputs /api/rate-limits shows as limit_window cost and pct. It needs a scan
of up to 7 days of events (about 0.5 s on the production copy), so it is
served from a background cache: /api/now never waits for it, and reports
null until the first build lands.

Binding window: the hero window of /api/runway (runway_rules.binding), so
the share and the hero's burn rate are points of one window. When that
window is not a Claude 7-day window (a Codex or OpenCode Go window, or the
month meter), the rate is None. Only personal scope has Claude
subscription windows.
"""
import json
import sys
import time

from .bg_cache import BackgroundCache
from .cost_windows import compute_window_cost_by_model
from .db import read_conn
from .limit_windows import _bucket_window_start, _iso_to_epoch
from .quota_freshness import MAX_OBSERVATION_AGE_S
from .usage_buckets import normalize_usage_buckets
from .usage_pace import MIN_USED_PCT, WEEK_S

# A rate is stable for minutes; the window cost moves a few dollars per
# minute against thousands, so two minutes of age is invisible.
# wait_s=0: the build takes ~0.5 s on production, so waiting a little
# never helps and only slows the 5 s poll; the next poll has the value.
_CACHE = BackgroundCache(ttl=120, wait_s=0.0)


def binding_rate(scope: str, now: float | None = None,
                 wait_s: float | None = None) -> dict | None:
    """{"bucket", "usd_per_point"} or None (not ready, not applicable)."""
    return _CACHE.get(scope, lambda: build_binding_rate(scope, now), wait_s)


def build_binding_rate(scope: str, now: float | None = None) -> dict | None:
    now = time.time() if now is None else now
    if scope != "personal" or sys.modules["app.config"].LOCKED_SCOPE == "enterprise":
        return None
    with read_conn(snapshot=True) as conn:
        stored = _oauth_row(conn)
        if stored is None:
            return None
        observed = stored.get("observed_at_epoch")
        if not isinstance(observed, (int, float)) or now - observed > MAX_OBSERVATION_AGE_S:
            return None
        source = stored.get("source")
        best = _binding_bucket(conn, stored.get("data") or {}, observed,
                               source, now, _hero_key(scope, conn))
        if best is None:
            return None
        key, pct, start = best
        by_model = compute_window_cost_by_model(conn, start, observed, scope=scope,
                                                anthropic_only=True)
    cost = sum(v for name, v in by_model.items() if _counts_for(key, name))
    if cost <= 0:
        return None                     # no evidence: a 0 rate would divide
    return {"bucket": key, "usd_per_point": cost / pct}


def _oauth_row(conn) -> dict | None:
    row = conn.execute("SELECT value FROM meta WHERE key='oauth_usage'").fetchone()
    if not row:
        return None
    try:
        stored = json.loads(row["value"])
    except (TypeError, ValueError):
        return None
    return stored if isinstance(stored, dict) else None


def _hero_key(scope, conn) -> str | None:
    """Bucket key of the /api/runway hero window when it is a Claude 7-day
    window (seven_day or scoped:*), else None."""
    from .runway import build_runway  # runway imports the api module
    hero = build_runway(scope, conn).get("binding") or ""
    provider, _, key = hero.partition(":")
    if provider != "claude" or not (key == "seven_day" or key.startswith("scoped:")):
        return None
    return key


def _binding_bucket(conn, usage, observed, source, now, hero_key):
    """(key, pct, window_start) of the hero window, or None.

    A window that has reset since the observation no longer binds."""
    if hero_key is None:
        return None
    for b in normalize_usage_buckets(usage):
        if b["key"] != hero_key:
            continue
        pct = b["utilization"]
        reset = _iso_to_epoch(b["resets_at"])
        if reset is None or reset <= max(observed, now) or pct < MIN_USED_PCT:
            return None
        start = _bucket_window_start(conn, hero_key, reset, WEEK_S, observed, source=source)
        return hero_key, pct, start
    return None


def _counts_for(key: str, display_model: str) -> bool:
    """A scoped window governs one model family; same stem rule as
    /api/rate-limits (scoped:opus_4_8 matches every 'Opus' display name)."""
    if key == "seven_day":
        return True
    family = key.split(":", 1)[1].split("_")[0].lower()
    return bool(family) and family in display_model.lower()
