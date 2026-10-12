"""Limit-window helpers shared by the quota surfaces. No routes.

Importers: app/api.py (/api/rate-limits), app/ha.py (/api/ha),
app/limit_readings.py and app/spend_history.py. They live outside the route
modules so no route module depends on another for them.

Import rule: this module imports nothing from app at module scope.
limit_readings imports _iso_to_epoch and _scrub_to_minute_or_none from here
at module level, so _bucket_window_start must import limit_readings lazily.
"""

from datetime import datetime, timezone
from typing import Optional


def _iso_to_epoch(iso_str: Optional[str]) -> Optional[float]:
    """Parse an ISO-8601 string to epoch seconds; None on ANY failure.

    One copy for every importer listed in the module docstring, rather than
    a duplicate per module (Fix 9).
    """
    if not isinstance(iso_str, str) or not iso_str:
        return None
    try:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def _scrub_to_minute_or_none(iso_str: Optional[str]) -> Optional[str]:
    """Truncate an ISO-8601 timestamp to whole-minute UTC precision.

    Removes subsecond and second precision so a per-account microsecond
    offset can't fingerprint the account across responses.

    Fails CLOSED: returns None when the input is empty or unparseable, so a
    raw (full-precision, fingerprintable) value can never pass through to
    the response.
    """
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return (
        dt.astimezone(timezone.utc).replace(second=0, microsecond=0).isoformat()
    )


def _bucket_window_start(conn, bucket_key, resets_epoch, window_s, observed=None, *, source=None):
    """Start epoch of the CURRENT limit window for one historized bucket.

    resets_at − window_s (minute-floored — limit timestamps never leave the
    server at sub-minute precision), pushed forward to the latest persistent
    GRANTED reset recorded for that bucket: a granted mid-window reset voids
    pre-grant usage, so spend anchored at resets_at − window would overcount
    (F1). corroborated_resets = persistent_resets (not raw detect_resets: a
    stale client replay must not move the cost window, review M1) plus
    account-level resets borrowed from sibling buckets — a grant zeroes
    every bucket at once, and a bucket whose own decrease is too small for
    detection still had its window cut (2026-07-09 incident). Lazy import —
    limit_readings imports this module at module level, so this module must
    never import it back at module scope.
    """
    start = ((resets_epoch // 60) * 60.0) - window_s
    from .limit_readings import corroborated_resets, floor_reset_events
    granted = floor_reset_events(corroborated_resets(
        conn, bucket_key, start, until_epoch=observed, source=source))
    if granted:
        start = max(start, granted[-1]["at_epoch"])
    return start
