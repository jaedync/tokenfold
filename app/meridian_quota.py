"""Meridian /v1/usage/quota -> POST /api/usage/claude payload.

Port of dotfleet pi/lib/claude-usage-payload.ts (parseClaudeUsage). Both feed
the same endpoint, so they must emit identical payloads for identical input;
app/tests/test_meridian_quota.py mirrors the TypeScript tests against the
shared client fixture to hold that parity.
"""
import math
import re

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
MAX_BUCKETS = 16
MAX_SKEW_MS = 300_000
MAX_AGE_MS = 86_400_000


def _record(value):
    return value if isinstance(value, dict) else {}


def _finite(value):
    # bool is an int subclass in Python but never a JSON number in TypeScript.
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # a huge JSON integer; JavaScript reads it as Infinity
        return False


def _identity(kind):
    if kind == "five_hour":
        return "five_hour", "5-Hour"
    if kind == "seven_day":
        return "seven_day", "7-Day"
    if not isinstance(kind, str) or not kind.startswith("seven_day_"):
        return None
    slug = kind[len("seven_day_"):]
    # fullmatch: Python's '$' also matches before a trailing newline; JS's does not.
    if len(slug) > 57 or not _SLUG_RE.fullmatch(slug):
        return None
    return f"scoped:{slug}", slug[0].upper() + slug[1:].replace("_", " ")


def _bucket(raw, fetched_at):
    """One validated bucket, or None to skip it. Meridian times are ms."""
    bucket = _record(raw)
    # Never let SDK metadata masquerade as the shared OAuth observation.
    if bucket.get("observedAt") != fetched_at or not _finite(bucket.get("observedAt")):
        return None
    if "source" in bucket and bucket["source"] != "oauth":
        return None
    identity = _identity(bucket.get("type"))
    utilization, resets_at = bucket.get("utilization"), bucket.get("resetsAt")
    if (identity is None or not _finite(utilization) or not 0 <= utilization <= 1
            or not _finite(resets_at) or resets_at <= 0 or resets_at / 1000 > 1e11):
        return None
    key, label = identity
    return {"key": key, "label": label, "pct": utilization * 100,
            "resets_at_epoch": resets_at / 1000}


def parse_meridian_quota(body, machine, now_epoch):
    """Return the strict ingest payload, or None when the body is unusable.

    Utilization is always fractional. The observation time is the OAuth
    fetch time, never the receipt time, so a cached or stale Meridian answer
    cannot look newer than it is.
    """
    # JavaScript trim() also strips U+FEFF; Python's strip() does not.
    if (not isinstance(machine, str) or not machine.replace("\ufeff", "").strip()
            or len(machine) > 128 or _CONTROL_RE.search(machine)):
        return None
    quota = _record(body)
    if "profile" not in quota or quota["profile"] not in (None, "default"):
        return None
    fetched_at = _record(_record(quota.get("sources")).get("oauth")).get("fetchedAt")
    now_ms = now_epoch * 1000
    if (not _finite(fetched_at) or fetched_at <= 0 or not _finite(now_ms)
            or fetched_at > now_ms + MAX_SKEW_MS or now_ms - fetched_at > MAX_AGE_MS):
        return None
    raw_buckets = quota.get("buckets")
    if not isinstance(raw_buckets, list) or len(raw_buckets) > MAX_BUCKETS:
        return None
    buckets = []
    for raw in raw_buckets:
        bucket = _bucket(raw, fetched_at)
        if bucket is None:
            continue
        if any(b["key"] == bucket["key"] for b in buckets):
            return None
        buckets.append(bucket)
    keys = {b["key"] for b in buckets}
    if not {"five_hour", "seven_day"} <= keys:
        return None
    payload = {"machine": machine, "account_class": "personal",
               "source": "meridian-oauth", "source_profile": "default",
               "observed_at_epoch": fetched_at / 1000, "buckets": buckets}
    # Source currency units are not established. Explicit disabled is safe;
    # enabled spend must not invent zero dollars or a monthly reset.
    if _record(quota.get("extraUsage")).get("isEnabled") is False:
        payload["extra_usage"] = {"enabled": False}
    return payload
