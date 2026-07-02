"""Normalize the fluid OAuth usage payload into a stable bucket list.

The Anthropic OAuth usage API is mid-migration: per-model limits now arrive
ONLY in a new ``data.limits[]`` array (kind session/weekly_all/weekly_scoped)
while the legacy flat dict buckets (``five_hour``, ``seven_day``,
``seven_day_<model>``) still populate on some plans and are null noise on
others. This module is the single source of truth for merging both shapes
(reused by /api/rate-limits, /api/ha, and future limit historization).

Contract:
- ``limits[]`` entries are the PRIMARY source; legacy dict buckets are the
  FALLBACK: they fill keys limits[] did not provide, plus a null resets_at
  on a limits[] bucket that shares a key with a legacy one.
- Output entries: {"key", "label", "utilization", "resets_at"} with resets_at
  RAW (minute-scrubbing happens at the API boundaries, never here) and
  utilization clamped to >= 0.0.
- The payload shape is fluid: null values, non-dict buckets, missing or
  non-numeric utilization, and unknown kinds are silently skipped. This
  function NEVER raises on unknown shapes.
"""

import math
import re

# Non-bucket keys consumed elsewhere (api.py reads extra_usage directly) or
# irrelevant to utilization — never treated as buckets even if dict-shaped.
_IGNORED_KEYS = frozenset({"spend", "extra_usage", "member_dashboard_available"})

_SLUG_RE = re.compile(r"[^a-z0-9_]+")

_FIXED_KEYS = ("five_hour", "seven_day")
_FIXED_LABELS = {"five_hour": "5-Hour", "seven_day": "7-Day"}


def _finite_number(value):
    """Return float(value) for finite int/float inputs; None otherwise.

    Bools are explicitly rejected (they are ints in Python, but a True
    utilization is garbage, not 100%).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def _slug(name):
    """Lowercase and reduce to [a-z0-9_]: 'Nova 9' -> 'nova_9'."""
    return _SLUG_RE.sub("_", name.strip().lower()).strip("_")


def _raw_resets_at(entry):
    """resets_at passes through RAW, but only if it is a string."""
    value = entry.get("resets_at")
    return value if isinstance(value, str) else None


def _bucket(key, label, utilization, resets_at):
    return {
        "key": key,
        "label": label,
        # Negative utilization is API noise, not credit — clamp to 0.0.
        "utilization": max(0.0, utilization),
        "resets_at": resets_at,
    }


def _buckets_from_limits(limits):
    """Parse the new limits[] array into {key: bucket}. Primary source."""
    out = {}
    if not isinstance(limits, list):
        return out
    for entry in limits:
        if not isinstance(entry, dict):
            continue
        pct = _finite_number(entry.get("percent"))
        if pct is None:
            continue
        kind = entry.get("kind")
        if kind in ("session", "weekly_all"):
            key = "five_hour" if kind == "session" else "seven_day"
            label = _FIXED_LABELS[key]
        elif kind == "weekly_scoped":
            scope = entry.get("scope")
            model = scope.get("model") if isinstance(scope, dict) else None
            display = model.get("display_name") if isinstance(model, dict) else None
            # scope.model.id may be null today or populated in the future —
            # the display_name is the identity we key on. No name, no bucket.
            if not isinstance(display, str) or not _slug(display):
                continue
            key = "scoped:" + _slug(display)
            label = display  # verbatim
        else:
            continue  # unknown kind — payload is fluid, skip silently
        if key not in out:  # first entry per key wins
            out[key] = _bucket(key, label, pct, _raw_resets_at(entry))
    return out


def _buckets_from_legacy(usage):
    """Parse legacy flat dict buckets into {key: bucket}. Fallback source."""
    out = {}
    for name, value in usage.items():
        if name in _IGNORED_KEYS or name == "limits":
            continue
        if not isinstance(value, dict):
            continue  # null noise keys (tangelo, iguana_necktie, ...)
        utilization = _finite_number(value.get("utilization"))
        if utilization is None:
            continue
        if name in _FIXED_KEYS:
            key, label = name, _FIXED_LABELS[name]
        elif name.startswith("seven_day_"):
            suffix = _slug(name[len("seven_day_"):])
            if not suffix:
                continue
            key = "scoped:" + suffix
            label = suffix.replace("_", " ").title()
        else:
            continue  # unknown dict shape — skip, never raise
        if key not in out:
            out[key] = _bucket(key, label, utilization, _raw_resets_at(value))
    return out


def normalize_usage_buckets(usage):
    """Merge limits[] (primary) and legacy dict buckets (fallback).

    Returns a new list of {"key", "label", "utilization", "resets_at"} dicts
    ordered five_hour, seven_day, then scoped:* keys sorted alphabetically
    (stable render order). Empty list for anything unparseable.
    """
    if not isinstance(usage, dict):
        return []
    primary = _buckets_from_limits(usage.get("limits"))
    fallback = _buckets_from_legacy(usage)
    # limits[] wins per field — it is the new API — but a null resets_at in a
    # limits[] entry is a data gap, not an override: fill it from the legacy
    # bucket when one exists. New dicts throughout; inputs are never mutated.
    merged = dict(fallback)
    for key, p in primary.items():
        f = fallback.get(key)
        merged[key] = ({**p, "resets_at": p["resets_at"] or f["resets_at"]}
                       if f is not None else p)
    ordered = [merged[k] for k in _FIXED_KEYS if k in merged]
    ordered += [merged[k] for k in sorted(merged) if k.startswith("scoped:")]
    return ordered
