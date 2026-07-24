"""Authoritative month-to-date figure for enterprise surfaces.

INVARIANT (enterprise scope): every user-facing month-to-date dollar figure
shows Anthropic's BILLED number when a fresh billing meter supplies one, and
any event-derived estimate shown alongside it is explicitly labelled as an
estimate. Tokenfold's own cost is a floor, not the bill: usage that never
reaches the ingest hook (claude.ai web, machines without the client, other
seats) is invisible to it.

Origin: on 2026-07-24 the dashboard hero showed the $926.85 estimate while
the meter read $999.48 against a $1,000 limit. The month limit was hit with
the headline number still implying ~$73 of headroom.

This module owns the choice in one place, unit-tested, with no DB access, so the
dashboard hero, the JSON API and the Home Assistant feed cannot drift apart.
Callers pass the event-derived estimate plus the meter payload built by
`extra_usage.build_meter_payload` and render the returned block verbatim.
"""

import math
from typing import Optional


def _finite(value) -> Optional[float]:
    """Coerce to a finite float, or None. Rejects bool (a bool is an int in
    Python and a `True` dollar amount is always a bug upstream)."""
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def month_hero_block(month_cost, meter: Optional[dict]) -> dict:
    """Return the month-to-date block the UI renders verbatim.

    `month_cost` is tokenfold's event-derived estimate for the UTC month.
    `meter` is the `build_meter_payload` dict (or None on personal scope /
    before the first capture); its `fresh` flag decides authority.

    Returns a NEW dict every call (never mutates `meter`) with keys:
      value            what the headline number must show
      source           'meter' (billed, authoritative) | 'estimate'
      measured_usd     tokenfold's event-derived estimate, always present
      unaccounted_usd  signed billed-minus-measured gap, None without a meter
      limit_usd        the account's monthly limit, None when unknown
      remaining_usd    headroom, floored at 0, None without a usable limit
      utilization      percent of limit used, None without a usable limit
    """
    measured = _finite(month_cost) or 0.0

    used = None
    if isinstance(meter, dict) and meter.get("fresh"):
        used = _finite(meter.get("used_usd"))

    if used is None:
        # No authoritative figure available: the estimate is all we have.
        return {
            "value": round(measured, 2),
            "source": "estimate",
            "measured_usd": round(measured, 2),
            "unaccounted_usd": None,
            "limit_usd": None,
            "remaining_usd": None,
            "utilization": None,
        }

    # A limit of 0 (or a missing/garbage one) is not a usable denominator;
    # headroom and utilization stay None rather than dividing by zero.
    limit = _finite(meter.get("limit_usd"))
    if limit is not None and limit <= 0:
        limit = None

    # Utilization is derived from the value actually on screen, not from the
    # meter's own coarse field, so the percentage can never contradict the
    # dollars printed next to it.
    remaining = round(max(0.0, limit - used), 2) if limit is not None else None
    utilization = round(used / limit * 100.0, 2) if limit is not None else None

    return {
        "value": round(used, 2),
        "source": "meter",
        "measured_usd": round(measured, 2),
        # Signed on purpose: tokenfold can over-measure too (pricing drift,
        # a machine double-pushing), and hiding that would mask a real bug.
        "unaccounted_usd": round(used - measured, 2),
        "limit_usd": round(limit, 2) if limit is not None else None,
        "remaining_usd": remaining,
        "utilization": utilization,
    }
