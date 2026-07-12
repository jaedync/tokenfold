"""Enterprise extra_usage meter: historize + derive.

The oauth/usage `extra_usage` block is Anthropic's own billing meter for the
org's current cycle — `monthly_limit` and `used_credits` in US cents
(self-described: currency=USD, decimal_places=2). Enterprise-account pushes
are rejected by the ingest stomp guard (no usable limit buckets), but the
guard siphons this block here, giving the enterprise side its only
billing-grade series without any manual entry.

Derivations:
- latest_meter: the authoritative MTD gauge (used/limit/utilization).
- daily_meter_deltas: per-UTC-day official spend increases, each interval
  compared against measured event cost over the same window — the gap is
  usage Tokenfold can't see (claude.ai web, mobile, Console).

A used_credits DROP between consecutive readings marks the billing-cycle
rollover; the newer reading's absolute value is the spend since the cycle
started (the meter counts up from zero).
"""

import math
import sqlite3
import time
from datetime import datetime, timezone
from typing import Optional

from .cost_windows import compute_window_cost
from .db import write_txn

# A meter older than this stops driving the gauge (no enterprise machine has
# pushed) — consumers fall back to the event-cost estimate.
METER_STALE_S = 48 * 3600


def log(msg, *args):
    formatted = msg % args if args else msg
    print(f"[extra_usage] {formatted}", flush=True)


def _finite_cents(value, minimum=0.0) -> Optional[float]:
    """Coerce an API cents field to a finite float >= minimum, else None.
    bool is an int subclass — reject it explicitly (True is not 1 cent)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        return None
    return value


def record_meter_reading(conn: sqlite3.Connection, machine: str,
                         extra: dict, fetched_epoch: float) -> bool:
    """Historize one captured extra_usage block. Returns True when a row
    landed; False when skipped (garbage used_credits, or the meter hasn't
    moved since the previous reading). Never raises — the caller is the
    ingest hot path."""
    used = _finite_cents(extra.get("used_credits"))
    if used is None:
        log("skipping reading: unusable used_credits %r (machine=%r)",
            extra.get("used_credits"), machine)
        return False
    # used_credits is the meter; a bad limit degrades to NULL rather than
    # losing the reading.
    limit = _finite_cents(extra.get("monthly_limit"), minimum=1e-9)
    utilization = extra.get("utilization")
    if isinstance(utilization, bool) or not isinstance(
            utilization, (int, float)) or not math.isfinite(
            float(utilization) if isinstance(utilization, (int, float))
            else 0.0):
        utilization = None

    # Dedupe check and insert share the write lock: two concurrent pushes of
    # the same reading must not both pass the check-then-insert.
    with write_txn(conn) as conn:
        prev = conn.execute(
            "SELECT used_cents, limit_cents FROM extra_usage_readings "
            "ORDER BY fetched_epoch DESC LIMIT 1").fetchone()
        if prev is not None and prev["used_cents"] == used \
                and prev["limit_cents"] == limit:
            return False

        conn.execute(
            "INSERT INTO extra_usage_readings"
            "(fetched_epoch, machine, used_cents, limit_cents, utilization) "
            "VALUES(?,?,?,?,?)",
            (fetched_epoch, machine, used, limit,
             float(utilization) if utilization is not None else None))
    return True


def latest_meter(conn: sqlite3.Connection) -> Optional[dict]:
    """Newest meter reading in dollars, or None when nothing captured yet."""
    row = conn.execute(
        "SELECT fetched_epoch, machine, used_cents, limit_cents, utilization "
        "FROM extra_usage_readings ORDER BY fetched_epoch DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    return {
        "used_usd": round(row["used_cents"] / 100.0, 2),
        "limit_usd": (round(row["limit_cents"] / 100.0, 2)
                      if row["limit_cents"] is not None else None),
        "utilization": row["utilization"],
        "fetched_epoch": row["fetched_epoch"],
        "machine": row["machine"],
    }


def meter_is_fresh(meter: Optional[dict],
                   now: Optional[float] = None) -> bool:
    if meter is None:
        return False
    now = time.time() if now is None else now
    return (now - meter["fetched_epoch"]) < METER_STALE_S


def build_meter_payload(conn: sqlite3.Connection, scope: str,
                        days: int = 30,
                        now: Optional[float] = None) -> Optional[dict]:
    """Dashboard payload: latest meter + daily official-vs-measured series.
    Enterprise scope only; None when nothing has been captured yet (the
    dashboard hides the section entirely)."""
    if scope != "enterprise":
        return None
    meter = latest_meter(conn)
    if meter is None:
        return None
    return {
        **meter,
        "fresh": meter_is_fresh(meter, now),
        "daily": daily_meter_deltas(conn, days, now),
    }


def daily_meter_deltas(conn: sqlite3.Connection, days: int = 30,
                       now: Optional[float] = None) -> list[dict]:
    """Per-UTC-day official spend increases vs measured event cost.

    Each consecutive reading pair contributes its interval to the UTC day of
    the LATER reading (the day the increase was observed — an interval can
    span a gap while no enterprise machine pushed, so this is attribution,
    not ground truth about when the spend happened). measured_usd prices the
    exact same [prev, cur) window from recorded enterprise events, so
    unaccounted_usd = official - measured isolates usage that never reaches
    Tokenfold. Sorted oldest-first.
    """
    now = time.time() if now is None else now
    since = now - days * 86400.0
    # One reading BEFORE the window seeds the first in-window delta.
    seed = conn.execute(
        "SELECT fetched_epoch, used_cents FROM extra_usage_readings "
        "WHERE fetched_epoch < ? ORDER BY fetched_epoch DESC LIMIT 1",
        (since,)).fetchone()
    rows = conn.execute(
        "SELECT fetched_epoch, used_cents FROM extra_usage_readings "
        "WHERE fetched_epoch >= ? ORDER BY fetched_epoch", (since,)).fetchall()
    readings = ([dict(seed)] if seed else []) + [dict(r) for r in rows]
    if len(readings) < 2:
        return []

    by_day: dict[str, dict] = {}
    for prev, cur in zip(readings, readings[1:]):
        delta_cents = cur["used_cents"] - prev["used_cents"]
        if delta_cents < 0:
            # Billing-cycle rollover: meter restarted from zero.
            delta_cents = cur["used_cents"]
        if delta_cents == 0:
            continue  # dedupe should prevent this, but limit-only rows exist
        measured = compute_window_cost(
            conn, prev["fetched_epoch"], cur["fetched_epoch"], "enterprise")
        day = datetime.fromtimestamp(
            cur["fetched_epoch"], tz=timezone.utc).strftime("%Y-%m-%d")
        slot = by_day.setdefault(
            day, {"day": day, "official_usd": 0.0, "measured_usd": 0.0})
        slot["official_usd"] += delta_cents / 100.0
        slot["measured_usd"] += measured

    out = []
    for day in sorted(by_day):
        slot = by_day[day]
        out.append({
            "day": day,
            "official_usd": round(slot["official_usd"], 2),
            "measured_usd": round(slot["measured_usd"], 2),
            "unaccounted_usd": round(
                slot["official_usd"] - slot["measured_usd"], 2),
        })
    return out
