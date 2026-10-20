"""Enterprise extra_usage meter: historize + derive.

The oauth/usage `extra_usage` block is Anthropic's own billing meter for the
org's current cycle — `monthly_limit` and `used_credits` in US cents
(self-described: currency=USD, decimal_places=2). Enterprise-account pushes
are rejected by the ingest stomp guard (no usable limit buckets), but the
guard siphons this block here, giving the enterprise side its only
billing-grade series without any manual entry.

Derivations:
- latest_meter: the authoritative MTD gauge (used/limit/utilization).
- daily_meter_deltas: per-local-day official spend increases, each interval
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
from zoneinfo import ZoneInfo

from .config import TZ_NAME
from .cost_windows import compute_window_cost
from .db import write_txn
from .meter_gap import local_days, meter_days, read_meter, window_cost

_TZ = ZoneInfo(TZ_NAME)

# A meter older than this is no longer shown as the billed figure by itself.
# Inside its own UTC month it stays the base of the month-to-date figure
# (meter_anchor); after the month turns over consumers use the estimate.
METER_STALE_S = 48 * 3600
# A meter limit older than this no longer overrides the stored budget, so
# the budget becomes editable again once no enterprise machine reports.
REMEMBERED_LIMIT_MAX_AGE_S = 92 * 86400


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
    # A reading from last month is not this month's billed figure, even when
    # it is only minutes old: the cycle resets at the UTC month boundary.
    return ((now - meter["fetched_epoch"]) < METER_STALE_S
            and meter["fetched_epoch"] >= _utc_month_start(now))


def remembered_limit_usd(conn: sqlite3.Connection,
                         now: Optional[float] = None) -> Optional[float]:
    """Newest positive monthly limit in dollars, up to
    REMEMBERED_LIMIT_MAX_AGE_S old.

    The limit is an org setting, not a spend sample, so it stays valid while
    no enterprise machine pushes, and across a reading whose limit was
    unusable (stored NULL)."""
    now = time.time() if now is None else now
    row = conn.execute(
        "SELECT limit_cents FROM extra_usage_readings "
        "WHERE limit_cents IS NOT NULL AND limit_cents > 0 "
        "AND fetched_epoch >= ? "
        "ORDER BY fetched_epoch DESC LIMIT 1",
        (now - REMEMBERED_LIMIT_MAX_AGE_S,)).fetchone()
    return round(row["limit_cents"] / 100.0, 2) if row else None


def _utc_month_start(epoch: float) -> float:
    d = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return d.replace(day=1, hour=0, minute=0, second=0,
                     microsecond=0).timestamp()


def meter_anchor(conn: sqlite3.Connection, meter: Optional[dict],
                 now: Optional[float] = None) -> Optional[dict]:
    """Month-to-date figure anchored on a stale same-month reading.

    Regression 2026-10-02: when the reporter stopped for 48h the gauge fell
    back to the event estimate, which misses claude.ai and any unreported
    machine, so the month figure dropped. Billing counts up from zero each
    cycle, so a same-month reading is still a hard floor: keep it as the
    base and add only the spend Tokenfold measured after it. None when the
    meter is fresh (shown as-is; this includes clock-skewed future readings)
    or from an earlier month.
    """
    if meter is None:
        return None
    now = time.time() if now is None else now
    fetched = meter["fetched_epoch"]
    if meter_is_fresh(meter, now) or fetched < _utc_month_start(now):
        return None
    # The meter bills Anthropic only: other work providers are not on it.
    since = compute_window_cost(conn, fetched, now, "enterprise", anthropic_only=True)
    return {
        "base_usd": meter["used_usd"],
        "measured_since_usd": round(since, 2),
        "used_usd": round(meter["used_usd"] + since, 2),
    }


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
        "anchored": meter_anchor(conn, meter, now),
        "remembered_limit_usd": remembered_limit_usd(conn, now),
        "daily": daily_meter_deltas(conn, days, now),
    }


def daily_meter_deltas(conn: sqlite3.Connection, days: int = 30,
                       now: Optional[float] = None) -> list[dict]:
    """Per-local-day official spend increases vs measured event cost.

    One source with the ledger's meter-gap line (app/meter_gap.meter_days):
    a day's bill runs from the last reading at or before its midnight to the
    last reading inside it, so an interval that spans a gap while no work
    machine pushed goes to the day of the later reading (attribution, not
    ground truth). measured_usd prices that same window from recorded work
    Claude events, so unaccounted_usd = official - measured isolates usage
    that never reaches Tokenfold. Sorted oldest-first.
    """
    now = time.time() if now is None else now
    since = now - days * 86400.0
    rows = meter_days(read_meter(conn, since, now), local_days(since, now, _TZ), _TZ, now,
                      window_cost(conn, now))
    return [{
        "day": r["day"],
        "official_usd": round(r["billed"], 2),
        "measured_usd": round(r["recorded"], 2),
        "unaccounted_usd": round(r["billed"] - r["recorded"], 2),
    } for r in rows if r["billed"]]
