"""Look-back spend analytics (Workstream F): weekly limit-window segments
and true-UTC monthly per-model costs, served by GET /api/spend-history.

Window segmentation is DERIVED ON READ from limit_readings (same philosophy
as detect_resets — no persisted events, heuristics stay tunable):

- NATURAL boundaries come from anchor transitions: when consecutive readings
  show resets_at_epoch moving from A to B and A had already passed by the
  later poll, the window ended at exactly A — its scheduled expiry — even if
  the server slept through the rollover.
- GRANTED boundaries are mid-window resets: the meter plunges while the
  anchor stays put, or the anchor jumps forward before the old one expired.
  Their epoch is the observing poll time (bounded by the poll interval).
- PRE-HISTORIZATION boundaries are inferred by stepping the earliest known
  window start back in 7-day increments; those segments carry inferred=True
  (their spend is exact, the cut points assume the anchor cadence held).

A single week can therefore contain MULTIPLE segments — a granted reset
splits it — which is exactly why entries are per reset-bounded segment,
never per calendar week.

Monthly costs use true UTC calendar months computed from raw events, NOT
the local-TZ day summaries: `events.day` buckets in TZ (America/Chicago by
default), so summing days would misplace every event in the hours around a
month boundary.
"""

import json
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

# Cycle check: api.py never imports this module (routers meet in main.py),
# and limit_readings -> api is the only edge below us — acyclic.
from .api import _iso_to_epoch
from .auth import require_dashboard_auth
from .config import scope_predicate
from .cost_windows import compute_window_cost, compute_window_cost_by_model
from .db import get_conn
# Thresholds single-sourced from limit_readings so a future tuning pass
# changes burn-segmentation and window-segmentation together.
from .limit_readings import RESET_DROP_PTS, RESET_JUMP_S
from .usage_buckets import normalize_usage_buckets

router = APIRouter()

WINDOW_S = 7 * 86400
MERGE_S = 3600       # boundaries closer than this are one real-world event
MAX_SEGMENTS = 60    # response/CPU bound; oldest segments drop first
MAX_MONTHS = 72      # ~6 years of bars; also caps SQL queries per request
_MIN_MODEL_COST = 0.005  # by_model entries below rounding visibility drop


def log(msg, *args):
    formatted = msg % args if args else msg
    print(f"[spend_history] {formatted}", flush=True)


def _floor_minute(epoch):
    """Minute-floor account-derived epochs (fingerprinting invariant — same
    rule as api._scrub_to_minute_or_none / limit_readings)."""
    return None if epoch is None else (epoch // 60) * 60.0


def _pair_boundary(prev, cur, nxt):
    """Boundary implied by one consecutive reading pair, or None.

    Sibling of limit_readings.detect_resets with one extra power: because
    this asks "when did the window END" (not "cut the burn series here"),
    an anchor transition whose old anchor already passed yields the EXACT
    scheduled expiry (kind natural) instead of the poll time.

    Stale-replay hardening (review M1): a lagging client can insert an
    out-of-order snapshot between two fresh rows. Two defenses:
    - only FORWARD anchor jumps count (a backward "jump" is by definition
      an older snapshot — the surrounding fresh rows re-derive the real
      transition, and _merge_boundaries dedupes it);
    - an in-place meter drop must PERSIST into the next reading (``nxt``);
      a real grant keeps the meter near zero, a replay recovers instantly.
    """
    p_anchor = prev["resets_at_epoch"]
    c_anchor = cur["resets_at_epoch"]
    if (p_anchor is not None and c_anchor is not None
            and c_anchor > p_anchor + RESET_JUMP_S):
        if p_anchor <= cur["fetched_epoch"]:
            # Old window ran to its scheduled end somewhere in this gap.
            return {"epoch": _floor_minute(p_anchor), "kind": "natural"}
        # Anchor replaced while the old window was still live — granted.
        return {"epoch": _floor_minute(cur["fetched_epoch"]),
                "kind": "granted"}
    # Meter wiped in place (anchor unchanged, window active) — granted,
    # but only if the drop persists into the next reading (or none exists
    # yet — kept provisionally, self-corrects on the next poll).
    if (p_anchor is not None and p_anchor > prev["fetched_epoch"]
            and cur["utilization"] is not None
            and prev["utilization"] is not None
            and cur["utilization"] <= prev["utilization"] - RESET_DROP_PTS):
        recovered = (nxt is not None and nxt["utilization"] is not None
                     and nxt["utilization"]
                     > prev["utilization"] - RESET_DROP_PTS)
        if not recovered:
            return {"epoch": _floor_minute(cur["fetched_epoch"]),
                    "kind": "granted"}
    return None


def _merge_boundaries(candidates):
    """Sort + collapse candidates within MERGE_S of each other.

    A collision is one real-world transition observed twice (detect-style
    double-fire, or an expiry re-detected as a drop). 'natural' wins the
    collision outright — its epoch is the exact scheduled expiry, while a
    granted epoch is only poll-bounded.
    """
    out = []
    for b in sorted(candidates, key=lambda x: x["epoch"]):
        if out and b["epoch"] - out[-1]["epoch"] < MERGE_S:
            if b["kind"] == "natural" and out[-1]["kind"] == "granted":
                out = out[:-1] + [dict(b)]
            continue
        out.append(dict(b))
    return out


def weekly_window_segments(conn, scope, now=None, anchor_epoch=None):
    """Reset-bounded seven_day window segments, oldest -> ongoing last.

    Each segment: {start_epoch, end_epoch (None while ongoing), end_kind
    natural|granted|ongoing, cost, peak_pct (None pre-historization),
    inferred, projected_end_epoch (ongoing only)}. All epochs minute-floored.
    Returns [] when no anchor is derivable (no oauth data ever seen).
    """
    now = time.time() if now is None else now
    rows = conn.execute(
        "SELECT bucket, fetched_epoch, utilization, resets_at_epoch "
        "FROM limit_readings WHERE bucket='seven_day' "
        "ORDER BY fetched_epoch ASC").fetchall()

    if anchor_epoch is None:
        for r in reversed(rows):
            if r["resets_at_epoch"] is not None:
                anchor_epoch = r["resets_at_epoch"]
                break
    if anchor_epoch is None:
        return []
    anchor_epoch = _floor_minute(anchor_epoch)

    # Clamp the look-back horizon (review H1): one event with a garbage
    # ancient ts_epoch (unset RTC, year-1 ISO string) must not drive an
    # unbounded back-step loop of per-segment SQL queries. MAX_SEGMENTS
    # windows is all we will ever emit, so nothing older can matter.
    oldest_event = _oldest_event_epoch(
        conn, scope, floor_epoch=now - (MAX_SEGMENTS + 2) * WINDOW_S)

    candidates = []
    prev = None
    for i, cur in enumerate(rows):
        if prev is not None:
            nxt = rows[i + 1] if i + 1 < len(rows) else None
            b = _pair_boundary(prev, cur, nxt)
            if b is not None and b["epoch"] is not None and b["epoch"] <= now:
                candidates.append(b)
        prev = cur
    derived = [dict(b, inferred=False) for b in _merge_boundaries(candidates)]

    # Earliest KNOWN window start: the earliest anchor ever observed names a
    # window that ENDS there, so that window STARTED one cadence earlier.
    # Every derived boundary sits above this base (anchors are within 7d of
    # their observing poll), so the inferred chain and the derived list
    # never interleave.
    earliest_anchor = min(
        (r["resets_at_epoch"] for r in rows
         if r["resets_at_epoch"] is not None),
        default=anchor_epoch)
    base = _floor_minute(earliest_anchor) - WINDOW_S
    inferred_epochs = [base]
    # len() cap is belt-and-braces on top of the horizon clamp above —
    # the chain must stay bounded even if a future edit widens the clamp.
    while (len(inferred_epochs) < MAX_SEGMENTS + 2
           and inferred_epochs[-1] > (oldest_event
                                      if oldest_event is not None
                                      else base)):
        inferred_epochs.append(inferred_epochs[-1] - WINDOW_S)
    inferred = [{"epoch": e, "kind": "natural", "inferred": True}
                for e in sorted(inferred_epochs)]

    bounds = inferred + derived
    segments = []
    for b0, b1 in zip(bounds, bounds[1:]):
        if oldest_event is not None and b1["epoch"] <= oldest_event:
            continue  # entirely before any event data — nothing to show
        segments.append(_build_segment(conn, scope, b0, b1["epoch"],
                                       b1["kind"],
                                       b0["inferred"] or b1["inferred"]))
    last = bounds[-1]
    # The ongoing segment is never 'inferred' (review L1): it covers LIVE
    # data — its start is either an observed boundary or the exact
    # current-anchor − 7d; rendering the live window faded would wrongly
    # imply its spend is assumed.
    ongoing = _build_segment(conn, scope, last, None, "ongoing",
                             False, now=now)
    ongoing["projected_end_epoch"] = anchor_epoch
    segments.append(ongoing)

    if len(segments) > MAX_SEGMENTS:
        log("dropping %d oldest segments (MAX_SEGMENTS=%d)",
            len(segments) - MAX_SEGMENTS, MAX_SEGMENTS)
        segments = segments[-MAX_SEGMENTS:]
    return segments


def _oldest_event_epoch(conn, scope, floor_epoch):
    """Earliest priced event visible to THIS scope (same predicates as
    compute_window_cost) — scope-filtered so an enterprise view never gets
    a history axis derived from personal activity, and vice versa.

    Clamped to ``floor_epoch`` (review H1): ingest never range-validates
    timestamps, so a single event with an ancient ts_epoch (epoch 0,
    year-1) would otherwise drive tens of thousands of look-back
    iterations — each one a SQL query — on every dashboard load.
    """
    row = conn.execute(
        "SELECT MIN(ts_epoch) AS t FROM events WHERE type='assistant' "
        "AND model IS NOT NULL AND model != '<synthetic>' "
        "AND request_id IS NOT NULL "
        f"AND {scope_predicate(scope)}").fetchone()
    if not row or row["t"] is None:
        return None
    return max(row["t"], floor_epoch)


def _build_segment(conn, scope, b0, end_epoch, end_kind, inferred, now=None):
    cost_end = end_epoch if end_epoch is not None else (
        time.time() if now is None else now)
    peak = conn.execute(
        "SELECT MAX(utilization) AS p FROM limit_readings "
        "WHERE bucket='seven_day' AND fetched_epoch >= ? "
        "AND fetched_epoch < ?", (b0["epoch"], cost_end)).fetchone()
    return {
        "start_epoch": b0["epoch"],
        "end_epoch": end_epoch,
        "end_kind": end_kind,
        "cost": round(compute_window_cost(conn, b0["epoch"], cost_end,
                                          scope=scope), 2),
        "peak_pct": peak["p"] if peak else None,
        "inferred": bool(inferred),
    }


def monthly_costs(conn, scope, now=None):
    """Per-UTC-calendar-month per-model cost, oldest month -> current.

    total is the rounded sum of UNROUNDED model costs, so it can differ
    from the sum of the rounded by_model values by a cent — chart stacks
    use by_model, total is tooltip copy.
    """
    now = time.time() if now is None else now
    # 32-day stride over-covers each month so the clamp can never trim a
    # month that MAX_MONTHS iterations would still emit (review H1).
    oldest = _oldest_event_epoch(
        conn, scope, floor_epoch=now - MAX_MONTHS * 32 * 86400)
    if oldest is None:
        return []
    d = datetime.fromtimestamp(oldest, tz=timezone.utc)
    y, m = d.year, d.month
    months = []
    while len(months) < MAX_MONTHS:
        start = datetime(y, m, 1, tzinfo=timezone.utc).timestamp()
        if start > now:
            break
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        end = datetime(ny, nm, 1, tzinfo=timezone.utc).timestamp()
        raw = compute_window_cost_by_model(conn, start, end, scope=scope)
        months.append({
            "month": "%04d-%02d" % (y, m),
            "total": round(sum(raw.values()), 2),
            "by_model": {k: round(v, 2) for k, v in sorted(raw.items())
                         if v >= _MIN_MODEL_COST},
            "partial": end > now,
        })
        y, m = ny, nm
    return months


@router.get("/api/spend-history",
            dependencies=[Depends(require_dashboard_auth)])
async def spend_history(scope: Optional[str] = Query(default=None)):
    """Look-back spend: months always (scope-filtered); 'windows' ONLY for
    personal scope on a non-enterprise-locked instance (personal Max limit
    data — mirrors the rate-limits oauth-key gating; enterprise callers
    never learn the key exists)."""
    import sys
    cfg = sys.modules["app.config"]  # fresh read — importlib.reload safety
    try:
        effective = cfg.resolve_scope(scope)
    except cfg.InvalidScope:
        raise HTTPException(status_code=400, detail="invalid scope")
    except cfg.ScopeLocked:
        raise HTTPException(status_code=404, detail="not found")

    conn = get_conn()
    now = time.time()
    # Same narrow-blast-radius treatment as windows below (review H1): a
    # surprise in month math (e.g. platform-specific fromtimestamp range
    # errors) must degrade to an empty chart, never a 500.
    out = {"months": []}
    try:
        out = {"months": monthly_costs(conn, effective, now)}
    except Exception as e:
        log("monthly_costs failed: %s", e)

    if effective == "personal" and cfg.LOCKED_SCOPE != "enterprise":
        # Narrow blast radius (api.py Fix 6 pattern): a segmentation bug
        # must only drop 'windows', never the months chart.
        try:
            anchor = None
            row = conn.execute(
                "SELECT value FROM meta WHERE key='oauth_usage'").fetchone()
            if row:
                usage = json.loads(row["value"]).get("data", {})
                for b in normalize_usage_buckets(usage):
                    if b["key"] == "seven_day":
                        anchor = _iso_to_epoch(b["resets_at"])
                        break
            windows = weekly_window_segments(conn, "personal", now=now,
                                             anchor_epoch=anchor)
            if windows:
                out["windows"] = windows
        except Exception as e:
            log("window segmentation failed: %s", e)
    return out
