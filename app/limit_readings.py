"""Append-only OAuth usage-limit history: writer, reset detection, retention,
and GET /api/limit-history.

Both oauth_usage writers (server poller, client POST /api/usage) append one
row per normalized bucket per poll into limit_readings, so mid-window limit
resets survive the next INSERT OR REPLACE of the meta snapshot. Reset events
are DERIVED ON READ, never persisted: the heuristic stays tunable without a
re-migration, and the append-only source rows remain the truth.
"""

import re
import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

# No import cycle: api.py never imports this module at module level (its one
# dependency, .limit_trends -> bucket_trend/distinct_buckets, is imported
# LAZILY inside the route body — see app/api.py), so both shared helpers
# (fail-closed minute-scrub, ISO-to-epoch parse) are imported rather than
# duplicated (Fix 9: this used to have its own _epoch_or_none, byte-identical
# to api._iso_to_epoch — consolidated onto the one copy).
from .api import _scrub_to_minute_or_none, _iso_to_epoch
from .auth import require_dashboard_auth
from .db import get_conn, write_txn
from .usage_buckets import normalize_usage_buckets

router = APIRouter()

# Tunable heuristics for detect_resets — chosen against the 600s poll cadence.
RESET_DROP_PTS = 10.0  # utilization drop that can't be jitter (1-2pt dips seen)
RESET_JUMP_S = 1200    # resets_at forward jump > 2x the 600s poll interval
                       # (tolerates poll jitter without flagging every refresh)
# persistent_resets replay fingerprint: the reading AFTER a candidate reset
# recovering to >= this fraction of the pre-reset level means the "reset" was
# a stale out-of-order snapshot (a real grant restarts the meter near zero).
# Proportional, not a fixed point offset: "within RESET_DROP_PTS of before"
# was vacuously true for any utilization_before < 10, so low-utilization
# grants could never survive the filter (2026-07-09 incident).
RESET_RECOVERY_FRACTION = 0.8

# F7: 400 days (was 90) — the spend-history window chart derives peak-%
# per limit window from these rows, so retention IS the chart's horizon.
# Volume stays trivial: ~450 rows/day => ~180k rows/400d.
RETENTION_DAYS = 400

HOURS_DEFAULT = 168   # one weekly window
HOURS_MAX = 9600      # 400 days — matches retention, larger asks are noise
# The checklist said [a-z0-9_]; ':' is deliberately added because scoped
# bucket keys look like 'scoped:fable' (see usage_buckets normalizer).
_BUCKET_RE = re.compile(r"^[a-z0-9_:]{1,64}$")


def log(msg, *args):
    """Print with prefix (uvicorn swallows custom loggers by default)."""
    formatted = msg % args if args else msg
    print(f"[limit_readings] {formatted}", flush=True)


def _floor_to_minute_or_none(epoch: Optional[float]) -> Optional[float]:
    """Floor an epoch-seconds value to whole-minute precision; None passes
    through unchanged. Mirrors _scrub_to_minute_or_none (app/api.py): a raw
    resets_at_epoch is account-derived, and its sub-minute offset can
    fingerprint the account across responses, so it must never leave the
    server at full precision.
    """
    return None if epoch is None else (epoch // 60) * 60.0


def floor_reset_events(events):
    """Return a NEW list of reset-event dicts (never mutates ``events`` or
    its dicts) with ``at_epoch`` floored to the minute, matching every other
    epoch-valued field emitted on this surface (series, eta_100_epoch,
    resets_at_epoch_before/after — the latter already floored inside
    detect_resets). detect_resets itself stays full-precision so callers that
    only COUNT/FILTER by at_epoch (e.g. compute_burn's resets_in_window and
    segment cutoff) are unaffected; only response-shaping call sites
    (bucket_trend, GET /api/limit-history) floor their own served copy.
    """
    return [dict(e, at_epoch=_floor_to_minute_or_none(e["at_epoch"]))
            for e in events]


def record_limit_readings(conn, usage_dict, fetched_epoch, source):
    """Append one limit_readings row per normalized bucket. NEVER raises.

    Every-poll writes, NO dedupe-on-change: a "still N% at time T" row is
    exactly what bounds each integer step-crossing to one poll interval for
    the burn-rate interpolation (Workstream D). Volume math: ~3 buckets x
    144 polls/day ~= 450 rows/day — trivial for SQLite.

    Bucket-level validation is delegated to normalize_usage_buckets: invalid
    or garbage buckets are skipped there; only valid ones are recorded.
    """
    try:
        buckets = normalize_usage_buckets(usage_dict)
        if not buckets:
            return
        with write_txn(conn) as conn:
            for b in buckets:
                conn.execute(
                    "INSERT INTO limit_readings(fetched_epoch, source, bucket, "
                    "utilization, resets_at, resets_at_epoch) VALUES(?,?,?,?,?,?)",
                    (fetched_epoch, source, b["key"], b["utilization"],
                     b["resets_at"], _iso_to_epoch(b["resets_at"])))
    except Exception as e:  # writer must never break the poll/ingest path
        log("record_limit_readings failed (source=%s): %s", source, e)


def detect_resets(rows):
    """Detect account-level limit resets in a SINGLE bucket's readings.

    Contract: ``rows`` are dicts or sqlite3.Rows for ONE bucket, ordered by
    fetched_epoch ascending, each carrying bucket / fetched_epoch /
    utilization / resets_at_epoch. The CALLER filters by bucket.

    A consecutive pair (prev, cur) is a reset iff prev's window was still
    active (prev.resets_at_epoch > prev.fetched_epoch — rules out natural
    expiry, where utilization returns to ~0 after resets_at passes) AND one
    of:
    - utilization dropped >= RESET_DROP_PTS,
    - resets_at jumped forward by more than RESET_JUMP_S, or
    - utilization wiped to exactly 0 from a nonzero level while the window
      is still active at CUR's poll time (prev.resets_at_epoch >
      cur.fetched_epoch — the stricter guard, so a stale anchor whose expiry
      fell between the two polls stays a natural rollover). Utilization is a
      monotonic meter within a window, so a wipe is a grant at ANY
      magnitude — the 2026-07-09 account-wide grant zeroed a bucket at 9%,
      invisible to the magnitude rule.

    Returns event dicts with epoch fields only (no raw resets_at strings).

    Double-fire is possible for one real-world reset: a straddling
    transition can trip the resets_at-jump rule on one consecutive pair and
    then the >=10pt-drop rule on the very next pair, and a stale client push
    can replay a pre-reset snapshot out of order. This is a known heuristic
    limitation, not a bug — consumers should treat two events for the same
    bucket within RESET_JUMP_S of each other as one reset.
    """
    events = []
    prev = None
    for cur in rows:
        if prev is not None:
            prev_re = prev["resets_at_epoch"]
            if prev_re is not None and prev_re > prev["fetched_epoch"]:
                cur_re = cur["resets_at_epoch"]
                dropped = (cur["utilization"]
                           <= prev["utilization"] - RESET_DROP_PTS)
                jumped = cur_re is not None and cur_re > prev_re + RESET_JUMP_S
                zeroed = (cur["utilization"] == 0
                          and prev["utilization"] is not None
                          and prev["utilization"] > 0
                          and prev_re > cur["fetched_epoch"])
                if dropped or jumped or zeroed:
                    events.append({
                        "bucket": cur["bucket"],
                        "at_epoch": cur["fetched_epoch"],
                        "utilization_before": prev["utilization"],
                        "utilization_after": cur["utilization"],
                        # Floored to whole minutes (fingerprinting risk);
                        # see _floor_to_minute_or_none docstring above.
                        "resets_at_epoch_before":
                            _floor_to_minute_or_none(prev_re),
                        "resets_at_epoch_after":
                            _floor_to_minute_or_none(cur_re),
                    })
        prev = cur
    return events


def persistent_resets(rows):
    """detect_resets minus one-poll flukes, for COST-BEARING consumers.

    A stale/out-of-order client push (a lagging machine POSTs a snapshot
    captured minutes earlier) can replay a lower utilization for one row,
    which detect_resets reads as a >=10pt drop — a fake granted reset. The
    tell: on the very next reading the meter is back near its pre-"reset"
    level, which a REAL grant makes impossible (usage restarts near zero).
    So: drop any event whose next reading recovered to at least
    RESET_RECOVERY_FRACTION of utilization_before. Proportional on purpose:
    the old fixed offset ("within RESET_DROP_PTS of before") was vacuously
    true whenever utilization_before < RESET_DROP_PTS, so a low-utilization
    grant could NEVER survive this filter. Events with no subsequent
    reading yet are kept provisionally (self-corrects next poll).

    Burn/trend consumers keep using raw detect_resets — a spurious segment
    cut there only shortens the interpolation span, while here a spurious
    event moves limit_window.start_epoch and undercounts the headline
    spend (M1).
    """
    events = detect_resets(rows)
    kept = []
    for e in events:
        nxt = next((r for r in rows
                    if r["fetched_epoch"] > e["at_epoch"]), None)
        if (nxt is not None and nxt["utilization"] is not None
                and nxt["utilization"]
                >= e["utilization_before"] * RESET_RECOVERY_FRACTION):
            continue
        kept.append(e)
    return kept


def corroborated_resets(conn, bucket, since_epoch):
    """persistent_resets for one bucket, PLUS account-level resets
    corroborated by sibling buckets.

    An account-wide grant zeroes every bucket in the same poll, but a bucket
    whose meter merely DECREASED (e.g. 9 -> 1 because usage resumed inside
    the poll gap) clears none of the per-bucket rules. When a sibling bucket
    has a persistent granted reset at time T, and this bucket's own pair
    straddling T shows any utilization decrease while its window was still
    active, that decrease inherits the sibling's event (utilization is
    monotonic within a window — a decrease coinciding with a sibling grant
    is the same account-level reset).

    Two guards keep siblings honest:
    - only sibling events whose OWN window was still active at event time
      (resets_at_epoch_before > at_epoch) corroborate — five_hour natural
      expiries routinely masquerade as persistent "resets" and must never
      cut a weekly window;
    - sibling times within RESET_JUMP_S of an own event are skipped (same
      real-world reset, already counted).

    Returns detect_resets-shaped dicts sorted by at_epoch; borrowed events
    additionally carry corroborated_by=<sibling bucket>. Full-precision
    at_epoch like detect_resets — response-shaping callers floor their own
    served copy (floor_reset_events).
    """
    def _rows(b):
        return conn.execute(
            "SELECT bucket, fetched_epoch, utilization, resets_at_epoch "
            "FROM limit_readings WHERE bucket=? AND fetched_epoch>=? "
            "ORDER BY fetched_epoch ASC", (b, since_epoch)).fetchall()

    own_rows = _rows(bucket)
    events = persistent_resets(own_rows)
    siblings = [r[0] for r in conn.execute(
        "SELECT DISTINCT bucket FROM limit_readings "
        "WHERE fetched_epoch>=? AND bucket<>?", (since_epoch, bucket))]
    candidates = []
    for sib in siblings:
        for e in persistent_resets(_rows(sib)):
            before = e["resets_at_epoch_before"]
            if before is not None and before > e["at_epoch"]:
                candidates.append((e["at_epoch"], sib))
    for at, sib in sorted(candidates):
        if any(abs(e["at_epoch"] - at) <= RESET_JUMP_S for e in events):
            continue
        prev = None
        for cur in own_rows:
            if (prev is not None
                    and prev["fetched_epoch"] < at <= cur["fetched_epoch"]):
                prev_re = prev["resets_at_epoch"]
                if (prev_re is not None and prev_re > cur["fetched_epoch"]
                        and cur["utilization"] is not None
                        and prev["utilization"] is not None
                        and cur["utilization"] < prev["utilization"]):
                    events.append({
                        "bucket": bucket,
                        "at_epoch": cur["fetched_epoch"],
                        "utilization_before": prev["utilization"],
                        "utilization_after": cur["utilization"],
                        "resets_at_epoch_before":
                            _floor_to_minute_or_none(prev_re),
                        "resets_at_epoch_after": _floor_to_minute_or_none(
                            cur["resets_at_epoch"]),
                        "corroborated_by": sib,
                    })
                break
            prev = cur
    events.sort(key=lambda e: e["at_epoch"])
    return events


def prune_limit_readings(conn, now_epoch=None):
    """Delete readings older than RETENTION_DAYS. Returns rows deleted."""
    now = time.time() if now_epoch is None else now_epoch
    with write_txn(conn) as conn:
        cur = conn.execute(
            "DELETE FROM limit_readings WHERE fetched_epoch < ?",
            (now - RETENTION_DAYS * 86400,))
    if cur.rowcount:
        log("pruned %d readings older than %dd", cur.rowcount, RETENTION_DAYS)
    return cur.rowcount


@router.get("/api/limit-history",
            dependencies=[Depends(require_dashboard_auth)])
async def limit_history(bucket: str = Query(...),
                        hours: int = Query(default=HOURS_DEFAULT),
                        scope: Optional[str] = Query(default=None)):
    """Utilization history + derived reset events for one bucket.

    Personal-Max data only: enterprise scope or an enterprise-locked
    instance gets a neutral 404 — mirroring how /api/rate-limits simply
    omits the oauth block, a 404 avoids advertising the feature at all.
    """
    import sys
    cfg = sys.modules["app.config"]  # fresh read — importlib.reload safety
    if cfg.LOCKED_SCOPE == "enterprise":
        raise HTTPException(status_code=404, detail="not found")
    try:
        effective = cfg.resolve_scope(scope)
    except cfg.InvalidScope:
        raise HTTPException(status_code=400, detail="invalid scope")
    except cfg.ScopeLocked:
        raise HTTPException(status_code=404, detail="not found")
    if effective == "enterprise":
        raise HTTPException(status_code=404, detail="not found")

    # fullmatch (not match): '$' matches before a trailing newline, so
    # .match with a '$'-anchored pattern would let 'seven_day\n' through.
    if not _BUCKET_RE.fullmatch(bucket or ""):
        raise HTTPException(status_code=400, detail="invalid bucket")
    hours = max(1, min(HOURS_MAX, hours))  # clamp, never error

    conn = get_conn()
    since = time.time() - hours * 3600
    rows = conn.execute(
        "SELECT bucket, fetched_epoch, utilization, resets_at, "
        "resets_at_epoch FROM limit_readings "
        "WHERE bucket=? AND fetched_epoch>=? ORDER BY fetched_epoch ASC",
        (bucket, since)).fetchall()
    return {
        "bucket": bucket,
        "readings": [
            {
                "t": r["fetched_epoch"],
                "pct": r["utilization"],
                # raw in storage, minute-scrubbed (fail-closed) at this boundary
                "resets_at": _scrub_to_minute_or_none(r["resets_at"]),
            }
            for r in rows
        ],
        # Minute-floored like every other epoch field this endpoint emits
        # (resets_at above, resets_at_epoch_before/after inside each event) —
        # detect_resets itself stays full-precision (Fix 2). Corroborated
        # sibling events are merged in (skipping any within RESET_JUMP_S of
        # a raw event — same real-world reset) so the chart's markers agree
        # with the dollar-window anchors in /api/rate-limits.
        "resets": floor_reset_events(_resets_with_corroboration(
            conn, bucket, since, rows)),
    }


def _resets_with_corroboration(conn, bucket, since, rows):
    """Raw detect_resets events for the served window, plus
    corroborated-only events (borrowed from sibling buckets) that no raw
    event already covers. Sorted by at_epoch, full precision."""
    raw = detect_resets(rows)
    extras = [
        e for e in corroborated_resets(conn, bucket, since)
        if e.get("corroborated_by")
        and not any(abs(e["at_epoch"] - r["at_epoch"]) <= RESET_JUMP_S
                    for r in raw)
    ]
    return sorted(raw + extras, key=lambda e: e["at_epoch"])
