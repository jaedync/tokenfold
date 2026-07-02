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
from .db import get_conn
from .usage_buckets import normalize_usage_buckets

router = APIRouter()

# Tunable heuristics for detect_resets — chosen against the 600s poll cadence.
RESET_DROP_PTS = 10.0  # utilization drop that can't be jitter (1-2pt dips seen)
RESET_JUMP_S = 1200    # resets_at forward jump > 2x the 600s poll interval
                       # (tolerates poll jitter without flagging every refresh)

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
        for b in buckets:
            conn.execute(
                "INSERT INTO limit_readings(fetched_epoch, source, bucket, "
                "utilization, resets_at, resets_at_epoch) VALUES(?,?,?,?,?,?)",
                (fetched_epoch, source, b["key"], b["utilization"],
                 b["resets_at"], _iso_to_epoch(b["resets_at"])))
        conn.commit()
    except Exception as e:  # writer must never break the poll/ingest path
        # A failed batch must not leave pending writes on the shared
        # module-global connection for the next unrelated caller to commit.
        try:
            conn.rollback()
        except Exception:
            pass
        log("record_limit_readings failed (source=%s): %s", source, e)


def detect_resets(rows):
    """Detect account-level limit resets in a SINGLE bucket's readings.

    Contract: ``rows`` are dicts or sqlite3.Rows for ONE bucket, ordered by
    fetched_epoch ascending, each carrying bucket / fetched_epoch /
    utilization / resets_at_epoch. The CALLER filters by bucket.

    A consecutive pair (prev, cur) is a reset iff prev's window was still
    active (prev.resets_at_epoch > prev.fetched_epoch — rules out natural
    expiry, where utilization returns to ~0 after resets_at passes) AND
    either utilization dropped >= RESET_DROP_PTS or resets_at jumped forward
    by more than RESET_JUMP_S.

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
                if dropped or jumped:
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
    So: drop any event whose next reading recovered to within
    RESET_DROP_PTS of utilization_before. Events with no subsequent
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
                > e["utilization_before"] - RESET_DROP_PTS):
            continue
        kept.append(e)
    return kept


def prune_limit_readings(conn, now_epoch=None):
    """Delete readings older than RETENTION_DAYS. Returns rows deleted."""
    now = time.time() if now_epoch is None else now_epoch
    cur = conn.execute(
        "DELETE FROM limit_readings WHERE fetched_epoch < ?",
        (now - RETENTION_DAYS * 86400,))
    conn.commit()
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
        # detect_resets itself stays full-precision (Fix 2).
        "resets": floor_reset_events(detect_resets(rows)),
    }
