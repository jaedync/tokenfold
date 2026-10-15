"""Limit-window history for /next: the current window's trajectory and the
past windows of one bucket. No routes (app/runway.py serves them).

Both read limit_readings with the same rules as the existing surfaces:
window starts come from limit_windows._bucket_window_start (granted resets
move the start), boundaries from spend_history's pair rules, grants from
limit_readings.persistent_resets. Every epoch that leaves here is
minute-floored (limit timestamps can fingerprint the account).
"""

import json
import time

from . import db
from .cost_windows import compute_window_cost, compute_window_cost_by_model
from .limit_readings import RESET_JUMP_S, persistent_resets
from .limit_trends import downsample
from .limit_windows import _bucket_window_start, _iso_to_epoch
from .quota_history import active_history_source, history_source_filter
from .read_cache import ReadCache
from .runway_rules import MIN_PCT_FOR_RATE
from .spend_history import _merge_boundaries, _pair_boundary
from .usage_buckets import normalize_usage_buckets
from .usage_pace import FIVE_HOUR_S, WEEK_S

TRAJECTORY_MAX_POINTS = 400
MAX_WINDOWS = 60       # newest past windows served; bounds the cost queries
# A closed window's spend changes only through late ingest (a machine that
# was offline pushes its backlog), so it may stay cached much longer than
# the 30 s of live feeds.
_window_cost_cache = ReadCache(ttl=3600, max_entries=4096)


def _floor(epoch):
    return None if epoch is None else (epoch // 60) * 60.0


def window_seconds(bucket):
    return FIVE_HOUR_S if bucket == "five_hour" else WEEK_S


def _rows(conn, bucket, since, until, source):
    predicate, params = history_source_filter(source)
    return conn.execute(
        "SELECT bucket, fetched_epoch, utilization, resets_at_epoch "
        "FROM limit_readings WHERE bucket=? AND fetched_epoch>=? "
        f"AND fetched_epoch<=? AND {predicate} ORDER BY fetched_epoch ASC",
        (bucket, since, until) + params).fetchall()


def claude_snapshot(conn):
    """(buckets by key, observed epoch, history source) from the stored
    OAuth snapshot, with the observation precedence of /api/rate-limits."""
    row = conn.execute(
        "SELECT value FROM meta WHERE key='oauth_usage'").fetchone()
    if not row:
        return {}, None, None
    stored = json.loads(row["value"])
    by_key = {b["key"]: b for b in normalize_usage_buckets(
        stored.get("data") or {})}
    observed = stored.get("observed_at_epoch")
    if not isinstance(observed, (int, float)) or isinstance(observed, bool):
        observed = _iso_to_epoch(stored.get("updated_at"))
    return by_key, observed, stored.get("source")


# ── trajectory ───────────────────────────────────────────────────────────

def five_hour_resets(conn, start, end, source):
    """5 h window boundaries (natural expiries and grants) in (start, end]."""
    rows = _rows(conn, "five_hour", start - FIVE_HOUR_S, end, source)
    candidates = []
    for i in range(1, len(rows)):
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        b = _pair_boundary(rows[i - 1], rows[i], nxt)
        if b is not None and b["epoch"] is not None:
            candidates.append(b)
    return [b["epoch"] for b in _merge_boundaries(candidates)
            if start < b["epoch"] <= end]


def claude_trajectory(conn, bucket, now):
    """{points, resets, start, end} for the current window of one Claude
    bucket, or None when the snapshot has no such bucket or reset time."""
    by_key, observed, source = claude_snapshot(conn)
    b = by_key.get(bucket)
    resets_epoch = _iso_to_epoch((b or {}).get("resets_at"))
    if resets_epoch is None:
        return None
    window_s = window_seconds(bucket)
    end = _floor(resets_epoch)
    start = _bucket_window_start(conn, bucket, resets_epoch, window_s,
                                 observed, source=source)
    # Rows of this window only: a lagging poll just after a natural start
    # can still carry the previous window's anchor.
    rows = [r for r in _rows(conn, bucket, start, now, source)
            if r["resets_at_epoch"] is None
            or abs(r["resets_at_epoch"] - resets_epoch) <= RESET_JUMP_S]
    points = downsample([[_floor(r["fetched_epoch"]), r["utilization"]]
                         for r in rows], TRAJECTORY_MAX_POINTS)
    resets = ([] if bucket == "five_hour"
              else five_hour_resets(conn, start, min(now, end), source))
    return {"points": points, "resets": resets, "start": _floor(start),
            "end": end}


# ── past windows ─────────────────────────────────────────────────────────

def _group_by_anchor(rows):
    """Consecutive runs of readings that share one reset anchor.

    A forward anchor move of more than RESET_JUMP_S starts a new window;
    smaller moves are OAuth jitter. A backward move is a stale replay of an
    older snapshot and is dropped.
    """
    groups = []
    for r in rows:
        anchor = r["resets_at_epoch"]
        if anchor is None:
            continue
        if groups and anchor <= groups[-1]["anchor"] + RESET_JUMP_S:
            if anchor >= groups[-1]["anchor"] - RESET_JUMP_S:
                groups[-1]["rows"].append(r)
            continue
        groups.append({"anchor": anchor, "rows": [r]})
    return groups


def _points_used(rows, grants):
    """Points consumed across a window: the peak of each segment between
    granted resets (a grant restarts the meter, so the peaks add)."""
    total, peak, cuts = 0.0, 0.0, list(grants)
    for r in rows:
        while cuts and r["fetched_epoch"] >= cuts[0]:
            total, peak = total + peak, 0.0
            cuts.pop(0)
        peak = max(peak, r["utilization"] or 0.0)
    return total + peak


def past_windows(rows, window_s, now):
    """Ended windows (oldest first) as {start, end, rows, grants}.

    end is the anchor (a natural expiry), or the first reading of the next
    window when that came before the anchor (a grant replaced the window).
    """
    groups = _group_by_anchor(rows)
    out, prev_end = [], None
    for i, g in enumerate(groups):
        anchor = _floor(g["anchor"])
        nxt = groups[i + 1] if i + 1 < len(groups) else None
        end = anchor
        if nxt is not None and nxt["rows"][0]["fetched_epoch"] < anchor:
            end = _floor(nxt["rows"][0]["fetched_epoch"])
        if nxt is None and end > now:
            break  # the ongoing window: /api/runway shows it
        start = anchor - window_s
        if prev_end is not None:
            start = max(start, prev_end)
        grants = [_floor(e["at_epoch"]) for e in persistent_resets(g["rows"])
                  if start < e["at_epoch"] < end]
        out.append({"start": start, "end": end, "rows": g["rows"],
                    "grants": grants})
        prev_end = end
    return out


def _window_usd(conn, scope, bucket, start, end):
    """Claude-subscription spend over [start, end); a scoped bucket counts
    its model family only (the /api/rate-limits scoped rule)."""
    def build():
        if not bucket.startswith("scoped:"):
            return compute_window_cost(conn, start, end, scope=scope,
                                       anthropic_only=True)
        family = bucket.split(":", 1)[1].split("_")[0].lower()
        by_model = compute_window_cost_by_model(conn, start, end, scope=scope,
                                                anthropic_only=True)
        return sum(v for k, v in by_model.items() if family in k.lower())
    return _window_cost_cache.get((db.DB_PATH, scope, bucket, start, end), build)


def window_history(conn, scope, bucket, now=None):
    """{"windows": [...]} for the newest MAX_WINDOWS ended windows."""
    now = time.time() if now is None else now
    source = active_history_source(conn)
    rows = _rows(conn, bucket, 0, now, source)
    windows = past_windows(rows, window_seconds(bucket), now)[-MAX_WINDOWS:]
    out = []
    for w in windows:
        peak = max((r["utilization"] or 0.0) for r in w["rows"])
        usd = _window_usd(conn, scope, bucket, w["start"], w["end"])
        used = _points_used(w["rows"], w["grants"])
        out.append({
            "start": w["start"], "end": w["end"], "peak_pct": peak,
            "hit_limit": peak >= 100, "granted_resets": w["grants"],
            "usd": usd,
            "usd_per_point": (usd / used if usd > 0 and used >= MIN_PCT_FOR_RATE
                              else None),
        })
    return {"windows": out}
