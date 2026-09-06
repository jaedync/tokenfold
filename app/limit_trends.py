"""Sub-window burn-rate & projection math over historized limit_readings.

Utilization arrives integer-quantized and is written every poll (~600s server
cadence plus client pushes), so a "still N% at time T" row bounds each integer
step-crossing to one poll interval. compute_burn reconstructs a continuous
utilization curve û(t) by piecewise-linear interpolation through the recorded
(fetched_epoch, utilization) points and reads the average burn off it.

Step-crossing rationale: when utilization steps from u to u+k somewhere between
two polls, the exact crossing instant is unknown within that interval. A linear
segment between the two readings places each integer crossing at the unbiased
midpoint position implied by the ±poll-interval uncertainty; because rows are
written EVERY poll (not only on change), each crossing is bounded to a single
interval rather than smeared across a long quiet stretch. Averaging over the
whole window then cancels the per-crossing residuals.

Bucket-name-generic: nothing here hardcodes bucket names except the five_hour
window sizing the caller passes in — a future scoped bucket flows through with
zero code change.
"""

from .limit_readings import detect_resets, floor_reset_events

# A segment must span at least this long, with at least this many distinct
# readings, before an average burn is meaningful (a single poll pair over a few
# minutes is noise given ±poll-interval crossing uncertainty).
MIN_SPAN_S = 900.0
MIN_SAMPLES = 2

SERIES_MAX_POINTS = 200
FIVE_HOUR_KEY = "five_hour"


def _load_dedup_rows(conn, bucket, boundary, end):
    """Load one bucket's readings at/after ``boundary`` plus the single latest
    reading strictly before it (the straddler), through ``end`` only,
    ascending, de-duplicated by
    fetched_epoch keeping the last row for each second.

    Server and client can land the same second; a zero-width [t, t] segment
    would divide by zero in interpolation, so duplicates collapse to one point.
    """
    in_window = conn.execute(
        "SELECT bucket, fetched_epoch, utilization, resets_at_epoch "
        "FROM limit_readings WHERE bucket=? AND fetched_epoch>=? AND fetched_epoch<=? "
        "ORDER BY fetched_epoch ASC",
        (bucket, boundary, end)).fetchall()
    straddler = conn.execute(
        "SELECT bucket, fetched_epoch, utilization, resets_at_epoch "
        "FROM limit_readings WHERE bucket=? AND fetched_epoch<? "
        "ORDER BY fetched_epoch DESC LIMIT 1",
        (bucket, boundary)).fetchone()
    rows = ([straddler] if straddler is not None else []) + list(in_window)
    # Dedup by fetched_epoch keeping the last (rows already ascending).
    by_epoch = {}
    for r in rows:
        by_epoch[r["fetched_epoch"]] = r
    return [by_epoch[k] for k in sorted(by_epoch)]


def _interp(points, t):
    """Piecewise-linear û(t) through ascending (x, y) points, clamping to the
    endpoint value beyond either end (utilization holds between polls, and the
    value at 'now' is simply the last reading)."""
    if t <= points[0][0]:
        return points[0][1]
    if t >= points[-1][0]:
        return points[-1][1]
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        if t <= x1:
            # x1 != x0: rows are de-duplicated by fetched_epoch.
            frac = (t - x0) / (x1 - x0)
            return y0 + frac * (y1 - y0)
    return points[-1][1]


def compute_burn(conn, bucket, now, window_s):
    """Average utilization burn (percentage-points per hour) over the trailing
    ``window_s`` seconds for one bucket.

    Returns {"pct_per_hr": float|None, "samples": int, "resets_in_window": int}.

    Method:
    - Load the window plus one straddling reading each side (de-duplicated).
    - Detect resets over those rows; resets_in_window counts events inside the
      window. Keep ONLY the trailing post-reset segment (readings at/after the
      most recent reset), because utilization restarts at a reset and pre-reset
      points would corrupt the slope.
    - Build û(t) over the segment and take
        (û(now) − û(boundary)) / (effective_s / 3600),
      where boundary = now − window_s and
        effective_s = now − max(boundary, segment[0].fetched_epoch)
      is the OBSERVED span the segment actually covers. û(boundary)
      interpolates between the straddler and the first in-window reading when
      the boundary falls between them, else clamps to the segment's first
      reading (e.g. when a reset trimmed the segment start to after the
      boundary).
      When a straddler exists and the boundary interpolates
      (segment[0].fetched_epoch <= boundary), effective_s == window_s and
      this is the plain (û(now) − û(boundary)) / (window_s / 3600) formula.
      When a reset (or cold start) trims/clamps the segment so its first
      point sits INSIDE the window (segment[0].fetched_epoch > boundary),
      dividing by the full window_s would dilute the burn over the dead time
      before the segment started; effective_s uses only the span actually
      observed instead.
    - pct_per_hr is None when the segment has < MIN_SAMPLES distinct
      readings, spans < MIN_SPAN_S seconds, or effective_s <= 0 (defensive;
      only reachable if ``now`` precedes the segment). samples is the
      segment length.
    """
    boundary = now - window_s
    rows = _load_dedup_rows(conn, bucket, boundary, now)

    events = detect_resets(rows)
    # >=: a reset landing exactly ON the boundary is IN-window (Fix 8 —
    # a strict '>' silently dropped the edge case).
    resets_in_window = sum(1 for e in events if e["at_epoch"] >= boundary)

    if events:
        cutoff = events[-1]["at_epoch"]
        segment = [r for r in rows if r["fetched_epoch"] >= cutoff]
    else:
        segment = rows

    result = {"pct_per_hr": None, "samples": len(segment),
              "resets_in_window": resets_in_window}
    if len(segment) < MIN_SAMPLES:
        return result

    points = [(r["fetched_epoch"], r["utilization"]) for r in segment]
    if points[-1][0] - points[0][0] < MIN_SPAN_S:
        return result

    # OBSERVED span, not the requested window: a reset (or cold start) can
    # trim/clamp the segment start to strictly after boundary, and dividing
    # the resulting delta by the full window_s would dilute the burn over
    # dead time the segment never covered (Fix 4).
    effective_s = now - max(boundary, points[0][0])
    if effective_s <= 0:
        return result

    delta = _interp(points, now) - _interp(points, boundary)
    result["pct_per_hr"] = delta / (effective_s / 3600.0)
    return result


def downsample(points, max_points=SERIES_MAX_POINTS):
    """Uniformly downsample a list to <= max_points, always keeping the first
    and last element. Returns a new list; input is never mutated."""
    n = len(points)
    if n <= max_points:
        return list(points)
    step = (n - 1) / (max_points - 1)
    idxs = [round(i * step) for i in range(max_points)]
    idxs[0] = 0
    idxs[-1] = n - 1
    seen = set()
    out = []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            out.append(points[i])
    return out


def _latest_utilization(conn, bucket, now):
    """Utilization of the bucket's most recent reading; None if it has none."""
    row = conn.execute(
        "SELECT utilization FROM limit_readings WHERE bucket=? AND fetched_epoch<=? "
        "ORDER BY fetched_epoch DESC LIMIT 1", (bucket, now)).fetchone()
    return row["utilization"] if row is not None else None


def distinct_buckets(conn, now, within_s=7 * 86400):
    """Distinct bucket names seen in limit_readings within the trailing window,
    sorted for stable output ordering."""
    return [r["bucket"] for r in conn.execute(
        "SELECT DISTINCT bucket FROM limit_readings WHERE fetched_epoch>=? "
        "ORDER BY bucket", (now - within_s,)).fetchall()]


def bucket_trend(conn, bucket, now):
    """Per-bucket trend dict for /api/rate-limits oauth.trend[bucket].

    Fields: burn_1h_pct_per_hr, burn_6h_pct_per_hr (2-dp, null when
    unavailable), eta_100_epoch (minute-floored, null when the relevant burn is
    non-positive/None or current pct unknown), pace ('under'|'on'|'over'|null),
    series ([[minute_epoch, pct], ...] downsampled to <= 200 pts), resets
    (detect_resets events over the served series window; already minute-floored).

    The relevant burn is the 1h burn for the five_hour bucket and the 6h burn
    for every other bucket; pace compares it to the even-drain rate
    100/window_hours (5h for five_hour, 168h otherwise) within a +/-10% deadband.
    """
    # The caller supplies the provider observation, not transport receipt time.
    # Later history must not leak into this observation's pace or chart.
    is_five = bucket == FIVE_HOUR_KEY

    b1 = compute_burn(conn, bucket, now, 3600)
    b6 = compute_burn(conn, bucket, now, 21600)
    burn_1h = (round(b1["pct_per_hr"], 2)
               if b1["pct_per_hr"] is not None else None)
    burn_6h = (round(b6["pct_per_hr"], 2)
               if b6["pct_per_hr"] is not None else None)

    relevant = burn_1h if is_five else burn_6h
    window_hours = 5.0 if is_five else 168.0
    even_drain = 100.0 / window_hours

    current_pct = _latest_utilization(conn, bucket, now)
    eta = None
    if relevant is not None and relevant > 0 and current_pct is not None:
        eta = ((now + (100.0 - current_pct) / relevant * 3600.0) // 60) * 60.0

    pace = None
    if relevant is not None:
        if relevant < even_drain * 0.9:
            pace = "under"
        elif relevant > even_drain * 1.1:
            pace = "over"
        else:
            pace = "on"

    series_hours = 24 if is_five else 168
    window_rows = conn.execute(
        "SELECT bucket, fetched_epoch, utilization, resets_at_epoch "
        "FROM limit_readings WHERE bucket=? AND fetched_epoch>=? AND fetched_epoch<=? "
        "ORDER BY fetched_epoch ASC",
        (bucket, now - series_hours * 3600, now)).fetchall()
    series = downsample(
        [[(r["fetched_epoch"] // 60) * 60.0, r["utilization"]]
         for r in window_rows])

    return {
        "burn_1h_pct_per_hr": burn_1h,
        "burn_6h_pct_per_hr": burn_6h,
        "eta_100_epoch": eta,
        "pace": pace,
        "series": series,
        # Minute-floored like every other epoch field on this surface (series
        # above, eta_100_epoch): detect_resets itself stays full-precision
        # (its contract for other callers, e.g. resets_in_window inside
        # compute_burn, is untouched) — only this served copy is floored.
        "resets": floor_reset_events(detect_resets(window_rows)),
    }
