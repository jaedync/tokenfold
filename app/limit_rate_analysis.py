"""Exchange rate between list-price dollars and usage-limit points. No I/O.

Port of the limit-xray method (~/.cache/limit-xray, REPORT.md):
1. intervals between quota readings of one window (7 d: >= 3 h, 5 h: >= 30 min);
2. dollars of deduped requests in each interval, split by family and type;
3. change points: granted resets, or a +-35% move of the rolling 24 h
   dollars-per-point against the period median; cuts whose two sides differ
   by less than MERGE_TOL on both meters are merged away again;
4. in the current period: dollars per point per meter (ratio estimator),
   token-type weights of the main family and family weights, both with a
   base rate per 7-day window (app/limit_family.py), and day-block
   bootstrap 90% bands;
5. a 5-hour peak factor from family-fit residuals (app/limit_peak.py); the
   5-hour family weights scale peak-hour dollars by it.

Inputs: readings {bucket: [(t, pct, resets_at_epoch)]}, requests
[(t, family, [usd_in, usd_out, usd_cache_write, usd_cache_read], tokens)]
sorted by t (tokens, the same four types, are optional: without them no
per-token weight), granted reset epochs. Output: the /api/limit-rate payload minus the
serve-time peak state.
"""

import logging
from datetime import datetime, timezone
from typing import NamedTuple, Optional, Sequence

from . import limit_family as lfam
from . import limit_fit as lf
from .limit_peak import compute_peak, empty_peak, peak_hours, peak_state  # noqa: F401 (re-export)

logger = logging.getLogger(__name__)

HOUR = 3600.0
SEVEN_DAY_MIN_S = 3 * HOUR
FIVE_HOUR_MIN_S = 0.5 * HOUR
MAX_INTERVAL_S = 72 * HOUR
ROLLING_S = 24 * HOUR
ROLLING_MIN_PTS = 5.0       # a quiet day gives no rolling value (rounding noise)
RATE_MOVE = 0.35            # REPORT.md change-point rule
CONFIRM = 3                 # consecutive moved values before a cut (~9 h)
MIN_PERIOD_VALUES = 4       # rolling values before the period median counts
REFINE_BACK_S = 48 * HOUR
MERGE_TOL = 0.20            # smaller rate differences are not a step
GRANT_SHADOW_S = 12 * HOUR  # a rate cut this close to a grant is the grant
SEGMENT_MIN_PTS = 10.0      # points a segment needs for its rate to count
FEATURE_MIN_USD = 5.0       # smaller features only add noise to the fit
TYPE_MIN_SHARE = 0.02       # of the main family's dollars, for a token type column
PEAK_LOOKBACK_S = 28 * 86400
BOOT_DRAWS = 200
BOOT_SEED = 20261015
TYPE_OUT, TYPE_CR = 1, 3


class Series(NamedTuple):
    intervals: list   # [lf.Interval]
    feats: list       # per interval: usd by family x type (4 per family)
    usd: list         # per interval: total usd


class Cut(NamedTuple):
    at: float
    cause: str


def _floor_min(t: Optional[float]) -> Optional[float]:
    return None if t is None else (t // 60) * 60.0


def _day(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")


def _r(v: Optional[float], nd: int = 4) -> Optional[float]:
    return None if v is None else round(v, nd)


def _series(readings, min_s, rows, width) -> Series:
    ivs = lf.build_intervals(readings, min_s, MAX_INTERVAL_S)
    feats = lf.assign_to_intervals(ivs, rows, width)
    return Series(ivs, feats, [sum(f) for f in feats])


def _idx(series: Series, lo: float, hi: float) -> list:
    return [i for i, iv in enumerate(series.intervals) if iv.ta >= lo and iv.tb <= hi]


def _rate(series: Series, idx: Sequence) -> Optional[float]:
    pts = sum(series.intervals[i].points for i in idx)
    if pts < SEGMENT_MIN_PTS:
        return None
    return sum(series.usd[i] for i in idx) / pts


# ── change points ──────────────────────────────────────────────────────────

def _first_move(ivs, usd, s) -> Optional[int]:
    """Index where a new rate period starts after ``s``, or None."""
    vals, run, first_dev = [], 0, None
    lo, su, sp = s, 0.0, 0.0
    for i in range(s, len(ivs)):
        su += usd[i]
        sp += ivs[i].points
        while ivs[lo].tb <= ivs[i].tb - ROLLING_S:
            su -= usd[lo]
            sp -= ivs[lo].points
            lo += 1
        if sp < ROLLING_MIN_PTS:
            continue
        r = su / sp
        med = lf.median(vals) if len(vals) >= MIN_PERIOD_VALUES else None
        if med and abs(r / med - 1) > RATE_MOVE:
            first_dev = i if run == 0 else first_dev
            run += 1
            if run >= CONFIRM:
                return _refine(ivs, usd, s, first_dev, i)
            continue  # moved values do not pull the period median
        run = 0
        vals.append(r)
    return None


def _refine(ivs, usd, s, first_dev, end) -> int:
    """The rolling window lags the step; put the cut at the split with the
    least squared error of two points-per-dollar slopes."""
    wlo = s
    while ivs[wlo].tb < ivs[first_dev].tb - REFINE_BACK_S:
        wlo += 1

    def sse(a, b):
        sxy = sum(usd[k] * ivs[k].points for k in range(a, b))
        sxx = sum(usd[k] ** 2 for k in range(a, b))
        syy = sum(ivs[k].points ** 2 for k in range(a, b))
        return syy - (sxy * sxy / sxx if sxx > 0 else 0.0)

    best, best_k = None, first_dev
    for k in range(max(wlo, s) + 1, end + 1):
        e = sse(wlo, k) + sse(k, end + 1)
        if best is None or e < best:
            best, best_k = e, k
    return best_k


def _rate_cuts(series: Series, lo: float, hi: float) -> list:
    idx = _idx(series, lo, hi)
    ivs = [series.intervals[i] for i in idx]
    usd = [series.usd[i] for i in idx]
    cuts, s = [], 0
    while s < len(ivs):
        k = _first_move(ivs, usd, s)
        if k is None or k <= s:
            break
        cuts.append(Cut(ivs[k].ta, "rate_change"))
        s = k
    return cuts


def change_points(s7: Series, granted: Sequence) -> list:
    if not s7.intervals:
        return []
    first, last = s7.intervals[0].ta, s7.intervals[-1].tb
    grants = [Cut(g, "granted_reset") for g in sorted(granted) if first < g < last]
    edges = [float("-inf")] + [g.at for g in grants] + [float("inf")]
    rate = [c for a, b in zip(edges, edges[1:]) for c in _rate_cuts(s7, a, b)]
    rate = [c for c in rate if all(abs(c.at - g.at) > GRANT_SHADOW_S for g in grants)]
    return sorted(grants + rate)


def _rate_rel_se(series: Series, idx) -> Optional[float]:
    """Relative standard error of the ratio estimator: residual spread of
    points around usd / rate, over the total points. It ignores the
    within-day correlation, so it is a floor, not a band."""
    rate = _rate(series, idx)
    if rate is None:
        return None
    pts = sum(series.intervals[i].points for i in idx)
    sse = sum((series.intervals[i].points - series.usd[i] / rate) ** 2 for i in idx)
    return sse ** 0.5 / pts


def _cut_score(series: Series, left, right) -> Optional[float]:
    """Relative move across a cut, or 0.0 when it is inside the noise
    (2 standard errors), or None when a side has too few points."""
    l, r = _rate(series, left), _rate(series, right)
    if l is None or r is None:
        return None
    move = abs(r / l - 1)
    noise = 2 * (_rate_rel_se(series, left) ** 2 + _rate_rel_se(series, right) ** 2) ** 0.5
    return move if move > noise else 0.0


def merge_cuts(cuts: Sequence, s7: Series, s5: Series) -> list:
    """Drop cuts whose sides do not differ on either meter (less than
    MERGE_TOL, or inside 2 standard errors), and cuts next to a segment too
    thin to have a 7-day rate. Smallest difference first, so one noisy cut
    cannot hide a real one."""
    cuts = list(cuts)
    while cuts:
        edges = [float("-inf")] + [c.at for c in cuts] + [float("inf")]
        segs = [(_idx(s7, a, b), _idx(s5, a, b)) for a, b in zip(edges, edges[1:])]
        scores = []
        for j in range(len(cuts)):
            m7 = _cut_score(s7, segs[j][0], segs[j + 1][0])
            m5 = _cut_score(s5, segs[j][1], segs[j + 1][1])
            scores.append(-1.0 if m7 is None else max(m7, m5 or 0.0))
        j = min(range(len(cuts)), key=lambda x: scores[x])
        if scores[j] >= MERGE_TOL:
            break
        cuts.pop(j)
    return cuts


def steps_payload(cuts, s7, s5) -> list:
    edges = [float("-inf")] + [c.at for c in cuts] + [float("inf")]
    out = []
    for j, c in enumerate(cuts):
        for name, s in (("seven_day", s7), ("five_hour", s5)):
            out.append({
                "at": _floor_min(c.at), "bucket": name,
                "from_usd_per_point": _r(_rate(s, _idx(s, edges[j], edges[j + 1]))),
                "to_usd_per_point": _r(_rate(s, _idx(s, edges[j + 1], edges[j + 2]))),
                "cause": c.cause})
    return out


# ── current-period fits ────────────────────────────────────────────────────

def _blocks(series: Series, idx, make_row) -> list:
    by_day: dict = {}
    for i in idx:
        by_day.setdefault(_day(series.intervals[i].tb), []).append(i)
    return [lf.gram_vector([make_row(i) for i in ii],
                           [series.intervals[i].points for i in ii])
            for _, ii in sorted(by_day.items())]


def usd_per_point(series: Series, idx) -> dict:
    blocks = _blocks(series, idx, lambda i: [series.usd[i]])
    total = lf.add_vectors(blocks)
    if not total:
        return {"value": None, "lo": None, "hi": None}
    st = lf.unpack_gram(total, 1)
    if st.sum_y < SEGMENT_MIN_PTS:
        return {"value": None, "lo": None, "hi": None}

    def stat(v):
        g = lf.unpack_gram(v, 1)
        return g.sum_x[0] / g.sum_y if g.sum_y > 0 else None

    boot = lf.block_bootstrap(blocks, stat, BOOT_DRAWS, BOOT_SEED)
    return {"value": _r(st.sum_x[0] / st.sum_y), "lo": _r(lf.percentile(boot, 5)),
            "hi": _r(lf.percentile(boot, 95))}


def nnls_fit(series: Series, idx, cols: Sequence) -> Optional[dict]:
    """NNLS of points on the chosen feature columns, with bootstrap bands.

    cols: list of lists of feature indexes; each fitted feature is the sum
    of its columns. Returns coef, lo, hi (points per $) and r2."""
    def row(i):
        f = series.feats[i]
        return [sum(f[c] for c in group) for group in cols]

    blocks = _blocks(series, idx, row)
    if not blocks:
        return None
    k = len(cols)
    st = lf.unpack_gram(lf.add_vectors(blocks), k)
    coef = lf.nnls(st.ata, st.aty)
    boot = lf.block_bootstrap(
        blocks, lambda v: lf.nnls(*lf.unpack_gram(v, k)[:2]), BOOT_DRAWS, BOOT_SEED)
    lo = [lf.percentile([b[c] for b in boot], 5) for c in range(k)]
    hi = [lf.percentile([b[c] for b in boot], 95) for c in range(k)]
    return {"coef": coef, "lo": lo, "hi": hi, "r2": lf.r_squared(st, coef),
            "usd": st.sum_x, "boot": boot}


def _type_fit(series: Series, idx, n_fams, main, blocks, nb) -> tuple:
    """Windowed fit (app/limit_family.py) of the main family's four token
    types, pivot output; other families enter as one dollar column so their
    use does not leak in. Returns (fit or None, kept columns; 0-3 are token
    types, 4 is the other families)."""
    groups = [[main * 4 + t] for t in range(4)]
    other = [f * 4 + t for f in range(n_fams) if f != main for t in range(4)]
    groups = groups + ([other] if other else [])
    totals = [sum(sum(series.feats[i][c] for c in g) for i in idx) for g in groups]
    main_usd = sum(totals[:4])
    # A token type with a tiny share (input, about 0.01% in production) only
    # adds noise; it keeps its price weight.
    keep = [j for j, v in enumerate(totals)
            if v >= FEATURE_MIN_USD and (j == 4 or v >= TYPE_MIN_SHARE * main_usd)]
    if TYPE_OUT not in keep or TYPE_CR not in keep:
        return None, keep

    def cols(i):
        f = series.feats[i]
        return [sum(f[c] for c in groups[j]) for j in keep]

    res = lfam.fit(series, idx, cols, len(keep), keep.index(TYPE_OUT), blocks, nb,
                   BOOT_DRAWS, BOOT_SEED)
    return res, keep


def type_coefs(tf) -> Optional[dict]:
    """{token type: limit weight per $ relative to output}, from _type_fit."""
    res, keep = tf
    if res is None:
        return None
    return {t: res["m"][j] for j, t in enumerate(keep) if t < 4}


def cache_read_payload(tf, fams, main) -> dict:
    """Cache-read limit weight per $ relative to output, main family only.

    1.0 = cache reads use the limit in line with their API price."""
    none = {"value": None, "lo": None, "hi": None, "family": None}
    if main is None:
        return none
    res, keep = tf
    if res is None:
        return dict(none, family=fams[main])
    jc = keep.index(TYPE_CR)
    boot = [mb[jc] for _, mb in res["boot"]]
    return {"value": _r(res["m"][jc]), "lo": _r(lf.percentile(boot, 5)),
            "hi": _r(lf.percentile(boot, 95)), "family": fams[main]}


# ── orchestration ──────────────────────────────────────────────────────────

def _guard(part: str, fn, default, notes: list):
    """One failing part degrades to its default, never the whole payload."""
    try:
        return fn()
    except Exception:
        logger.exception("limit-rate %s failed", part)
        notes.append(f"{part} unavailable (internal error)")
        return default


def _peak(s5: Series, cuts, n_fams: int, now: float) -> dict:
    """Pool 5 h residuals over the last 4 weeks, one family fit per period
    (the limit size differs between periods, the time pattern does not)."""
    edges = [float("-inf")] + [c.at for c in cuts] + [float("inf")]
    since = now - PEAK_LOOKBACK_S
    samples = []
    for a, b in zip(edges, edges[1:]):
        idx = [i for i in _idx(s5, a, b) if s5.intervals[i].ta >= since]
        cols = _family_cols(s5, idx, n_fams)
        fit = nnls_fit(s5, idx, cols) if cols else None
        if fit is None:
            continue
        for i in idx:
            f = s5.feats[i]
            exp = sum(c * sum(f[x] for x in g) for c, g in zip(fit["coef"], cols))
            iv = s5.intervals[i]
            samples.append(((iv.ta + iv.tb) / 2, iv.points, exp))
    return compute_peak(samples) if samples else empty_peak()


def _family_cols(series: Series, idx, n_fams: int, out_used=None) -> list:
    cols = []
    for fi in range(n_fams):
        group = [fi * 4 + t for t in range(4)]
        if sum(series.feats[i][c] for i in idx for c in group) >= FEATURE_MIN_USD:
            if out_used is not None:
                out_used[fi] = len(cols)
            cols.append(group)
    return cols


def _peak_rows(rows: Sequence, peak: dict) -> list:
    """Request vectors with peak-hour dollars scaled by the peak factor, so
    the 5-hour family fit is not misled by when each model ran."""
    factor = (peak or {}).get("factor")
    if not factor:
        return list(rows)
    is_peak = peak_hours(peak)
    return [(t, [x * factor for x in vec]) if is_peak(t) else (t, vec) for t, vec in rows]


def _family_fits(readings, rows, width, s7, idx7, start, n_fams, main, w, peak) -> dict:
    """Per meter: the windowed family fit (app/limit_family.py) on effective
    dollars, or None."""
    ends = lfam.window_ends(readings.get("seven_day") or [])
    s5 = _series(readings.get("five_hour") or [], FIVE_HOUR_MIN_S, _peak_rows(rows, peak), width)
    idx5 = _idx(s5, start, float("inf")) if start is not None else []
    out = {}
    for name, s, idx in (("seven_day", s7, idx7), ("five_hour", s5, idx5)):
        fitted = [fi for fi in range(n_fams)
                  if sum(s.feats[i][fi * 4 + t] for i in idx for t in range(4)) >= FEATURE_MIN_USD]
        if main is None or main not in fitted:
            out[name] = None
            continue
        blocks, nb = lfam.block_ids(s.intervals, idx, ends)

        def cols(i, s=s, fitted=fitted):
            return [lfam.effective(s.feats[i], fi, w) for fi in fitted]

        res = lfam.fit(s, idx, cols, len(fitted), fitted.index(main), blocks, nb,
                       BOOT_DRAWS, BOOT_SEED)
        out[name] = None if res is None else {
            "res": res, "fitted": fitted,
            "e": {fi: lfam.eff_ratio(s, idx, fi, w) for fi in fitted}}
    return out


def _period_sums(requests, start, now) -> tuple:
    """Spend, dollars by type and tokens by type per family in the period.
    Requests may carry tokens as a 4th element; without them no family has
    a per-token weight."""
    spend: dict = {}
    usd_t: dict = {}
    tok_t: dict = {}
    for req in requests:
        t, fam, parts = req[0], req[1], req[2]
        if start is None or not start <= t <= now:
            continue
        spend[fam] = spend.get(fam, 0.0) + sum(parts)
        usd_t[fam] = [a + b for a, b in zip(usd_t.get(fam, [0.0] * 4), parts)]
        if len(req) > 3 and req[3] is not None:
            tok_t[fam] = [a + b for a, b in zip(tok_t.get(fam, [0.0] * 4), req[3])]
    return spend, usd_t, tok_t


def analyze(readings: dict, requests: Sequence, granted: Sequence, now: float) -> dict:
    fams = sorted({r[1] for r in requests})
    fidx = {f: i for i, f in enumerate(fams)}
    width = 4 * len(fams)
    rows = []
    for req in requests:
        t, fam, parts = req[0], req[1], req[2]
        vec = [0.0] * width
        vec[fidx[fam] * 4:fidx[fam] * 4 + 4] = parts
        rows.append((t, vec))
    r7 = readings.get("seven_day") or []
    s7 = _series(r7, SEVEN_DAY_MIN_S, rows, width)
    s5 = _series(readings.get("five_hour") or [], FIVE_HOUR_MIN_S, rows, width)
    notes: list = []
    cuts = _guard("steps", lambda: merge_cuts(change_points(s7, granted), s7, s5), [], notes)
    steps = _guard("steps", lambda: steps_payload(cuts, s7, s5), [], notes)
    start = cuts[-1].at if cuts else (r7[0][0] if r7 else None)
    idx7 = _idx(s7, start, float("inf")) if start is not None else []
    idx5 = _idx(s5, start, float("inf")) if start is not None else []
    spend, usd_t, tok_t = _period_sums(requests, start, now)
    by_spend = sorted(fams, key=lambda f: -spend.get(f, 0.0))
    nullband = {"value": None, "lo": None, "hi": None}
    upp = {"seven_day": _guard("usd_per_point", lambda: usd_per_point(s7, idx7), nullband, notes),
           "five_hour": _guard("usd_per_point", lambda: usd_per_point(s5, idx5), nullband, notes)}
    main = fidx[by_spend[0]] if by_spend and spend.get(by_spend[0]) else None
    blocks7, nb7 = lfam.block_ids(s7.intervals, idx7, lfam.window_ends(r7))
    tf = _guard("cache_read_weight",
                lambda: _type_fit(s7, idx7, len(fams), main, blocks7, nb7)
                if main is not None else (None, []), (None, []), notes)
    crw = _guard("cache_read_weight", lambda: cache_read_payload(tf, fams, main),
                 {"value": None, "lo": None, "hi": None, "family": None}, notes)
    w = lfam.type_weights(type_coefs(tf))
    peak = _guard("peak", lambda: _peak(s5, cuts, len(fams), now), empty_peak(), notes)
    fits = _guard("families", lambda: _family_fits(readings, rows, width, s7, idx7, start,
                                                   len(fams), main, w, peak), {}, notes)
    families = _guard("families", lambda: lfam.families_payload(
        fams, spend, usd_t, tok_t, main, w, fits), [], notes)
    status = "ok"
    if upp["seven_day"]["value"] is None:
        status = "insufficient_data"
        notes.insert(0, "too few 7-day points since the period start to fit")
    seven = fits.get("seven_day")
    r2 = seven["res"]["r2"] if seven else None
    return {
        "generated_at": now,
        "period_start": _floor_min(start),
        "fit_quality": {"r2": _r(r2), "intervals": len(idx7), "status": status,
                        "note": "; ".join(notes) or None},
        "usd_per_point": upp,
        "steps": steps,
        "family_main": fams[main] if main is not None else None,
        "families": families,
        "peak": peak,
        "cache_read_weight": crw,
    }
