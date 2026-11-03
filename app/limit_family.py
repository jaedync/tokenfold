"""Model-family weights on the limit meters, with a base rate per 7-day
window. No I/O; part of /api/limit-rate (app/limit_rate_analysis.py).

Why a base rate per window: the main family's dollars per point drift from
week to week (production, Oct 2026: $25.1 to $22.8 per 7-day point of Opus
5.5). One rate for the whole period gives that drift to any family whose use
grew in the same weeks: Sonnet 5.5 read 1.42x the Opus limit per dollar with
one rate and 1.22x with a rate per window. The model is

    points_i = k_b(i) * sum_f m_f * x_if

k_b: points per effective main-family dollar in 7-day window b.
m_f: limit use of family f per effective dollar, relative to the main family
     (m_main = 1).
x_if: effective dollars of family f in interval i. Each token type is
     weighed with the main family's type weights (cache reads count at about
     0.4 of their price), so a family with fewer cache reads is not called
     heavier for that alone.

The fit alternates two linear steps (k given m, m given k) on per-day sums,
so the day-block bootstrap only adds sums, as in app/limit_fit.py. Intervals
never span two windows, so the sums are block-diagonal per window.
"""

from bisect import bisect_left
from datetime import datetime, timezone
from typing import Optional, Sequence

from . import limit_fit as lf

ALS_START_ITERS = 8
ALS_ITERS = 100
ALS_TOL = 1e-10
BOOT_ITERS = 40
# Dollars (or units) a column and the pivot need in one window to share it.
IDENT_MIN = 5.0
# A weight this far from price-proportional is a collinearity artifact of
# the type fit (cache reads and writes move together), not a measurement.
TYPE_WEIGHT_MIN, TYPE_WEIGHT_MAX = 0.05, 3.0
N_TYPES = 4  # input, output, cache write, cache read
TYPE_OUT = 1


def window_ends(readings7: Sequence) -> list:
    """Sorted distinct resets_at of the 7-day meter: one entry per window."""
    ends: list = []
    for _, _, reset in readings7:
        if reset is None:
            continue
        if ends and abs(reset - ends[-1]) <= lf.SAME_WINDOW_S:
            continue
        ends.append(float(reset))
    return sorted(ends)


def block_ids(intervals: Sequence, idx: Sequence, ends: Sequence) -> tuple:
    """Map each interval in idx to a compact window index; return
    ({interval index: block}, block count)."""
    raw = {i: bisect_left(ends, (intervals[i].ta + intervals[i].tb) / 2) for i in idx}
    order = {b: n for n, b in enumerate(sorted(set(raw.values())))}
    return {i: order[b] for i, b in raw.items()}, len(order)


def type_weights(coef_by_type: Optional[dict]) -> list:
    """Per-type limit weight per dollar relative to output, from the main
    family's type fit ({type index: weight relative to output}). Types the
    fit left out (too little spend) keep the price weight 1.0; a fit
    without output, or with a weight outside the plausible range,
    falls back to price weights for every type."""
    flat = [1.0] * N_TYPES
    if not coef_by_type or coef_by_type.get(TYPE_OUT) != 1.0:
        return flat
    w = list(flat)
    for t, c in coef_by_type.items():
        w[t] = c
    if any(not TYPE_WEIGHT_MIN <= v <= TYPE_WEIGHT_MAX for v in w):
        return flat
    return w


def effective(feats_row: Sequence, fi: int, w: Sequence) -> float:
    return sum(w[t] * feats_row[fi * N_TYPES + t] for t in range(N_TYPES))


def _day(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")


def day_vectors(series, idx, cols, F: int, blocks: dict, nb: int) -> list:
    """Per-day flat sums of the F columns ``cols(i)``. Layout per window b
    (stride F*F + 2F): the F x F gram, X'y (F), sum x (F); then yty,
    sum y, n."""
    per = F * F + 2 * F
    size = nb * per + 3
    by_day: dict = {}
    for i in idx:
        iv = series.intervals[i]
        x = cols(i)
        v = by_day.setdefault(_day(iv.tb), [0.0] * size)
        base = blocks[i] * per
        y = iv.points
        for a in range(F):
            if not x[a]:
                continue
            v[base + F * F + a] += x[a] * y
            v[base + F * F + F + a] += x[a]
            row = base + a * F
            for c in range(F):
                v[row + c] += x[a] * x[c]
        v[-3] += y * y
        v[-2] += y
        v[-1] += 1.0
    return [v for _, v in sorted(by_day.items())]


class _View:
    """Index helpers over one summed vector."""

    def __init__(self, vec, nb, F):
        self.v, self.nb, self.F, self.per = vec, nb, F, F * F + 2 * F

    def g(self, b, a, c):
        return self.v[b * self.per + a * self.F + c]

    def aty(self, b, a):
        return self.v[b * self.per + self.F * self.F + a]

    def sx(self, b, a):
        return self.v[b * self.per + self.F * self.F + self.F + a]


def _k_step(s: _View, m) -> list:
    k = []
    for b in range(s.nb):
        num = sum(m[a] * s.aty(b, a) for a in range(s.F))
        den = sum(m[a] * m[c] * s.g(b, a, c) for a in range(s.F) for c in range(s.F))
        k.append(max(0.0, num / den) if den > 0 else 0.0)
    return k


def _m_step(s: _View, k, m, main, others) -> list:
    ata = [[sum(k[b] ** 2 * s.g(b, a, c) for b in range(s.nb)) for c in others] for a in others]
    aty = [sum(k[b] * s.aty(b, a) - k[b] ** 2 * s.g(b, a, main) for b in range(s.nb))
           for a in others]
    m = list(m)
    for n, a in zip(lf.nnls(ata, aty), others):
        m[a] = n
    return m


def _sse(s: _View, k, m) -> float:
    out = s.v[-3]
    for b in range(s.nb):
        out -= 2 * k[b] * sum(m[a] * s.aty(b, a) for a in range(s.F))
        out += k[b] ** 2 * sum(m[a] * m[c] * s.g(b, a, c)
                               for a in range(s.F) for c in range(s.F))
    return out


def _m_gradient(s: _View, k, m, a: int) -> float:
    """-1/2 d(SSE)/d(m_a)."""
    return sum(k[b] * (s.aty(b, a) - k[b] * sum(s.g(b, a, c) * m[c] for c in range(s.F)))
               for b in range(s.nb))


def _lm_step(s: _View, k, m, free_k, free_m, lam) -> tuple:
    """One damped Gauss-Newton step on the free parameters (normal
    equations of the Jacobian, all from the window sums)."""
    F, nb = s.F, s.nb
    gm = [[sum(s.g(b, a, c) * m[c] for c in range(F)) for a in range(F)] for b in range(nb)]
    mgm = [sum(m[a] * gm[b][a] for a in range(F)) for b in range(nb)]
    ma = [sum(m[a] * s.aty(b, a) for a in range(F)) for b in range(nb)]
    params = [("k", b) for b in free_k] + [("m", a) for a in free_m]
    n = len(params)
    jtj = [[0.0] * n for _ in range(n)]
    jtr = [0.0] * n
    for i, (ti, xi) in enumerate(params):
        if ti == "k":
            jtr[i] = ma[xi] - k[xi] * mgm[xi]
        else:
            jtr[i] = sum(k[b] * (s.aty(b, xi) - k[b] * gm[b][xi]) for b in range(nb))
        for j, (tj, xj) in enumerate(params):
            if ti == "k" and tj == "k":
                jtj[i][j] = mgm[xi] if xi == xj else 0.0
            elif ti == "k":
                jtj[i][j] = k[xi] * gm[xi][xj]
            elif tj == "k":
                jtj[i][j] = k[xj] * gm[xj][xi]
            else:
                jtj[i][j] = sum(k[b] ** 2 * s.g(b, xi, xj) for b in range(nb))
    for i in range(n):
        jtj[i][i] *= 1.0 + lam
    delta = lf._solve(jtj, jtr)
    k, m = list(k), list(m)
    for (t, x), d in zip(params, delta):
        if t == "k":
            k[x] = max(0.0, k[x] + d)
        else:
            m[x] = max(0.0, m[x] + d)
    return k, m


def solve(vec, nb: int, F: int, main: int, start=None, iters: int = ALS_ITERS) -> tuple:
    """(k per window, m per column) minimising the squared error, m[main] = 1.

    Alternating steps find the region (and keep m >= 0 through NNLS), then
    Levenberg-Marquardt finishes: alternating steps alone crawl when k and m
    are correlated (one window holds most of a family's use)."""
    s = _View(vec, nb, F)
    m = list(start) if start else [1.0] * F
    m[main] = 1.0
    others = [a for a in range(F) if a != main]
    k = _k_step(s, m)
    for _ in range(ALS_START_ITERS):
        if not others:
            break
        m = _m_step(s, k, m, main, others)
        k = _k_step(s, m)
    free_k = [b for b in range(nb) if any(s.g(b, a, a) > 0 for a in range(F))]
    sse, lam = _sse(s, k, m), 1e-3
    for _ in range(iters):
        # A weight clipped to 0 comes back when the error falls as it grows.
        free_m = [a for a in others if m[a] > 0 or _m_gradient(s, k, m, a) > 0]
        nk, nm = _lm_step(s, k, m, free_k, free_m, lam)
        new = _sse(s, nk, nm)
        if new <= sse:
            done = sse - new <= ALS_TOL * max(sse, 1e-12)
            k, m, sse, lam = nk, nm, new, max(lam / 3, 1e-9)
            if done:
                break
        else:
            lam *= 4
            if lam > 1e8:
                break
    return k, m


def _r2(vec, nb, F, k, m) -> Optional[float]:
    yty, sum_y, n = vec[-3], vec[-2], vec[-1]
    if n < 2:
        return None
    sst = yty - sum_y ** 2 / n
    return 1.0 - _sse(_View(vec, nb, F), k, m) / sst if sst > 0 else None


def _kbar(vec, nb, F, k, main) -> Optional[float]:
    """Period base rate: k per window weighted by the pivot column's sum
    in it (the main family's effective dollars)."""
    s = _View(vec, nb, F)
    e = [s.sx(b, main) for b in range(nb)]
    tot = sum(e)
    return sum(kb * eb for kb, eb in zip(k, e)) / tot if tot > 0 else None


def fit(series, idx, cols, F: int, pivot: int, blocks: dict, nb: int,
        draws: int, seed: int) -> Optional[dict]:
    """Weights m of the F columns ``cols(i)`` relative to column ``pivot``,
    the period base rate kbar (points per pivot unit), bootstrap draws
    [(kbar, m)] and R^2. None without data."""
    if not idx or F == 0:
        return None
    days = day_vectors(series, idx, cols, F, blocks, nb)
    total = lf.add_vectors(days)
    if not total or total[-2] <= 0:
        return None
    k, m = solve(total, nb, F, pivot)
    kbar = _kbar(total, nb, F, k, pivot)
    if not kbar:
        return None

    def stat(v):
        kb, mb = solve(v, nb, F, pivot, start=m, iters=BOOT_ITERS)
        kk = _kbar(v, nb, F, kb, pivot)
        return (kk, mb) if kk else None

    boot = lf.block_bootstrap(days, stat, draws, seed)
    return {"k": k, "m": m, "kbar": kbar, "boot": boot, "identified": identified(total, nb, F, pivot),
            "r2": _r2(total, nb, F, k, m), "intervals": len(idx)}


def identified(vec, nb: int, F: int, pivot: int) -> list:
    """Per column: does it have spend in windows where the pivot has spend?
    In a window without the pivot only k_b * m_f is known, so a column seen
    only there gets its start value back, with a band of zero width."""
    s = _View(vec, nb, F)
    shared = [b for b in range(nb) if s.sx(b, pivot) >= IDENT_MIN]
    return [a == pivot or sum(s.sx(b, a) for b in shared) >= IDENT_MIN for a in range(F)]


def band(values: Sequence) -> tuple:
    return lf.percentile(values, 5), lf.percentile(values, 95)


def token_ratio(usd_t: Sequence, tok_t: Sequence, main_usd_t: Sequence,
                main_tok_t: Sequence, w: Sequence) -> Optional[float]:
    """Effective price of a family's own tokens over the effective price of
    the same tokens at the main family's rates (Sonnet 5.5 vs Opus 5.5: 0.5).
    Only types both families used count."""
    num = den = 0.0
    for t in range(N_TYPES):
        if tok_t[t] > 0 and main_tok_t[t] > 0:
            num += w[t] * usd_t[t]
            den += w[t] * tok_t[t] * main_usd_t[t] / main_tok_t[t]
    return num / den if den > 0 else None


# ── payload ────────────────────────────────────────────────────────────────

FAMILY_MIN_USD = 100.0      # BUILD.md: show a family only from $100


def _r(v: Optional[float], nd: int = 4) -> Optional[float]:
    return None if v is None else round(v, nd)


def _null() -> dict:
    return {"value": None, "lo": None, "hi": None}


def eff_ratio(series, idx, fi: int, w) -> Optional[float]:
    """Effective dollars per own dollar of family fi in the fitted intervals."""
    raw = sum(sum(series.feats[i][fi * N_TYPES:(fi + 1) * N_TYPES]) for i in idx)
    eff = sum(effective(series.feats[i], fi, w) for i in idx)
    return eff / raw if raw > 0 else None


def _per_point(res, pos: int, e: Optional[float]) -> Optional[tuple]:
    """Own dollars per point at the period base rate, with its 90% band."""
    m = res["m"][pos]
    if not m or not e:
        return None
    lo_p, hi_p = band([kk * mb[pos] * e for kk, mb in res["boot"]])
    return (1.0 / (res["kbar"] * m * e), 1.0 / hi_p if hi_p else None,
            1.0 / lo_p if lo_p else None)


def _per_token(fit_entry, fi: int, main: int, q: Optional[float]) -> dict:
    if fi == main:
        return {"value": 1.0, "lo": 1.0, "hi": 1.0}
    if fit_entry is None or q is None or fi not in fit_entry["fitted"]:
        return _null()
    res = fit_entry["res"]
    pos = fit_entry["fitted"].index(fi)
    if not res["identified"][pos]:
        return _null()
    lo, hi = band([mb[pos] * q for _, mb in res["boot"]])
    return {"value": _r(res["m"][pos] * q), "lo": _r(lo), "hi": _r(hi)}


def families_payload(fams, spend, usd_t, tok_t, main, w, fits: dict) -> list:
    """One entry per family with spend, by spend. ``fits``: bucket ->
    {"res", "fitted", "e"} or None; ``usd_t``/``tok_t``: family -> 4 sums
    by token type over the period (tokens may be missing: no per-token)."""
    out = []
    seven = fits.get("seven_day")
    main_name = fams[main] if main is not None else None
    for fi, fam in enumerate(fams):
        usd = spend.get(fam, 0.0)
        if usd <= 0:
            continue
        entry = {"family": fam, "usd_per_point": None, "lo": None, "hi": None,
                 "spend_usd": round(usd, 2),
                 "per_token": {"seven_day": _null(), "five_hour": _null()}}
        out.append(entry)
        if (usd < FAMILY_MIN_USD or seven is None or fi not in seven["fitted"]
                or not seven["res"]["identified"][seven["fitted"].index(fi)]):
            entry["note"] = "not enough data"
            continue
        pos = seven["fitted"].index(fi)
        pp = _per_point(seven["res"], pos, seven["e"].get(fi))
        if pp is None:
            entry["note"] = "no measurable limit use"
            continue
        entry.update(usd_per_point=_r(pp[0]), lo=_r(pp[1]), hi=_r(pp[2]))
        q = None
        if main_name and tok_t.get(fam) and tok_t.get(main_name):
            q = token_ratio(usd_t[fam], tok_t[fam], usd_t[main_name], tok_t[main_name], w)
        entry["per_token"] = {b: _per_token(fits.get(b), fi, main, q)
                              for b in ("seven_day", "five_hour")}
    out.sort(key=lambda f: -f["spend_usd"])
    return out
