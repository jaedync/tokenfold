"""Small pure-Python fitting kit for /api/limit-rate. No routes, no I/O.

Why no numpy: every fit here has at most 8 features, so the regression works
on the k x k normal equations, and a day-block bootstrap only adds up
precomputed per-day sums. That is cheap in plain Python and keeps the image
free of a 30+ MB dependency.

Data shapes:
- a reading is (fetched_epoch, utilization_pct, resets_at_epoch or None);
- an Interval is a span (ta, tb] between two readings of one limit window;
- a "gram vector" is a flat list of sums over intervals (see gram_vector),
  so blocks can be added together with add_vectors.
"""

import random
from bisect import bisect_right
from typing import Callable, NamedTuple, Optional, Sequence

# Two readings belong to one limit window when their resets_at anchors agree
# within this tolerance (OAuth jitters resets_at between polls; the reset
# detector in app/limit_readings.py uses the same 1200 s jump rule).
SAME_WINDOW_S = 1200.0
# Utilization is a whole-point meter that can wobble by a fraction between
# sources; a smaller dip is not a reset.
DROP_TOLERANCE_PTS = 0.5


class Interval(NamedTuple):
    ta: float
    tb: float
    points: float


class GramStats(NamedTuple):
    ata: list
    aty: list
    yty: float
    sum_y: float
    n: float
    sum_x: list


def _same_window(ra, rb) -> bool:
    return ra is not None and rb is not None and abs(rb - ra) <= SAME_WINDOW_S


def build_intervals(readings: Sequence, min_s: float, max_s: float) -> list:
    """Greedy, non-overlapping intervals of at least ``min_s`` seconds.

    Port of limit-xray lib.intervals: start at a reading, walk forward until
    ``min_s`` has passed; give up early on a window change (resets_at moved)
    or a drop (a granted reset). Long spans hold whole-point rounding noise
    small relative to the points gained. Spans over ``max_s`` are collector
    gaps and are skipped.
    """
    out = []
    n = len(readings)
    i = 0
    while i < n - 1:
        ta, ua, ra = readings[i]
        j = i + 1
        while j < n and readings[j][0] - ta < min_s:
            _, uj, rj = readings[j]
            if not _same_window(ra, rj) or uj < ua - DROP_TOLERANCE_PTS:
                break
            j += 1
        if j >= n:
            break
        tb, ub, rb = readings[j]
        ok = (_same_window(ra, rb) and ub >= ua - DROP_TOLERANCE_PTS
              and tb - ta >= min_s)
        if ok and tb - ta <= max_s:
            out.append(Interval(ta, tb, ub - ua))
        i = j
    return out


def assign_to_intervals(intervals: Sequence, rows: Sequence, width: int) -> list:
    """Sum each row's vector into the interval (ta, tb] holding its time.

    ``rows`` are (t, vector) sorted by t; intervals are sorted and disjoint
    (build_intervals guarantees it), so one bisect per interval is enough.
    """
    times = [r[0] for r in rows]
    sums = []
    for iv in intervals:
        acc = [0.0] * width
        lo = bisect_right(times, iv.ta)
        hi = bisect_right(times, iv.tb)
        for k in range(lo, hi):
            vec = rows[k][1]
            for c in range(width):
                acc[c] += vec[c]
        sums.append(acc)
    return sums


def normal_equations(rows: Sequence, ys: Sequence) -> tuple:
    k = len(rows[0]) if rows else 0
    ata = [[0.0] * k for _ in range(k)]
    aty = [0.0] * k
    for row, y in zip(rows, ys):
        for a in range(k):
            ra = row[a]
            if ra == 0.0:
                continue
            aty[a] += ra * y
            ata_a = ata[a]
            for b in range(k):
                ata_a[b] += ra * row[b]
    return ata, aty


def gram_vector(rows: Sequence, ys: Sequence) -> list:
    """Flat sums for one block: AtA, Aty, yty, sum y, n, sum x per feature."""
    ata, aty = normal_equations(rows, ys)
    k = len(aty)
    sum_x = [sum(r[c] for r in rows) for c in range(k)]
    flat = [v for row in ata for v in row]
    return flat + aty + [sum(y * y for y in ys), float(sum(ys)), float(len(ys))] + sum_x


def unpack_gram(vec: Sequence, k: int) -> GramStats:
    ata = [list(vec[r * k:(r + 1) * k]) for r in range(k)]
    p = k * k
    aty = list(vec[p:p + k])
    yty, sum_y, n = vec[p + k], vec[p + k + 1], vec[p + k + 2]
    sum_x = list(vec[p + k + 3:p + 2 * k + 3])
    return GramStats(ata, aty, yty, sum_y, n, sum_x)


def add_vectors(vectors: Sequence) -> list:
    if not vectors:
        return []
    out = [0.0] * len(vectors[0])
    for v in vectors:
        for i, x in enumerate(v):
            out[i] += x
    return out


def _solve(m: list, b: list) -> list:
    """Gaussian elimination with partial pivoting on a small dense system.

    A tiny ridge on a near-singular pivot keeps collinear features (cache
    reads and cache writes move together) from raising; NNLS then splits the
    weight between them, and the bootstrap band shows the ambiguity.
    """
    n = len(b)
    a = [list(row) + [b[i]] for i, row in enumerate(m)]
    scale = max((abs(a[i][i]) for i in range(n)), default=1.0) or 1.0
    eps = scale * 1e-12
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[piv] = a[piv], a[col]
        if abs(a[col][col]) < eps:
            a[col][col] = eps
        for r in range(col + 1, n):
            f = a[r][col] / a[col][col]
            if f:
                for c in range(col, n + 1):
                    a[r][c] -= f * a[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = a[r][n] - sum(a[r][c] * x[c] for c in range(r + 1, n))
        x[r] = s / a[r][r]
    return x


def nnls(ata: list, aty: list, max_iter: int = 200) -> list:
    """Lawson-Hanson non-negative least squares on the normal equations.

    min ||Ax - y||^2 subject to x >= 0 only needs AtA and Aty, which is what
    lets the bootstrap add up per-day blocks instead of refitting raw rows.
    """
    k = len(aty)
    if k == 0:
        return []
    x = [0.0] * k
    passive: set = set()
    tol = 1e-10 * max(1.0, max(abs(v) for v in aty))

    def gradient(xv):
        return [aty[i] - sum(ata[i][j] * xv[j] for j in range(k)) for i in range(k)]

    def ls_on(pset):
        idx = sorted(pset)
        sol = _solve([[ata[i][j] for j in idx] for i in idx], [aty[i] for i in idx])
        z = [0.0] * k
        for i, v in zip(idx, sol):
            z[i] = v
        return z

    for _ in range(max_iter):
        w = gradient(x)
        free = [i for i in range(k) if i not in passive and w[i] > tol]
        if not free:
            break
        passive.add(max(free, key=lambda i: w[i]))
        for _ in range(max_iter):
            z = ls_on(passive)
            if all(z[i] > 0 for i in passive):
                x = z
                break
            alpha = min(x[i] / (x[i] - z[i]) for i in passive if z[i] <= 0)
            x = [x[i] + alpha * (z[i] - x[i]) for i in range(k)]
            passive = {i for i in passive if x[i] > 1e-15}
            x = [x[i] if i in passive else 0.0 for i in range(k)]
            if not passive:
                break
    return [max(0.0, v) for v in x]


def r_squared(stats: GramStats, x: Sequence) -> Optional[float]:
    """1 - SSE/SST from the summed block statistics (no raw rows needed)."""
    k = len(x)
    if stats.n < 2:
        return None
    xatx = sum(x[i] * stats.ata[i][j] * x[j] for i in range(k) for j in range(k))
    sse = stats.yty - 2 * sum(x[i] * stats.aty[i] for i in range(k)) + xatx
    sst = stats.yty - stats.sum_y ** 2 / stats.n
    if sst <= 0:
        return None
    return 1.0 - sse / sst


def block_bootstrap(blocks: Sequence, stat: Callable, draws: int, seed: int) -> list:
    """Resample whole blocks (days) with replacement; return stat per draw.

    Day blocks keep the autocorrelation of one working day together, so the
    band is honest about how few independent days the fit has. Seeded: the
    same data gives the same band on every recompute. Draws where ``stat``
    returns None (for example no points) are dropped.
    """
    rng = random.Random(seed)
    n = len(blocks)
    out = []
    if n == 0:
        return out
    for _ in range(draws):
        picked = [blocks[rng.randrange(n)] for _ in range(n)]
        value = stat(add_vectors(picked))
        if value is not None:
            out.append(value)
    return out


def percentile(values: Sequence, q: float) -> Optional[float]:
    """Linear-interpolated percentile (numpy's default method)."""
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def median(values: Sequence) -> Optional[float]:
    return percentile(values, 50)
