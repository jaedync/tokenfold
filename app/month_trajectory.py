"""Month-to-date billed line for the work scope's claude:month window
(/api/runway/trajectory). No routes (app/runway.py serves it).

Source: extra_usage_readings, Anthropic's billing meter captured from
enterprise pushes (app/extra_usage.py). When the month has no meter reading,
the hand-entered org-page figures in billing_readings are the fallback.
Points are a percent of the same budget as the month window, so the chart's
y scale and the window's pct agree. Epochs are minute-floored like every
other trajectory.

The split divides the area under the billed line by model. The meter has no
model detail, so the layers are the recorded work Claude spend per model
(the sessions Tokenfold has), summed from the month start to each point.
Billed spend above the recorded total is its own layer. When the recorded
total is above the billed line (the meter lags the calls a little), the
model layers scale down to the line. The layers always add up to the point.
"""
import logging
from bisect import bisect_right

from . import db
from .aggregator import get_cache_version
from .cost_windows import compute_window_cost_by_model_buckets
from .limit_trends import downsample
from .read_cache import ReadCache

log = logging.getLogger(__name__)

MAX_POINTS = 400
# Recorded spend is summed in 10 min steps: a point counts the steps that
# ended at or before it. Finer than the meter's own lag, and one month is
# at most ~4,500 steps for the one query.
SPLIT_BUCKET_S = 600
# The largest models keep their own layer; the rest share one.
SPLIT_MAX_MODELS = 5
OTHER_MODELS = "Other models"
# Pi rows name the provider ("Anthropic / Opus 5.5"), Claude Code rows do not
# ("Opus 5.5"). The meter bills Anthropic only, so both are one layer.
_PI_PREFIX = "Anthropic / "
# The month's spend query takes ~0.4 s on the live database, and every tab
# polls the trajectory. Built again after an ingest (the version) or a new
# meter reading (the key).
_split_cache = ReadCache(ttl=120, max_entries=4, version=get_cache_version)


def _floor(epoch):
    return (epoch // 60) * 60.0


def _current_cycle(readings):
    """Readings after the newest drop in used dollars.

    The meter counts up from zero each billing cycle. Its rollover can come
    a little after the UTC month starts, so readings before a drop belong
    to the previous cycle.
    """
    start = 0
    for i in range(1, len(readings)):
        if readings[i][1] < readings[i - 1][1]:
            start = i
    return readings[start:]


def _meter_readings(conn, month_start, now):
    rows = conn.execute(
        "SELECT fetched_epoch, used_cents FROM extra_usage_readings "
        "WHERE fetched_epoch >= ? AND fetched_epoch <= ? "
        "ORDER BY fetched_epoch", (month_start, now)).fetchall()
    return [(r["fetched_epoch"], r["used_cents"] / 100.0) for r in rows]


def _manual_readings(conn, month_label, now):
    rows = conn.execute(
        "SELECT recorded_epoch, amount_usd FROM billing_readings "
        "WHERE scope = 'enterprise' AND month = ? AND recorded_epoch <= ? "
        "ORDER BY recorded_epoch", (month_label, now)).fetchall()
    return [(r["recorded_epoch"], r["amount_usd"]) for r in rows]


def month_points(conn, month_start, month_label, budget_usd, now):
    """[[t, pct_of_budget], ...] for this month, oldest first (<= 400)."""
    if not budget_usd or budget_usd <= 0:
        return []
    readings = (_meter_readings(conn, month_start, now)
                or _manual_readings(conn, month_label, now))
    points = [[_floor(t), round(usd / budget_usd * 100, 4)]
              for t, usd in _current_cycle(readings)]
    return downsample(points, MAX_POINTS)


def month_split(conn, month_start, points, budget_usd):
    """The model split of points (see the module doc), or None.

    None when there is nothing to split or the spend query fails: the
    billed line must still draw.
    """
    if not points or not budget_usd or budget_usd <= 0:
        return None
    end = points[-1][0]
    try:
        buckets = _split_cache.get((db.DB_PATH, month_start, end), lambda: _merge_names(
            compute_window_cost_by_model_buckets(
                conn, month_start, end, SPLIT_BUCKET_S,
                scope="enterprise", anthropic_only=True)))
    except Exception as e:
        log.warning("month split failed: %r", e)
        return None
    return split_layers(points, buckets, month_start, SPLIT_BUCKET_S, budget_usd)


def _merge_names(buckets):
    out = {}
    for i, models in buckets.items():
        merged = {}
        for m, usd in models.items():
            name = m[len(_PI_PREFIX):] if m.startswith(_PI_PREFIX) else m
            merged[name] = merged.get(name, 0.0) + usd
        out[i] = merged
    return out


def _cumulative(buckets):
    """(bucket ends, [{model: usd so far}]) in bucket order."""
    ends, totals, acc = [], [], {}
    for i in sorted(buckets):
        acc = {m: acc.get(m, 0.0) + buckets[i].get(m, 0.0)
               for m in {*acc, *buckets[i]}}
        ends.append(i + 1)
        totals.append(acc)
    return ends, totals


def _top_models(final):
    """Model names to keep, largest first, and the ones merged into Other."""
    ranked = sorted((m for m, v in final.items() if v > 0),
                    key=lambda m: (-final[m], m))
    if len(ranked) <= SPLIT_MAX_MODELS + 1:
        return ranked, []
    return ranked[:SPLIT_MAX_MODELS], ranked[SPLIT_MAX_MODELS:]


def split_layers(points, buckets, start, bucket_s, budget_usd):
    """{models: [{model, pct: [...]}], unrecorded: [...]} aligned to points.

    pct values are layer heights (not stacked) in percent of the budget.
    """
    ends, totals = _cumulative(buckets)
    if not totals:
        return None
    keep, merged = _top_models(totals[-1])
    if not keep:
        return None
    names = keep + ([OTHER_MODELS] if merged else [])
    layers = {m: [] for m in names}
    unrecorded = []
    for t, pct in points:
        k = bisect_right(ends, int((t - start) // bucket_s)) - 1
        spent = totals[k] if k >= 0 else {}
        usd = {m: spent.get(m, 0.0) for m in keep}
        if merged:
            usd[OTHER_MODELS] = sum(spent.get(m, 0.0) for m in merged)
        recorded = sum(usd.values()) / budget_usd * 100
        scale = pct / recorded if recorded > pct else 1.0
        for m in names:
            layers[m].append(round(usd[m] / budget_usd * 100 * scale, 4))
        unrecorded.append(round(max(0.0, pct - recorded), 4))
    return {"models": [{"model": m, "pct": layers[m]} for m in names],
            "unrecorded": unrecorded}
