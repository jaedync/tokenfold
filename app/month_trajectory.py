"""Month-to-date billed line for the work scope's claude:month window
(/api/runway/trajectory). No routes (app/runway.py serves it).

Source: extra_usage_readings, Anthropic's billing meter captured from
enterprise pushes (app/extra_usage.py). When the month has no meter reading,
the hand-entered org-page figures in billing_readings are the fallback.
Points are a percent of the same budget as the month window, so the chart's
y scale and the window's pct agree. Epochs are minute-floored like every
other trajectory.
"""

from .limit_trends import downsample

MAX_POINTS = 400


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
