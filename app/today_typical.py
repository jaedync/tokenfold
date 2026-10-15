"""Typical spend by this time of day, for /api/today. No routes.

"Typical by now" is the median, over the last 28 complete local days, of
each day's cumulative spend at the current local wall-clock time. That
needs spend per day at sub-day grain, which daily_summary does not keep,
so this module scans events once per local day and keeps a 15-minute slot
profile per day (past days do not change, apart from late ingest, which a
one-hour refresh absorbs). The scan prices 28 days of requests (~1.2 s on
the production copy), so it runs in a background cache: /api/today returns
null for typical_by_now_usd until the first build lands.

15-minute slots are absolute epoch buckets. Every IANA offset in use is a
multiple of 15 minutes, so each bucket maps to exactly one local slot.
"""
import statistics
import sys
from datetime import datetime, timedelta

from .bg_cache import BackgroundCache
from .db import read_conn
from .event_dedupe import canonical_assistant_predicate
from .pricing import (REPORTED_COST_SUM_SQL, compute_cost,
                      display_model_for_row, effective_geo, reported_cost)

SLOT_S = 900
TYPICAL_DAYS = 28

_PROFILE = BackgroundCache(ttl=3600, wait_s=0.05, max_entries=4)

_SLOT_SQL = (
    "SELECT CAST(first_ts / 900 AS INTEGER) AS bucket, model, provider, source_client, "
    "speed, inference_geo, SUM(inp) AS inp, SUM(outp) AS outp, SUM(cc) AS cc, "
    "SUM(cr) AS cr, SUM(c5m) AS c5m, SUM(c1h) AS c1h, SUM(ws) AS ws, "
    "SUM(reported_input) AS reported_input, SUM(reported_output) AS reported_output, "
    "SUM(reported_cache_read) AS reported_cache_read, "
    "SUM(reported_cache_write) AS reported_cache_write, "
    f"{REPORTED_COST_SUM_SQL} AS reported_total "
    "FROM ("
    "  SELECT MIN(ts_epoch) AS first_ts, model, provider, source_client, request_id, "
    "  MAX(speed) AS speed, MAX(inference_geo) AS inference_geo, "
    "  MAX(input_tokens) AS inp, MAX(output_tokens) AS outp, "
    "  MAX(cache_creation_tokens) AS cc, MAX(cache_read_tokens) AS cr, "
    "  MAX(cache_ephemeral_5m) AS c5m, MAX(cache_ephemeral_1h) AS c1h, "
    "  MAX(web_search_requests) AS ws, "
    "  MAX(reported_cost_input) AS reported_input, MAX(reported_cost_output) AS reported_output, "
    "  MAX(reported_cost_cache_read) AS reported_cache_read, "
    "  MAX(reported_cost_cache_write) AS reported_cache_write, "
    "  MAX(reported_cost_total) AS reported_total "
    "  FROM events WHERE type='assistant' AND model IS NOT NULL "
    "  AND model != '<synthetic>' AND request_id IS NOT NULL "
    "  AND {canonical} AND {pred} AND ts_epoch >= ? AND ts_epoch < ? "
    "  GROUP BY model, provider, source_client, request_id"
    ") GROUP BY bucket, model, provider, source_client, speed, inference_geo")


def history_days(today: str, first_day: str | None) -> list[str]:
    """The last 28 complete local days, from the first day with data on."""
    if not first_day or first_day >= today:
        return []
    base = datetime.strptime(today, "%Y-%m-%d")
    days = [(base - timedelta(days=n)).strftime("%Y-%m-%d")
            for n in range(TYPICAL_DAYS, 0, -1)]
    return [d for d in days if d >= first_day]


def _row_cost(r, enterprise: bool) -> float:
    cost = reported_cost(r)
    if cost is not None:
        return cost
    dm = display_model_for_row(r["model"], r["provider"], r["source_client"])
    return compute_cost(
        dm, r["inp"] or 0, r["outp"] or 0, r["cc"] or 0, r["cr"] or 0, r["speed"],
        effective_geo(r["inference_geo"], enterprise=enterprise),
        cw_5m=r["c5m"] or 0, cw_1h=r["c1h"] or 0, web_search=r["ws"] or 0,
        ts_epoch=r["bucket"] * SLOT_S)


def slot_profile(conn, scope: str, start: float, end: float, tz) -> dict:
    """{local day: {slot index: usd}} over [start, end)."""
    pred = sys.modules["app.config"].scope_predicate(scope)
    sql = _SLOT_SQL.format(canonical=canonical_assistant_predicate(), pred=pred)
    out: dict[str, dict[int, float]] = {}
    for r in conn.execute(sql, (start, end)):
        local = datetime.fromtimestamp(r["bucket"] * SLOT_S, tz)
        slot = (local.hour * 60 + local.minute) // 15
        day = out.setdefault(local.strftime("%Y-%m-%d"), {})
        day[slot] = day.get(slot, 0.0) + _row_cost(r, scope == "enterprise")
    return out


def cumulative_at(slots: dict[int, float], seconds_into_day: float) -> float:
    """Spend from midnight to this wall-clock time; the current slot counts
    pro rata (spend inside a slot is assumed even)."""
    k, rest = divmod(seconds_into_day, SLOT_S)
    k = int(k)
    done = sum(v for s, v in slots.items() if s < k)
    return done + slots.get(k, 0.0) * (rest / SLOT_S)


def build_profile(scope: str, days: list[str], tz) -> dict:
    if not days:
        return {}
    start = datetime.strptime(days[0], "%Y-%m-%d").replace(tzinfo=tz).timestamp()
    end = (datetime.strptime(days[-1], "%Y-%m-%d").replace(tzinfo=tz)
           + timedelta(days=1)).timestamp()
    with read_conn() as conn:
        return slot_profile(conn, scope, start, end, tz)


def typical_by_now(scope: str, days: list[str], seconds_into_day: float, tz) -> float | None:
    """Median cumulative spend at this time of day, or None (no history, or
    the background profile is not built yet)."""
    if not days:
        return None
    key = (scope, days[0], days[-1])
    profile = _PROFILE.get(key, lambda: build_profile(scope, days, tz))
    if profile is None:
        return None
    return statistics.median(cumulative_at(profile.get(d, {}), seconds_into_day)
                             for d in days)


def clear_cache() -> None:
    """Test support."""
    _PROFILE.clear()
