"""SQL loaders for /api/limit-rate: quota readings, granted resets and
deduped request dollars. Read-only; the caller owns the connection.
"""

import sqlite3
from typing import Optional

from .config import scope_predicate
from .cost_windows import ANTHROPIC_ONLY_SQL
from .event_dedupe import canonical_assistant_predicate
from .limit_readings import corroborated_resets
from .pricing import compute_cost, display_model, effective_geo
from .quota_history import history_source_filter

_FAMILIES = ("fable", "opus", "sonnet", "haiku")


def family_of(model: Optional[str]) -> str:
    m = (model or "").lower()
    return next((f for f in _FAMILIES if f in m), "other")


def load_readings(conn: sqlite3.Connection, bucket: str, since: float,
                  source: Optional[str], until: float = float("inf")) -> list:
    """(t, pct, resets_at_epoch) ascending for one bucket of the active
    quota feed. Legacy and managed feeds are not comparable
    (app/quota_history.py), so only one is read. Rows at the same second
    collapse to the last one, as limit-xray did."""
    predicate, params = history_source_filter(source)
    rows = conn.execute(
        "SELECT fetched_epoch, utilization, resets_at_epoch FROM limit_readings "
        "WHERE bucket=? AND fetched_epoch>=? AND fetched_epoch<=? "
        f"AND utilization IS NOT NULL AND {predicate} "
        "ORDER BY fetched_epoch, id", (bucket, since, until) + params).fetchall()
    out: list = []
    for t, pct, reset in rows:
        reading = (float(t), float(pct), None if reset is None else float(reset))
        if out and round(out[-1][0]) == round(reading[0]):
            out[-1] = reading
        else:
            out.append(reading)
    return out


# A reset this close to (or after) the window's own resets_at is the natural
# weekly rollover, not a grant.
NATURAL_ROLLOVER_S = 300.0


def load_granted(conn: sqlite3.Connection, since: float, source: Optional[str],
                 until: Optional[float] = None) -> list:
    """Granted mid-window resets of the 7-day meter, sibling-corroborated:
    the same evidence the spend windows use (app/limit_windows.py).

    detect_resets also flags the natural rollover (resets_at jumps a week
    while the previous poll was still inside the window). The spend windows
    do not mind, the rollover is their start anyway, but here every week
    would become a change point, so only resets well before the window end
    count."""
    return [e["at_epoch"] for e in
            corroborated_resets(conn, "seven_day", since, until_epoch=until, source=source)
            if e["resets_at_epoch_before"] is not None
            and e["resets_at_epoch_before"] > e["at_epoch"] + NATURAL_ROLLOVER_S]


# Inner query: one row per request (MAX per streaming repeat), canonical
# twin filter so a Meridian Pi request and its SDK copy count once. Same
# request key as app/cost_windows.py, so this endpoint and the dashboard
# count the same requests. The canonical predicate also drops Pi rows copied
# to another machine or path (not_pi_copy).
# Outer query: per-minute sums, so a 45-day scan returns tens of thousands of
# rows, not one per request. Cost is linear in tokens, so pricing the sums
# is exact; one minute is far below the 30-minute shortest interval. Rows
# with a positive Pi-reported cost are summed apart (``rep``): their
# reported components are the dollars, as in app/cost_windows.py.
_REP = ("COALESCE(MAX(reported_cost_total), MAX(reported_cost_input) + "
        "MAX(reported_cost_output) + MAX(reported_cost_cache_write) + "
        "MAX(reported_cost_cache_read), 0) > 0")
_REQUESTS_SQL = (
    "SELECT model, speed, inference_geo, rep, MIN(first_ts) AS first_ts, "
    "MAX(last_ts) AS t, SUM(inp) AS inp, SUM(outp) AS outp, SUM(cc) AS cc, "
    "SUM(cr) AS cr, SUM(c5m) AS c5m, SUM(c1h) AS c1h, SUM(ws) AS ws, "
    "SUM(r_in) AS r_in, SUM(r_out) AS r_out, SUM(r_cw) AS r_cw, "
    "SUM(r_cr) AS r_cr, SUM(r_total) AS r_total FROM ("
    "  SELECT model, MIN(ts_epoch) AS first_ts, MAX(ts_epoch) AS last_ts, "
    "  MAX(speed) AS speed, MAX(inference_geo) AS inference_geo, "
    "  MAX(input_tokens) AS inp, MAX(output_tokens) AS outp, "
    "  MAX(cache_creation_tokens) AS cc, MAX(cache_read_tokens) AS cr, "
    "  MAX(cache_ephemeral_5m) AS c5m, MAX(cache_ephemeral_1h) AS c1h, "
    "  MAX(web_search_requests) AS ws, "
    "  MAX(reported_cost_input) AS r_in, MAX(reported_cost_output) AS r_out, "
    "  MAX(reported_cost_cache_write) AS r_cw, MAX(reported_cost_cache_read) AS r_cr, "
    "  MAX(reported_cost_total) AS r_total, " + _REP + " AS rep "
    "  FROM events WHERE type='assistant' AND request_id IS NOT NULL "
    "  AND model IS NOT NULL AND model != '<synthetic>' "
    "  AND ts_epoch >= ? AND ts_epoch < ? "
    "  AND {canonical} AND {scope} AND {anthropic} "
    "  GROUP BY model, provider, source_client, request_id"
    ") GROUP BY model, speed, inference_geo, rep, CAST(last_ts / 60 AS INTEGER)")


def _token_cost(r) -> list:
    """List-price dollars by token type from tokens, priced by the raw
    Anthropic model id (Pi rows too). Web search fees go with input."""
    name = display_model(r["model"])
    geo = effective_geo(r["inference_geo"], enterprise=False)
    kw = {"speed": r["speed"], "inference_geo": geo, "ts_epoch": r["first_ts"]}
    return [
        compute_cost(name, r["inp"] or 0, 0, 0, 0, web_search=r["ws"] or 0, **kw),
        compute_cost(name, 0, r["outp"] or 0, 0, 0, **kw),
        compute_cost(name, 0, 0, r["cc"] or 0, 0, cw_5m=r["c5m"] or 0,
                     cw_1h=r["c1h"] or 0, **kw),
        compute_cost(name, 0, 0, 0, r["cr"] or 0, **kw),
    ]


def _split_cost(r) -> list:
    """Dollars by token type for one per-minute group.

    Pi-reported costs win when positive (the canonical cost rule): Pi
    prices 1 h cache writes at 2x input but stores no 1 h token split, so
    token pricing would undercount its cache writes by about a third
    (prod, 7 Oct 2026: $297 reported vs $186 at the 5 m rate). The total
    wins over the components; components only give the split."""
    tokens = _token_cost(r)
    if not r["rep"]:
        return tokens
    comps = [r["r_in"] or 0.0, r["r_out"] or 0.0, r["r_cw"] or 0.0, r["r_cr"] or 0.0]
    total = r["r_total"] if r["r_total"] is not None else sum(comps)
    shape = comps if sum(comps) > 0 else tokens
    base = sum(shape)
    if base <= 0:
        return [total, 0.0, 0.0, 0.0]
    return [v * total / base for v in shape]


def load_requests(conn: sqlite3.Connection, since: float, until: float,
                  scope: str = "personal") -> list:
    """[(t, family, [usd_in, usd_out, usd_cache_write, usd_cache_read])] by t.

    Personal scope and Anthropic-bound rows only: the subscription meter
    never sees Codex, OpenCode or work-account traffic."""
    sql = _REQUESTS_SQL.format(canonical=canonical_assistant_predicate(),
                               scope=scope_predicate(scope),
                               anthropic=ANTHROPIC_ONLY_SQL)
    cur = conn.cursor()
    cur.row_factory = sqlite3.Row  # per cursor: never touch a shared connection
    rows = cur.execute(sql, (since, until)).fetchall()
    out = [(float(r["t"]), family_of(r["model"]), _split_cost(r)) for r in rows]
    out.sort(key=lambda x: x[0])
    return out
