"""Shared cost accounting for bounded event windows.

Deduplicates streaming-API token repeats by taking MAX() per request_id,
then sums costs across models using pricing.compute_cost().
"""

import sqlite3

from .config import DEFAULT_SCOPE, scope_predicate
from .pricing import (compute_cost, display_model_for_row, effective_geo,
                      era_boundaries, reported_cost)


def compute_window_cost(
    conn: sqlite3.Connection,
    start_epoch: float,
    end_epoch: float,
    scope: str = DEFAULT_SCOPE,
) -> float:
    """Sum assistant-event cost over [start_epoch, end_epoch) for the given scope.

    Thin wrapper over compute_window_cost_by_model — one query, identical
    dedupe/era/geo semantics; this just collapses the per-model split.

    Returns 0.0 if the window is empty. Does not round — callers round
    to their preferred precision.
    """
    return sum(compute_window_cost_by_model(
        conn, start_epoch, end_epoch, scope).values())


def compute_window_cost_by_model(
    conn: sqlite3.Connection,
    start_epoch: float,
    end_epoch: float,
    scope: str = DEFAULT_SCOPE,
) -> dict:
    """Per-display-model cost over [start_epoch, end_epoch) for the given scope.

    Streaming API chunks repeat token counts on every message; we dedupe
    with MAX(tokens) per (model, request_id), then sum costs per model.
    Synthetic events and rows missing model/request_id are excluded.

    Defaults to enterprise scope (fail-closed for /api/ha and HA integrations).
    Pass scope='personal' for the personal usage view.

    Returns {} if the window is empty; values are unrounded floats keyed by
    display_model() names (e.g. 'Sonnet 5'), so raw model-id aliases that
    share a display name are already merged.
    """
    pred = scope_predicate(scope)
    # Era-split: the outer GROUP BY discards timestamps, so a window straddling
    # a pricing-era boundary would price both sides at one era. A SQL-side
    # bucket column ((first_ts >= b1) + (first_ts >= b2) + ...) splits groups
    # at each boundary — every row in a group then shares every boundary side,
    # so the group's MIN(first_ts) is a valid pricing representative. Kept in
    # SQL (no per-request Python loop): this is a hot path.
    bounds = era_boundaries()
    if bounds:
        era_sel = ("(" + " + ".join("(first_ts >= ?)" for _ in bounds)
                   + ") as era, MIN(first_ts) as first_ts, ")
        inner_ts = "MIN(ts_epoch) as first_ts, "
        era_grp = ", era"
        params: tuple = (*bounds, start_epoch, end_epoch)
    else:
        # No era-listed models: query is structurally identical to the
        # pre-era version.
        era_sel = inner_ts = era_grp = ""
        params = (start_epoch, end_epoch)
    by_model: dict = {}
    for r in conn.execute(
        "SELECT model, provider, source_client, speed, inference_geo, "
        f"{era_sel}"
        "SUM(inp) as inp, SUM(outp) as outp, "
        "SUM(cc) as cc, SUM(cr) as cr, SUM(c5m) as c5m, SUM(c1h) as c1h, "
        "SUM(ws) as ws, "
        "SUM(reported_input) as reported_input, SUM(reported_output) as reported_output, "
        "SUM(reported_cache_read) as reported_cache_read, SUM(reported_cache_write) as reported_cache_write, "
        "SUM(reported_total) as reported_total "
        "FROM ("
        f"  SELECT model, provider, source_client, request_id, {inner_ts}"
        "  MAX(speed) as speed, MAX(inference_geo) as inference_geo, "
        "  MAX(input_tokens) as inp, MAX(output_tokens) as outp, "
        "  MAX(cache_creation_tokens) as cc, MAX(cache_read_tokens) as cr, "
        "  MAX(cache_ephemeral_5m) as c5m, MAX(cache_ephemeral_1h) as c1h, "
        "  MAX(web_search_requests) as ws, "
        "  MAX(reported_cost_input) as reported_input, MAX(reported_cost_output) as reported_output, "
        "  MAX(reported_cost_cache_read) as reported_cache_read, MAX(reported_cost_cache_write) as reported_cache_write, "
        "  MAX(reported_cost_total) as reported_total "
        "  FROM events WHERE type='assistant' AND model IS NOT NULL "
        "  AND model != '<synthetic>' AND request_id IS NOT NULL "
        f"  AND {pred} "
        "  AND ts_epoch >= ? AND ts_epoch < ? "
        "  GROUP BY model, provider, source_client, request_id"
        f") GROUP BY model, provider, source_client, speed, inference_geo{era_grp}",
        params,
    ):
        dm = display_model_for_row(r["model"], r["provider"], r["source_client"])
        req_cost = reported_cost(r)
        if req_cost is None:
            req_cost = compute_cost(
                dm,
                r["inp"] or 0,
                r["outp"] or 0,
                r["cc"] or 0,
                r["cr"] or 0,
                r["speed"],
                effective_geo(r["inference_geo"], enterprise=(scope == "enterprise")),
                cw_5m=r["c5m"] or 0,
                cw_1h=r["c1h"] or 0,
                web_search=r["ws"] or 0,
                ts_epoch=r["first_ts"] if bounds else None,
            )
        by_model[dm] = by_model.get(dm, 0.0) + req_cost
    return by_model
