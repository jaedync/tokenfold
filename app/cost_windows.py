"""Shared cost accounting for bounded event windows.

Deduplicates streaming-API token repeats by taking MAX() per request_id,
then sums costs across models using pricing.compute_cost().
"""

import sqlite3

from .config import ENTERPRISE_PRED
from .pricing import compute_cost, display_model


def compute_window_cost(
    conn: sqlite3.Connection,
    start_epoch: float,
    end_epoch: float,
) -> float:
    """Sum assistant-event cost over [start_epoch, end_epoch).

    Streaming API chunks repeat token counts on every message; we dedupe
    with MAX(tokens) per (model, request_id), then sum costs per model.
    Synthetic events and rows missing model/request_id are excluded.

    Scope is fail-closed to verified-enterprise usage (config.ENTERPRISE_PRED)
    with no opt-out: this feeds the user-facing /api/ha REST endpoint and must
    never blend consumer-account spend.

    Returns 0.0 if the window is empty. Does not round — callers round
    to their preferred precision.
    """
    total = 0.0
    for r in conn.execute(
        "SELECT model, speed, inference_geo, "
        "SUM(inp) as inp, SUM(outp) as outp, "
        "SUM(cc) as cc, SUM(cr) as cr "
        "FROM ("
        "  SELECT model, request_id, "
        "  MAX(speed) as speed, MAX(inference_geo) as inference_geo, "
        "  MAX(input_tokens) as inp, MAX(output_tokens) as outp, "
        "  MAX(cache_creation_tokens) as cc, MAX(cache_read_tokens) as cr "
        "  FROM events WHERE type='assistant' AND model IS NOT NULL "
        "  AND model != '<synthetic>' AND request_id IS NOT NULL "
        f"  AND {ENTERPRISE_PRED} "
        "  AND ts_epoch >= ? AND ts_epoch < ? "
        "  GROUP BY model, request_id"
        ") GROUP BY model, speed, inference_geo",
        (start_epoch, end_epoch),
    ):
        dm = display_model(r["model"])
        total += compute_cost(
            dm,
            r["inp"] or 0,
            r["outp"] or 0,
            r["cc"] or 0,
            r["cr"] or 0,
            r["speed"],
            r["inference_geo"],
        )
    return total
