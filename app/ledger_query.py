"""SQL readers for the ledger (/api/ledger, /api/sessions/{id}).

Costs follow app/cost_windows.py exactly: MAX() per request_id, the canonical
assistant predicate for Meridian/Pi twins, reported Pi cost first, then list
price per pricing era. Activity (sessions, prompts, active time, tools) also
drops whole Meridian twin sessions through counts_as_activity, as the daily
rollup and the old recent-sessions card do.

The activity scan is the expensive part: tens of thousands of rows per day.
It reads the covering index idx_events_session_activity (app/db.py) instead of
the table, so it cannot test the scope predicate per row. The session list
comes from the billable query, which does test it, and a Claude Code session
with any row of the other scope (mixed_scope) takes the exact path instead.
"""

import json
import sqlite3
from collections import Counter
from itertools import groupby
from operator import itemgetter

from .aggregator import _make_display_names
from .config import IDLE_THRESHOLD_S
from .event_dedupe import canonical_assistant_predicate, counts_as_activity
from .pricing import (REPORTED_COST_SUM_SQL, compute_cost,
                      display_model_for_row, effective_geo, era_boundaries,
                      reported_cost)

ACTIVITY_INDEX = "idx_events_session_activity"
# json_each takes one bound parameter, so a long id list is one statement.
_IDS = "(SELECT value FROM json_each(?))"


def _billable_sql(pred: str, bounds: list) -> str:
    era = ""
    if bounds:
        era = "(" + " + ".join("(first_ts >= ?)" for _ in bounds) + ")"
    return (
        "SELECT session_id, agent_id, model, provider, source_client, speed, "
        "inference_geo, "
        f"{era + ' AS era, ' if era else ''}"
        "COUNT(*) AS requests, MIN(first_ts) AS first_ts, MAX(last_ts) AS last_ts, "
        "SUM(inp) AS inp, SUM(outp) AS outp, SUM(cc) AS cc, SUM(cr) AS cr, "
        "SUM(c5m) AS c5m, SUM(c1h) AS c1h, SUM(ws) AS ws, "
        "SUM(reported_input) AS reported_input, SUM(reported_output) AS reported_output, "
        "SUM(reported_cache_read) AS reported_cache_read, "
        "SUM(reported_cache_write) AS reported_cache_write, "
        f"{REPORTED_COST_SUM_SQL} AS reported_total, "
        "MAX(machine) AS machine, MAX(project_dir) AS project_dir, "
        "MAX(org_type) AS org_type "
        "FROM ("
        "  SELECT session_id, agent_id, model, provider, source_client, request_id, "
        "  MIN(ts_epoch) AS first_ts, MAX(ts_epoch) AS last_ts, "
        "  MAX(speed) AS speed, MAX(inference_geo) AS inference_geo, "
        "  MAX(input_tokens) AS inp, MAX(output_tokens) AS outp, "
        "  MAX(cache_creation_tokens) AS cc, MAX(cache_read_tokens) AS cr, "
        "  MAX(cache_ephemeral_5m) AS c5m, MAX(cache_ephemeral_1h) AS c1h, "
        "  MAX(web_search_requests) AS ws, "
        "  MAX(reported_cost_input) AS reported_input, "
        "  MAX(reported_cost_output) AS reported_output, "
        "  MAX(reported_cost_cache_read) AS reported_cache_read, "
        "  MAX(reported_cost_cache_write) AS reported_cache_write, "
        "  MAX(reported_cost_total) AS reported_total, "
        "  MAX(source_machine) AS machine, MAX(project_dir) AS project_dir, "
        "  MAX(org_type) AS org_type "
        # Term order matters: the twin test is a primary-key probe and drops
        # most Meridian rows before the canonical EXISTS probes run.
        "  FROM events WHERE type='assistant' AND request_id IS NOT NULL "
        "  AND ts_epoch >= ? AND ts_epoch < ? AND session_id IS NOT NULL "
        "  AND model IS NOT NULL AND model != '<synthetic>' "
        f"  AND {pred} AND {counts_as_activity()} AND {canonical_assistant_predicate()} "
        "  GROUP BY session_id, agent_id, model, provider, source_client, request_id"
        ") GROUP BY session_id, agent_id, model, provider, source_client, speed, "
        f"inference_geo{', era' if era else ''}"
    )


def billable_groups(conn: sqlite3.Connection, pred: str, start: float,
                    end: float, session_ids: list[str] | None = None) -> list[sqlite3.Row]:
    """Deduped billable groups per (session, agent, model, pricing inputs).

    The optional session filter is for the detail route (it then probes the
    session index per id).
    """
    bounds = era_boundaries()
    sql = _billable_sql(pred, bounds)
    params: list = [*bounds, start, end]
    if session_ids is not None:
        sql = sql.replace("AND session_id IS NOT NULL", f"AND session_id IN {_IDS}", 1)
        params.append(json.dumps(session_ids))
    return conn.execute(sql, params).fetchall()


def group_cost(row, enterprise: bool) -> float:
    """One group's dollars, priced as compute_window_cost_by_model prices it."""
    dm = display_model_for_row(row["model"], row["provider"], row["source_client"])
    cost = reported_cost(row)
    if cost is not None:
        return cost
    era_ts = row["first_ts"] if "era" in row.keys() else None
    return compute_cost(
        dm, row["inp"] or 0, row["outp"] or 0, row["cc"] or 0, row["cr"] or 0,
        row["speed"], effective_geo(row["inference_geo"], enterprise=enterprise),
        cw_5m=row["c5m"] or 0, cw_1h=row["c1h"] or 0,
        web_search=row["ws"] or 0, ts_epoch=era_ts)


def mixed_scope(conn: sqlite3.Connection, session_ids: list[str], start: float,
                end: float, pred: str) -> set[str]:
    """Claude Code sessions in the list with any user/assistant row outside
    the scope in [start, end) (an account switch mid-session).

    Pi ids carry the account class in their namespace, so a Pi session is
    never mixed; callers pass Claude Code ids only, which keeps this short.
    """
    if not session_ids:
        return set()
    return {r[0] for r in conn.execute(
        f"SELECT DISTINCT session_id FROM events WHERE session_id IN {_IDS} "
        "AND type IN ('user','assistant') AND ts_epoch >= ? AND ts_epoch < ? "
        f"AND NOT ({pred})", (json.dumps(session_ids), start, end))}


def accumulate_activity(rows) -> dict:
    """(session_id, agent_id, ts, is_human_prompt) rows, ordered by session,
    agent, time -> {(session, agent): {active_s, start, end, prompts}}.

    Active time is the summarizer's rule: gaps shorter than IDLE_THRESHOLD_S
    between consecutive user/assistant events of one thread. This loop sees
    every event of the window (300k+ for 30 days), so it stays plain.
    """
    out: dict = {}
    idle = IDLE_THRESHOLD_S
    for key, group in groupby(rows, key=itemgetter(0, 1)):
        _sid, _aid, start, prompts = next(group)
        prev, active, prompts = start, 0.0, prompts or 0
        for _s, _a, ts, prompt in group:
            gap = ts - prev
            if 0 < gap < idle:
                active += gap
            prev = ts
            if prompt:
                prompts += prompt
        out[key] = {"active_s": active, "start": start, "end": prev, "prompts": prompts}
    return out


def activity(conn: sqlite3.Connection, session_ids: list[str], start: float,
             end: float, exact_pred: str | None = None) -> dict:
    """Active time, span and human prompts per (session, agent) in [start, end).

    exact_pred filters each row by scope (mixed-scope sessions only): it
    reads the table, so keep that list short.
    """
    if not session_ids:
        return {}
    if exact_pred is None:
        source, scope = f"events INDEXED BY {ACTIVITY_INDEX}", ""
    else:
        source, scope = "events", f" AND {exact_pred}"
    rows = conn.execute(
        # A sidechain row is never a human prompt, whatever its flag says.
        "SELECT session_id, agent_id, ts_epoch, "
        f"CASE WHEN agent_id IS NULL THEN is_human_prompt ELSE 0 END FROM {source} "
        f"WHERE session_id IN {_IDS} AND type IN ('user','assistant') "
        f"AND ts_epoch >= ? AND ts_epoch < ?{scope} "
        "ORDER BY session_id, agent_id, ts_epoch",
        (json.dumps(session_ids), start, end))
    return accumulate_activity(rows)


def tool_counts(conn: sqlite3.Connection, session_ids: list[str],
                start: float, end: float) -> Counter:
    """Tool calls per session in [start, end)."""
    if not session_ids:
        return Counter()
    return Counter({r[0]: r[1] for r in conn.execute(
        f"SELECT session_id, COUNT(*) FROM tool_uses WHERE session_id IN {_IDS} "
        "AND ts_epoch >= ? AND ts_epoch < ? GROUP BY session_id",
        (json.dumps(session_ids), start, end))})


def run_tool_counts(conn: sqlite3.Connection, session_ids: list[str],
                    start: float, end: float) -> Counter:
    """Tool calls per (session, agent) for sessions with inline subagent rows.

    tool_uses has no agent_id, so this joins each call to its event. Claude
    Code only: a Pi child is its own session and needs no join.
    """
    if not session_ids:
        return Counter()
    return Counter({(r[0], r[1]): r[2] for r in conn.execute(
        "SELECT t.session_id, e.agent_id, COUNT(*) FROM tool_uses AS t "
        "JOIN events AS e ON e.uuid = t.event_uuid "
        f"WHERE t.session_id IN {_IDS} AND t.ts_epoch >= ? AND t.ts_epoch < ? "
        "AND e.agent_id IS NOT NULL GROUP BY t.session_id, e.agent_id",
        (json.dumps(session_ids), start, end))})


def titles(conn: sqlite3.Connection, session_ids: list[str]) -> dict:
    """AI titles, overlaid by explicit Claude Desktop titles (as the old card)."""
    if not session_ids:
        return {}
    ids = json.dumps(session_ids)
    out = {r[0]: r[1] for r in conn.execute(
        f"SELECT session_id, title FROM session_titles WHERE session_id IN {_IDS}", (ids,))}
    out.update({r[0]: r[1] for r in conn.execute(
        "SELECT cli_session_id, title FROM desktop_sessions "
        f"WHERE cli_session_id IN {_IDS} AND title IS NOT NULL AND title != ''", (ids,))})
    return out


def _path_parts(path: str) -> list[str]:
    parts = [p for p in path.split("/") if p]
    # A bare home folder says nothing about the project; never show the user name.
    if len(parts) == 2 and parts[0] in ("home", "Users"):
        return ["~"]
    return parts or ["/"]


def project_names(dirs) -> dict:
    """Short display names for project_dir values.

    Claude Code stores the dash-encoded folder; the dashboard's existing
    names apply. Pi stores an absolute path: show the leaf, with its parent
    when two paths share a leaf. Never show the full path (it names the user).
    """
    dirs = {d for d in dirs if d}
    encoded = sorted(d for d in dirs if not d.startswith("/"))
    out = dict(_make_display_names(encoded))
    paths = {d: _path_parts(d) for d in dirs if d.startswith("/")}
    leaves = Counter(parts[-1] for parts in paths.values())
    for d, parts in paths.items():
        leaf = parts[-1]
        out[d] = "/".join(parts[-2:]) if leaves[leaf] > 1 and len(parts) > 1 else leaf
    return out

