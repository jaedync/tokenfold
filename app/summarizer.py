"""Compute daily_summary rows from raw events for specific days."""

import json
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .config import ENTERPRISE_PRED, IDLE_THRESHOLD_S, TZ_NAME
from .db import get_conn, write_txn
from .pricing import (compute_cost, display_model, display_model_for_row,
                      effective_geo, load_pricing, reported_cost)

TZ = ZoneInfo(TZ_NAME)


def _reported_cost_parts(
        row, total: float) -> tuple[float, float, float, float, float] | None:
    """Return Pi-reported cost components plus any undecomposed residual."""
    if row["source_client"] != "pi-agent":
        return None
    keys = ("reported_input", "reported_output", "reported_cache_read",
            "reported_cache_write", "reported_total")
    if not any(row[key] is not None for key in keys):
        return None
    parts = tuple(float(row[key] or 0) for key in keys[:4])
    residual = max(0.0, float(total) - sum(parts))
    return (*parts, residual)


def _accumulate(conn, days: list[str], placeholders: str, account: str) -> dict:
    """Build per-day accumulators for a single account.

    Returns day_data: dict[str, dict] keyed by day string, with the same
    structure as before (model/project/machine/tool/prompt_model/gen + scalars).
    Every events query is scoped to the given account via
    AND COALESCE(account_email,'unknown') = ?
    tool_uses queries are scoped via the account's session_ids.
    """
    acct_filter = "AND COALESCE(account_email,'unknown') = ?"

    # ── Tool counts per day — scoped to account's sessions ──
    # tool_uses has no account_email; scope via session_id subquery.
    session_subq = (
        f"AND session_id IN ("
        f"SELECT session_id FROM events "
        f"WHERE COALESCE(account_email,'unknown') = ? AND day IN ({placeholders})"
        f")"
    )
    daily_tool_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    daily_tool_totals: dict[str, int] = defaultdict(int)
    for r in conn.execute(
        f"SELECT day, name, COUNT(*) as cnt FROM tool_uses "
        f"WHERE day IN ({placeholders}) {session_subq} GROUP BY day, name",
        days + [account] + days,
    ):
        daily_tool_counts[r["day"]][r["name"]] = r["cnt"]
        daily_tool_totals[r["day"]] += r["cnt"]

    # ── Q1: Token dedup per request_id, scoped to account + target days ──
    requests = conn.execute(
        f"SELECT request_id, COALESCE(project_dir,'unknown') as project_dir, "
        f"source_machine, session_id, day, model, provider, source_client, is_sidechain, agent_id, "
        f"speed, inference_geo, ({ENTERPRISE_PRED}) as is_ent, "
        f"MAX(input_tokens) as inp, MAX(output_tokens) as out, "
        f"MAX(cache_creation_tokens) as cc, MAX(cache_read_tokens) as cr, "
        f"MAX(cache_ephemeral_5m) as c5m, MAX(cache_ephemeral_1h) as c1h, "
        f"MAX(web_search_requests) as ws, MAX(web_fetch_requests) as wf, "
        f"MAX(reported_cost_input) as reported_input, MAX(reported_cost_output) as reported_output, "
        f"MAX(reported_cost_cache_read) as reported_cache_read, MAX(reported_cost_cache_write) as reported_cache_write, "
        f"MAX(reported_cost_total) as reported_total, "
        f"MIN(ts_epoch) as first_ts, MAX(ts_epoch) as last_ts "
        f"FROM events "
        f"WHERE type='assistant' AND model IS NOT NULL AND model != '<synthetic>' "
        f"AND request_id IS NOT NULL AND day IN ({placeholders}) {acct_filter} "
        f"GROUP BY request_id, provider, source_client",
        days + [account],
    ).fetchall()

    # Per-day accumulators
    day_data: dict[str, dict] = {}
    for d in days:
        day_data[d] = {
            "sessions": 0, "human_prompts": 0, "tool_calls": daily_tool_totals.get(d, 0),
            "input_tokens": 0, "output_tokens": 0,
            "cache_creation_tokens": 0, "cache_read_tokens": 0,
            "active_s": 0.0, "thinking_s": 0.0, "tool_exec_s": 0.0,
            "subagent_s": 0.0, "agent_runs": 0, "cost": 0.0,
            "model": defaultdict(lambda: {
                "input": 0, "output": 0, "cache_write": 0, "cache_read": 0,
                "cache_5m": 0, "cache_1h": 0,
                "web_search": 0, "web_fetch": 0,
                "api_calls": 0, "main_api_calls": 0, "main_cost": 0.0,
                "cost": 0.0, "has_reported_cost": False,
                "reported_cost_input": 0.0, "reported_cost_output": 0.0,
                "reported_cost_cache_read": 0.0,
                "reported_cost_cache_write": 0.0, "reported_cost_other": 0.0,
                "main_prompts": 0, "agent_invocations": 0,
                "active_s": 0.0, "gen_s": 0.0, "gen_out": 0,
            }),
            "project": defaultdict(lambda: {"seconds": 0.0, "cost": 0.0}),
            "machine": defaultdict(lambda: {
                "input": 0, "output": 0, "cache_write": 0, "cache_read": 0,
                "calls": 0, "prompts": 0, "tool_calls": 0, "cost": 0.0,
            }),
            "tool": dict(daily_tool_counts.get(d, {})),
            "prompt_model": defaultdict(int),
            "gen": defaultdict(lambda: {"gen_s": 0.0, "gen_out": 0}),
        }

    # Process Q1 results
    for r in requests:
        d = r["day"]
        if d not in day_data:
            continue
        dd = day_data[d]
        inp, out = r["inp"] or 0, r["out"] or 0
        cc, cr = r["cc"] or 0, r["cr"] or 0
        dm = display_model_for_row(r["model"], r["provider"], r["source_client"])
        machine = r["source_machine"]

        dd["input_tokens"] += inp
        dd["output_tokens"] += out
        dd["cache_creation_tokens"] += cc
        dd["cache_read_tokens"] += cr

        # Era selection keys on the request's first_ts, so re-summarizing an
        # old day after a pricing-era flip keeps its original rates. A local-TZ
        # day can straddle the UTC era boundary by a few hours; pricing each
        # request at its own first_ts is the accepted approximation.
        req_cost = reported_cost(r)
        if req_cost is None:
            req_cost = compute_cost(dm, inp, out, cc, cr, r["speed"],
                                    effective_geo(r["inference_geo"],
                                                  enterprise=bool(r["is_ent"])),
                                    cw_5m=r["c5m"] or 0, cw_1h=r["c1h"] or 0,
                                    web_search=r["ws"] or 0,
                                    ts_epoch=r["first_ts"])
        dd["cost"] += req_cost
        dd["project"][r["project_dir"]]["cost"] += req_cost

        ms = dd["model"][dm]
        ms["input"] += inp
        ms["output"] += out
        ms["cache_write"] += cc
        ms["cache_5m"] += r["c5m"] or 0
        ms["cache_1h"] += r["c1h"] or 0
        ms["web_search"] += r["ws"] or 0
        ms["web_fetch"] += r["wf"] or 0
        ms["cache_read"] += cr
        ms["api_calls"] += 1
        ms["cost"] += req_cost
        reported_parts = _reported_cost_parts(r, req_cost)
        if reported_parts is not None:
            ms["has_reported_cost"] = True
            for key, value in zip(
                    ("reported_cost_input", "reported_cost_output",
                     "reported_cost_cache_read", "reported_cost_cache_write",
                     "reported_cost_other"), reported_parts):
                ms[key] += value
        if not r["is_sidechain"]:
            ms["main_api_calls"] += 1
            ms["main_cost"] += req_cost

        mt = dd["machine"][machine]
        mt["input"] += inp
        mt["output"] += out
        mt["cache_write"] += cc
        mt["cache_read"] += cr
        mt["calls"] += 1
        mt["cost"] += req_cost

    # ── Q2: Sessions per day ──
    for r in conn.execute(
        f"SELECT day, COUNT(DISTINCT session_id) as cnt "
        f"FROM events WHERE agent_id IS NULL AND day IN ({placeholders}) "
        f"{acct_filter} GROUP BY day",
        days + [account],
    ):
        if r["day"] in day_data:
            day_data[r["day"]]["sessions"] = r["cnt"]

    # ── Q3: Human prompts per day ──
    for r in conn.execute(
        f"SELECT day, COUNT(*) as cnt "
        f"FROM events WHERE is_human_prompt=1 AND is_sidechain=0 "
        f"AND day IN ({placeholders}) {acct_filter} GROUP BY day",
        days + [account],
    ):
        if r["day"] in day_data:
            day_data[r["day"]]["human_prompts"] = r["cnt"]

    # ── Machine prompts per day ──
    for r in conn.execute(
        f"SELECT day, source_machine, COUNT(*) as cnt "
        f"FROM events WHERE is_human_prompt=1 AND day IN ({placeholders}) "
        f"{acct_filter} GROUP BY day, source_machine",
        days + [account],
    ):
        if r["day"] in day_data:
            day_data[r["day"]]["machine"][r["source_machine"]]["prompts"] += r["cnt"]

    # ── Machine tool calls per day — scoped to account's sessions ──
    for r in conn.execute(
        f"SELECT day, source_machine, COUNT(*) as cnt "
        f"FROM tool_uses WHERE day IN ({placeholders}) {session_subq} "
        f"GROUP BY day, source_machine",
        days + [account] + days,
    ):
        if r["day"] in day_data:
            day_data[r["day"]]["machine"][r["source_machine"]]["tool_calls"] += r["cnt"]

    # ── Q4: Prompt→model attribution ──
    pending_prompts = 0
    current_session = None
    for r in conn.execute(
        f"SELECT session_id, day, ts_epoch, type, is_human_prompt, model, provider, source_client "
        f"FROM events "
        f"WHERE is_sidechain=0 AND agent_id IS NULL "
        f"AND ("
        f"  (type='user' AND is_human_prompt=1) OR "
        f"  (type='assistant' AND model IS NOT NULL AND model != '<synthetic>')"
        f") AND day IN ({placeholders}) {acct_filter} "
        f"ORDER BY session_id, ts_epoch",
        days + [account],
    ):
        if r["session_id"] != current_session:
            pending_prompts = 0
            current_session = r["session_id"]
        if r["is_human_prompt"]:
            pending_prompts += 1
        elif r["type"] == "assistant":
            dm = display_model_for_row(r["model"], r["provider"], r["source_client"])
            d = r["day"]
            if d in day_data:
                day_data[d]["prompt_model"][dm] += pending_prompts
                day_data[d]["model"][dm]["main_prompts"] += pending_prompts
            pending_prompts = 0

    # ── Q5: Main session active time gaps ──
    # For day-boundary accuracy, also fetch the last event before the earliest
    # target day for each session that spans the boundary.
    earliest_day = min(days)
    prev_day = (datetime.strptime(earliest_day, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    # Get sessions (for this account) that have events on target days
    target_sessions = conn.execute(
        f"SELECT DISTINCT session_id FROM events "
        f"WHERE is_sidechain=0 AND agent_id IS NULL AND type IN ('user','assistant') "
        f"AND day IN ({placeholders}) {acct_filter}",
        days + [account],
    ).fetchall()
    target_session_ids = {r["session_id"] for r in target_sessions}

    # Build context: last event per session from the day before earliest target
    session_context: dict[str, dict] = {}
    if target_session_ids:
        ctx_placeholders = ",".join("?" for _ in target_session_ids)
        for r in conn.execute(
            f"SELECT session_id, project_dir, day, ts_epoch, type, model, provider, source_client, "
            f"has_tool_use, has_tool_result "
            f"FROM events "
            f"WHERE is_sidechain=0 AND agent_id IS NULL "
            f"AND type IN ('user','assistant') "
            f"AND day = ? AND session_id IN ({ctx_placeholders}) "
            f"AND COALESCE(account_email,'unknown') = ? "
            f"ORDER BY session_id, ts_epoch",
            [prev_day] + list(target_session_ids) + [account],
        ):
            # Keep overwriting — last event per session on prev_day
            session_context[r["session_id"]] = dict(r)

    prev_main = None
    for r in conn.execute(
        f"SELECT session_id, project_dir, day, ts_epoch, type, model, provider, source_client, "
        f"has_tool_use, has_tool_result "
        f"FROM events "
        f"WHERE is_sidechain=0 AND agent_id IS NULL "
        f"AND type IN ('user','assistant') "
        f"AND day IN ({placeholders}) {acct_filter} "
        f"ORDER BY session_id, ts_epoch",
        days + [account],
    ):
        if prev_main is None or prev_main["session_id"] != r["session_id"]:
            # Seed with context from previous day if available
            prev_main = session_context.get(r["session_id"])

        if prev_main and prev_main["session_id"] == r["session_id"]:
            gap = r["ts_epoch"] - prev_main["ts_epoch"]
            if 0 < gap < IDLE_THRESHOLD_S:
                d = r["day"]
                if d in day_data:
                    dd = day_data[d]
                    proj_dir = r["project_dir"] or "unknown"
                    dd["active_s"] += gap
                    dd["project"][proj_dir]["seconds"] += gap

                    gap_model = None
                    if prev_main["type"] == "assistant":
                        pm = prev_main["model"] or ""
                        if pm and pm != "<synthetic>":
                            gap_model = display_model_for_row(
                                pm, prev_main["provider"],
                                prev_main["source_client"])
                    if not gap_model and r["type"] == "assistant":
                        cm = r["model"] or ""
                        if cm and cm != "<synthetic>":
                            gap_model = display_model_for_row(
                                cm, r["provider"], r["source_client"])
                    if gap_model:
                        dd["model"][gap_model]["active_s"] += gap

                    is_te = (prev_main["type"] == "assistant" and prev_main["has_tool_use"]
                             and r["type"] == "user" and r["has_tool_result"])
                    if is_te:
                        dd["tool_exec_s"] += gap
                    else:
                        dd["thinking_s"] += gap

        prev_main = dict(r)

    # ── Q6: Subagent active time gaps ──
    agent_ids_seen: dict[str, set] = defaultdict(set)
    prev_sub = None
    for r in conn.execute(
        f"SELECT agent_id, day, ts_epoch, type, model, provider, source_client "
        f"FROM events "
        f"WHERE agent_id IS NOT NULL AND type IN ('user','assistant') "
        f"AND day IN ({placeholders}) {acct_filter} "
        f"ORDER BY agent_id, ts_epoch",
        days + [account],
    ):
        if prev_sub and prev_sub["agent_id"] == r["agent_id"]:
            gap = r["ts_epoch"] - prev_sub["ts_epoch"]
            if 0 < gap < IDLE_THRESHOLD_S:
                d = r["day"]
                if d in day_data:
                    day_data[d]["subagent_s"] += gap
                    agent_ids_seen[d].add(r["agent_id"])

                    gap_model = None
                    if prev_sub["type"] == "assistant":
                        pm = prev_sub["model"] or ""
                        if pm and pm != "<synthetic>":
                            gap_model = display_model_for_row(
                                pm, prev_sub["provider"], prev_sub["source_client"])
                    if not gap_model and r["type"] == "assistant":
                        cm = r["model"] or ""
                        if cm and cm != "<synthetic>":
                            gap_model = display_model_for_row(
                                cm, r["provider"], r["source_client"])
                    if gap_model:
                        day_data[d]["model"][gap_model]["active_s"] += gap
        prev_sub = r

    for d in days:
        if d in day_data:
            day_data[d]["agent_runs"] = len(agent_ids_seen.get(d, set()))

    # ── Q7: Agent invocations per model per day ──
    for r in conn.execute(
        f"SELECT day, agent_id, model, provider, source_client FROM events "
        f"WHERE agent_id IS NOT NULL AND type='assistant' "
        f"AND model IS NOT NULL AND model != '<synthetic>' "
        f"AND day IN ({placeholders}) {acct_filter} "
        f"GROUP BY day, agent_id, model, provider, source_client",
        days + [account],
    ):
        d = r["day"]
        if d in day_data:
            dm = display_model_for_row(r["model"], r["provider"], r["source_client"])
            day_data[d]["model"][dm]["agent_invocations"] += 1

    # ── Q8: Generation time ──
    main_user_ts: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for r in conn.execute(
        f"SELECT session_id, ts_epoch, day FROM events "
        f"WHERE type='user' AND is_sidechain=0 AND agent_id IS NULL "
        f"AND day IN ({placeholders}) {acct_filter} "
        f"ORDER BY session_id, ts_epoch",
        days + [account],
    ):
        main_user_ts[r["session_id"]].append((r["ts_epoch"], r["day"]))

    agent_user_ts: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for r in conn.execute(
        f"SELECT agent_id, ts_epoch, day FROM events "
        f"WHERE type='user' AND agent_id IS NOT NULL "
        f"AND day IN ({placeholders}) {acct_filter} "
        f"ORDER BY agent_id, ts_epoch",
        days + [account],
    ):
        agent_user_ts[r["agent_id"]].append((r["ts_epoch"], r["day"]))

    for r in requests:
        out_tok = r["out"] or 0
        if out_tok < 50 or not r["model"]:
            continue
        d = r["day"]
        if d not in day_data:
            continue
        aid = r["agent_id"]
        if aid:
            ts_list = agent_user_ts.get(aid)
        else:
            ts_list = main_user_ts.get(r["session_id"])
        if not ts_list:
            continue
        epochs = [t[0] for t in ts_list]
        idx = bisect_right(epochs, r["first_ts"])
        if idx == 0:
            continue
        preceding_user_ts = epochs[idx - 1]
        gen_time = r["last_ts"] - preceding_user_ts
        if gen_time < 0.5 or gen_time > 120:
            continue
        dm = display_model_for_row(r["model"], r["provider"], r["source_client"])
        day_data[d]["gen"][dm]["gen_s"] += gen_time
        day_data[d]["gen"][dm]["gen_out"] += out_tok

    return day_data


def summarize_days(days: list[str] | None = None):
    """Recompute daily_summary rows for the given days (or all days if None).

    Each (day, account_email) pair gets a self-contained summary row derived
    from the events and tool_uses tables.  The result is written to the
    daily_summary table via DELETE + INSERT.
    """
    load_pricing()
    conn = get_conn()

    if days is None:
        days = [r["day"] for r in conn.execute(
            "SELECT DISTINCT day FROM events ORDER BY day"
        )]

    if not days:
        return

    placeholders = ",".join("?" for _ in days)

    # Discover all (day, account) pairs with their per-day plan/org.
    # Using MAX(plan) per (day, acct) instead of across all days prevents a
    # cross-day lexicographic-MAX from poisoning earlier/later days' plan stamp.
    acct_day_rows = conn.execute(
        f"SELECT day, COALESCE(account_email,'unknown') AS acct, "
        f"MAX(plan) AS plan, MAX(org_name) AS org, "
        f"MAX(org_type) AS org_type, MAX(org_uuid) AS org_uuid "
        f"FROM events WHERE day IN ({placeholders}) GROUP BY day, acct",
        days,
    ).fetchall()

    # Build (day, acct) -> (plan, org, org_type, org_uuid) lookup and unique account set
    day_acct_meta: dict[tuple, tuple] = {}
    accounts_set: set[str] = set()
    for r in acct_day_rows:
        day_acct_meta[(r["day"], r["acct"])] = (r["plan"], r["org"], r["org_type"], r["org_uuid"])
        accounts_set.add(r["acct"])
    accounts = list(accounts_set)

    if not accounts:
        return

    # Phase 1 — READ ONLY: accumulate every account's rows before touching
    # the table. The 2026-07-12 incident: DELETE ran first, minutes of
    # accumulate followed, and a concurrent summarize's INSERT landed inside
    # that window — the final INSERT died on UNIQUE(day, account_email) with
    # the DELETE already persisted by other threads' commits.
    now = datetime.now(TZ).isoformat()
    rows = []

    for account in accounts:
        day_data = _accumulate(conn, days, placeholders, account)
        for d in days:
            dd = day_data.get(d)
            if dd is None:
                continue
            # Skip days with zero activity for this account
            if (dd["cost"] == 0 and dd["sessions"] == 0 and dd["human_prompts"] == 0
                    and dd["tool_calls"] == 0 and dd["active_s"] == 0):
                continue
            # Use per-day plan/org/org_type/org_uuid (not cross-day MAX)
            plan, org, org_type, org_uuid = day_acct_meta.get(
                (d, account), (None, None, None, None))
            rows.append((
                d, account, plan, org, org_type, org_uuid,
                dd["sessions"], dd["human_prompts"],
                dd["tool_calls"], dd["input_tokens"], dd["output_tokens"],
                dd["cache_creation_tokens"], dd["cache_read_tokens"],
                dd["active_s"], dd["thinking_s"], dd["tool_exec_s"],
                dd["subagent_s"], dd["agent_runs"], dd["cost"],
                json.dumps({k: dict(v) for k, v in dd["model"].items()}),
                json.dumps({k: dict(v) for k, v in dd["project"].items()}),
                json.dumps({k: dict(v) for k, v in dd["machine"].items()}),
                json.dumps(dict(dd["tool"])),
                json.dumps(dict(dd["prompt_model"])),
                json.dumps({k: dict(v) for k, v in dd["gen"].items()}),
                now,
            ))

    # Phase 2 — atomic destructive write, serialized by the write lock:
    # DELETE and re-INSERT commit together, so a concurrent summarize can
    # neither abort this rebuild nor observe (and persist) a half-rebuilt
    # table. OR REPLACE is belt-and-braces for any writer not yet routed
    # through write_txn.
    with write_txn() as wconn:
        wconn.execute(
            f"DELETE FROM daily_summary WHERE day IN ({placeholders})", days)
        wconn.executemany(
            "INSERT OR REPLACE INTO daily_summary "
            "(day, account_email, plan, org_name, org_type, org_uuid, sessions, "
            "human_prompts, tool_calls, input_tokens, output_tokens, "
            "cache_creation_tokens, cache_read_tokens, "
            "active_s, thinking_s, tool_exec_s, subagent_s, agent_runs, cost, "
            "model_json, project_json, machine_json, tool_json, prompt_model_json, "
            "gen_json, updated_at) VALUES (" + ",".join("?" * 26) + ")",
            rows,
        )
