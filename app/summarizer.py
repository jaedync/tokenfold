"""Compute daily_summary rows from raw events for specific days."""

import json
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .config import IDLE_THRESHOLD_S, TZ_NAME
from .db import get_conn
from .pricing import compute_cost, display_model, load_pricing

TZ = ZoneInfo(TZ_NAME)


def summarize_days(days: list[str] | None = None):
    """Recompute daily_summary rows for the given days (or all days if None).

    Each day gets a self-contained summary row derived from the events and
    tool_uses tables, scoped to that day.  The result is written to the
    daily_summary table via INSERT OR REPLACE.
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

    # ── Tool counts per day ──
    daily_tool_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    daily_tool_totals: dict[str, int] = defaultdict(int)
    for r in conn.execute(
        f"SELECT day, name, COUNT(*) as cnt FROM tool_uses "
        f"WHERE day IN ({placeholders}) GROUP BY day, name",
        days,
    ):
        daily_tool_counts[r["day"]][r["name"]] = r["cnt"]
        daily_tool_totals[r["day"]] += r["cnt"]

    # ── Q1: Token dedup per request_id, scoped to target days ──
    requests = conn.execute(
        f"SELECT request_id, COALESCE(project_dir,'unknown') as project_dir, "
        f"source_machine, session_id, day, model, is_sidechain, agent_id, "
        f"MAX(input_tokens) as inp, MAX(output_tokens) as out, "
        f"MAX(cache_creation_tokens) as cc, MAX(cache_read_tokens) as cr, "
        f"MIN(ts_epoch) as first_ts, MAX(ts_epoch) as last_ts "
        f"FROM events "
        f"WHERE type='assistant' AND model IS NOT NULL AND model != '<synthetic>' "
        f"AND request_id IS NOT NULL AND day IN ({placeholders}) "
        f"GROUP BY request_id",
        days,
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
                "api_calls": 0, "main_api_calls": 0, "main_cost": 0.0,
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
        dm = display_model(r["model"])
        machine = r["source_machine"]

        dd["input_tokens"] += inp
        dd["output_tokens"] += out
        dd["cache_creation_tokens"] += cc
        dd["cache_read_tokens"] += cr

        req_cost = compute_cost(dm, inp, out, cc, cr)
        dd["cost"] += req_cost
        dd["project"][r["project_dir"]]["cost"] += req_cost

        ms = dd["model"][dm]
        ms["input"] += inp
        ms["output"] += out
        ms["cache_write"] += cc
        ms["cache_read"] += cr
        ms["api_calls"] += 1
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
        f"FROM events WHERE agent_id IS NULL AND day IN ({placeholders}) GROUP BY day",
        days,
    ):
        if r["day"] in day_data:
            day_data[r["day"]]["sessions"] = r["cnt"]

    # ── Q3: Human prompts per day ──
    for r in conn.execute(
        f"SELECT day, COUNT(*) as cnt "
        f"FROM events WHERE is_human_prompt=1 AND is_sidechain=0 AND day IN ({placeholders}) GROUP BY day",
        days,
    ):
        if r["day"] in day_data:
            day_data[r["day"]]["human_prompts"] = r["cnt"]

    # ── Machine prompts per day ──
    for r in conn.execute(
        f"SELECT day, source_machine, COUNT(*) as cnt "
        f"FROM events WHERE is_human_prompt=1 AND day IN ({placeholders}) GROUP BY day, source_machine",
        days,
    ):
        if r["day"] in day_data:
            day_data[r["day"]]["machine"][r["source_machine"]]["prompts"] += r["cnt"]

    # ── Machine tool calls per day ──
    for r in conn.execute(
        f"SELECT day, source_machine, COUNT(*) as cnt "
        f"FROM tool_uses WHERE day IN ({placeholders}) GROUP BY day, source_machine",
        days,
    ):
        if r["day"] in day_data:
            day_data[r["day"]]["machine"][r["source_machine"]]["tool_calls"] += r["cnt"]

    # ── Q4: Prompt→model attribution ──
    pending_prompts = 0
    current_session = None
    for r in conn.execute(
        f"SELECT session_id, day, ts_epoch, type, is_human_prompt, model "
        f"FROM events "
        f"WHERE is_sidechain=0 AND agent_id IS NULL "
        f"AND ("
        f"  (type='user' AND is_human_prompt=1) OR "
        f"  (type='assistant' AND model IS NOT NULL AND model != '<synthetic>')"
        f") AND day IN ({placeholders}) "
        f"ORDER BY session_id, ts_epoch",
        days,
    ):
        if r["session_id"] != current_session:
            pending_prompts = 0
            current_session = r["session_id"]
        if r["is_human_prompt"]:
            pending_prompts += 1
        elif r["type"] == "assistant":
            dm = display_model(r["model"])
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

    # Get sessions that have events on target days
    target_sessions = conn.execute(
        f"SELECT DISTINCT session_id FROM events "
        f"WHERE is_sidechain=0 AND agent_id IS NULL AND type IN ('user','assistant') "
        f"AND day IN ({placeholders})",
        days,
    ).fetchall()
    target_session_ids = {r["session_id"] for r in target_sessions}

    # Build context: last event per session from the day before earliest target
    session_context: dict[str, dict] = {}
    if target_session_ids:
        ctx_placeholders = ",".join("?" for _ in target_session_ids)
        for r in conn.execute(
            f"SELECT session_id, project_dir, day, ts_epoch, type, model, "
            f"has_tool_use, has_tool_result "
            f"FROM events "
            f"WHERE is_sidechain=0 AND agent_id IS NULL "
            f"AND type IN ('user','assistant') "
            f"AND day = ? AND session_id IN ({ctx_placeholders}) "
            f"ORDER BY session_id, ts_epoch",
            [prev_day] + list(target_session_ids),
        ):
            # Keep overwriting — last event per session on prev_day
            session_context[r["session_id"]] = dict(r)

    prev_main = None
    for r in conn.execute(
        f"SELECT session_id, project_dir, day, ts_epoch, type, model, "
        f"has_tool_use, has_tool_result "
        f"FROM events "
        f"WHERE is_sidechain=0 AND agent_id IS NULL "
        f"AND type IN ('user','assistant') "
        f"AND day IN ({placeholders}) "
        f"ORDER BY session_id, ts_epoch",
        days,
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
                            gap_model = display_model(pm)
                    if not gap_model and r["type"] == "assistant":
                        cm = r["model"] or ""
                        if cm and cm != "<synthetic>":
                            gap_model = display_model(cm)
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
        f"SELECT agent_id, day, ts_epoch, type, model "
        f"FROM events "
        f"WHERE agent_id IS NOT NULL AND type IN ('user','assistant') "
        f"AND day IN ({placeholders}) "
        f"ORDER BY agent_id, ts_epoch",
        days,
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
                            gap_model = display_model(pm)
                    if not gap_model and r["type"] == "assistant":
                        cm = r["model"] or ""
                        if cm and cm != "<synthetic>":
                            gap_model = display_model(cm)
                    if gap_model:
                        day_data[d]["model"][gap_model]["active_s"] += gap
        prev_sub = r

    for d in days:
        if d in day_data:
            day_data[d]["agent_runs"] = len(agent_ids_seen.get(d, set()))

    # ── Q7: Agent invocations per model per day ──
    for r in conn.execute(
        f"SELECT day, agent_id, model FROM events "
        f"WHERE agent_id IS NOT NULL AND type='assistant' "
        f"AND model IS NOT NULL AND model != '<synthetic>' "
        f"AND day IN ({placeholders}) "
        f"GROUP BY day, agent_id, model",
        days,
    ):
        d = r["day"]
        if d in day_data:
            dm = display_model(r["model"])
            day_data[d]["model"][dm]["agent_invocations"] += 1

    # ── Q8: Generation time ──
    main_user_ts: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for r in conn.execute(
        f"SELECT session_id, ts_epoch, day FROM events "
        f"WHERE type='user' AND is_sidechain=0 AND agent_id IS NULL "
        f"AND day IN ({placeholders}) "
        f"ORDER BY session_id, ts_epoch",
        days,
    ):
        main_user_ts[r["session_id"]].append((r["ts_epoch"], r["day"]))

    agent_user_ts: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for r in conn.execute(
        f"SELECT agent_id, ts_epoch, day FROM events "
        f"WHERE type='user' AND agent_id IS NOT NULL "
        f"AND day IN ({placeholders}) "
        f"ORDER BY agent_id, ts_epoch",
        days,
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
        dm = display_model(r["model"])
        day_data[d]["gen"][dm]["gen_s"] += gen_time
        day_data[d]["gen"][dm]["gen_out"] += out_tok

    # ── Write summary rows ──
    now = datetime.now(TZ).isoformat()
    rows = []
    for d in days:
        dd = day_data.get(d)
        if dd is None:
            continue
        rows.append((
            d,
            dd["sessions"], dd["human_prompts"], dd["tool_calls"],
            dd["input_tokens"], dd["output_tokens"],
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

    conn.executemany(
        "INSERT OR REPLACE INTO daily_summary "
        "(day, sessions, human_prompts, tool_calls, "
        "input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens, "
        "active_s, thinking_s, tool_exec_s, subagent_s, agent_runs, cost, "
        "model_json, project_json, machine_json, tool_json, prompt_model_json, gen_json, "
        "updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
