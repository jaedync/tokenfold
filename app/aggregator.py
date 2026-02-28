"""Rebuild aggregated data from events/tool_uses tables.

Produces the same JSON structure as generate-dashboard.py's parse_and_compute(),
querying from SQLite instead of walking JSONL files.
"""

import re
import threading
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .config import IDLE_THRESHOLD_S, RECENCY_DAYS, TZ_NAME
from .db import get_conn
from .pricing import (
    MODEL_BENCHMARKS, MODEL_ORDER, compute_cost, display_model, get_pricing,
    load_pricing, model_sort_key,
)

TZ = ZoneInfo(TZ_NAME)

# In-memory cache - rebuilt only after ingest or on first request
_cache_lock = threading.Lock()
_cached_data: dict | None = None
_cache_version: int = 0


# Patterns that identify a "home directory" prefix to strip.
# Linux: -home-user-   macOS: -Users-user-   Windows: C--Users-user-
# Mounts: -mnt-vol-
_PREFIX_RES = [
    re.compile(r"^-(?:home|Users)-[^-]+(?:-|$)"),   # Linux / macOS
    re.compile(r"^[A-Z]--Users-[^-]+(?:-|$)"),       # Windows (C--Users-X-)
    re.compile(r"^-mnt-[^-]+(?:-|$)"),                # mount paths
]
_STRUCTURAL_PARENTS = (
    "services-", "development-", "projects-", "code-", "work-",
    "repos-", "src-", "github-", "Desktop-", "Documents-", "Downloads-",
    "AppData-Local-", "clawd-",
)


def _strip_prefix(dir_name: str):
    """Strip the home/system prefix, returning (remainder, matched_regex) or None."""
    for regex in _PREFIX_RES:
        m = regex.match(dir_name)
        if m:
            return dir_name[m.end():], regex
    return None


def _leaf_project_name(dir_name: str) -> str:
    """Extract the leaf project name from a dash-encoded project path.

    ``-home-user-services-my-project``  → ``my-project``
    ``-Users-user-development-my-app``  → ``my-app``
    ``C--Users-User-Documents-my-code`` → ``my-code``
    """
    result = _strip_prefix(dir_name)
    if result is None:
        return dir_name.lstrip("-") or dir_name
    remainder = result[0]
    if not remainder:
        return "~ (home)"
    for parent in _STRUCTURAL_PARENTS:
        if remainder.startswith(parent):
            leaf = remainder[len(parent):]
            if leaf:
                return leaf
    return remainder


def _make_display_names(raw_dirs: list[str]) -> dict[str, str]:
    """Map raw project_dir keys → short display names, disambiguating collisions."""
    short = {d: _leaf_project_name(d) for d in raw_dirs}
    counts = Counter(short.values())
    for d in raw_dirs:
        if counts[short[d]] > 1:
            result = _strip_prefix(d)
            if result and result[0]:
                # Show as parent/leaf for disambiguation
                remainder = result[0]
                for parent in _STRUCTURAL_PARENTS:
                    if remainder.startswith(parent) and remainder[len(parent):]:
                        short[d] = parent.rstrip("-") + "/" + remainder[len(parent):]
                        break
                else:
                    short[d] = remainder
            else:
                # Home dir only - disambiguate by extracting username
                um = re.search(r"(?:home|Users)-([^-]+)", d)
                short[d] = "~ (" + um.group(1) + ")" if um else d.strip("-")
    return short


def invalidate_cache():
    """Clear cached dashboard data, forcing rebuild on next request."""
    global _cached_data, _cache_version
    with _cache_lock:
        _cached_data = None
        _cache_version += 1


def get_cache_version() -> int:
    """Return current cache version (incremented on each invalidation)."""
    with _cache_lock:
        return _cache_version


def aggregate_days(touched_days: set[str] | None = None):
    """Incremental stub - dashboard always queries live."""
    pass


def build_dashboard_data() -> dict:
    """Return cached dashboard data, rebuilding if invalidated."""
    global _cached_data
    with _cache_lock:
        if _cached_data is not None:
            return _cached_data
    # Build outside the lock to avoid blocking concurrent readers
    data = _build_dashboard_data_inner()
    with _cache_lock:
        _cached_data = data
    return data


def _build_dashboard_data_inner() -> dict:
    """Query events + tool_uses and produce the full dashboard JSON blob."""
    load_pricing()
    conn = get_conn()
    cutoff_date = (datetime.now(TZ) - timedelta(days=RECENCY_DAYS)).strftime("%Y-%m-%d")

    # ── Pre-aggregate what we can in SQL ──
    # Tool counts - all-time (top 20) and recent (top 20)
    tool_counts_rows = conn.execute(
        "SELECT name, COUNT(*) as cnt FROM tool_uses GROUP BY name ORDER BY cnt DESC LIMIT 20"
    ).fetchall()
    tool_counts = {r["name"]: r["cnt"] for r in tool_counts_rows}
    recent_tool_rows = conn.execute(
        "SELECT name, COUNT(*) as cnt FROM tool_uses WHERE day>=? GROUP BY name ORDER BY cnt DESC LIMIT 20",
        (cutoff_date,),
    ).fetchall()
    recent_tool_counts = {r["name"]: r["cnt"] for r in recent_tool_rows}

    # Tool counts by day+session (for daily tool_calls)
    daily_tool_counts = defaultdict(int)
    for r in conn.execute("SELECT day, COUNT(*) as cnt FROM tool_uses GROUP BY day"):
        daily_tool_counts[r["day"]] = r["cnt"]
    total_tool_calls = sum(daily_tool_counts.values())

    # Last active timestamp (most recent event epoch)
    last_active_row = conn.execute("SELECT MAX(ts_epoch) as ts FROM events").fetchone()
    last_active_ts = last_active_row["ts"] if last_active_row and last_active_row["ts"] else None

    # Source machines with last activity
    _machine_last_active = {}
    for r in conn.execute(
        "SELECT source_machine, MAX(ts_epoch) as last_ts FROM events GROUP BY source_machine"
    ):
        _machine_last_active[r["source_machine"]] = r["last_ts"]
    machine_list = list(_machine_last_active.keys())

    # ── Bulk SQL queries (replace per-session N+1 loop) ──

    daily = defaultdict(lambda: {
        "sessions": 0, "human_prompts": 0, "input_tokens": 0,
        "output_tokens": 0, "cache_creation_tokens": 0, "cache_read_tokens": 0,
        "tool_calls": 0, "active_s": 0.0, "thinking_s": 0.0,
        "tool_exec_s": 0.0, "cost": 0.0,
    })
    model_stats = defaultdict(lambda: {
        "input": 0, "output": 0, "cache_write": 0, "cache_read": 0,
        "api_calls": 0, "main_api_calls": 0, "main_cost": 0.0,
        "main_prompts": 0, "agent_invocations": 0, "active_s": 0.0,
        "gen_s": 0.0, "gen_out": 0,
        "recent_active_s": 0.0, "recent_gen_s": 0.0, "recent_gen_out": 0,
        "recent_input": 0, "recent_output": 0, "recent_cache_write": 0,
        "recent_cache_read": 0, "recent_main_cost": 0.0, "last_seen": "",
    })
    project_seconds = Counter()
    project_cost = Counter()
    recent_project_seconds = Counter()
    recent_project_cost = Counter()
    tot = {"thinking_s": 0.0, "tool_exec_s": 0.0, "active_s": 0.0,
           "subagent_s": 0.0, "agent_runs": 0,
           "recent_subagent_s": 0.0, "recent_agent_runs": 0,
           "recent_thinking_s": 0.0, "recent_tool_exec_s": 0.0,
           "tokens": 0, "human_prompts": 0, "tool_calls": total_tool_calls}
    models_seen = set()

    # ── Q1: Token dedup per request_id ──
    # Single query replaces the per-session req_data dict accumulation.
    # GROUP BY request_id with MAX() deduplicates streaming token repeats.
    requests = conn.execute(
        "SELECT request_id, COALESCE(project_dir,'unknown') as project_dir, "
        "source_machine, session_id, day, model, is_sidechain, agent_id, "
        "MAX(input_tokens) as inp, MAX(output_tokens) as out, "
        "MAX(cache_creation_tokens) as cc, MAX(cache_read_tokens) as cr, "
        "MIN(ts_epoch) as first_ts, MAX(ts_epoch) as last_ts "
        "FROM events "
        "WHERE type='assistant' AND model IS NOT NULL AND model != '<synthetic>' "
        "AND request_id IS NOT NULL "
        "GROUP BY request_id"
    ).fetchall()

    # Per-machine accumulators (built from Q1 to avoid separate queries)
    _mach_tokens = defaultdict(lambda: [0, 0, 0, 0, 0])  # inp,out,cc,cr,calls
    _mach_daily_cost = defaultdict(lambda: defaultdict(float))  # machine→day→cost
    _recent_mach_tokens = defaultdict(lambda: [0, 0, 0, 0, 0])
    _recent_mach_daily_cost = defaultdict(float)  # machine→cost

    for r in requests:
        inp, out = r["inp"] or 0, r["out"] or 0
        cc, cr = r["cc"] or 0, r["cr"] or 0
        day = r["day"]
        dm = display_model(r["model"])
        proj_dir = r["project_dir"]
        machine = r["source_machine"]

        daily[day]["input_tokens"] += inp
        daily[day]["output_tokens"] += out
        daily[day]["cache_creation_tokens"] += cc
        daily[day]["cache_read_tokens"] += cr
        tot["tokens"] += inp + out + cc + cr

        req_cost = compute_cost(dm, inp, out, cc, cr)
        daily[day]["cost"] += req_cost
        project_cost[proj_dir] += req_cost
        if day >= cutoff_date:
            recent_project_cost[proj_dir] += req_cost

        models_seen.add(dm)
        ms = model_stats[dm]
        ms["input"] += inp
        ms["output"] += out
        ms["cache_write"] += cc
        ms["cache_read"] += cr
        ms["api_calls"] += 1
        if not r["is_sidechain"]:
            ms["main_api_calls"] += 1
            ms["main_cost"] += req_cost
            if day >= cutoff_date:
                ms["recent_main_cost"] += req_cost
        if day >= cutoff_date:
            ms["recent_input"] += inp
            ms["recent_output"] += out
            ms["recent_cache_write"] += cc
            ms["recent_cache_read"] += cr
        if day > ms["last_seen"]:
            ms["last_seen"] = day

        # Machine stats (avoids separate per-machine SQL queries)
        mt = _mach_tokens[machine]
        mt[0] += inp; mt[1] += out; mt[2] += cc; mt[3] += cr; mt[4] += 1
        _mach_daily_cost[machine][day] += req_cost
        if day >= cutoff_date:
            rmt = _recent_mach_tokens[machine]
            rmt[0] += inp; rmt[1] += out; rmt[2] += cc; rmt[3] += cr; rmt[4] += 1
            _recent_mach_daily_cost[machine] += req_cost

    # ── Q2: Daily sessions + total sessions ──
    for r in conn.execute(
        "SELECT day, COUNT(DISTINCT session_id) as cnt "
        "FROM events WHERE agent_id IS NULL GROUP BY day"
    ):
        daily[r["day"]]["sessions"] = r["cnt"]

    sessions_count = (conn.execute(
        "SELECT COUNT(DISTINCT session_id) FROM events"
    ).fetchone()[0] or 0)

    # ── Q3: Human prompts per day ──
    for r in conn.execute(
        "SELECT day, COUNT(*) as cnt "
        "FROM events WHERE is_human_prompt=1 AND is_sidechain=0 GROUP BY day"
    ):
        daily[r["day"]]["human_prompts"] = r["cnt"]
        tot["human_prompts"] += r["cnt"]

    # ── Q4: Prompt→model attribution ──
    # For each human prompt, find the next assistant event's model in the same
    # session. Fetches all relevant events in one query (instead of per-session)
    # and iterates once to attribute prompts to models.
    pending_prompts = 0
    current_session = None
    for r in conn.execute(
        "SELECT session_id, ts_epoch, type, is_human_prompt, model "
        "FROM events "
        "WHERE is_sidechain=0 AND agent_id IS NULL "
        "AND ("
        "  (type='user' AND is_human_prompt=1) OR "
        "  (type='assistant' AND model IS NOT NULL AND model != '<synthetic>')"
        ") "
        "ORDER BY session_id, ts_epoch"
    ):
        if r["session_id"] != current_session:
            pending_prompts = 0
            current_session = r["session_id"]
        if r["is_human_prompt"]:
            pending_prompts += 1
        elif r["type"] == "assistant":
            dm = display_model(r["model"])
            model_stats[dm]["main_prompts"] += pending_prompts
            pending_prompts = 0

    # ── Q5: Main session active time gaps ──
    # Sorted fetch + Python-side gap computation (faster than LAG window).
    prev_main = None
    for r in conn.execute(
        "SELECT session_id, project_dir, day, ts_epoch, type, model, "
        "has_tool_use, has_tool_result "
        "FROM events "
        "WHERE is_sidechain=0 AND agent_id IS NULL "
        "AND type IN ('user','assistant') "
        "ORDER BY session_id, ts_epoch"
    ):
        if prev_main and prev_main["session_id"] == r["session_id"]:
            gap = r["ts_epoch"] - prev_main["ts_epoch"]
            if 0 < gap < IDLE_THRESHOLD_S:
                day = prev_main["day"]
                proj_dir = r["project_dir"] or "unknown"

                # Model attribution: prefer prev (if assistant), then curr
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
                    model_stats[gap_model]["active_s"] += gap
                    if day >= cutoff_date:
                        model_stats[gap_model]["recent_active_s"] += gap

                daily[day]["active_s"] += gap
                tot["active_s"] += gap
                project_seconds[proj_dir] += gap
                if day >= cutoff_date:
                    recent_project_seconds[proj_dir] += gap

                # Classify gap: tool execution vs thinking
                is_te = (prev_main["type"] == "assistant" and prev_main["has_tool_use"]
                         and r["type"] == "user" and r["has_tool_result"])
                if is_te:
                    daily[day]["tool_exec_s"] += gap
                    tot["tool_exec_s"] += gap
                    if day >= cutoff_date:
                        tot["recent_tool_exec_s"] += gap
                else:
                    daily[day]["thinking_s"] += gap
                    tot["thinking_s"] += gap
                    if day >= cutoff_date:
                        tot["recent_thinking_s"] += gap
        prev_main = r

    # ── Q6: Subagent active time gaps ──
    # Sorted fetch + Python-side gap computation (faster than LAG window).
    agent_has_recent = set()
    prev_sub = None
    for r in conn.execute(
        "SELECT agent_id, day, ts_epoch, type, model "
        "FROM events "
        "WHERE agent_id IS NOT NULL AND type IN ('user','assistant') "
        "ORDER BY agent_id, ts_epoch"
    ):
        if prev_sub and prev_sub["agent_id"] == r["agent_id"]:
            gap = r["ts_epoch"] - prev_sub["ts_epoch"]
            if 0 < gap < IDLE_THRESHOLD_S:
                day = prev_sub["day"]

                tot["subagent_s"] += gap
                if day >= cutoff_date:
                    tot["recent_subagent_s"] += gap
                    agent_has_recent.add(r["agent_id"])

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
                    model_stats[gap_model]["active_s"] += gap
                    if day >= cutoff_date:
                        model_stats[gap_model]["recent_active_s"] += gap
        prev_sub = r

    tot["agent_runs"] = (conn.execute(
        "SELECT COUNT(DISTINCT agent_id) FROM events "
        "WHERE agent_id IS NOT NULL AND type IN ('user','assistant')"
    ).fetchone()[0] or 0)
    tot["recent_agent_runs"] = len(agent_has_recent)

    # ── Q7: Agent invocations per model ──
    agent_model_pairs = set()
    for r in conn.execute(
        "SELECT agent_id, model FROM events "
        "WHERE agent_id IS NOT NULL AND type='assistant' "
        "AND model IS NOT NULL AND model != '<synthetic>' "
        "GROUP BY agent_id, model"
    ):
        dm = display_model(r["model"])
        pair = (r["agent_id"], dm)
        if pair not in agent_model_pairs:
            agent_model_pairs.add(pair)
            model_stats[dm]["agent_invocations"] += 1

    # ── Q8: Generation time ──
    # Build user timestamp indices for bisect lookup (two bulk queries).
    main_user_ts = defaultdict(list)
    for r in conn.execute(
        "SELECT session_id, ts_epoch FROM events "
        "WHERE type='user' AND is_sidechain=0 AND agent_id IS NULL "
        "ORDER BY session_id, ts_epoch"
    ):
        main_user_ts[r["session_id"]].append(r["ts_epoch"])

    agent_user_ts_map = defaultdict(list)
    for r in conn.execute(
        "SELECT agent_id, ts_epoch FROM events "
        "WHERE type='user' AND agent_id IS NOT NULL "
        "ORDER BY agent_id, ts_epoch"
    ):
        agent_user_ts_map[r["agent_id"]].append(r["ts_epoch"])

    for r in requests:
        out_tok = r["out"] or 0
        if out_tok < 50 or not r["model"]:
            continue
        aid = r["agent_id"]
        if aid:
            ts_list = agent_user_ts_map.get(aid)
        else:
            ts_list = main_user_ts.get(r["session_id"])
        if not ts_list:
            continue
        idx = bisect_right(ts_list, r["first_ts"])
        if idx == 0:
            continue
        preceding_user_ts = ts_list[idx - 1]
        gen_time = r["last_ts"] - preceding_user_ts
        if gen_time < 0.5 or gen_time > 120:
            continue
        dm = display_model(r["model"])
        model_stats[dm]["gen_s"] += gen_time
        model_stats[dm]["gen_out"] += out_tok
        if r["day"] >= cutoff_date:
            model_stats[dm]["recent_gen_s"] += gen_time
            model_stats[dm]["recent_gen_out"] += out_tok

    # Merge SQL-computed daily tool counts
    for day, cnt in daily_tool_counts.items():
        daily[day]["tool_calls"] = cnt

    # ── Build date range ──
    if daily:
        all_dates = sorted(daily.keys())
        start = datetime.strptime(all_dates[0], "%Y-%m-%d")
        end = datetime.strptime(all_dates[-1], "%Y-%m-%d")
        date_range = []
        cur = start
        while cur <= end:
            date_range.append(cur.strftime("%Y-%m-%d"))
            cur += timedelta(days=1)
    else:
        date_range = []

    num_days = len(date_range) or 1
    make_empty_day = lambda: {
        "sessions": 0, "human_prompts": 0, "input_tokens": 0,
        "output_tokens": 0, "cache_creation_tokens": 0, "cache_read_tokens": 0,
        "tool_calls": 0, "active_s": 0.0, "thinking_s": 0.0,
        "tool_exec_s": 0.0, "cost": 0.0,
    }
    daily_list = []
    for d in date_range:
        dd = daily.get(d, make_empty_day())
        daily_list.append({
            "date": d,
            "sessions": dd["sessions"],
            "prompts": dd["human_prompts"],
            "tool_calls": dd["tool_calls"],
            "active_minutes": round(dd["active_s"] / 60),
            "input_tokens": dd["input_tokens"],
            "output_tokens": dd["output_tokens"],
            "cache_creation_tokens": dd["cache_creation_tokens"],
            "cache_read_tokens": dd["cache_read_tokens"],
            "cost": round(dd["cost"], 2),
        })

    # ── Model breakdown ──
    total_cost = 0.0
    model_breakdown = []
    for name in sorted(model_stats, key=lambda m: model_sort_key(m)):
        ms = model_stats[name]
        total_tok = ms["input"] + ms["output"] + ms["cache_write"] + ms["cache_read"]
        cost = compute_cost(name, ms["input"], ms["output"], ms["cache_write"], ms["cache_read"])
        p = get_pricing(name)
        total_cost += cost
        main_cost = round(ms["main_cost"], 2)
        agent_cost = round(cost - ms["main_cost"], 2)
        avg_cost_per_turn = (ms["main_cost"] / ms["main_prompts"]
                             if ms["main_prompts"] > 0 else None)
        avg_cost_per_agent = (agent_cost / ms["agent_invocations"]
                              if ms["agent_invocations"] > 0 else None)
        active_hours = ms["active_s"] / 3600
        recent_cost = compute_cost(name, ms["recent_input"], ms["recent_output"],
                                   ms["recent_cache_write"], ms["recent_cache_read"])
        recent_hours = ms["recent_active_s"] / 3600
        if recent_hours >= 0.5:
            cost_per_hour = recent_cost / recent_hours
        elif active_hours > 0:
            cost_per_hour = cost / active_hours
        else:
            cost_per_hour = None
        if ms["recent_gen_s"] > 0:
            output_tok_per_s = ms["recent_gen_out"] / ms["recent_gen_s"]
        elif ms["gen_s"] > 0:
            output_tok_per_s = ms["gen_out"] / ms["gen_s"]
        else:
            output_tok_per_s = None
        recent_active_hours = recent_hours
        recent_cost_per_hour = (recent_cost / recent_hours) if recent_hours >= 0.5 else None
        all_cost_per_hour = (cost / active_hours) if active_hours > 0 else None
        recent_output_tok_per_s = (ms["recent_gen_out"] / ms["recent_gen_s"]) if ms["recent_gen_s"] > 0 else None
        all_output_tok_per_s = (ms["gen_out"] / ms["gen_s"]) if ms["gen_s"] > 0 else None
        model_breakdown.append({
            "model": name, "api_calls": ms["api_calls"],
            "input": ms["input"], "output": ms["output"],
            "cache_write": ms["cache_write"], "cache_read": ms["cache_read"],
            "total_tokens": total_tok, "cost": round(cost, 2),
            "main_cost": main_cost, "agent_cost": agent_cost,
            "avg_cost_per_turn": round(avg_cost_per_turn, 4) if avg_cost_per_turn is not None else None,
            "avg_cost_per_agent": round(avg_cost_per_agent, 4) if avg_cost_per_agent is not None else None,
            "main_prompts": ms["main_prompts"],
            "agent_invocations": ms["agent_invocations"],
            "active_hours": round(active_hours, 1),
            "cost_per_hour": round(cost_per_hour, 2) if cost_per_hour is not None else None,
            "output_tok_per_s": round(output_tok_per_s, 1) if output_tok_per_s is not None else None,
            "cost_input": round(ms["input"] * p[0] / 1e6, 2),
            "cost_output": round(ms["output"] * p[1] / 1e6, 2),
            "cost_cache_write": round(ms["cache_write"] * p[2] / 1e6, 2),
            "cost_cache_read": round(ms["cache_read"] * p[3] / 1e6, 2),
            "last_seen": ms["last_seen"],
            "recent": ms["last_seen"] >= cutoff_date,
            "recent_input": ms["recent_input"], "recent_output": ms["recent_output"],
            "recent_cache_write": ms["recent_cache_write"], "recent_cache_read": ms["recent_cache_read"],
            "recent_total_tokens": ms["recent_input"] + ms["recent_output"] + ms["recent_cache_write"] + ms["recent_cache_read"],
            "recent_cost": round(recent_cost, 2),
            "recent_main_cost": round(ms["recent_main_cost"], 2),
            "recent_agent_cost": round(recent_cost - ms["recent_main_cost"], 2),
            "recent_cost_input": round(ms["recent_input"] * p[0] / 1e6, 2),
            "recent_cost_output": round(ms["recent_output"] * p[1] / 1e6, 2),
            "recent_cost_cache_write": round(ms["recent_cache_write"] * p[2] / 1e6, 2),
            "recent_cost_cache_read": round(ms["recent_cache_read"] * p[3] / 1e6, 2),
            "recent_active_hours": round(recent_active_hours, 1),
            "recent_cost_per_hour": round(recent_cost_per_hour, 2) if recent_cost_per_hour is not None else None,
            "recent_output_tok_per_s": round(recent_output_tok_per_s, 1) if recent_output_tok_per_s is not None else None,
            "all_cost_per_hour": round(all_cost_per_hour, 2) if all_cost_per_hour is not None else None,
            "all_output_tok_per_s": round(all_output_tok_per_s, 1) if all_output_tok_per_s is not None else None,
        })

    models_donut = {
        name: ms["input"] + ms["output"] + ms["cache_write"] + ms["cache_read"]
        for name, ms in sorted(model_stats.items(),
            key=lambda kv: -(kv[1]["input"] + kv[1]["output"] +
                             kv[1]["cache_write"] + kv[1]["cache_read"]))
    }

    # ── Per-machine stats (derived from Q1 pre-computed accumulators) ──
    machine_stats = {}
    for m, mt in _mach_tokens.items():
        machine_stats[m] = {
            "api_calls": mt[4],
            "input_tokens": mt[0], "output_tokens": mt[1],
            "cache_creation_tokens": mt[2], "cache_read_tokens": mt[3],
        }
    # Prompts per machine (lightweight - one event per prompt, no dedup)
    for r in conn.execute(
        "SELECT source_machine, COUNT(*) as prompts "
        "FROM events WHERE is_human_prompt=1 GROUP BY source_machine"
    ):
        m = r["source_machine"]
        if m in machine_stats:
            machine_stats[m]["prompts"] = r["prompts"]
        else:
            machine_stats[m] = {"api_calls": 0, "prompts": r["prompts"],
                                "input_tokens": 0, "output_tokens": 0,
                                "cache_creation_tokens": 0, "cache_read_tokens": 0}
    # Tool calls per machine
    for r in conn.execute(
        "SELECT source_machine, COUNT(*) as cnt FROM tool_uses GROUP BY source_machine"
    ):
        m = r["source_machine"]
        if m in machine_stats:
            machine_stats[m]["tool_calls"] = r["cnt"]
    machine_daily = _mach_daily_cost  # already computed in Q1 loop
    # Build machine summary list
    machine_summary = []
    for m_name in sorted(machine_stats, key=lambda x: -(machine_stats[x].get("prompts", 0))):
        ms = machine_stats[m_name]
        total_tok = ms["input_tokens"] + ms["output_tokens"] + ms["cache_creation_tokens"] + ms["cache_read_tokens"]
        m_cost = sum(machine_daily.get(m_name, {}).values())
        machine_summary.append({
            "machine": m_name,
            "prompts": ms.get("prompts", 0),
            "api_calls": ms["api_calls"],
            "tool_calls": ms.get("tool_calls", 0),
            "total_tokens": total_tok,
            "cost": round(m_cost, 2),
        })
    # ── Recent per-machine stats (derived from Q1 pre-computed accumulators) ──
    recent_machine_stats = {}
    for m, rmt in _recent_mach_tokens.items():
        recent_machine_stats[m] = {
            "api_calls": rmt[4],
            "input_tokens": rmt[0], "output_tokens": rmt[1],
            "cache_creation_tokens": rmt[2], "cache_read_tokens": rmt[3],
        }
    for r in conn.execute(
        "SELECT source_machine, COUNT(*) as prompts "
        "FROM events WHERE is_human_prompt=1 AND day>=? GROUP BY source_machine",
        (cutoff_date,),
    ):
        m = r["source_machine"]
        if m in recent_machine_stats:
            recent_machine_stats[m]["prompts"] = r["prompts"]
        else:
            recent_machine_stats[m] = {"api_calls": 0, "prompts": r["prompts"],
                                        "input_tokens": 0, "output_tokens": 0,
                                        "cache_creation_tokens": 0, "cache_read_tokens": 0}
    for r in conn.execute(
        "SELECT source_machine, COUNT(*) as cnt FROM tool_uses WHERE day>=? GROUP BY source_machine",
        (cutoff_date,),
    ):
        m = r["source_machine"]
        if m in recent_machine_stats:
            recent_machine_stats[m]["tool_calls"] = r["cnt"]
    recent_machine_summary = []
    for m_name in sorted(recent_machine_stats, key=lambda x: -(recent_machine_stats[x].get("prompts", 0))):
        rms = recent_machine_stats[m_name]
        total_tok = rms["input_tokens"] + rms["output_tokens"] + rms["cache_creation_tokens"] + rms["cache_read_tokens"]
        m_cost = _recent_mach_daily_cost.get(m_name, 0.0)
        recent_machine_summary.append({
            "machine": m_name,
            "prompts": rms.get("prompts", 0),
            "api_calls": rms["api_calls"],
            "tool_calls": rms.get("tool_calls", 0),
            "total_tokens": total_tok,
            "cost": round(m_cost, 2),
        })

    # Build machine daily series for stacked chart
    machine_daily_series = {}
    for m_name in machine_stats:
        series = []
        for d in date_range:
            series.append(round(machine_daily.get(m_name, {}).get(d, 0.0), 2))
        machine_daily_series[m_name] = series

    # ── Hourly activity: 24h grid anchored at 1am ──
    # Today's hours (1am → current) = blue, yesterday fills the rest = yellow.
    # Grid always runs 1am → 12am (24 cells).  Each cell pulls from today
    # if that hour has occurred, otherwise from yesterday.
    now_h = datetime.now(TZ)
    current_hour_num = now_h.hour  # 0-23
    today = now_h.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(days=1)
    N_HOURS = 24

    hourly_list = []
    for slot in range(N_HOURS):
        clock_hour = slot  # slot 0 = 12am, slot 23 = 11pm
        is_today = clock_hour <= current_hour_num
        h = (today if is_today else yesterday) + timedelta(hours=clock_hour)
        hn = clock_hour % 12 or 12
        hourly_list.append({
            "label": f"{hn}{'a' if clock_hour < 12 else 'p'}",
            "date": h.strftime("%Y-%m-%d"),
            "day_short": h.strftime("%a").upper(),
            "prompts": 0, "tool_calls": 0, "cost": 0.0,
            "is_now": is_today and clock_hour == current_hour_num,
            "is_future": False,
            "period": "recent" if is_today else "past",
            "_epoch": h.timestamp(),
        })

    # Query window spans all referenced hours
    all_epochs = [hl["_epoch"] for hl in hourly_list]
    h_start_epoch = min(all_epochs)
    h_end_epoch = max(all_epochs) + 3600
    # Build epoch→index lookup for mapping query results back
    epoch_to_idx = {hl["_epoch"]: i for i, hl in enumerate(hourly_list)}

    # Helper: map a raw epoch from SQL to the hourly_list index
    def _epoch_to_slot(raw_epoch):
        floor_ep = int(raw_epoch // 3600) * 3600
        return epoch_to_idx.get(floor_ep)

    for r in conn.execute(
        "SELECT CAST(ts_epoch / 3600 AS INTEGER) * 3600 as bucket, COUNT(*) as cnt "
        "FROM events WHERE is_human_prompt=1 AND ts_epoch>=? AND ts_epoch<? "
        "GROUP BY bucket", (h_start_epoch, h_end_epoch),
    ):
        idx = epoch_to_idx.get(r["bucket"])
        if idx is not None:
            hourly_list[idx]["prompts"] = r["cnt"]

    for r in conn.execute(
        "SELECT CAST(ts_epoch / 3600 AS INTEGER) * 3600 as bucket, COUNT(*) as cnt "
        "FROM tool_uses WHERE ts_epoch>=? AND ts_epoch<? "
        "GROUP BY bucket", (h_start_epoch, h_end_epoch),
    ):
        idx = epoch_to_idx.get(r["bucket"])
        if idx is not None:
            hourly_list[idx]["tool_calls"] = r["cnt"]

    for r in conn.execute(
        "SELECT CAST(first_ts / 3600 AS INTEGER) * 3600 as bucket, model, "
        "SUM(inp) as inp, SUM(outp) as outp, SUM(cc) as cc, SUM(cr) as cr "
        "FROM ("
        "  SELECT MIN(ts_epoch) as first_ts, model, request_id, "
        "  MAX(input_tokens) as inp, MAX(output_tokens) as outp, "
        "  MAX(cache_creation_tokens) as cc, MAX(cache_read_tokens) as cr "
        "  FROM events WHERE type='assistant' AND model IS NOT NULL "
        "  AND model != '<synthetic>' AND request_id IS NOT NULL "
        "  AND ts_epoch>=? AND ts_epoch<? "
        "  GROUP BY model, request_id"
        ") GROUP BY bucket, model",
        (h_start_epoch, h_end_epoch),
    ):
        idx = epoch_to_idx.get(r["bucket"])
        if idx is not None:
            dm = display_model(r["model"])
            hourly_list[idx]["cost"] += compute_cost(
                dm, r["inp"] or 0, r["outp"] or 0, r["cc"] or 0, r["cr"] or 0)

    for hl in hourly_list:
        hl["cost"] = round(hl["cost"], 2)
        del hl["_epoch"]  # strip internal field

    # ── Weekly usage budget gauge (from OAuth API data) ──
    import json as _json
    oauth_row = conn.execute(
        "SELECT value FROM meta WHERE key='oauth_usage'"
    ).fetchone()
    weekly_budget = None
    if oauth_row:
        try:
            stored = _json.loads(oauth_row["value"])
            usage = stored.get("data", {})
            updated_at = stored.get("updated_at", "")

            # Build gauge data from real Anthropic utilization
            seven_day = usage.get("seven_day") or {}
            five_hour = usage.get("five_hour") or {}
            seven_day_sonnet = usage.get("seven_day_sonnet") or {}
            seven_day_opus = usage.get("seven_day_opus") or {}
            extra = usage.get("extra_usage") or {}

            resets_at_iso = seven_day.get("resets_at", "")
            weekly_budget = {
                "source": "oauth",
                "weekly_pct": seven_day.get("utilization", 0),
                "weekly_resets_at": resets_at_iso,
                "five_hour_pct": five_hour.get("utilization", 0),
                "five_hour_resets_at": five_hour.get("resets_at", ""),
                "sonnet_pct": seven_day_sonnet.get("utilization", 0) if seven_day_sonnet else None,
                "opus_pct": seven_day_opus.get("utilization", 0) if seven_day_opus else None,
                "extra_usage": {
                    "enabled": extra.get("is_enabled", False),
                    "monthly_limit_cents": extra.get("monthly_limit", 0),
                    "used_cents": extra.get("used_credits", 0),
                    "pct": extra.get("utilization", 0),
                } if extra else None,
                "updated_at": updated_at,
            }

            # ── Precise weekly window stats (cost + active time) ──
            # Compute from raw events using exact epoch boundaries instead
            # of daily buckets, so partial-day boundaries are handled correctly.
            if resets_at_iso:
                try:
                    reset_dt = datetime.fromisoformat(resets_at_iso.replace("Z", "+00:00"))
                    reset_epoch = reset_dt.timestamp()
                    week_start_epoch = reset_epoch - 7 * 24 * 3600

                    # Cost: deduplicate by request_id, filter by epoch window
                    week_cost = 0.0
                    for r in conn.execute(
                        "SELECT model, SUM(inp) as inp, SUM(out) as out, "
                        "SUM(cc) as cc, SUM(cr) as cr "
                        "FROM ("
                        "  SELECT model, request_id, "
                        "  MAX(input_tokens) as inp, MAX(output_tokens) as out, "
                        "  MAX(cache_creation_tokens) as cc, MAX(cache_read_tokens) as cr "
                        "  FROM events WHERE type='assistant' AND model IS NOT NULL "
                        "  AND model != '<synthetic>' AND request_id IS NOT NULL "
                        "  AND ts_epoch>=? AND ts_epoch<? "
                        "  GROUP BY model, request_id"
                        ") GROUP BY model",
                        (week_start_epoch, reset_epoch),
                    ):
                        dm = display_model(r["model"])
                        week_cost += compute_cost(
                            dm, r["inp"] or 0, r["out"] or 0,
                            r["cc"] or 0, r["cr"] or 0)

                    # Active time: sum gaps from session timelines within window
                    # Use a SQL approach: get all relevant user/assistant events
                    # in the window, ordered by session, then compute gaps.
                    week_active_s = 0.0
                    prev_evt = None
                    for e in conn.execute(
                        "SELECT session_id, ts_epoch, type, is_sidechain, "
                        "has_tool_use, has_tool_result, agent_id "
                        "FROM events "
                        "WHERE ts_epoch>=? AND ts_epoch<? "
                        "AND type IN ('user','assistant') "
                        "AND is_sidechain=0 AND agent_id IS NULL "
                        "ORDER BY session_id, ts_epoch",
                        (week_start_epoch, reset_epoch),
                    ):
                        if (prev_evt and
                                prev_evt["session_id"] == e["session_id"]):
                            gap = e["ts_epoch"] - prev_evt["ts_epoch"]
                            if 0 < gap < IDLE_THRESHOLD_S:
                                week_active_s += gap
                        prev_evt = e

                    weekly_budget["week_cost"] = round(week_cost, 2)
                    weekly_budget["week_active_s"] = round(week_active_s)
                except (ValueError, TypeError, OSError):
                    pass
        except (ValueError, KeyError):
            pass

    # ── Project display names (short, disambiguated) ──
    _top_project_dirs = sorted(project_cost, key=lambda x: -project_cost[x])[:15]
    _proj_display = _make_display_names(_top_project_dirs)
    _projects_list = [
        {"name": _proj_display[k], "minutes": round(project_seconds[k] / 60),
         "cost": round(project_cost[k], 2),
         "recent_minutes": round(recent_project_seconds[k] / 60),
         "recent_cost": round(recent_project_cost.get(k, 0), 2)}
        for k in _top_project_dirs
    ]

    return {
        "cards": {
            "sessions": sessions_count,
            "human_prompts": tot["human_prompts"],
            "total_tokens": tot["tokens"],
            "active_time_s": round(tot["active_s"]),
            "tool_calls": tot["tool_calls"],
            "models_used": len(models_seen),
            "avg_prompts_day": round(tot["human_prompts"] / num_days),
            "avg_active_day_s": round(tot["active_s"] / num_days),
        },
        "daily": daily_list,
        "models": models_donut,
        "tools": tool_counts,
        "recent_tools": recent_tool_counts,
        "time_breakdown": {
            "thinking": round(tot["thinking_s"]),
            "tool_execution": round(tot["tool_exec_s"]),
            "subagent": round(tot["subagent_s"]),
            "agent_runs": tot["agent_runs"],
            "recent_subagent": round(tot["recent_subagent_s"]),
            "recent_agent_runs": tot["recent_agent_runs"],
            "recent_thinking": round(tot["recent_thinking_s"]),
            "recent_tool_execution": round(tot["recent_tool_exec_s"]),
        },
        "projects": _projects_list,
        "model_breakdown": model_breakdown,
        "total_cost": round(total_cost, 2),
        "total_orch_cost": round(sum(m["main_cost"] for m in model_breakdown), 2),
        "total_agent_cost": round(sum(m["agent_cost"] for m in model_breakdown), 2),
        "benchmarks": {
            name: MODEL_BENCHMARKS.get(name, {})
            for name in model_stats if MODEL_BENCHMARKS.get(name)
        },
        "output_pricing": {name: get_pricing(name)[1] for name in model_stats},
        "model_pricing": {name: {"input": p[0], "output": p[1], "cache_write": p[2], "cache_read": p[3]}
                          for name in model_stats for p in [get_pricing(name)]},
        "cutoff_date": cutoff_date,
        "recency_days": RECENCY_DAYS,
        "generation_time": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "data_range": (f"since {datetime.strptime(date_range[0], '%Y-%m-%d').strftime('%b %-d, %Y')}" if date_range else "No data"),
        "machines": machine_list,
        "machine_last_active": _machine_last_active,
        "machine_summary": machine_summary,
        "recent_machine_summary": recent_machine_summary,
        "machine_daily_cost": machine_daily_series,
        "model_order": MODEL_ORDER,
        "hourly": hourly_list,
        "weekly_budget": weekly_budget,
        "last_active_ts": last_active_ts,
        "version": get_cache_version(),
    }
