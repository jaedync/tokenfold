"""Rebuild aggregated dashboard data from daily_summary rows.

Reads pre-computed per-day summaries (built by summarizer.py) and merges them
into the same JSON structure the old event-scanning code produced.  Hourly
activity is still computed live from events/tool_uses (48h window).
"""

import json
import random
import re
import threading
import time as _time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .config import ENTERPRISE_PRED as _ENT_PRED, RECENCY_DAYS, TZ_NAME
from .db import get_conn
from .pricing import (
    MODEL_BENCHMARKS, MODEL_ORDER, compute_cost, display_model, get_pricing,
    load_pricing, model_sort_key,
)
from .water import compute_energy_wh, compute_water_ml

TZ = ZoneInfo(TZ_NAME)

# Enterprise scope predicate (_ENT_PRED) is imported from config — single source
# of truth shared with api.py and cost_windows.py to prevent drift.

# In-memory cache — rebuilt only after ingest or on first request
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

    ``-home-jaedy-services-claude-stats-v2``  → ``claude-stats-v2``
    ``-Users-jaedy-development-caldera-mcp``  → ``caldera-mcp``
    ``C--Users-Acme-Documents-aswp-claude`` → ``aswp-claude``
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
                # Home dir only — disambiguate by extracting username
                um = re.search(r"(?:home|Users)-([^-]+)", d)
                short[d] = "~ (" + um.group(1) + ")" if um else d.strip("-")
    return short


_rebuilding = False


def invalidate_cache():
    """Trigger eager background rebuild. Serves stale cache during rebuild."""
    trigger_eager_rebuild()


def trigger_eager_rebuild():
    """Rebuild cache in background thread. Serves previous cache during rebuild."""
    global _rebuilding
    with _cache_lock:
        if _rebuilding:
            return  # rebuild already in progress
        _rebuilding = True
        _cache_version_bump()

    def _rebuild():
        global _cached_data, _rebuilding
        try:
            data = _build_dashboard_data_inner()
            with _cache_lock:
                _cached_data = data
        finally:
            with _cache_lock:
                _rebuilding = False

    threading.Thread(target=_rebuild, daemon=True).start()


def _cache_version_bump():
    """Increment cache version (caller must hold _cache_lock)."""
    global _cache_version
    _cache_version += 1


def get_cache_version() -> int:
    """Return current cache version (incremented on each invalidation)."""
    with _cache_lock:
        return _cache_version


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


def _empty_dashboard(cutoff_date: str) -> dict:
    """Return the dashboard structure with no data (used when no summaries exist)."""
    now = datetime.now(TZ)
    return {
        "cards": {
            "sessions": 0, "human_prompts": 0, "total_tokens": 0,
            "active_time_s": 0, "tool_calls": 0, "models_used": 0,
            "avg_prompts_day": 0, "avg_active_day_s": 0,
        },
        "daily": [], "tools": {}, "recent_tools": {},
        "time_breakdown": {
            "thinking": 0, "tool_execution": 0, "subagent": 0, "agent_runs": 0,
            "recent_subagent": 0, "recent_agent_runs": 0,
            "recent_thinking": 0, "recent_tool_execution": 0,
        },
        "projects": [], "model_breakdown": [],
        "total_cost": 0, "total_orch_cost": 0, "total_agent_cost": 0,
        "benchmarks": {}, "output_pricing": {}, "model_pricing": {},
        "cutoff_date": cutoff_date,
        "generation_time": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "data_range": "No data",
        "machines": [], "machine_last_active": {},
        "machine_summary": [], "recent_machine_summary": [],
        "machine_daily_cost": {}, "model_order": MODEL_ORDER,
        "hourly": [],
        "last_active_ts": None, "version": get_cache_version(),
        "today": {"cost": 0.0, "model_breakdown": [], "time_breakdown": {
            "thinking": 0, "tool_execution": 0, "subagent": 0, "agent_runs": 0,
        }, "tools": {}, "projects": [], "machine_summary": []},
        "org_name": "",
        "plan_scope": "enterprise",
    }


def _merge_summary_rows(rows) -> dict:
    """Merge multiple enterprise daily_summary rows into a single dict.

    Scalar columns are summed. JSON blob columns are merged by deep-summing
    numeric fields per top-level key.  The returned dict is indexable like a
    sqlite3.Row for all fields consumed by _build_today_data.
    """
    scalar_cols = (
        "sessions", "human_prompts", "tool_calls",
        "input_tokens", "output_tokens", "cache_creation_tokens", "cache_read_tokens",
        "active_s", "thinking_s", "tool_exec_s", "subagent_s", "agent_runs", "cost",
    )
    blob_cols = (
        "model_json", "project_json", "machine_json",
        "tool_json", "prompt_model_json", "gen_json",
    )

    merged: dict = {}
    # Sum scalars
    for col in scalar_cols:
        merged[col] = sum(r[col] or 0 for r in rows)

    # Merge blob columns by deep-summing numeric leaf values
    for col in blob_cols:
        combined: dict = {}
        for row in rows:
            outer = json.loads(row[col] or "{}")
            for key, val in outer.items():
                if isinstance(val, dict):
                    target = combined.setdefault(key, {})
                    for inner_key, inner_val in val.items():
                        target[inner_key] = target.get(inner_key, 0) + (inner_val or 0)
                else:
                    # tool_json and prompt_model_json: {name: int}
                    combined[key] = combined.get(key, 0) + (val or 0)
        merged[col] = json.dumps(combined)

    return merged


def _build_hourly(conn) -> list[dict]:
    """Build the 24-slot hourly activity grid from live event data (48h window)."""
    now_h = datetime.now(TZ)
    current_hour_num = now_h.hour
    today = now_h.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(days=1)

    hourly_list = []
    for slot in range(24):
        clock_hour = slot
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

    all_epochs = [hl["_epoch"] for hl in hourly_list]
    h_start_epoch = min(all_epochs)
    h_end_epoch = max(all_epochs) + 3600
    epoch_to_idx = {hl["_epoch"]: i for i, hl in enumerate(hourly_list)}

    for r in conn.execute(
        f"SELECT CAST(ts_epoch / 3600 AS INTEGER) * 3600 as bucket, COUNT(*) as cnt "
        f"FROM events WHERE is_human_prompt=1 AND {_ENT_PRED} AND ts_epoch>=? AND ts_epoch<? "
        "GROUP BY bucket", (h_start_epoch, h_end_epoch),
    ):
        idx = epoch_to_idx.get(r["bucket"])
        if idx is not None:
            hourly_list[idx]["prompts"] = r["cnt"]

    for r in conn.execute(
        f"SELECT CAST(ts_epoch / 3600 AS INTEGER) * 3600 as bucket, COUNT(*) as cnt "
        f"FROM tool_uses WHERE ts_epoch>=? AND ts_epoch<? "
        f"AND session_id IN (SELECT session_id FROM events WHERE {_ENT_PRED}) "
        "GROUP BY bucket", (h_start_epoch, h_end_epoch),
    ):
        idx = epoch_to_idx.get(r["bucket"])
        if idx is not None:
            hourly_list[idx]["tool_calls"] = r["cnt"]

    for r in conn.execute(
        f"SELECT CAST(first_ts / 3600 AS INTEGER) * 3600 as bucket, model, speed, inference_geo, "
        "SUM(inp) as inp, SUM(outp) as outp, SUM(cc) as cc, SUM(cr) as cr "
        "FROM ("
        f"  SELECT MIN(ts_epoch) as first_ts, model, request_id, "
        "  MAX(speed) as speed, MAX(inference_geo) as inference_geo, "
        "  MAX(input_tokens) as inp, MAX(output_tokens) as outp, "
        "  MAX(cache_creation_tokens) as cc, MAX(cache_read_tokens) as cr "
        f"  FROM events WHERE type='assistant' AND model IS NOT NULL "
        f"  AND model != '<synthetic>' AND request_id IS NOT NULL "
        f"  AND {_ENT_PRED} "
        "  AND ts_epoch>=? AND ts_epoch<? "
        "  GROUP BY model, request_id"
        ") GROUP BY bucket, model, speed, inference_geo",
        (h_start_epoch, h_end_epoch),
    ):
        idx = epoch_to_idx.get(r["bucket"])
        if idx is not None:
            dm = display_model(r["model"])
            hourly_list[idx]["cost"] += compute_cost(
                dm, r["inp"] or 0, r["outp"] or 0, r["cc"] or 0, r["cr"] or 0,
                r["speed"], r["inference_geo"])

    for hl in hourly_list:
        hl["cost"] = round(hl["cost"], 2)
        del hl["_epoch"]

    return hourly_list


def _build_today_data(conn, today_str: str) -> dict:
    """Build the 'today' sub-object from today's daily_summary row.

    Returns model_breakdown, time_breakdown, tools, projects, and
    machine_summary scoped to just today.
    """
    ent_rows = conn.execute(
        f"SELECT * FROM daily_summary WHERE day = ? AND {_ENT_PRED}", (today_str,)
    ).fetchall()

    if not ent_rows:
        return {
            "cost": 0.0,
            "model_breakdown": [], "time_breakdown": {
                "thinking": 0, "tool_execution": 0, "subagent": 0, "agent_runs": 0,
            },
            "tools": {}, "projects": [], "machine_summary": [],
        }

    row = _merge_summary_rows(ent_rows)

    # Model breakdown for today
    model_data = json.loads(row["model_json"] or "{}")
    gen_data = json.loads(row["gen_json"] or "{}")
    today_mb = []
    for mname, md in sorted(model_data.items(), key=lambda kv: model_sort_key(kv[0])):
        inp = md.get("input", 0)
        out = md.get("output", 0)
        cw = md.get("cache_write", 0)
        cr = md.get("cache_read", 0)
        cost = md.get("cost", 0.0)
        p = get_pricing(mname)
        main_cost = md.get("main_cost", 0.0)
        agent_cost = round(cost - main_cost, 2)
        main_prompts = md.get("main_prompts", 0)
        agent_invocations = md.get("agent_invocations", 0)
        avg_cost_per_turn = (main_cost / main_prompts) if main_prompts > 0 else None
        avg_cost_per_agent = (agent_cost / agent_invocations) if agent_invocations > 0 else None
        active_hours = md.get("active_s", 0.0) / 3600
        gd = gen_data.get(mname, {})
        gen_s = gd.get("gen_s", 0.0)
        gen_out = gd.get("gen_out", 0)
        energy = compute_energy_wh(mname, inp, out)
        water = compute_water_ml(mname, inp, out)
        today_mb.append({
            "model": mname,
            "api_calls": md.get("api_calls", 0),
            "input": inp, "output": out,
            "cache_write": cw, "cache_read": cr,
            "total_tokens": inp + out + cw + cr,
            "cost": round(cost, 2),
            "main_cost": round(main_cost, 2),
            "agent_cost": agent_cost,
            "avg_cost_per_turn": round(avg_cost_per_turn, 4) if avg_cost_per_turn is not None else None,
            "avg_cost_per_agent": round(avg_cost_per_agent, 4) if avg_cost_per_agent is not None else None,
            "main_prompts": main_prompts,
            "agent_invocations": agent_invocations,
            "active_hours": round(active_hours, 1),
            "cost_per_hour": round(cost / active_hours, 2) if active_hours > 0 else None,
            "all_cost_per_hour": round(cost / active_hours, 2) if active_hours > 0 else None,
            "output_tok_per_s": round(gen_out / gen_s, 1) if gen_s > 0 else None,
            "all_output_tok_per_s": round(gen_out / gen_s, 1) if gen_s > 0 else None,
            "cost_input": round(inp * p[0] / 1e6, 2),
            "cost_output": round(out * p[1] / 1e6, 2),
            "cost_cache_write": round(cw * p[2] / 1e6, 2),
            "cost_cache_read": round(cr * p[3] / 1e6, 2),
            # Today view uses the same keys as recent/all for compatibility
            "recent_cost": round(cost, 2),
            "recent_main_cost": round(main_cost, 2),
            "recent_agent_cost": agent_cost,
            "recent_cost_per_hour": round(cost / active_hours, 2) if active_hours > 0 else None,
            "recent_output_tok_per_s": round(gen_out / gen_s, 1) if gen_s > 0 else None,
            "recent_active_hours": round(active_hours, 1),
            "recent_cost_input": round(inp * p[0] / 1e6, 2),
            "recent_cost_output": round(out * p[1] / 1e6, 2),
            "recent_cost_cache_write": round(cw * p[2] / 1e6, 2),
            "recent_cost_cache_read": round(cr * p[3] / 1e6, 2),
            "recent_input": inp, "recent_output": out,
            "recent_cache_write": cw, "recent_cache_read": cr,
            "recent_total_tokens": inp + out + cw + cr,
            "last_seen": today_str, "recent": True,
            "energy_wh": round(energy, 1),
            "water_ml": round(water, 1),
            "recent_energy_wh": round(energy, 1),
            "recent_water_ml": round(water, 1),
        })

    # Time breakdown for today
    time_breakdown = {
        "thinking": round(row["thinking_s"]),
        "tool_execution": round(row["tool_exec_s"]),
        "subagent": round(row["subagent_s"]),
        "agent_runs": row["agent_runs"],
    }

    # Tools for today
    tools = json.loads(row["tool_json"] or "{}")

    # Projects for today
    proj_data = json.loads(row["project_json"] or "{}")
    raw_dirs = sorted(proj_data, key=lambda x: -proj_data[x].get("cost", 0))[:15]
    proj_display = _make_display_names(raw_dirs)
    projects = [
        {"name": proj_display[k], "minutes": round(proj_data[k].get("seconds", 0) / 60),
         "cost": round(proj_data[k].get("cost", 0), 2),
         "recent_minutes": round(proj_data[k].get("seconds", 0) / 60),
         "recent_cost": round(proj_data[k].get("cost", 0), 2)}
        for k in raw_dirs
    ]

    # Machine summary for today
    mach_data = json.loads(row["machine_json"] or "{}")
    machine_summary = []
    for mname in sorted(mach_data, key=lambda x: -mach_data[x].get("prompts", 0)):
        mv = mach_data[mname]
        total_tok = mv.get("input", 0) + mv.get("output", 0) + mv.get("cache_write", 0) + mv.get("cache_read", 0)
        machine_summary.append({
            "machine": mname,
            "prompts": mv.get("prompts", 0),
            "api_calls": mv.get("calls", 0),
            "tool_calls": mv.get("tool_calls", 0),
            "total_tokens": total_tok,
            "cost": round(mv.get("cost", 0), 2),
        })

    return {
        "cost": round(row["cost"] or 0.0, 2),
        "model_breakdown": today_mb,
        "time_breakdown": time_breakdown,
        "tools": tools,
        "projects": projects,
        "machine_summary": machine_summary,
    }


def _build_dashboard_data_inner() -> dict:
    """Read daily_summary rows and produce the full dashboard JSON blob."""
    load_pricing()
    conn = get_conn()
    cutoff_date = (datetime.now(TZ) - timedelta(days=RECENCY_DAYS)).strftime("%Y-%m-%d")

    # ── Read verified-enterprise summary rows only ──
    rows = conn.execute(
        f"SELECT * FROM daily_summary WHERE {_ENT_PRED} ORDER BY day"
    ).fetchall()

    if not rows:
        return _empty_dashboard(cutoff_date)

    # ── Collect distinct enterprise orgs ──
    orgs = sorted({row["org_name"] for row in rows if row["org_name"]})

    # ── Accumulators ──
    model_stats = defaultdict(lambda: {
        "input": 0, "output": 0, "cache_write": 0, "cache_read": 0,
        "api_calls": 0, "main_api_calls": 0, "main_cost": 0.0,
        "main_prompts": 0, "agent_invocations": 0, "active_s": 0.0,
        "gen_s": 0.0, "gen_out": 0,
        "recent_active_s": 0.0, "recent_gen_s": 0.0, "recent_gen_out": 0,
        "recent_input": 0, "recent_output": 0, "recent_cache_write": 0,
        "recent_cache_read": 0, "recent_main_cost": 0.0, "last_seen": "",
        "cost": 0.0, "recent_cost": 0.0,
    })
    project_seconds: Counter = Counter()
    project_cost: Counter = Counter()
    recent_project_seconds: Counter = Counter()
    recent_project_cost: Counter = Counter()
    tot = {
        "thinking_s": 0.0, "tool_exec_s": 0.0, "active_s": 0.0,
        "subagent_s": 0.0, "agent_runs": 0,
        "recent_subagent_s": 0.0, "recent_agent_runs": 0,
        "recent_thinking_s": 0.0, "recent_tool_exec_s": 0.0,
        "tokens": 0, "human_prompts": 0, "tool_calls": 0, "sessions": 0,
    }
    models_seen: set[str] = set()
    all_tool_counts: Counter = Counter()
    recent_tool_counts: Counter = Counter()

    # Per-machine accumulators
    mach_all: dict[str, dict] = defaultdict(lambda: {
        "input": 0, "output": 0, "cache_write": 0, "cache_read": 0,
        "calls": 0, "prompts": 0, "tool_calls": 0, "cost": 0.0,
    })
    mach_recent: dict[str, dict] = defaultdict(lambda: {
        "input": 0, "output": 0, "cache_write": 0, "cache_read": 0,
        "calls": 0, "prompts": 0, "tool_calls": 0, "cost": 0.0,
    })
    mach_daily_cost: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    machine_set: set[str] = set()

    daily_map: dict[str, dict] = {}

    # ── Merge each day's summary ──
    for row in rows:
        day = row["day"]
        is_recent = day >= cutoff_date

        # Scalar daily fields — accumulate to support multiple enterprise accounts per day
        dm = daily_map.setdefault(day, {
            "sessions": 0, "human_prompts": 0, "input_tokens": 0, "output_tokens": 0,
            "cache_creation_tokens": 0, "cache_read_tokens": 0, "tool_calls": 0,
            "active_s": 0.0, "thinking_s": 0.0, "tool_exec_s": 0.0, "cost": 0.0,
        })
        dm["sessions"] += row["sessions"]
        dm["human_prompts"] += row["human_prompts"]
        dm["input_tokens"] += row["input_tokens"]
        dm["output_tokens"] += row["output_tokens"]
        dm["cache_creation_tokens"] += row["cache_creation_tokens"]
        dm["cache_read_tokens"] += row["cache_read_tokens"]
        dm["tool_calls"] += row["tool_calls"]
        dm["active_s"] += row["active_s"]
        dm["thinking_s"] += row["thinking_s"]
        dm["tool_exec_s"] += row["tool_exec_s"]
        dm["cost"] += row["cost"]

        tot["sessions"] += row["sessions"]
        tot["human_prompts"] += row["human_prompts"]
        tot["tool_calls"] += row["tool_calls"]
        tot["tokens"] += (row["input_tokens"] + row["output_tokens"]
                          + row["cache_creation_tokens"] + row["cache_read_tokens"])
        tot["active_s"] += row["active_s"]
        tot["thinking_s"] += row["thinking_s"]
        tot["tool_exec_s"] += row["tool_exec_s"]
        tot["subagent_s"] += row["subagent_s"]
        tot["agent_runs"] += row["agent_runs"]
        if is_recent:
            tot["recent_thinking_s"] += row["thinking_s"]
            tot["recent_tool_exec_s"] += row["tool_exec_s"]
            tot["recent_subagent_s"] += row["subagent_s"]
            tot["recent_agent_runs"] += row["agent_runs"]

        # ── model_json ──
        model_data = json.loads(row["model_json"] or "{}")
        for mname, md in model_data.items():
            models_seen.add(mname)
            ms = model_stats[mname]
            ms["input"] += md.get("input", 0)
            ms["output"] += md.get("output", 0)
            ms["cache_write"] += md.get("cache_write", 0)
            ms["cache_read"] += md.get("cache_read", 0)
            ms["api_calls"] += md.get("api_calls", 0)
            ms["main_api_calls"] += md.get("main_api_calls", 0)
            ms["main_cost"] += md.get("main_cost", 0.0)
            ms["main_prompts"] += md.get("main_prompts", 0)
            ms["agent_invocations"] += md.get("agent_invocations", 0)
            ms["active_s"] += md.get("active_s", 0.0)
            ms["cost"] += md.get("cost", 0.0)
            if day > ms["last_seen"]:
                ms["last_seen"] = day
            if is_recent:
                ms["recent_input"] += md.get("input", 0)
                ms["recent_output"] += md.get("output", 0)
                ms["recent_cache_write"] += md.get("cache_write", 0)
                ms["recent_cache_read"] += md.get("cache_read", 0)
                ms["recent_main_cost"] += md.get("main_cost", 0.0)
                ms["recent_active_s"] += md.get("active_s", 0.0)
                ms["recent_cost"] += md.get("cost", 0.0)

        # ── gen_json (generation time — stored separately from model_json) ──
        gen_data = json.loads(row["gen_json"] or "{}")
        for mname, gd in gen_data.items():
            ms = model_stats[mname]
            ms["gen_s"] += gd.get("gen_s", 0.0)
            ms["gen_out"] += gd.get("gen_out", 0)
            if is_recent:
                ms["recent_gen_s"] += gd.get("gen_s", 0.0)
                ms["recent_gen_out"] += gd.get("gen_out", 0)

        # ── project_json ──
        proj_data = json.loads(row["project_json"] or "{}")
        for pdir, pd in proj_data.items():
            project_seconds[pdir] += pd.get("seconds", 0.0)
            project_cost[pdir] += pd.get("cost", 0.0)
            if is_recent:
                recent_project_seconds[pdir] += pd.get("seconds", 0.0)
                recent_project_cost[pdir] += pd.get("cost", 0.0)

        # ── machine_json ──
        mach_data = json.loads(row["machine_json"] or "{}")
        for mname, md in mach_data.items():
            machine_set.add(mname)
            ma = mach_all[mname]
            ma["input"] += md.get("input", 0)
            ma["output"] += md.get("output", 0)
            ma["cache_write"] += md.get("cache_write", 0)
            ma["cache_read"] += md.get("cache_read", 0)
            ma["calls"] += md.get("calls", 0)
            ma["prompts"] += md.get("prompts", 0)
            ma["tool_calls"] += md.get("tool_calls", 0)
            ma["cost"] += md.get("cost", 0.0)
            mach_daily_cost[mname][day] += md.get("cost", 0.0)
            if is_recent:
                mr = mach_recent[mname]
                mr["input"] += md.get("input", 0)
                mr["output"] += md.get("output", 0)
                mr["cache_write"] += md.get("cache_write", 0)
                mr["cache_read"] += md.get("cache_read", 0)
                mr["calls"] += md.get("calls", 0)
                mr["prompts"] += md.get("prompts", 0)
                mr["tool_calls"] += md.get("tool_calls", 0)
                mr["cost"] += md.get("cost", 0.0)

        # ── tool_json ──
        tool_data = json.loads(row["tool_json"] or "{}")
        for tname, cnt in tool_data.items():
            all_tool_counts[tname] += cnt
            if is_recent:
                recent_tool_counts[tname] += cnt

    # ── Build date range (fill gaps) ──
    all_dates = sorted(daily_map.keys())
    start = datetime.strptime(all_dates[0], "%Y-%m-%d")
    end = datetime.strptime(all_dates[-1], "%Y-%m-%d")
    date_range: list[str] = []
    cur = start
    while cur <= end:
        date_range.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)

    num_days = len(date_range) or 1

    # ── daily list ──
    daily_list = []
    for d in date_range:
        dd = daily_map.get(d)
        if dd:
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
        else:
            daily_list.append({
                "date": d, "sessions": 0, "prompts": 0, "tool_calls": 0,
                "active_minutes": 0, "input_tokens": 0, "output_tokens": 0,
                "cache_creation_tokens": 0, "cache_read_tokens": 0, "cost": 0,
            })

    # ── Tool counts (top 20) ──
    tool_counts = dict(all_tool_counts.most_common(20))
    recent_tools = dict(recent_tool_counts.most_common(20))

    # ── Model breakdown ──
    total_cost = 0.0
    model_breakdown = []
    for name in sorted(model_stats, key=lambda m: model_sort_key(m)):
        ms = model_stats[name]
        total_tok = ms["input"] + ms["output"] + ms["cache_write"] + ms["cache_read"]
        cost = ms["cost"]
        p = get_pricing(name)
        total_cost += cost
        main_cost = round(ms["main_cost"], 2)
        agent_cost = round(cost - ms["main_cost"], 2)
        avg_cost_per_turn = (ms["main_cost"] / ms["main_prompts"]
                             if ms["main_prompts"] > 0 else None)
        avg_cost_per_agent = (agent_cost / ms["agent_invocations"]
                              if ms["agent_invocations"] > 0 else None)
        active_hours = ms["active_s"] / 3600
        recent_cost = ms["recent_cost"]
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
        energy = compute_energy_wh(name, ms["input"], ms["output"])
        water = compute_water_ml(name, ms["input"], ms["output"])
        recent_energy = compute_energy_wh(name, ms["recent_input"], ms["recent_output"])
        recent_water = compute_water_ml(name, ms["recent_input"], ms["recent_output"])
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
            "energy_wh": round(energy, 1),
            "water_ml": round(water, 1),
            "recent_energy_wh": round(recent_energy, 1),
            "recent_water_ml": round(recent_water, 1),
        })

    # ── Machine summaries ──
    machine_list = sorted(machine_set)
    machine_summary = []
    for m_name in sorted(mach_all, key=lambda x: -mach_all[x]["prompts"]):
        ma = mach_all[m_name]
        total_tok = ma["input"] + ma["output"] + ma["cache_write"] + ma["cache_read"]
        machine_summary.append({
            "machine": m_name,
            "prompts": ma["prompts"],
            "api_calls": ma["calls"],
            "tool_calls": ma["tool_calls"],
            "total_tokens": total_tok,
            "cost": round(ma["cost"], 2),
        })

    recent_machine_summary = []
    for m_name in sorted(mach_recent, key=lambda x: -mach_recent[x]["prompts"]):
        mr = mach_recent[m_name]
        total_tok = mr["input"] + mr["output"] + mr["cache_write"] + mr["cache_read"]
        recent_machine_summary.append({
            "machine": m_name,
            "prompts": mr["prompts"],
            "api_calls": mr["calls"],
            "tool_calls": mr["tool_calls"],
            "total_tokens": total_tok,
            "cost": round(mr["cost"], 2),
        })

    # ── Machine daily cost series ──
    machine_daily_series: dict[str, list[float]] = {}
    for m_name in mach_all:
        series = []
        for d in date_range:
            series.append(round(mach_daily_cost.get(m_name, {}).get(d, 0.0), 2))
        machine_daily_series[m_name] = series

    # ── Last active timestamp (global + per machine) ──
    # daily_summary can't answer "active in last 15 minutes" — needs minute-grain
    # timestamps from raw events. Indexed on source_machine, cheap.
    machine_last_active: dict[str, float] = {}
    last_active_ts = None
    for r in conn.execute(
        f"SELECT source_machine, MAX(ts_epoch) as ts FROM events "
        f"WHERE {_ENT_PRED} "
        "GROUP BY source_machine"
    ):
        if r["ts"]:
            machine_last_active[r["source_machine"]] = r["ts"]
            if last_active_ts is None or r["ts"] > last_active_ts:
                last_active_ts = r["ts"]

    # ── Sessions count (total distinct, from summary) ──
    # Note: session counts from summaries are per-day distinct, so the total
    # may overcount sessions spanning midnight.  Use the sum as a close approx.
    sessions_count = tot["sessions"]

    # ── Hourly (live query) ──
    hourly_list = _build_hourly(conn)

    # ── Project display names ──
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
        "tools": tool_counts,
        "recent_tools": recent_tools,
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
        "generation_time": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "data_range": (f"since {datetime.strptime(date_range[0], '%Y-%m-%d').strftime('%b %-d, %Y')}" if date_range else "No data"),
        "machines": machine_list,
        "machine_last_active": machine_last_active,
        "machine_summary": machine_summary,
        "recent_machine_summary": recent_machine_summary,
        "machine_daily_cost": machine_daily_series,
        "model_order": MODEL_ORDER,
        "hourly": hourly_list,
        "last_active_ts": last_active_ts,
        "version": get_cache_version(),
        "today": _build_today_data(conn, datetime.now(TZ).strftime("%Y-%m-%d")),
        "org_name": " · ".join(orgs),
        "plan_scope": "enterprise",
    }


# ── Background sweep timers ──────────────────────────────────────────────────

_sweep_timers: list[threading.Timer] = []
_last_full_sweep: float = 0.0

PERIODIC_SWEEP_INTERVAL = 3600    # 1 hour
FULL_SWEEP_INTERVAL = 86400       # 24 hours
SWEEP_JITTER_MAX = 600            # 0-10 minutes random offset


def _jitter() -> float:
    """Random delay 60-600 seconds to avoid landing on the hour."""
    return random.uniform(60, SWEEP_JITTER_MAX)


def start_sweeps():
    """Start the periodic and full sweep background timers."""
    _schedule_periodic_sweep()
    _schedule_full_sweep()


def stop_sweeps():
    """Cancel all pending sweep timers."""
    for t in _sweep_timers:
        t.cancel()
    _sweep_timers.clear()


def _schedule_periodic_sweep():
    delay = PERIODIC_SWEEP_INTERVAL + _jitter()
    t = threading.Timer(delay, _run_periodic_sweep)
    t.daemon = True
    t.start()
    _sweep_timers.append(t)


def _schedule_full_sweep():
    delay = FULL_SWEEP_INTERVAL + _jitter()
    t = threading.Timer(delay, _run_full_sweep)
    t.daemon = True
    t.start()
    _sweep_timers.append(t)


def _run_periodic_sweep():
    """Recompute last 7 days of summaries, unless a full sweep ran recently."""
    global _last_full_sweep
    # Skip if a full sweep ran within the last hour
    if _time.time() - _last_full_sweep < PERIODIC_SWEEP_INTERVAL:
        _schedule_periodic_sweep()
        return

    try:
        from .summarizer import summarize_days
        today = datetime.now(TZ)
        days = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
        summarize_days(days)
        trigger_eager_rebuild()
    except Exception:
        pass  # Don't crash the timer on transient errors
    _schedule_periodic_sweep()


def _run_full_sweep():
    """Recompute all daily summaries."""
    global _last_full_sweep
    try:
        from .summarizer import summarize_days
        summarize_days(None)  # all days
        _last_full_sweep = _time.time()
        trigger_eager_rebuild()
    except Exception:
        pass
    _schedule_full_sweep()
