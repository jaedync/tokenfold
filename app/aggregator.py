"""Rebuild aggregated dashboard data from daily_summary rows.

Reads pre-computed per-day summaries (built by summarizer.py) and merges them
into the same JSON structure the old event-scanning code produced.  Hourly
activity is still computed live from events/tool_uses (48h window).
"""

import json
import logging
import os
import random
import re
import threading
import time as _time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from datetime import timezone as _timezone
from zoneinfo import ZoneInfo

from .extra_usage import build_meter_payload
from .month_hero import month_hero_block
from .config import DEFAULT_SCOPE, RECENCY_DAYS, TZ_NAME, scope_predicate
from .cost_windows import compute_window_cost
from .db import checkpoint_wal, get_conn
from .pricing import (
    MODEL_BENCHMARKS, MODEL_ORDER, WEB_SEARCH_PER_1K, compute_cost,
    REPORTED_COST_SUM_SQL, display_model, display_model_for_row, effective_geo,
    get_pricing, is_priced, load_pricing, reported_cost,
    model_sort_key,
)
from .served_models import served_model_chips
from .water import compute_energy_wh, compute_water_ml

# Per-request web-search fee for the cost-component breakdowns.
_WS_FEE = WEB_SEARCH_PER_1K / 1000.0

logger = logging.getLogger(__name__)

TZ = ZoneInfo(TZ_NAME)

# scope_predicate(scope) is imported from config — single source of truth for
# SQL predicates shared with api.py and cost_windows.py to prevent drift.

# In-memory cache — rebuilt only after ingest or on first request.
# Keyed by scope string ("enterprise" / "personal") so each scope slot is independent.
_cache_lock = threading.Lock()
_cached_data: dict[str, dict] = {}
_cache_version: int = 0

# Scope-aware pre-warm bookkeeping. Invalidation clears _cached_data for ALL
# scopes, but the drain-loop worker used to rebuild ONLY DEFAULT_SCOPE — so a
# dashboard tab viewing the non-default scope hit a cold cache (a synchronous
# request-thread build) on every SSE-driven refetch, once per ~1s under active
# realtime ingest. We track which scopes have been requested recently and have
# the worker rebuild the union of {DEFAULT_SCOPE} ∪ {scopes seen within the TTL}
# so every actively-viewed scope stays warm. Maps scope → last-request monotonic
# timestamp; protected by _cache_lock.
WARM_SCOPE_TTL_S = 900  # 15 minutes: a scope idle longer than this stops warming
_warm_scopes: dict[str, float] = {}


def _mark_scope_requested(scope: str) -> None:
    """Record that `scope` was just requested (caller must NOT hold _cache_lock).

    Feeds the worker's warm-scope set so an actively-viewed non-default scope is
    pre-warmed on each drain iteration instead of rebuilt synchronously per SSE
    refetch. Uses a monotonic clock (immune to wall-clock jumps)."""
    with _cache_lock:
        _warm_scopes[scope] = _time.monotonic()


def _warm_scope_set() -> set[str]:
    """Scopes the worker should rebuild: DEFAULT_SCOPE plus every scope requested
    within WARM_SCOPE_TTL_S. Caller must hold _cache_lock. Prunes stale entries."""
    now = _time.monotonic()
    stale = [s for s, ts in _warm_scopes.items() if now - ts > WARM_SCOPE_TTL_S]
    for s in stale:
        del _warm_scopes[s]
    return {DEFAULT_SCOPE} | set(_warm_scopes)


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


# ── Machine identity normalization (UX P1-7) ────────────────────────────────
# The same physical machine reports under hostname variants (bare hostname,
# Tailscale/mDNS FQDN, different casing) and was counted as several machines.
# Normalize at READ time only — database rows keep the raw reported name.
#
# Variants that differ structurally (not just by domain suffix / case) need an
# alias entry. Extend via the MACHINE_ALIASES env var: "alias=canonical,...".
_DEFAULT_MACHINE_ALIASES = {
    "macbook-pro": "jaedyns-macbook-pro",
}


def _load_machine_aliases() -> dict[str, str]:
    aliases = dict(_DEFAULT_MACHINE_ALIASES)
    for pair in os.environ.get("MACHINE_ALIASES", "").split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            k, v = k.strip().lower(), v.strip().lower()
            if k and v:
                aliases[k] = v
    return aliases


MACHINE_ALIASES = _load_machine_aliases()


def canonical_machine(name):
    """Collapse hostname variants of one machine to a canonical display name.

    lowercase -> keep only the first DNS label (drops .ts.net/.local/etc.)
    -> alias map. None/empty pass through unchanged.
    """
    if not name:
        return name
    base = name.strip().lower().split(".", 1)[0]
    return MACHINE_ALIASES.get(base, base)


# ── MCP tool display naming (UX P2-15) ──────────────────────────────────────

def display_tool_name(name):
    """Short display name for MCP tool ids: ``server · tool``.

    ``mcp__plugin_playwright_playwright__browser_take_screenshot``
    → ``playwright · browser_take_screenshot``
    (mcp__ prefix stripped, repeated server tokens deduped, generic leading
    'plugin' token dropped). Non-MCP names pass through unchanged.
    """
    if not name or not name.startswith("mcp__"):
        return name
    parts = name[5:].split("__", 1)
    if len(parts) == 1:
        return parts[0]
    server, tool = parts
    tokens: list[str] = []
    for t in server.split("_"):
        if t and (not tokens or tokens[-1] != t):
            tokens.append(t)
    if len(tokens) > 1 and tokens[0] == "plugin":
        tokens = tokens[1:]
    return "{} · {}".format("_".join(tokens) or server, tool)


def _display_tool_counts(counter, top: int = 20):
    """Collapse a raw tool counter to display names.

    Returns (top-N {display: count} ordered by count desc,
             {display: [raw ids]} for renamed tools — tooltip fallback).
    """
    merged: Counter = Counter()
    fulls: dict[str, set] = defaultdict(set)
    for raw, cnt in counter.items():
        disp = display_tool_name(raw)
        merged[disp] += cnt
        if disp != raw:
            fulls[disp].add(raw)
    top_items = dict(merged.most_common(top))
    full_map = {d: sorted(fulls[d]) for d in top_items if d in fulls}
    return top_items, full_map


_rebuilding = False


def invalidate_cache():
    """Trigger eager background rebuild. Serves stale cache during rebuild."""
    trigger_eager_rebuild()


_cache_gen = 0


def trigger_eager_rebuild():
    """Rebuild cache in background thread. Serves previous cache during rebuild.

    Invalidation (version bump + cache clear) happens UNCONDITIONALLY — even
    when a rebuild is already in flight. The original early-return made a
    write that raced an ingest rebuild invisible until the next unrelated
    invalidation.

    The worker runs a DRAIN LOOP rather than a single build: after building
    every warm scope it checks, under the lock, whether the generation still
    matches the one it started with. If so, it stores all results and exits; if
    a newer invalidation landed while it was building, it adopts that generation
    and builds again. This coalesces any number of mid-build invalidations into
    exactly one follow-up build (not N), never drops the last one, and — because
    only the first invalidation to find no worker running spawns a thread —
    keeps at most one worker alive regardless of ingest cadence. Under ~1s
    realtime ingest a single early-return would have left the cache cold almost
    permanently; the loop guarantees the cache converges on data built from the
    final generation.

    Scope-aware: each iteration rebuilds the union {DEFAULT_SCOPE} ∪ {scopes
    requested within WARM_SCOPE_TTL_S}, so an actively-viewed non-default scope
    stays warm instead of forcing a synchronous request-thread build on every
    SSE refetch. All warm scopes are gen-checked together under ONE lock hold,
    so a stale iteration never writes any scope's pre-invalidation data.
    """
    global _rebuilding, _cache_gen
    with _cache_lock:
        _cache_gen += 1
        _cache_version_bump()
        _cached_data.clear()  # invalidate all scopes immediately
        if _rebuilding:
            return  # the running worker's drain loop will pick up this gen
        _rebuilding = True
        gen = _cache_gen

    def _rebuild():
        global _rebuilding, _cache_gen
        current_gen = gen
        cleared = False
        try:
            while True:
                # Snapshot the warm-scope set for THIS iteration under the lock,
                # then build every scope outside the lock so readers aren't
                # blocked. A transient build error (e.g. DB hiccup) in ANY scope
                # is swallowed below rather than crashing the worker; the finally
                # still clears _rebuilding so a later invalidation can retry —
                # matching the sweep timers' "don't die on transient errors"
                # stance. Failing one scope fails the whole iteration and drops
                # this generation; the next invalidation retries (same semantics
                # as the pre-scope-aware single-scope worker).
                with _cache_lock:
                    scopes = _warm_scope_set()
                try:
                    built = {s: _build_dashboard_data_inner(s) for s in scopes}
                except Exception:
                    # Dropping the generation is deliberate (next invalidation
                    # retries) — hiding the failure is not.
                    logger.exception(
                        "dashboard rebuild failed (scopes=%s, gen=%s) — "
                        "generation dropped, next invalidation retries",
                        sorted(scopes), current_gen)
                    return  # finally clears the flag
                # Storing the results and clearing _rebuilding must be atomic
                # under ONE lock hold: an invalidation slipping in between the
                # two would see _rebuilding=True, early-return, then be stranded
                # when we clear the flag — its generation never serviced. All
                # warm scopes are written together so none can carry stale data.
                with _cache_lock:
                    if _cache_gen == current_gen:
                        # No invalidation landed while building — store every
                        # warm scope, release the worker slot, and stop.
                        _cached_data.update(built)
                        _rebuilding = False
                        cleared = True
                        return
                    # A newer invalidation arrived mid-build; drain it with one
                    # more build rather than writing this now-stale result. Stay
                    # _rebuilding so no second worker spawns.
                    current_gen = _cache_gen
        finally:
            # Guarantees the flag is cleared on ANY abnormal exit (build raised,
            # including non-Exception BaseExceptions) — never leave it stuck True.
            if not cleared:
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




def _tiered_cw_parts(cw, c1h, p):
    """Cache-write dollar components honoring the tier split: 1h tokens at 2x
    base input, everything else (5m + unsplit legacy) at the 5m rate p[2].
    Returns (cost_5m_bucket, cost_1h)."""
    cw = cw or 0
    c1h = min(c1h or 0, cw)
    return ((cw - c1h) * p[2] / 1e6, c1h * p[0] * 2.0 / 1e6)


def _tiered_cw_cost(cw, c1h, p):
    c5, c1 = _tiered_cw_parts(cw, c1h, p)
    return c5 + c1


def _summary_model_identity(name: str) -> tuple[str, str]:
    """Internal provider/model identity for a daily-summary display key."""
    if " / " in name:
        provider, model = name.split(" / ", 1)
        return provider, model
    # The original ingest path is Claude-only and historically stored bare
    # Anthropic model names. Treat it as the same identity as Pi/Anthropic.
    return "Anthropic", name


def _conditional_model_names(
        identities: set[tuple[str, str]]) -> dict[tuple[str, str], str]:
    """Show providers only when one normalized model has multiple providers."""
    providers: dict[str, set[str]] = defaultdict(set)
    for provider, model in identities:
        providers[model].add(provider)
    return {
        identity: (f"{identity[0]} / {identity[1]}"
                   if len(providers[identity[1]]) > 1 else identity[1])
        for identity in identities
    }


def _round_visible_cost(value: float) -> float:
    """Retain sub-cent reported costs instead of presenting them as zero."""
    return round(value, 4 if 0 < abs(value) < 0.01 else 2)


def _summary_cost_parts(md, pricing):
    """Cost-chart parts, preferring exact Pi-reported component dollars."""
    if md.get("has_reported_cost"):
        return {
            "input": md.get("reported_cost_input", 0),
            "output": md.get("reported_cost_output", 0),
            "cache_5m": 0, "cache_1h": 0,
            "cache_write_reported": md.get("reported_cost_cache_write", 0),
            "cache_read": md.get("reported_cost_cache_read", 0),
            "other": md.get("reported_cost_other", 0),
        }
    c5, c1 = _tiered_cw_parts(md.get("cache_write", 0),
                              md.get("cache_1h", 0), pricing)
    return {
        "input": md.get("input", 0) * pricing[0] / 1e6,
        "output": md.get("output", 0) * pricing[1] / 1e6,
        "cache_5m": c5, "cache_1h": c1, "cache_write_reported": 0,
        "cache_read": md.get("cache_read", 0) * pricing[3] / 1e6,
        "other": 0,
    }


def _build_recent_sessions(conn, pred, limit=25, enterprise=False):
    """Per-session rollup over the recent window: cost, tokens, wall duration,
    burn rate ($/hr). Titles come from desktop_sessions (Claude Desktop pushes
    those); CLI sessions show their project instead. Scope-partitioned via pred.

    Burn rate needs >= 5 min of wall time — extrapolating a one-shot session
    to an hourly rate produces absurd numbers."""
    cutoff_epoch = (datetime.now(TZ) - timedelta(days=RECENCY_DAYS)).timestamp()
    rows = conn.execute(
        "SELECT session_id, model, provider, source_client, speed, inference_geo, COUNT(*) as reqs, "
        "MIN(first_ts) as min_ts, MAX(last_ts) as max_ts, "
        "SUM(inp) as inp, SUM(outp) as outp, SUM(cc) as cc, SUM(cr) as cr, "
        "SUM(c5m) as c5m, SUM(c1h) as c1h, SUM(ws) as ws, "
        "SUM(reported_input) as reported_input, SUM(reported_output) as reported_output, "
        "SUM(reported_cache_read) as reported_cache_read, SUM(reported_cache_write) as reported_cache_write, "
        f"{REPORTED_COST_SUM_SQL} as reported_total, "
        "MAX(machine) as machine, MAX(project_dir) as project_dir "
        "FROM ("
        "  SELECT session_id, model, provider, source_client, request_id, "
        "  MAX(speed) as speed, MAX(inference_geo) as inference_geo, "
        "  MIN(ts_epoch) as first_ts, MAX(ts_epoch) as last_ts, "
        "  MAX(input_tokens) as inp, MAX(output_tokens) as outp, "
        "  MAX(cache_creation_tokens) as cc, MAX(cache_read_tokens) as cr, "
        "  MAX(cache_ephemeral_5m) as c5m, MAX(cache_ephemeral_1h) as c1h, "
        "  MAX(web_search_requests) as ws, "
        "  MAX(reported_cost_input) as reported_input, MAX(reported_cost_output) as reported_output, "
        "  MAX(reported_cost_cache_read) as reported_cache_read, MAX(reported_cost_cache_write) as reported_cache_write, "
        "  MAX(reported_cost_total) as reported_total, "
        "  MAX(source_machine) as machine, MAX(project_dir) as project_dir "
        "  FROM events WHERE type='assistant' AND model IS NOT NULL "
        "  AND model != '<synthetic>' AND request_id IS NOT NULL "
        "  AND session_id IS NOT NULL "
        f"  AND {pred} AND ts_epoch >= ? "
        "  GROUP BY session_id, model, provider, source_client, request_id"
        ") GROUP BY session_id, model, provider, source_client, speed, inference_geo",
        (cutoff_epoch,),
    ).fetchall()

    sessions: dict = {}
    for r in rows:
        sid = r["session_id"]
        st = sessions.setdefault(sid, {
            "session_id": sid, "cost": 0.0, "total_tokens": 0,
            "input": 0, "output": 0, "cache_write": 0, "cache_read": 0,
            "api_calls": 0,
            "min_ts": r["min_ts"], "max_ts": r["max_ts"],
            "machine": r["machine"], "project_dir": r["project_dir"],
            "_model_cost": {},
            "_parts": {"input": 0.0, "output": 0.0, "cache_5m": 0.0,
                       "cache_1h": 0.0, "cache_read": 0.0, "web_search": 0.0},
        })
        dm = display_model_for_row(r["model"], r["provider"], r["source_client"])
        # Group min_ts as the era representative: data-derived, so displayed
        # historical session costs can't move when the wall clock crosses a
        # pricing-era boundary. A session straddling the boundary prices
        # entirely at its start era — including sessions resumed days later
        # under the same session_id. Accepted coarseness; bounded to this
        # card, daily totals are unaffected.
        c = reported_cost(r)
        if c is None:
            c = compute_cost(dm, r["inp"] or 0, r["outp"] or 0, r["cc"] or 0,
                             r["cr"] or 0, r["speed"],
                             effective_geo(r["inference_geo"], enterprise=enterprise),
                             cw_5m=r["c5m"] or 0, cw_1h=r["c1h"] or 0,
                             web_search=r["ws"] or 0, ts_epoch=r["min_ts"])
        st["cost"] += c
        identity = _summary_model_identity(dm)
        st["_model_cost"][identity] = st["_model_cost"].get(identity, 0.0) + c
        # Prefer reported component dollars for Pi. Claude keeps the
        # historical list-price breakdown.
        parts = st["_parts"]
        if r["source_client"] == "pi-agent":
            if any(r[k] is not None for k in ("reported_input", "reported_output",
                                               "reported_cache_read", "reported_cache_write")):
                parts["input"] += r["reported_input"] or 0
                parts["output"] += r["reported_output"] or 0
                parts["cache_read"] += r["reported_cache_read"] or 0
                parts["cache_5m"] += r["reported_cache_write"] or 0
        else:
            p = get_pricing(dm, r["min_ts"])
            if p:
                c5, c1 = _tiered_cw_parts(r["cc"] or 0, r["c1h"] or 0, p)
                parts["input"] += (r["inp"] or 0) * p[0] / 1e6
                parts["output"] += (r["outp"] or 0) * p[1] / 1e6
                parts["cache_5m"] += c5
                parts["cache_1h"] += c1
                parts["cache_read"] += (r["cr"] or 0) * p[3] / 1e6
        # the web-search fee is model-independent and bills even unpriced models
        parts["web_search"] += (r["ws"] or 0) * _WS_FEE
        st["total_tokens"] += (r["inp"] or 0) + (r["outp"] or 0) + (r["cc"] or 0) + (r["cr"] or 0)
        st["input"] += r["inp"] or 0
        st["output"] += r["outp"] or 0
        st["cache_write"] += r["cc"] or 0
        st["cache_read"] += r["cr"] or 0
        st["api_calls"] += r["reqs"] or 0
        st["min_ts"] = min(st["min_ts"], r["min_ts"])
        st["max_ts"] = max(st["max_ts"], r["max_ts"])

    # AI-assigned names (from ai-title transcript records) as the base layer;
    # explicit Claude Desktop titles overlay them.
    prompt_counts = {r["session_id"]: r["n"] for r in conn.execute(
        f"SELECT session_id, COUNT(*) as n FROM events "
        f"WHERE is_human_prompt=1 AND session_id IS NOT NULL AND {pred} "
        f"AND ts_epoch >= ? GROUP BY session_id", (cutoff_epoch,))}

    titles = {r["session_id"]: r["title"] for r in conn.execute(
        "SELECT session_id, title FROM session_titles")}
    titles.update({r["cli_session_id"]: r["title"] for r in conn.execute(
        "SELECT cli_session_id, title FROM desktop_sessions "
        "WHERE title IS NOT NULL AND title != ''")})

    project_names = _make_display_names(
        sorted({st["project_dir"] for st in sessions.values() if st["project_dir"]}))

    session_identities = {
        identity for session in sessions.values()
        for identity in session["_model_cost"]
    }
    display_names = _conditional_model_names(session_identities)
    out = []
    for st in sorted(sessions.values(), key=lambda x: -x["max_ts"])[:limit]:
        duration_s = max(0.0, st["max_ts"] - st["min_ts"])
        burn = round(st["cost"] / (duration_s / 3600), 2) if duration_s >= 300 else None
        primary_identity = (max(st["_model_cost"], key=st["_model_cost"].get)
                            if st["_model_cost"] else None)
        primary_model = display_names.get(primary_identity)
        out.append({
            "session_id": st["session_id"],
            "title": titles.get(st["session_id"]),
            "project": project_names.get(st["project_dir"], st["project_dir"]),
            "machine": canonical_machine(st["machine"]),
            "model": primary_model,
            "cost": _round_visible_cost(st["cost"]),
            "total_tokens": st["total_tokens"],
            "input": st["input"], "output": st["output"],
            "cache_write": st["cache_write"], "cache_read": st["cache_read"],
            "api_calls": st["api_calls"],
            "prompts": prompt_counts.get(st["session_id"], 0),
            "models": [{"model": display_names[identity],
                        "cost": _round_visible_cost(cost)}
                       for identity, cost in sorted(
                           st["_model_cost"].items(), key=lambda item: -item[1])],
            "first_ts": st["min_ts"],
            "duration_s": round(duration_s),
            "burn_per_hr": burn,
            "last_ts": st["max_ts"],
            "cost_parts": {k: round(v, 4) for k, v in st["_parts"].items()},
        })
    return out


def build_dashboard_data(scope: str = DEFAULT_SCOPE) -> dict:
    """Return cached dashboard data for the given scope, rebuilding if missing."""
    # Record the request so the drain-loop worker keeps this scope pre-warmed
    # (see _warm_scopes) — otherwise a non-default scope refetches cold every ~1s.
    _mark_scope_requested(scope)
    with _cache_lock:
        cached = _cached_data.get(scope)
        if cached is not None:
            return cached
        gen_at_start = _cache_gen
    # Build outside the lock to avoid blocking concurrent readers.
    data = _build_dashboard_data_inner(scope)
    with _cache_lock:
        # Only store if no invalidation landed while we were building: a slow
        # request-thread build that straddled an invalidation would otherwise
        # clobber the cache with pre-invalidation data. Return the built data to
        # the caller either way — it is correct as of when the build started.
        if _cache_gen == gen_at_start:
            _cached_data[scope] = data
    return data


def _geo_assumed(scope: str) -> bool:
    """True when this enterprise view bills at the assumed US 1.1x rate —
    surfaced in the payload so the UI can disclose the assumption."""
    import app.config as _config
    return scope == "enterprise" and _config.ENTERPRISE_ASSUME_GEO == "us"


def _empty_dashboard(cutoff_date: str, scope: str = DEFAULT_SCOPE) -> dict:
    """Return the dashboard structure with no data (used when no summaries exist)."""
    now = datetime.now(TZ)
    # Built once and shared with month_hero_block: two build_meter_payload
    # calls could straddle a fresh capture and hand the hero a different
    # reading than the meter panel renders.
    _meter = build_meter_payload(get_conn(), scope)
    return {
        "cards": {
            "sessions": 0, "human_prompts": 0, "total_tokens": 0,
            "active_time_s": 0, "tool_calls": 0, "models_used": 0,
            "avg_prompts_day": 0, "avg_active_day_s": 0,
        },
        "daily": [], "tools": {}, "recent_tools": {}, "tool_full_names": {},
        "time_breakdown": {
            "thinking": 0, "tool_execution": 0, "subagent": 0, "agent_runs": 0,
            "recent_subagent": 0, "recent_agent_runs": 0,
            "recent_thinking": 0, "recent_tool_execution": 0,
        },
        "projects": [], "recent_sessions": [], "model_breakdown": [],
        "served_models": {"all": {}, "14d": {}, "today": {}},
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
        "scope": scope,
        "month_cost": 0.0,
        "month_label": datetime.now(_timezone.utc).strftime("%B %Y") + " · UTC",
        # Meter readings are first-class data, independent of daily summaries —
        # they must survive the no-summaries-yet path (fresh enterprise view).
        "meter": _meter,
        # A fresh enterprise instance with no summaries can still have a
        # billed figure: the meter is captured independently of ingest.
        "month_hero": month_hero_block(0.0, _meter),
        "geo_assumed": _geo_assumed(scope),
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


def _build_hourly(conn, pred: str, enterprise: bool = False) -> list[dict]:
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
        f"FROM events WHERE is_human_prompt=1 AND {pred} AND ts_epoch>=? AND ts_epoch<? "
        "GROUP BY bucket", (h_start_epoch, h_end_epoch),
    ):
        idx = epoch_to_idx.get(r["bucket"])
        if idx is not None:
            hourly_list[idx]["prompts"] = r["cnt"]

    for r in conn.execute(
        f"SELECT CAST(ts_epoch / 3600 AS INTEGER) * 3600 as bucket, COUNT(*) as cnt "
        f"FROM tool_uses WHERE ts_epoch>=? AND ts_epoch<? "
        f"AND session_id IN (SELECT session_id FROM events WHERE {pred}) "
        "GROUP BY bucket", (h_start_epoch, h_end_epoch),
    ):
        idx = epoch_to_idx.get(r["bucket"])
        if idx is not None:
            hourly_list[idx]["tool_calls"] = r["cnt"]

    for r in conn.execute(
        f"SELECT CAST(first_ts / 3600 AS INTEGER) * 3600 as bucket, model, provider, source_client, speed, inference_geo, "
        "SUM(inp) as inp, SUM(outp) as outp, SUM(cc) as cc, SUM(cr) as cr, "
        "SUM(c5m) as c5m, SUM(c1h) as c1h, SUM(ws) as ws, "
        "SUM(reported_input) as reported_input, SUM(reported_output) as reported_output, "
        "SUM(reported_cache_read) as reported_cache_read, SUM(reported_cache_write) as reported_cache_write, "
        f"{REPORTED_COST_SUM_SQL} as reported_total "
        "FROM ("
        f"  SELECT MIN(ts_epoch) as first_ts, model, provider, source_client, request_id, "
        "  MAX(speed) as speed, MAX(inference_geo) as inference_geo, "
        "  MAX(input_tokens) as inp, MAX(output_tokens) as outp, "
        "  MAX(cache_creation_tokens) as cc, MAX(cache_read_tokens) as cr, "
        "  MAX(cache_ephemeral_5m) as c5m, MAX(cache_ephemeral_1h) as c1h, "
        "  MAX(web_search_requests) as ws, "
        "  MAX(reported_cost_input) as reported_input, MAX(reported_cost_output) as reported_output, "
        "  MAX(reported_cost_cache_read) as reported_cache_read, MAX(reported_cost_cache_write) as reported_cache_write, "
        "  MAX(reported_cost_total) as reported_total "
        f"  FROM events WHERE type='assistant' AND model IS NOT NULL "
        f"  AND model != '<synthetic>' AND request_id IS NOT NULL "
        f"  AND {pred} "
        "  AND ts_epoch>=? AND ts_epoch<? "
        "  GROUP BY model, provider, source_client, request_id"
        ") GROUP BY bucket, model, provider, source_client, speed, inference_geo",
        (h_start_epoch, h_end_epoch),
    ):
        idx = epoch_to_idx.get(r["bucket"])
        if idx is not None:
            dm = display_model_for_row(r["model"], r["provider"], r["source_client"])
            # hour-bucket epoch as era representative for this group
            c = reported_cost(r)
            if c is None:
                c = compute_cost(
                    dm, r["inp"] or 0, r["outp"] or 0, r["cc"] or 0, r["cr"] or 0,
                    r["speed"],
                    effective_geo(r["inference_geo"], enterprise=enterprise),
                    cw_5m=r["c5m"] or 0, cw_1h=r["c1h"] or 0,
                    web_search=r["ws"] or 0, ts_epoch=r["bucket"])
            hourly_list[idx]["cost"] += c

    for hl in hourly_list:
        hl["cost"] = round(hl["cost"], 2)
        del hl["_epoch"]

    return hourly_list


def _build_today_data(conn, today_str: str, pred: str) -> dict:
    """Build the 'today' sub-object from today's daily_summary row.

    Returns model_breakdown, time_breakdown, tools, projects, and
    machine_summary scoped to just today.
    """
    ent_rows = conn.execute(
        f"SELECT * FROM daily_summary WHERE day = ? AND {pred}", (today_str,)
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

    # Model breakdown for today. Merge storage keys that describe the same
    # provider/model identity, while decomposing cost before the merge so
    # reported Pi components and priced Claude components both survive.
    raw_model_data = json.loads(row["model_json"] or "{}")
    raw_gen_data = json.loads(row["gen_json"] or "{}")
    model_data: dict[tuple[str, str], dict] = defaultdict(dict)
    cost_parts: dict[tuple[str, str], dict] = defaultdict(
        lambda: defaultdict(float))
    gen_data: dict[tuple[str, str], dict] = defaultdict(
        lambda: defaultdict(float))
    for storage_name, source in raw_model_data.items():
        identity = _summary_model_identity(storage_name)
        target = model_data[identity]
        for key, value in source.items():
            if key == "has_reported_cost":
                target[key] = bool(target.get(key)) or bool(value)
            elif isinstance(value, (int, float)):
                target[key] = target.get(key, 0) + value
        parts = _summary_cost_parts(source, get_pricing(storage_name))
        for key, value in parts.items():
            cost_parts[identity][key] += value
    for storage_name, source in raw_gen_data.items():
        target = gen_data[_summary_model_identity(storage_name)]
        target["gen_s"] += source.get("gen_s", 0.0)
        target["gen_out"] += source.get("gen_out", 0)

    display_names = _conditional_model_names(set(model_data))
    today_mb = []
    for identity in sorted(model_data, key=lambda item: model_sort_key(item[1])):
        _provider, base_name = identity
        mname = display_names[identity]
        md = model_data[identity]
        inp = md.get("input", 0)
        out = md.get("output", 0)
        cw = md.get("cache_write", 0)
        cr = md.get("cache_read", 0)
        cost = md.get("cost", 0.0)
        parts = cost_parts[identity]
        main_cost = md.get("main_cost", 0.0)
        agent_cost = round(cost - main_cost, 2)
        main_prompts = md.get("main_prompts", 0)
        agent_invocations = md.get("agent_invocations", 0)
        avg_cost_per_turn = (main_cost / main_prompts) if main_prompts > 0 else None
        avg_cost_per_agent = (agent_cost / agent_invocations) if agent_invocations > 0 else None
        active_hours = md.get("active_s", 0.0) / 3600
        gd = gen_data.get(identity, {})
        gen_s = gd.get("gen_s", 0.0)
        gen_out = gd.get("gen_out", 0)
        energy = compute_energy_wh(base_name, inp, out)
        water = compute_water_ml(base_name, inp, out)
        today_mb.append({
            "model": mname,
            "unpriced": not is_priced(base_name) and not md.get("has_reported_cost"),
            "has_reported_cost": bool(md.get("has_reported_cost")),
            "api_calls": md.get("api_calls", 0),
            "input": inp, "output": out,
            "cache_write": cw, "cache_read": cr,
            "total_tokens": inp + out + cw + cr,
            "cost": _round_visible_cost(cost),
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
            "cache_5m": md.get("cache_5m", 0),
            "cache_1h": md.get("cache_1h", 0),
            "cost_input": _round_visible_cost(parts["input"]),
            "cost_output": _round_visible_cost(parts["output"]),
            "cost_cache_write": _round_visible_cost(
                parts["cache_5m"] + parts["cache_1h"]
                + parts["cache_write_reported"]),
            "cost_cache_5m": _round_visible_cost(parts["cache_5m"]),
            "cost_cache_1h": _round_visible_cost(parts["cache_1h"]),
            "cost_cache_write_reported": _round_visible_cost(
                parts["cache_write_reported"]),
            "cost_cache_read": _round_visible_cost(parts["cache_read"]),
            "cost_other": _round_visible_cost(parts["other"]),
            "web_search": md.get("web_search", 0),
            "web_fetch": md.get("web_fetch", 0),
            "cost_web_search": round(md.get("web_search", 0) * _WS_FEE, 2),
            # Today view uses the same keys as recent/all for compatibility
            "recent_cost": _round_visible_cost(cost),
            "recent_main_cost": round(main_cost, 2),
            "recent_agent_cost": agent_cost,
            "recent_cost_per_hour": round(cost / active_hours, 2) if active_hours > 0 else None,
            "recent_output_tok_per_s": round(gen_out / gen_s, 1) if gen_s > 0 else None,
            "recent_active_hours": round(active_hours, 1),
            "recent_cost_input": _round_visible_cost(parts["input"]),
            "recent_cost_output": _round_visible_cost(parts["output"]),
            "recent_cost_cache_write": _round_visible_cost(
                parts["cache_5m"] + parts["cache_1h"]
                + parts["cache_write_reported"]),
            "recent_cost_cache_5m": _round_visible_cost(parts["cache_5m"]),
            "recent_cost_cache_1h": _round_visible_cost(parts["cache_1h"]),
            "recent_cost_cache_write_reported": _round_visible_cost(
                parts["cache_write_reported"]),
            "recent_cost_cache_read": _round_visible_cost(parts["cache_read"]),
            "recent_cost_other": _round_visible_cost(parts["other"]),
            "recent_cost_web_search": round(md.get("web_search", 0) * _WS_FEE, 2),
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

    # Tools for today (display-named like the all-time/recent counters)
    tools, today_tool_fulls = _display_tool_counts(
        Counter(json.loads(row["tool_json"] or "{}")))

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

    # Machine summary for today — accumulate per CANONICAL machine so hostname
    # variants of one box merge instead of appearing as separate rows.
    mach_data = json.loads(row["machine_json"] or "{}")
    mach_canon: dict[str, dict] = defaultdict(lambda: {
        "prompts": 0, "calls": 0, "tool_calls": 0, "total_tokens": 0, "cost": 0.0,
    })
    for raw_mname, mv in mach_data.items():
        mc = mach_canon[canonical_machine(raw_mname)]
        mc["prompts"] += mv.get("prompts", 0)
        mc["calls"] += mv.get("calls", 0)
        mc["tool_calls"] += mv.get("tool_calls", 0)
        mc["total_tokens"] += (mv.get("input", 0) + mv.get("output", 0)
                               + mv.get("cache_write", 0) + mv.get("cache_read", 0))
        mc["cost"] += mv.get("cost", 0)
    machine_summary = [
        {
            "machine": mname,
            "prompts": mc["prompts"],
            "api_calls": mc["calls"],
            "tool_calls": mc["tool_calls"],
            "total_tokens": mc["total_tokens"],
            "cost": round(mc["cost"], 2),
        }
        for mname, mc in sorted(mach_canon.items(), key=lambda kv: -kv[1]["prompts"])
    ]

    return {
        "cost": round(row["cost"] or 0.0, 2),
        "model_breakdown": today_mb,
        "time_breakdown": time_breakdown,
        "tools": tools,
        "tool_full_names": today_tool_fulls,
        "projects": projects,
        "machine_summary": machine_summary,
    }


def _served_model_chips_safe(conn, pred: str, cutoff_date: str) -> dict:
    """served_model_chips with a narrow blast radius (api.py Fix 6 pattern).

    The served-model capture is a best-effort observatory over a format
    Anthropic changes without notice; a surprise in it must cost the dashboard
    one small chip, never the whole page.
    """
    empty = {"all": {}, "14d": {}, "today": {}}
    try:
        return served_model_chips(
            conn, pred, cutoff_date, datetime.now(TZ).strftime("%Y-%m-%d"))
    except Exception:
        logger.exception("served-model chip build failed, chips omitted")
        return empty


def _build_dashboard_data_inner(scope: str = DEFAULT_SCOPE) -> dict:
    """Read daily_summary rows and produce the full dashboard JSON blob for the given scope."""
    load_pricing()
    conn = get_conn()
    cutoff_date = (datetime.now(TZ) - timedelta(days=RECENCY_DAYS)).strftime("%Y-%m-%d")
    pred = scope_predicate(scope)

    # ── Read summary rows for the requested scope ──
    rows = conn.execute(
        f"SELECT * FROM daily_summary WHERE {pred} ORDER BY day"
    ).fetchall()

    if not rows:
        return _empty_dashboard(cutoff_date, scope)

    # (org values are not served; server-side predicate uses org as filter only)

    # ── Accumulators ──
    model_stats = defaultdict(lambda: {
        "input": 0, "output": 0, "cache_write": 0, "cache_read": 0,
        "cache_5m": 0, "cache_1h": 0, "recent_cache_5m": 0, "recent_cache_1h": 0,
        "web_search": 0, "web_fetch": 0, "recent_web_search": 0,
        "api_calls": 0, "main_api_calls": 0, "main_cost": 0.0,
        "main_prompts": 0, "agent_invocations": 0, "active_s": 0.0,
        "gen_s": 0.0, "gen_out": 0,
        "recent_active_s": 0.0, "recent_gen_s": 0.0, "recent_gen_out": 0,
        "recent_input": 0, "recent_output": 0, "recent_cache_write": 0,
        "recent_cache_read": 0, "recent_main_cost": 0.0, "last_seen": "",
        "cost": 0.0, "recent_cost": 0.0, "has_reported_cost": False,
        # Dollar-parts decomposition, accumulated per-day at that day's era so
        # displayed historical costs never move when the wall clock crosses a
        # pricing-era boundary (see the model_json merge loop).
        "cost_input": 0.0, "cost_output": 0.0, "cost_cache_5m": 0.0,
        "cost_cache_1h": 0.0, "cost_cache_write_reported": 0.0,
        "cost_cache_read": 0.0, "cost_other": 0.0,
        "recent_cost_input": 0.0, "recent_cost_output": 0.0,
        "recent_cost_cache_5m": 0.0, "recent_cost_cache_1h": 0.0,
        "recent_cost_cache_write_reported": 0.0,
        "recent_cost_cache_read": 0.0, "recent_cost_other": 0.0,
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
        # Local-day start epoch: era representative for this day's dollar-parts
        # decomposition. Coarser than summarizer (which prices per-request at
        # first_ts): on a local day straddling the UTC era boundary, the
        # boundary-evening requests decompose at the day-start era while the
        # day's `cost` was priced per-request, so that one day's cost-mix may
        # not reconcile exactly. Accepted; stable across wall clock either way.
        day_epoch = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=TZ).timestamp()

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
            identity = _summary_model_identity(mname)
            models_seen.add(identity)
            ms = model_stats[identity]
            ms["input"] += md.get("input", 0)
            ms["output"] += md.get("output", 0)
            ms["cache_write"] += md.get("cache_write", 0)
            ms["cache_5m"] += md.get("cache_5m", 0)
            ms["cache_1h"] += md.get("cache_1h", 0)
            ms["web_search"] += md.get("web_search", 0)
            ms["web_fetch"] += md.get("web_fetch", 0)
            ms["cache_read"] += md.get("cache_read", 0)
            ms["api_calls"] += md.get("api_calls", 0)
            ms["main_api_calls"] += md.get("main_api_calls", 0)
            ms["main_cost"] += md.get("main_cost", 0.0)
            ms["main_prompts"] += md.get("main_prompts", 0)
            ms["agent_invocations"] += md.get("agent_invocations", 0)
            ms["active_s"] += md.get("active_s", 0.0)
            ms["cost"] += md.get("cost", 0.0)
            ms["has_reported_cost"] = (ms["has_reported_cost"]
                                       or bool(md.get("has_reported_cost")))
            # List-price dollar parts (no geo/fast modifiers), priced at THIS
            # day's era — a Sonnet 5 aggregate spanning the intro/standard flip
            # decomposes each day at its own rates.
            p = get_pricing(mname, day_epoch)
            parts = _summary_cost_parts(md, p)
            ms["cost_input"] += parts["input"]
            ms["cost_output"] += parts["output"]
            ms["cost_cache_5m"] += parts["cache_5m"]
            ms["cost_cache_1h"] += parts["cache_1h"]
            ms["cost_cache_write_reported"] += parts["cache_write_reported"]
            ms["cost_cache_read"] += parts["cache_read"]
            ms["cost_other"] += parts["other"]
            if day > ms["last_seen"]:
                ms["last_seen"] = day
            if is_recent:
                ms["recent_input"] += md.get("input", 0)
                ms["recent_output"] += md.get("output", 0)
                ms["recent_cache_write"] += md.get("cache_write", 0)
                ms["recent_cache_5m"] += md.get("cache_5m", 0)
                ms["recent_cache_1h"] += md.get("cache_1h", 0)
                ms["recent_web_search"] += md.get("web_search", 0)
                ms["recent_cache_read"] += md.get("cache_read", 0)
                ms["recent_main_cost"] += md.get("main_cost", 0.0)
                ms["recent_active_s"] += md.get("active_s", 0.0)
                ms["recent_cost"] += md.get("cost", 0.0)
                ms["recent_cost_input"] += parts["input"]
                ms["recent_cost_output"] += parts["output"]
                ms["recent_cost_cache_5m"] += parts["cache_5m"]
                ms["recent_cost_cache_1h"] += parts["cache_1h"]
                ms["recent_cost_cache_write_reported"] += parts["cache_write_reported"]
                ms["recent_cost_cache_read"] += parts["cache_read"]
                ms["recent_cost_other"] += parts["other"]

        # ── gen_json (generation time — stored separately from model_json) ──
        gen_data = json.loads(row["gen_json"] or "{}")
        for mname, gd in gen_data.items():
            ms = model_stats[_summary_model_identity(mname)]
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
        for raw_mname, md in mach_data.items():
            mname = canonical_machine(raw_mname)
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

    # ── Tool counts (top 20, MCP ids display-named; raw ids kept for tooltips) ──
    tool_counts, _tool_fulls = _display_tool_counts(all_tool_counts)
    recent_tools, _recent_fulls = _display_tool_counts(recent_tool_counts)

    # ── Model breakdown ──
    total_cost = 0.0
    model_breakdown = []
    display_names = _conditional_model_names(set(model_stats))
    for identity in sorted(model_stats, key=lambda item: model_sort_key(item[1])):
        _provider, base_name = identity
        name = display_names[identity]
        ms = model_stats[identity]
        total_tok = ms["input"] + ms["output"] + ms["cache_write"] + ms["cache_read"]
        cost = ms["cost"]
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
        energy = compute_energy_wh(base_name, ms["input"], ms["output"])
        water = compute_water_ml(base_name, ms["input"], ms["output"])
        recent_energy = compute_energy_wh(base_name, ms["recent_input"], ms["recent_output"])
        recent_water = compute_water_ml(base_name, ms["recent_input"], ms["recent_output"])
        model_breakdown.append({
            "model": name,
            "unpriced": not is_priced(base_name) and not ms["has_reported_cost"],
            "has_reported_cost": ms["has_reported_cost"],
            "api_calls": ms["api_calls"],
            "input": ms["input"], "output": ms["output"],
            "cache_write": ms["cache_write"], "cache_read": ms["cache_read"],
            "total_tokens": total_tok, "cost": _round_visible_cost(cost),
            "main_cost": main_cost, "agent_cost": agent_cost,
            "avg_cost_per_turn": round(avg_cost_per_turn, 4) if avg_cost_per_turn is not None else None,
            "avg_cost_per_agent": round(avg_cost_per_agent, 4) if avg_cost_per_agent is not None else None,
            "main_prompts": ms["main_prompts"],
            "agent_invocations": ms["agent_invocations"],
            "active_hours": round(active_hours, 1),
            "cost_per_hour": round(cost_per_hour, 2) if cost_per_hour is not None else None,
            "output_tok_per_s": round(output_tok_per_s, 1) if output_tok_per_s is not None else None,
            "cache_5m": ms["cache_5m"], "cache_1h": ms["cache_1h"],
            "recent_cache_5m": ms["recent_cache_5m"], "recent_cache_1h": ms["recent_cache_1h"],
            # dollar parts come from the per-day era-priced accumulators
            "cost_input": _round_visible_cost(ms["cost_input"]),
            "cost_output": _round_visible_cost(ms["cost_output"]),
            "cost_cache_write": _round_visible_cost(
                ms["cost_cache_5m"] + ms["cost_cache_1h"]
                + ms["cost_cache_write_reported"]),
            "cost_cache_5m": _round_visible_cost(ms["cost_cache_5m"]),
            "cost_cache_1h": _round_visible_cost(ms["cost_cache_1h"]),
            "cost_cache_write_reported": _round_visible_cost(
                ms["cost_cache_write_reported"]),
            "cost_cache_read": _round_visible_cost(ms["cost_cache_read"]),
            "cost_other": _round_visible_cost(ms["cost_other"]),
            "web_search": ms["web_search"],
            "web_fetch": ms["web_fetch"],
            "cost_web_search": round(ms["web_search"] * _WS_FEE, 2),
            "last_seen": ms["last_seen"],
            "recent": ms["last_seen"] >= cutoff_date,
            "recent_input": ms["recent_input"], "recent_output": ms["recent_output"],
            "recent_cache_write": ms["recent_cache_write"], "recent_cache_read": ms["recent_cache_read"],
            "recent_total_tokens": ms["recent_input"] + ms["recent_output"] + ms["recent_cache_write"] + ms["recent_cache_read"],
            "recent_cost": _round_visible_cost(recent_cost),
            "recent_main_cost": round(ms["recent_main_cost"], 2),
            "recent_agent_cost": round(recent_cost - ms["recent_main_cost"], 2),
            "recent_cost_input": _round_visible_cost(ms["recent_cost_input"]),
            "recent_cost_output": _round_visible_cost(ms["recent_cost_output"]),
            "recent_cost_cache_write": _round_visible_cost(
                ms["recent_cost_cache_5m"] + ms["recent_cost_cache_1h"]
                + ms["recent_cost_cache_write_reported"]),
            "recent_cost_cache_5m": _round_visible_cost(ms["recent_cost_cache_5m"]),
            "recent_cost_cache_1h": _round_visible_cost(ms["recent_cost_cache_1h"]),
            "recent_cost_cache_write_reported": _round_visible_cost(
                ms["recent_cost_cache_write_reported"]),
            "recent_cost_cache_read": _round_visible_cost(ms["recent_cost_cache_read"]),
            "recent_cost_other": _round_visible_cost(ms["recent_cost_other"]),
            "recent_cost_web_search": round(ms["recent_web_search"] * _WS_FEE, 2),
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

    # ── Month-to-date cost: UTC month boundaries, to match Anthropic billing ──
    # Anthropic accounts on UTC days (UTC midnight = ~7pm America/Chicago), so
    # bucketing by the local `day` column made this counter disagree with the
    # Claude account page for evening usage on month edges. Computed from raw
    # events via the shared window-cost path (request dedup + full pricing,
    # incl. server-tool fees). ONLY this metric is UTC — heatmap/daily tables
    # stay local by design.
    _now_utc = datetime.now(_timezone.utc)
    _month_start_utc = _now_utc.replace(day=1, hour=0, minute=0,
                                        second=0, microsecond=0)
    month_cost = round(compute_window_cost(
        conn, _month_start_utc.timestamp(), _now_utc.timestamp() + 1, scope), 2)
    month_label = _now_utc.strftime("%B %Y") + " · UTC"
    # Read once and reuse for both "meter" and "month_hero": a second call
    # could straddle a fresh capture and leave the hero disagreeing with the
    # meter panel rendered from the same payload.
    _meter = build_meter_payload(conn, scope)

    # ── Last active timestamp (global + per machine) ──
    # daily_summary can't answer "active in last 15 minutes" — needs minute-grain
    # timestamps from raw events. Indexed on source_machine, cheap.
    machine_last_active: dict[str, float] = {}
    last_active_ts = None
    for r in conn.execute(
        f"SELECT source_machine, MAX(ts_epoch) as ts FROM events "
        f"WHERE {pred} "
        "GROUP BY source_machine"
    ):
        if r["ts"]:
            mkey = canonical_machine(r["source_machine"])
            # variants of one machine merge: keep the most recent timestamp
            if r["ts"] > machine_last_active.get(mkey, 0):
                machine_last_active[mkey] = r["ts"]
            if last_active_ts is None or r["ts"] > last_active_ts:
                last_active_ts = r["ts"]

    # ── Sessions count (total distinct, from summary) ──
    # Note: session counts from summaries are per-day distinct, so the total
    # may overcount sessions spanning midnight.  Use the sum as a close approx.
    sessions_count = tot["sessions"]

    # ── Hourly (live query) ──
    hourly_list = _build_hourly(conn, pred, enterprise=(scope == "enterprise"))

    # ── Recent sessions (burn rate per task) ──
    recent_sessions = _build_recent_sessions(
        conn, pred, enterprise=(scope == "enterprise"))

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
        "tool_full_names": {**_tool_fulls, **_recent_fulls},
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
        "recent_sessions": recent_sessions,
        "model_breakdown": model_breakdown,
        # Per-mode chip text, keyed by display model name. Its own small query
        # over events: daily_summary has no served-model dimension, and this
        # signal is deliberately not rolled up (see app/served_models.py).
        "served_models": _served_model_chips_safe(conn, pred, cutoff_date),
        "unpriced_models": sorted(m["model"] for m in model_breakdown if m["unpriced"]),
        "total_cost": round(total_cost, 2),
        "total_orch_cost": round(sum(m["main_cost"] for m in model_breakdown), 2),
        "total_agent_cost": round(sum(m["agent_cost"] for m in model_breakdown), 2),
        "benchmarks": {
            display_names[identity]: MODEL_BENCHMARKS.get(identity[1], {})
            for identity in model_stats if MODEL_BENCHMARKS.get(identity[1])
        },
        # CURRENT list-price tables (rate columns on the dashboard): wall-clock
        # era is semantically intended — they show what a model costs now, not
        # a historical blend, so no ts_epoch here.
        "output_pricing": {
            display_names[identity]: get_pricing(identity[1])[1]
            for identity in model_stats
        },
        "model_pricing": {
            display_names[identity]: {
                "input": p[0], "output": p[1], "cache_write": p[2],
                "cache_write_1h": round(p[0] * 2.0, 4), "cache_read": p[3]
            }
            for identity in model_stats for p in [get_pricing(identity[1])]
        },
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
        "today": _build_today_data(conn, datetime.now(TZ).strftime("%Y-%m-%d"), pred),
        "scope": scope,
        "month_cost": month_cost,
        "meter": _meter,
        # Authoritative MTD figure: billed when a fresh meter exists, the
        # estimate otherwise. See app/month_hero.py for the invariant.
        "month_hero": month_hero_block(month_cost, _meter),
        "geo_assumed": _geo_assumed(scope),
        "month_label": month_label,
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
        # Swallowing keeps the timer alive; logging keeps the failure visible
        # (the 2026-07-12 wipe died silently behind a bare `pass`).
        logger.exception("hourly sweep failed — last-7-days re-roll aborted")
    try:
        checkpoint_wal()
    except Exception:
        logger.warning("hourly WAL checkpoint failed", exc_info=True)
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
        logger.exception("full sweep failed — all-days daily_summary rebuild "
                         "aborted (table preserved; rebuild retries in 24h)")
    _schedule_full_sweep()
