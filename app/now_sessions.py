"""Event-side facts for the live sessions of /api/now. No routes.

agent_state knows which sessions are live; events know what they cost. This
module joins the two:

- Claude Code hooks report the transcript session id, which is the events
  session_id as-is.
- Pi reports its native session id, but Pi ingest namespaces every id as
  pi:<hash(class, machine, file, sid)>:<sid> (ingest._pi_namespace). The
  hash needs the session file name, which notify never sees, so the ids are
  matched by suffix against Pi sessions with recent billable rows.

Costs use the canonical row predicate and MAX-per-request dedupe of
cost_windows, so a Pi request and its Meridian SDK twin bill once.
"""
import hashlib
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from .config import TZ_NAME, enterprise_predicate
from .event_dedupe import canonical_assistant_predicate
from .ingest import _pi_namespace
from .pricing import (REPORTED_COST_SUM_SQL, compute_cost,
                      display_model_for_row, effective_geo, era_boundaries,
                      reported_cost)

# Pi sessions are looked up among those with a billable row this recently.
# agent_state forgets a session 10 min (2 h when waiting) after its last
# hook event, so a live session with older usage only is rare; such a
# session shows without cost until its next turn lands.
PI_LOOKUP_WINDOW_S = 48 * 3600
_PI_SCAN_TTL_S = 15.0
BURN_WINDOW_S = 1800
_MAX_IDS = 400                     # bound the IN (...) lists

# Pi subagent cursors written this recently are parent-link candidates.
PI_SUBAGENT_LOOKBACK_S = 3 * 86400

_pi_lock = threading.Lock()
_pi_scan: dict = {"at": 0.0, "by_native": {}}
_sub_scan: dict = {"at": 0.0, "files": []}
_parent_of: dict[str, str] = {}      # child native sid -> parent native sid


def _recent_pi_sessions(conn, now: float) -> dict[str, list[str]]:
    """native Pi sid -> namespaced events session ids (recent rows only).

    One index range scan (~25 ms on production), reused for 15 s: /api/now
    polls every 5 s, and a new Pi session's first rows arrive at turn end."""
    with _pi_lock:
        if time.monotonic() - _pi_scan["at"] < _PI_SCAN_TTL_S:
            return _pi_scan["by_native"]
    by_native: dict[str, list[str]] = {}
    for (sid,) in conn.execute(
            "SELECT DISTINCT session_id FROM events INDEXED BY idx_events_billable_ts "
            "WHERE type='assistant' AND request_id IS NOT NULL AND ts_epoch >= ? "
            "AND session_id GLOB 'pi:*:*'", (now - PI_LOOKUP_WINDOW_S,)):
        native = sid.split(":", 2)[2]
        by_native.setdefault(native, []).append(sid)
    with _pi_lock:
        _pi_scan.update(at=time.monotonic(), by_native=by_native)
    return by_native


def reset_cache() -> None:
    """Test support."""
    with _pi_lock:
        _pi_scan.update(at=0.0, by_native={})
        _sub_scan.update(at=0.0, files=[])
        _parent_of.clear()


def _subagent_files(conn, now: float) -> list[tuple[str, str, str]]:
    """(class, machine, session_file) of recent Pi subagent cursors.

    Pi writes a subagent's transcript under subagents/<parent sid>/, and
    ingest keys its cursor by that file (~3 ms scan, reused for 15 s)."""
    with _pi_lock:
        if time.monotonic() - _sub_scan["at"] < _PI_SCAN_TTL_S:
            return _sub_scan["files"]
    # updated_at is the server's local ISO time; a 3-day margin absorbs
    # the offset change at a DST switch in this string comparison.
    since = datetime.fromtimestamp(now - PI_SUBAGENT_LOOKBACK_S,
                                   ZoneInfo(TZ_NAME)).isoformat()
    files = []
    for r in conn.execute(
            "SELECT machine, session_file FROM sync_cursors WHERE machine GLOB 'pi:*:*' "
            "AND session_file GLOB 'subagents/*/*' AND updated_at >= ?", (since,)):
        _, cls, machine = r["machine"].split(":", 2)
        files.append((cls, machine, r["session_file"]))
    with _pi_lock:
        _sub_scan.update(at=time.monotonic(), files=files)
    return files


def pi_parents(conn, children: dict[str, list[str]], now: float) -> dict[str, str]:
    """Pi subagent sid -> parent sid.

    A child's events id is pi:<ns>:<sid> with ns = ingest._pi_namespace(class,
    machine, session_file, sid); the cursor whose file reproduces ns names
    the parent folder. Found links are kept (they never change).

    Runs on worker threads (/api/now and /api/today at once): the link
    cache is read and written under _pi_lock."""
    with _pi_lock:
        out = {c: _parent_of[c] for c in children if c in _parent_of}
    todo = {c: eids for c, eids in children.items() if c not in out}
    if not todo:
        return out
    files = _subagent_files(conn, now)
    for child, eids in todo.items():
        suffix = ":" + child
        wanted = {e[:-len(suffix)] for e in eids
                  if e.startswith("pi:") and e.endswith(suffix)}
        for cls, machine, f in files:
            if _pi_namespace(cls, machine, f, child) in wanted:
                out[child] = f.split("/", 2)[1]
                break
    with _pi_lock:
        if len(_parent_of) > 4096:
            _parent_of.clear()
        _parent_of.update(out)
    return out


def resolve_event_sessions(conn, live_ids, now: float) -> dict[str, list[str]]:
    """agent_state sid -> events session ids (itself for Claude Code)."""
    by_native = _recent_pi_sessions(conn, now)
    return {sid: [sid, *by_native.get(sid, [])] for sid in live_ids}


def mirror_sessions(conn, ids) -> set[str]:
    """Meridian's SDK copies of Pi sessions (event_dedupe twin sessions)."""
    ids = list(ids)[:_MAX_IDS]
    if not ids:
        return set()
    marks = ",".join("?" for _ in ids)
    return {r[0] for r in conn.execute(
        f"SELECT session_id FROM pi_mirror_sessions WHERE session_id IN ({marks})", ids)}


def _rollup_sql(n_ids: int, bounds: list) -> str:
    era = ("(" + " + ".join("(first_ts >= ?)" for _ in bounds) + ")"
           if bounds else "0")
    marks = ",".join("?" for _ in range(n_ids))
    return (
        "SELECT session_id, model, provider, source_client, speed, inference_geo, "
        f"{era} AS era, (first_ts >= ?) AS recent, MAX(is_main) AS is_main, "
        "MAX(agent_id) AS agent_id, "
        "MIN(first_ts) AS first_ts, MAX(last_ts) AS last_ts, MAX(is_ent) AS is_ent, "
        "SUM(inp) AS inp, SUM(outp) AS outp, SUM(cc) AS cc, SUM(cr) AS cr, "
        "SUM(c5m) AS c5m, SUM(c1h) AS c1h, SUM(ws) AS ws, "
        "SUM(reported_input) AS reported_input, SUM(reported_output) AS reported_output, "
        "SUM(reported_cache_read) AS reported_cache_read, "
        "SUM(reported_cache_write) AS reported_cache_write, "
        f"{REPORTED_COST_SUM_SQL} AS reported_total "
        "FROM ("
        "  SELECT session_id, model, provider, source_client, request_id, "
        "  MIN(ts_epoch) AS first_ts, MAX(ts_epoch) AS last_ts, "
        "  MAX(agent_id IS NULL AND is_sidechain=0) AS is_main, MAX(agent_id) AS agent_id, "
        f"  MAX({enterprise_predicate()}) AS is_ent, "
        "  MAX(speed) AS speed, MAX(inference_geo) AS inference_geo, "
        "  MAX(input_tokens) AS inp, MAX(output_tokens) AS outp, "
        "  MAX(cache_creation_tokens) AS cc, MAX(cache_read_tokens) AS cr, "
        "  MAX(cache_ephemeral_5m) AS c5m, MAX(cache_ephemeral_1h) AS c1h, "
        "  MAX(web_search_requests) AS ws, "
        "  MAX(reported_cost_input) AS reported_input, MAX(reported_cost_output) AS reported_output, "
        "  MAX(reported_cost_cache_read) AS reported_cache_read, "
        "  MAX(reported_cost_cache_write) AS reported_cache_write, "
        "  MAX(reported_cost_total) AS reported_total "
        f"  FROM events WHERE session_id IN ({marks}) AND type='assistant' "
        "  AND model IS NOT NULL AND model != '<synthetic>' AND request_id IS NOT NULL "
        f"  AND {canonical_assistant_predicate()} "
        "  GROUP BY session_id, model, provider, source_client, request_id"
        ") GROUP BY session_id, model, provider, source_client, speed, inference_geo, "
        "era, recent")


def _row_cost(r, enterprise: bool) -> float:
    dm = display_model_for_row(r["model"], r["provider"], r["source_client"])
    cost = reported_cost(r)
    if cost is None:
        cost = compute_cost(
            dm, r["inp"] or 0, r["outp"] or 0, r["cc"] or 0, r["cr"] or 0,
            r["speed"], effective_geo(r["inference_geo"], enterprise=enterprise),
            cw_5m=r["c5m"] or 0, cw_1h=r["c1h"] or 0, web_search=r["ws"] or 0,
            ts_epoch=r["first_ts"])
    return cost


def _is_anthropic(r) -> bool:
    """Claude subscription usage (cost_windows.ANTHROPIC_ONLY_SQL rule)."""
    return r["source_client"] != "pi-agent" or (r["provider"] or "").lower() == "anthropic"


def session_rollups(conn, event_ids, now: float, scope: str) -> dict[str, dict]:
    """events session id -> {cost, cost_recent, anthropic_recent, first_ts,
    last_ts, enterprise, client, model, provider, agent_name} over the whole
    session. agent_name is a Pi subagent's run name (namespaced agent_id).

    cost_recent covers the last BURN_WINDOW_S. model is the newest
    main-thread model (newest of any thread when there is none)."""
    ids = sorted(set(event_ids))[:_MAX_IDS]
    if not ids:
        return {}
    bounds = era_boundaries()
    params = [*bounds, now - BURN_WINDOW_S, *ids]
    out: dict[str, dict] = {}
    for r in conn.execute(_rollup_sql(len(ids), bounds), params):
        agg = out.setdefault(r["session_id"], {
            "cost": 0.0, "cost_recent": 0.0, "anthropic_recent": 0.0,
            "first_ts": r["first_ts"], "last_ts": r["last_ts"],
            "enterprise": False, "_newest": (-1, 0.0), "client": None,
            "model": None, "provider": None, "agent_name": None})
        cost = _row_cost(r, scope == "enterprise")
        agg["cost"] += cost
        if r["recent"]:
            agg["cost_recent"] += cost
            if _is_anthropic(r):
                agg["anthropic_recent"] += cost
        agg["first_ts"] = min(agg["first_ts"], r["first_ts"])
        agg["last_ts"] = max(agg["last_ts"], r["last_ts"])
        agg["enterprise"] = agg["enterprise"] or bool(r["is_ent"])
        if r["source_client"] == "pi-agent" and (r["agent_id"] or "").count(":") >= 2:
            agg["agent_name"] = r["agent_id"].split(":", 2)[2]
        rank = (r["is_main"] or 0, r["last_ts"])
        if rank > agg["_newest"]:
            agg.update(_newest=rank, client=r["source_client"],
                       model=r["model"], provider=r["provider"])
    for agg in out.values():
        del agg["_newest"]
    return out


def session_titles(conn, ids) -> dict[str, str]:
    """Titles from session_titles, else Claude Desktop's session metadata."""
    ids = sorted(set(ids))[:_MAX_IDS]
    if not ids:
        return {}
    marks = ",".join("?" for _ in ids)
    titles = {r[0]: r[1] for r in conn.execute(
        "SELECT cli_session_id, title FROM desktop_sessions "
        f"WHERE cli_session_id IN ({marks}) AND COALESCE(title, '') != ''", ids)}
    titles.update({r[0]: r[1] for r in conn.execute(
        "SELECT session_id, title FROM session_titles "
        f"WHERE session_id IN ({marks}) AND title != ''", ids)})
    return titles
