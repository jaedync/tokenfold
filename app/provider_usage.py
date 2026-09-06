"""Best-effort provider quota snapshots from the Pi dotfleet extension.

Snapshots are keyed by dashboard scope (personal vs enterprise), derived from
the reporting machine's fleet account class. Every fleet machine posts to the
same Tokenfold, so without that key an enterprise Codex account on a work box
overwrote the personal Codex snapshot (2026-09-02 incident).
"""
import json
import math
import sys
import time

from .quota_freshness import quota_window_valid
from datetime import datetime, timezone

from .db import get_conn, write_txn

META_KEY = "provider_usage"
MAX_SNAPSHOT_AGE_S = 24 * 3600
# A bad client clock must not create an effectively immortal snapshot that
# rejects every later, correctly timestamped one.
MAX_CLOCK_SKEW_S = 300
_PROVIDER_EVENT_IDS = {
    "codex": ("openai-codex",),
    "opencode-go": ("opencode-go",),
    "opencode-zen": ("opencode", "opencode-zen"),
}
_SCOPE_FOR_ACCOUNT_CLASS = {"work": "enterprise", "personal": "personal"}


def scope_for_account_class(account_class):
    """Map a fleet account class onto a dashboard scope (same rule as Pi events)."""
    try:
        return _SCOPE_FOR_ACCOUNT_CLASS[account_class]
    except KeyError:
        raise ValueError(f"unknown account class: {account_class!r}") from None


def _load_scopes(conn):
    """Return the stored per-scope snapshot map, dropping legacy unscoped data.

    Pre-scope servers stored a flat ``providers`` dict. Those snapshots carry
    no account class, so they are the stomp bug in persisted form and are
    never served or carried forward.
    """
    row = conn.execute("SELECT value FROM meta WHERE key=?", (META_KEY,)).fetchone()
    try:
        stored = json.loads(row["value"]) if row else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        stored = {}
    scopes = stored.get("scopes") if isinstance(stored, dict) else None
    if not isinstance(scopes, dict):
        return {}
    return {
        scope: dict(providers)
        for scope, providers in scopes.items()
        if isinstance(providers, dict)
    }


def _merge_snapshot(providers, limit, machine, now):
    """Return ``providers`` with one snapshot merged in, newest observation wins."""
    data = limit.model_dump(exclude_none=True)
    observed = data.get("observed_at_epoch") or now
    if observed > now + MAX_CLOCK_SKEW_S:
        observed = now
    previous = providers.get(data["provider"]) or {}
    if observed < previous.get("observed_at_epoch", 0):
        return providers
    return {
        **providers,
        data["provider"]: {**data, "observed_at_epoch": observed, "machine": machine},
    }


def store_provider_usage(machine, account_class, limits, now=None):
    """Merge newer snapshots into the reporter's scope; peers are never touched."""
    now = now if now is not None else time.time()
    scope = scope_for_account_class(account_class)
    conn = get_conn()
    # Read-modify-write under the shared lock. Independent machines can report
    # different providers concurrently, and neither report may stomp its peer.
    with write_txn(conn):
        scopes = _load_scopes(conn)
        providers = scopes.get(scope, {})
        for limit in limits:
            providers = _merge_snapshot(providers, limit, machine, now)
        scopes = {**scopes, scope: providers}
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
            (META_KEY, json.dumps({"scopes": scopes})),
        )
    return sorted(providers)


def _minute_iso(epoch):
    if not _finite(epoch) or not 0 <= epoch <= 10**11:
        return None
    minute = int(epoch // 60) * 60
    return datetime.fromtimestamp(minute, timezone.utc).isoformat().replace(
        "+00:00", "Z")


def _month_reported_costs(conn, now, scope):
    """Return Pi's provider-reported API-equivalent cost for the UTC month."""
    current = datetime.fromtimestamp(now, timezone.utc)
    month_start = current.replace(day=1, hour=0, minute=0, second=0,
                                  microsecond=0).timestamp()
    pred = sys.modules["app.config"].scope_predicate(scope)
    rows = conn.execute(
        "SELECT provider, SUM(CASE WHEN reported_total IS NOT NULL "
        "THEN reported_total ELSE COALESCE(reported_input,0) + "
        "COALESCE(reported_output,0) + COALESCE(reported_cache_read,0) + "
        "COALESCE(reported_cache_write,0) END) AS cost FROM ("
        " SELECT provider, request_id, MAX(reported_cost_total) reported_total,"
        " MAX(reported_cost_input) reported_input,"
        " MAX(reported_cost_output) reported_output,"
        " MAX(reported_cost_cache_read) reported_cache_read,"
        " MAX(reported_cost_cache_write) reported_cache_write"
        " FROM events WHERE type='assistant' AND source_client='pi-agent'"
        " AND request_id IS NOT NULL AND ts_epoch>=? AND ts_epoch<?"
        f" AND {pred} GROUP BY provider, request_id"
        ") GROUP BY provider",
        (month_start, now),
    ).fetchall()
    raw = {row["provider"]: float(row["cost"] or 0) for row in rows}
    return {
        key: round(sum(raw.get(pid, 0) for pid in provider_ids), 2)
        for key, provider_ids in _PROVIDER_EVENT_IDS.items()
    }


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _codex_window_label(window):
    """Codex primary is a slot, not a duration (Pro Lite uses a weekly primary)."""
    seconds = window.get("window_seconds")
    if _finite(seconds) and seconds > 0:
        for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60)):
            if seconds % size == 0:
                return f"{int(seconds // size)}-{unit} limit"
        return f"{int(seconds)}-second limit"
    return {"primary": "Primary limit", "secondary": "Secondary limit"}.get(
        window.get("key"), "Usage limit")


def _window_reported_cost(conn, scope, provider, start, end):
    """Reported API-equivalent spend only; never reprice missing provider costs."""
    pred = sys.modules["app.config"].scope_predicate(scope)
    providers = _PROVIDER_EVENT_IDS[provider]
    slots = ",".join("?" for _ in providers)
    row = conn.execute(
        "SELECT SUM(CASE WHEN total IS NOT NULL THEN total ELSE "
        "COALESCE(inp,0)+COALESCE(outp,0)+COALESCE(cr,0)+COALESCE(cw,0) END) cost "
        "FROM (SELECT provider, request_id, MAX(reported_cost_total) total, "
        "MAX(reported_cost_input) inp, MAX(reported_cost_output) outp, "
        "MAX(reported_cost_cache_read) cr, MAX(reported_cost_cache_write) cw "
        "FROM events WHERE type='assistant' AND source_client='pi-agent' "
        f"AND provider IN ({slots}) AND {pred} AND request_id IS NOT NULL "
        "AND ts_epoch>=? AND ts_epoch<? GROUP BY provider, request_id)",
        (*providers, start, end),
    ).fetchone()
    return row["cost"]


def _fresh_windows(snapshot, provider, now, conn, scope, include_costs):
    result = []
    observed = snapshot.get("observed_at_epoch")
    for raw in snapshot.get("windows", []):
        if not isinstance(raw, dict):
            continue
        reset, duration, pct = (raw.get(k) for k in (
            "resets_at_epoch", "window_seconds", "pct"))
        window = {
            "key": raw.get("key"),
            "label": _codex_window_label(raw) if provider == "codex" else raw.get("label"),
            "pct": pct,
            "resets_at": _minute_iso(reset),
            "window_seconds": duration,
        }
        # Match spend to the exact sample, not now: transcript ingestion may
        # continue for hours after a quota snapshot. Expired/future windows
        # cannot explain that sample and must never yield a dollar projection.
        if (include_costs and all(_finite(v) for v in (reset, duration, pct, observed))
                and duration > 0 and 0 < pct <= 100 and reset > now
                and quota_window_valid(observed, now, reset, reset - duration)):
            start = reset - duration
            cost = _window_reported_cost(conn, scope, provider, start, observed)
            if _finite(cost) and cost > 0:
                window["window_cost"] = round(cost, 4)
                window["window_start_epoch"] = int(start // 60) * 60
                window["window_end_epoch"] = int(observed // 60) * 60
                # Same 5% noise floor as the Anthropic gauges.
                if pct >= 5:
                    capacity = cost * 100 / pct
                    window["estimated_capacity"] = round(capacity, 4)
                    window["estimated_remaining"] = round(max(0, capacity - cost), 4)
        result.append(window)
    return result


def provider_usage_block(scope, now=None, *, conn=None, include_costs=True):
    """Build one scope's provider block, omitting stale snapshots."""
    now = now if now is not None else time.time()
    conn = conn if conn is not None else get_conn()
    snapshots = _load_scopes(conn).get(scope, {})
    costs = _month_reported_costs(conn, now, scope) if include_costs else {}
    result = {}

    for provider in _PROVIDER_EVENT_IDS:
        snapshot = snapshots.get(provider) or {}
        observed = snapshot.get("observed_at_epoch")
        fresh = _finite(observed) and -MAX_CLOCK_SKEW_S <= now - observed <= MAX_SNAPSHOT_AGE_S
        windows = _fresh_windows(snapshot, provider, now, conn, scope, include_costs) if fresh else []
        cost = costs.get(provider, 0)
        if not windows and cost <= 0:
            continue
        plan = snapshot.get("plan") if fresh else None
        result[provider] = {
            "windows": windows,
            **({"month_cost": cost} if include_costs else {}),
            **({"updated_at_epoch": observed} if fresh else {}),
            **({"plan": plan} if isinstance(plan, str) else {}),
        }
    return result
