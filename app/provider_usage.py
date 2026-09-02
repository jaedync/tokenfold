"""Best-effort provider quota snapshots from the Pi dotfleet extension."""
import json
import time
from datetime import datetime, timezone

from .db import get_conn, write_txn

META_KEY = "provider_usage"
MAX_SNAPSHOT_AGE_S = 24 * 3600
_PROVIDER_EVENT_IDS = {
    "codex": ("openai-codex",),
    "opencode-go": ("opencode-go",),
    "opencode-zen": ("opencode", "opencode-zen"),
}


def store_provider_usage(machine, limits, now=None):
    """Merge newer provider snapshots so one missing credential cannot stomp peers."""
    now = now if now is not None else time.time()
    conn = get_conn()
    # Read-modify-write under the shared lock. Independent machines can report
    # different providers concurrently, and neither report may stomp its peer.
    with write_txn(conn):
        row = conn.execute("SELECT value FROM meta WHERE key=?", (META_KEY,)).fetchone()
        try:
            stored = json.loads(row["value"]) if row else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            stored = {}
        providers = stored.get("providers") if isinstance(stored, dict) else None
        providers = dict(providers) if isinstance(providers, dict) else {}

        for limit in limits:
            data = limit.model_dump()
            observed = data.get("observed_at_epoch") or now
            # A bad client clock must not create an effectively immortal row
            # that rejects every later, correctly timestamped snapshot.
            if observed > now + 300:
                observed = now
            previous = providers.get(data["provider"])
            previous_observed = (previous or {}).get("observed_at_epoch", 0)
            if observed < previous_observed:
                continue
            data["observed_at_epoch"] = observed
            data["machine"] = machine
            providers[data["provider"]] = data

        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
            (META_KEY, json.dumps({"providers": providers})),
        )
    return sorted(providers)


def _minute_iso(epoch):
    if not isinstance(epoch, (int, float)):
        return None
    minute = int(epoch // 60) * 60
    return datetime.fromtimestamp(minute, timezone.utc).isoformat().replace(
        "+00:00", "Z")


def _month_reported_costs(conn, now):
    """Return Pi's provider-reported API-equivalent cost for the UTC month."""
    current = datetime.fromtimestamp(now, timezone.utc)
    month_start = current.replace(day=1, hour=0, minute=0, second=0,
                                  microsecond=0).timestamp()
    import sys
    personal_pred = sys.modules["app.config"].scope_predicate("personal")
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
        f" AND {personal_pred} GROUP BY provider, request_id"
        ") GROUP BY provider",
        (month_start, now),
    ).fetchall()
    raw = {row["provider"]: float(row["cost"] or 0) for row in rows}
    return {
        key: round(sum(raw.get(pid, 0) for pid in provider_ids), 2)
        for key, provider_ids in _PROVIDER_EVENT_IDS.items()
    }


def provider_usage_block(now=None):
    """Build the personal-dashboard provider block, omitting stale snapshots."""
    now = now if now is not None else time.time()
    conn = get_conn()
    row = conn.execute("SELECT value FROM meta WHERE key=?", (META_KEY,)).fetchone()
    try:
        stored = json.loads(row["value"]) if row else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        stored = {}
    snapshots = stored.get("providers") if isinstance(stored, dict) else {}
    snapshots = snapshots if isinstance(snapshots, dict) else {}
    costs = _month_reported_costs(conn, now)
    result = {}

    for provider in _PROVIDER_EVENT_IDS:
        snapshot = snapshots.get(provider)
        observed = (snapshot or {}).get("observed_at_epoch")
        fresh = isinstance(observed, (int, float)) and now - observed <= MAX_SNAPSHOT_AGE_S
        windows = []
        if fresh:
            for window in snapshot.get("windows", []):
                if not isinstance(window, dict):
                    continue
                windows.append({
                    "key": window.get("key"),
                    "label": window.get("label"),
                    "pct": window.get("pct"),
                    "resets_at": _minute_iso(window.get("resets_at_epoch")),
                    "window_seconds": window.get("window_seconds"),
                })
        cost = costs.get(provider, 0)
        if not windows and cost <= 0:
            continue
        result[provider] = {
            "windows": windows,
            "month_cost": cost,
            **({"updated_at_epoch": observed} if fresh else {}),
        }
    return result
