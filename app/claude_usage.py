"""Personal metadata ingestion and atomic ownership of the OAuth snapshot.

After Meridian takes ownership, legacy sources may represent another account;
returning to them requires an explicit migration, never an age-based fallback.
"""
import json
import time
from datetime import datetime, timezone

from fastapi import HTTPException, Request

from .db import write_txn
from .quota_freshness import MAX_CLOCK_SKEW_S, MAX_OBSERVATION_AGE_S

MANAGED_SOURCE = "meridian-oauth"


async def parse_claude_usage(request: Request):
    """Validate locally without echoing rejected private fields or NaN inputs.

    Framework validation errors echo input values (and cannot serialize NaN).
    This metadata boundary deliberately returns only a bounded status message.
    """
    from .models import ClaudeUsageRequest
    try:
        return ClaudeUsageRequest.model_validate(await request.json())
    except (ValueError, TypeError):
        raise HTTPException(status_code=422, detail="invalid Claude quota metadata") from None


def _stored(conn):
    row = conn.execute("SELECT value FROM meta WHERE key='oauth_usage'").fetchone()
    try:
        value = json.loads(row[0]) if row else {}
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def managed_source_owns_usage(conn):
    return _stored(conn).get("source") == MANAGED_SOURCE


def store_snapshot(usage, observed, source, *, metadata=None, history=True):
    """Compare and write snapshot + history under the existing writer lock.

    Legacy sources have no reliable observation clock, so once managed they
    cannot even append history (which would corrupt reset/trend inference).
    """
    from .limit_readings import record_limit_readings
    updated_at = datetime.fromtimestamp(observed, timezone.utc).isoformat()
    with write_txn() as conn:
        # Serialize the comparison in SQLite too, not only within this process.
        # Nested writers already own a transaction through write_txn.
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")
        previous = _stored(conn)
        if previous.get("source") == MANAGED_SOURCE:
            if source != MANAGED_SOURCE:
                return None
            prior = previous.get("observed_at_epoch")
            if isinstance(prior, (float, int)) and observed <= prior:
                return None
            original = (metadata or {}).get("original_observed_at_epoch", observed)
            prior_original = previous.get("original_observed_at_epoch", prior)
            if isinstance(prior_original, (float, int)) and original <= prior_original:
                return None  # replaying a skew-clamped sample cannot rejuvenate it
        # First Meridian sample transfers ownership: legacy receipt stamps
        # are not comparable observations and must not starve the transfer.
        stored = {"data": usage, "updated_at": updated_at, "source": source,
                  "observed_at_epoch": observed, **(metadata or {})}
        conn.execute("INSERT OR REPLACE INTO meta (key,value) VALUES('oauth_usage',?)",
                     (json.dumps(stored),))
        if history:
            record_limit_readings(conn, usage, observed, source,
                                  strict=source == MANAGED_SOURCE)
    return updated_at


def store_claude_usage(req):
    import app.config as cfg
    if cfg.LOCKED_SCOPE == "enterprise":
        raise HTTPException(status_code=403, detail="personal quota is outside locked scope")
    now = time.time()
    original = req.observed_at_epoch
    if original < now - MAX_OBSERVATION_AGE_S or original > now + MAX_CLOCK_SKEW_S:
        raise HTTPException(status_code=422, detail="observation outside accepted time range")
    observed = min(original, now)
    usage = {}
    for bucket in req.buckets:
        key = ("seven_day_" + bucket.key.split(":", 1)[1]
               if bucket.key.startswith("scoped:") else bucket.key)
        usage[key] = {"utilization": bucket.pct, "label": bucket.label,
                      "resets_at": datetime.fromtimestamp(
                          bucket.resets_at_epoch, timezone.utc).isoformat()}
    if req.extra_usage is not None:
        fields = {"enabled": "is_enabled", "monthly_limit_cents": "monthly_limit",
                  "used_cents": "used_credits", "pct": "utilization"}
        usage["extra_usage"] = {fields[k]: v for k, v in
                                req.extra_usage.model_dump(exclude_none=True).items()}
    updated = store_snapshot(usage, observed, req.source, metadata={
        "source_profile": req.source_profile,
        "original_observed_at_epoch": original,
    })
    if updated:
        from .aggregator import trigger_eager_rebuild
        trigger_eager_rebuild()
    return {"status": "ok" if updated else "ignored_stale", "updated_at": updated}
