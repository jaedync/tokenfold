"""GET /api/served-models: which model actually served each request.

Anthropic sometimes serves a request with a model other than the one asked
for; the thinking-block signature header names it (app/sigheader.py). This
module owns every READ of that capture: the grouped API rows and the dashboard
chip data. Nothing here rolls up into daily_summary: the signal is a
best-effort observatory that is expected to change or vanish, so it is
computed on the fly against the partial index idx_events_served.
"""

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from zoneinfo import ZoneInfo

from .auth import require_dashboard_auth
from .config import TZ_NAME
from .db import get_conn
from .pricing import display_model

router = APIRouter()
TZ = ZoneInfo(TZ_NAME)

DAYS_DEFAULT = 30
DAYS_MAX = 400

# Every served-model id seen so far is claude-<name>-<hash>-v<n>-prod; the
# fixed affixes carry no information in a chip that is already next to the
# model name, so they are trimmed for display only (never in stored data).
_SLUG_PREFIX = "claude-"
_SLUG_SUFFIX = "-prod"


def slug(served_model: str) -> str:
    """Chip-sized name for a served model id. Display only."""
    name = served_model
    if name.startswith(_SLUG_PREFIX):
        name = name[len(_SLUG_PREFIX):]
    if name.endswith(_SLUG_SUFFIX):
        name = name[:-len(_SLUG_SUFFIX)]
    return name or served_model


def _since_day(days: int) -> str:
    """First local day included in a `days`-long window ending today."""
    return (datetime.now(TZ) - timedelta(days=days - 1)).strftime("%Y-%m-%d")


def served_model_rows(conn, pred: str, since_day: str) -> list[dict]:
    """Grouped (day, model, served_model, sig_version, sig_fields) rows.

    served_model NULL rows are included on purpose: they are the share of
    blocks whose header format no longer names a model (v4 dropped f6), and
    hiding them would make the named share look total.
    """
    rows = conn.execute(
        "SELECT day, model, served_model, sig_version, sig_fields, "
        "COUNT(*) AS blocks, "
        "COALESCE(SUM(COALESCE(sig_cipher_len, 0)), 0) AS cipher_bytes "
        "FROM events "
        "WHERE sig_header IS NOT NULL AND day >= ? "
        f"AND {pred} "
        "GROUP BY day, model, served_model, sig_version, sig_fields "
        "ORDER BY day DESC, model, blocks DESC",
        (since_day,),
    ).fetchall()
    return [
        {
            "day": r["day"],
            "model": r["model"],
            "served_model": r["served_model"],
            "sig_version": r["sig_version"],
            "sig_fields": r["sig_fields"],
            "blocks": r["blocks"],
            "cipher_bytes": r["cipher_bytes"],
        }
        for r in rows
    ]


def _chip_label(counts: Counter, signed: int) -> str:
    """'58% kettle-e2c95a10-v2', most common first, ' · ' between slugs.

    The denominator is every SIGNED block of the model (sig_header present),
    including v4 blocks whose header names no model: the same "share of the
    blocks" the statusline shows, so the two never disagree. Unnamed blocks
    dilute the share rather than inflate it; the API's null rows show their
    size separately. 1% floors a nonzero share so it never renders as '0%'.
    """
    parts = []
    for served, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        pct = max(1, round(100 * n / signed)) if signed else 0
        parts.append(f"{pct}% {slug(served)}")
    return " · ".join(parts)


def served_model_chips(conn, pred: str, cutoff_date: str,
                       today_str: str) -> dict:
    """Per-mode chip text keyed by the dashboard's DISPLAY model name.

    {"all": {"Fable 5": "58% kettle-e2c95a10-v2"}, "14d": {...}, "today": {...}}

    A model appears only when some block in that window was served by a
    DIFFERENT model, so the dashboard renders nothing when there is nothing to
    report. Model breakdown rows come from daily_summary, which has no notion
    of served models, so this is its own small query over events.
    """
    modes = ("all", "14d", "today")
    # mode -> display name -> Counter of served ids that differ from the model
    differing: dict = {m: defaultdict(Counter) for m in modes}
    # mode -> display name -> signed blocks (header present, named or not)
    signed: dict = {m: Counter() for m in modes}

    rows = conn.execute(
        "SELECT day, model, served_model, COUNT(*) AS blocks FROM events "
        "WHERE sig_header IS NOT NULL AND model IS NOT NULL "
        f"AND {pred} "
        "GROUP BY day, model, served_model"
    ).fetchall()

    for r in rows:
        served = r["served_model"]
        name = display_model(r["model"])
        in_modes = ["all"]
        if r["day"] >= cutoff_date:
            in_modes.append("14d")
        if r["day"] == today_str:
            in_modes.append("today")
        for mode in in_modes:
            signed[mode][name] += r["blocks"]
            if served and served != r["model"]:
                differing[mode][name][served] += r["blocks"]

    return {
        mode: {
            name: _chip_label(counts, signed[mode][name])
            for name, counts in differing[mode].items() if counts
        }
        for mode in modes
    }


@router.get("/api/served-models",
            dependencies=[Depends(require_dashboard_auth)])
async def served_models(days: int = Query(default=DAYS_DEFAULT),
                        scope: Optional[str] = Query(default=None)):
    """Grouped served-model capture for the last `days` local days.

    Personal scope only, gated exactly like /api/limit-history: an
    enterprise-locked instance and an explicit enterprise scope both get a
    neutral 404 rather than an empty body, so the feature is not advertised
    to callers who may never read it.
    """
    import sys
    cfg = sys.modules["app.config"]  # fresh read, importlib.reload safety
    if cfg.LOCKED_SCOPE == "enterprise":
        raise HTTPException(status_code=404, detail="not found")
    try:
        effective = cfg.resolve_scope(scope)
    except cfg.InvalidScope:
        raise HTTPException(status_code=400, detail="invalid scope")
    except cfg.ScopeLocked:
        raise HTTPException(status_code=404, detail="not found")
    if effective == "enterprise":
        raise HTTPException(status_code=404, detail="not found")

    days = max(1, min(DAYS_MAX, days))  # clamp, never error
    # effective can only be "personal" past the gate above.
    rows = served_model_rows(get_conn(), cfg.scope_predicate(effective),
                             _since_day(days))
    return {"days": days, "rows": rows}
