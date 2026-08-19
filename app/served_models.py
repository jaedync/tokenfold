"""Served-model reads: /api/served-models, its timeline, and the chip data.

Anthropic sometimes serves a request with a model other than the one asked
for; the thinking-block signature header names it (app/sigheader.py). This
module owns every READ of that capture: the grouped API rows, the timeline the
dashboard draws, and the chip text next to each model name. Nothing here rolls
up into daily_summary: the signal is a best-effort observatory that is expected
to change or vanish, so it is computed on the fly against the partial index
idx_events_served.

Reroutes arrive in RUNS: a session is served by one model for a stretch, flips,
and flips back, and the fleet-wide picture changes hour to hour. A single share
of a whole range ("3% kettle") hides exactly that, so the timeline reports runs
and transitions, and the chip reports a state and a date instead of a share.
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

# The state of a block: served by the model asked for, served by a named other
# model (its display slug), or served by a header that names nobody.
SELF = "self"
HIDDEN = "hidden"

# A state whose last block is this close to the model's newest block reads as
# current ("since Aug 17") rather than as history ("Aug 17-18"). A day is the
# smallest window that survives a night of not working.
LIVE_WINDOW_S = 86400

# Cell width per window length: half-hourly up close, daily across a year, so
# a strip is always a few hundred cells wide rather than tens of thousands.
BIN_LADDER = ((2, 1800), (14, 3600), (90, 21600))
BIN_FALLBACK = 86400

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


def _state(model: str, served_model: Optional[str]) -> str:
    """What served a block, as one comparable token."""
    if served_model is None:
        return HIDDEN
    if served_model == model:
        return SELF
    return slug(served_model)


def _since_day(days: int) -> str:
    """First local day included in a `days`-long window ending today."""
    return (datetime.now(TZ) - timedelta(days=days - 1)).strftime("%Y-%m-%d")


def _since_epoch(since_day: str) -> float:
    """Local midnight starting `since_day`, the left edge of a timeline."""
    return datetime.strptime(since_day, "%Y-%m-%d").replace(
        tzinfo=TZ).timestamp()


def _bin_seconds(days: int) -> int:
    """Cell width for a `days`-long window."""
    for upper, size in BIN_LADDER:
        if days <= upper:
            return size
    return BIN_FALLBACK


def _fmt_day(ts: float) -> str:
    """'Aug 17' for an epoch, in the instance's local time."""
    return datetime.fromtimestamp(ts, TZ).strftime("%b %-d")


def _span_label(first_ts: float, last_ts: float) -> str:
    """'Aug 17' | 'Aug 17-18' | 'Jul 31-Aug 2' for a first-to-last span.

    The month is written once when both ends share it, which is how a person
    says it and keeps the chip short in the common case.
    """
    first, last = _fmt_day(first_ts), _fmt_day(last_ts)
    if first == last:
        return first
    start = datetime.fromtimestamp(first_ts, TZ)
    end = datetime.fromtimestamp(last_ts, TZ)
    if (start.year, start.month) == (end.year, end.month):
        return f"{first}-{end.day}"
    return f"{first}-{last}"


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


# ── chip ──────────────────────────────────────────────────────────────────


def _chip_label(counts: Counter, signed: int) -> str:
    """'58% kettle-e2c95a10-v2 · 1% hidden', most common first.

    The denominator is every SIGNED block of the model (sig_header present),
    including the blocks whose header names no model: the same "share of the
    blocks" the statusline shows, so the two never disagree. Those unnamed
    blocks are now a state of their own rather than a silent divisor. 1%
    floors a nonzero share so it never renders as '0%'.
    """
    parts = []
    for state, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        pct = max(1, round(100 * n / signed)) if signed else 0
        parts.append(f"{pct}% {state}")
    return " · ".join(parts)


def _top_version(versions: Counter) -> Optional[int]:
    """Most common header version in a bag of blocks, lowest wins a tie."""
    ranked = sorted(versions.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[0][0] if ranked else None


def _chip_entry(state: str, seen: dict, latest_ts: float) -> str:
    """One state as the chip says it: 'kettle-e2c95a10-v2 since Aug 17'.

    A state still live at the model's newest block gets 'since <start>'; one
    that has stopped gets the span it covered, so a chip never implies that a
    finished reroute is still happening.
    """
    label = state
    if state == HIDDEN:
        version = _top_version(seen["versions"])
        label = HIDDEN if version is None else f"{HIDDEN} (v{version})"
    if seen["last"] >= latest_ts - LIVE_WINDOW_S:
        return f"{label} since {_fmt_day(seen['first'])}"
    return f"{label} {_span_label(seen['first'], seen['last'])}"


def _chip(states: dict, signed: int, latest_ts: float) -> dict:
    """{"text", "title"} for one model in one mode window."""
    ordered = sorted(states.items(), key=lambda kv: (-kv[1]["blocks"], kv[0]))
    counts = Counter({state: seen["blocks"] for state, seen in ordered})
    return {
        "text": " · ".join(_chip_entry(state, seen, latest_ts)
                           for state, seen in ordered),
        "title": _chip_label(counts, signed),
    }


def _note_state(bucket: dict, state: str, row) -> None:
    """Fold one grouped row into a model's per-state seen-when record."""
    seen = bucket.get(state)
    if seen is None:
        seen = {"blocks": 0, "first": row["first_ts"], "last": row["last_ts"],
                "versions": Counter()}
        bucket[state] = seen
    seen["blocks"] += row["blocks"]
    seen["first"] = min(seen["first"], row["first_ts"])
    seen["last"] = max(seen["last"], row["last_ts"])
    if row["sig_version"] is not None:
        seen["versions"][row["sig_version"]] += row["blocks"]


def served_model_chips(conn, pred: str, cutoff_date: str,
                       today_str: str) -> dict:
    """Per-mode chip data keyed by the dashboard's DISPLAY model name.

    {"all": {"Fable 5": {"text": "kettle-e2c95a10-v2 since Aug 17",
                         "title": "58% kettle-e2c95a10-v2"}}, "14d": ..., ...}

    A model appears only when some block in that window was served by anything
    other than itself (a named other model, or a header that hides it), so the
    dashboard renders nothing when there is nothing to report. Model breakdown
    rows come from daily_summary, which has no notion of served models, so this
    is its own small query over events.
    """
    modes = ("all", "14d", "today")
    # mode -> display name -> state -> {blocks, first, last, versions}
    states: dict = {m: defaultdict(dict) for m in modes}
    # mode -> display name -> signed blocks (header present, named or not)
    signed: dict = {m: Counter() for m in modes}
    # mode -> display name -> newest signed block, the "now" a chip compares to
    latest: dict = {m: {} for m in modes}

    rows = conn.execute(
        "SELECT day, model, served_model, sig_version, COUNT(*) AS blocks, "
        "MIN(ts_epoch) AS first_ts, MAX(ts_epoch) AS last_ts FROM events "
        "WHERE sig_header IS NOT NULL AND model IS NOT NULL "
        f"AND {pred} "
        "GROUP BY day, model, served_model, sig_version"
    ).fetchall()

    for r in rows:
        name = display_model(r["model"])
        state = _state(r["model"], r["served_model"])
        in_modes = ["all"]
        if r["day"] >= cutoff_date:
            in_modes.append("14d")
        if r["day"] == today_str:
            in_modes.append("today")
        for mode in in_modes:
            signed[mode][name] += r["blocks"]
            latest[mode][name] = max(latest[mode].get(name, r["last_ts"]),
                                     r["last_ts"])
            if state != SELF:
                _note_state(states[mode][name], state, r)

    return {
        mode: {
            name: _chip(bucket, signed[mode][name], latest[mode][name])
            for name, bucket in states[mode].items() if bucket
        }
        for mode in modes
    }


# ── timeline ──────────────────────────────────────────────────────────────


def _timeline_rows(conn, pred: str, since_day: str) -> list:
    """Every signed block in the window, oldest first.

    Shaped in Python rather than SQL: the window holds tens of thousands of
    rows at most, and runs, ledger and latest-state all need the same single
    ordered pass. `model IS NOT NULL` keeps a row with no requested model out
    of a display name it cannot have.
    """
    return conn.execute(
        "SELECT model, served_model, sig_version, sig_fields, ts_epoch, "
        "session_id, source_machine FROM events "
        "WHERE sig_header IS NOT NULL AND model IS NOT NULL AND day >= ? "
        f"AND {pred} "
        "ORDER BY ts_epoch, uuid",
        (since_day,),
    ).fetchall()


def _blank_tally() -> dict:
    return {
        "blocks": Counter(),            # display -> blocks
        "nonself": Counter(),           # display -> blocks not served by self
        "cells": Counter(),             # (display, bin_start, state) -> blocks
        "runs": defaultdict(list),      # session -> [[model, state, t0, t1, n]]
        "machines": defaultdict(Counter),   # session -> machine -> blocks
        "session_nonself": Counter(),   # session -> blocks not served by self
        "ledger": {},                   # combo key -> record
        "latest": {},                   # display -> trailing run
    }


def _tally_session(tally: dict, row, name: str, state: str, ts: float) -> None:
    """Extend the session's run list, or start a new run on a flip."""
    session = row["session_id"]
    if session is None:
        return  # no session to draw a bar for; the block still counts elsewhere
    tally["machines"][session][row["source_machine"]] += 1
    if state != SELF:
        tally["session_nonself"][session] += 1
    runs = tally["runs"][session]
    if runs and runs[-1][0] == name and runs[-1][1] == state:
        runs[-1][3] = ts
        runs[-1][4] += 1
    else:
        runs.append([name, state, ts, ts, 1])


def _tally_ledger(tally: dict, row, name: str, state: str, ts: float) -> None:
    """One record per (model, state, served model, header shape).

    State is part of the key because two requested model ids can share a
    display name (a dated id and its alias), and one of them can be self while
    the other is a reroute.
    """
    key = (name, state, row["served_model"], row["sig_version"],
           row["sig_fields"])
    record = tally["ledger"].get(key)
    if record is None:
        record = {
            "model": name, "state": state,
            "served_model": row["served_model"],
            "sig_version": row["sig_version"],
            "sig_fields": row["sig_fields"],
            "first_seen": ts, "last_seen": ts, "blocks": 0,
            "sessions": set(), "machines": set(),
            "first_session": row["session_id"],
            "first_machine": row["source_machine"],
        }
        tally["ledger"][key] = record
    record["last_seen"] = ts
    record["blocks"] += 1
    if row["session_id"] is not None:
        record["sessions"].add(row["session_id"])
    record["machines"].add(row["source_machine"])


def _tally_latest(tally: dict, name: str, state: str, ts: float) -> None:
    """Track the trailing fleet-wide run: state now, and since when."""
    run = tally["latest"].get(name)
    if run is None or run["state"] != state:
        tally["latest"][name] = {"state": state, "since": ts, "blocks": 1}
    else:
        tally["latest"][name] = {**run, "blocks": run["blocks"] + 1}


def _tally(rows: list, bin_seconds: int) -> dict:
    """Single ordered pass over the window; every section reads off this."""
    tally = _blank_tally()
    for row in rows:
        name = display_model(row["model"])
        state = _state(row["model"], row["served_model"])
        ts = row["ts_epoch"]
        tally["blocks"][name] += 1
        if state != SELF:
            tally["nonself"][name] += 1
        tally["cells"][(name, int(ts // bin_seconds) * bin_seconds, state)] += 1
        _tally_session(tally, row, name, state, ts)
        _tally_ledger(tally, row, name, state, ts)
        _tally_latest(tally, name, state, ts)
    return tally


def _timeline_models(tally: dict) -> list[str]:
    """Models worth a strip: any block not served by themselves, busiest first.

    Ranked by ALL of the model's signed blocks, self included, because that is
    what the strip draws.
    """
    return sorted((name for name, n in tally["nonself"].items() if n),
                  key=lambda name: (-tally["blocks"][name], name))


def _timeline_bins(tally: dict, models: list[str]) -> list:
    """[model, bin_start, state, blocks], only for drawn models, no empties."""
    rank = {name: i for i, name in enumerate(models)}
    cells = [(rank[key[0]], key[1], key[2], blocks)
             for key, blocks in tally["cells"].items() if key[0] in rank]
    return [[models[i], start, state, blocks]
            for i, start, state, blocks in sorted(cells)]


def _timeline_sessions(tally: dict) -> dict:
    """Sessions that saw a reroute, as compressed runs. Self-only ones are the
    normal case and would swamp the interesting ones."""
    return {
        session: {
            "machine": sorted(tally["machines"][session].items(),
                              key=lambda kv: (-kv[1], kv[0]))[0][0],
            "runs": runs,
        }
        for session, runs in tally["runs"].items()
        if tally["session_nonself"][session]
    }


def _ledger_order(item) -> tuple:
    """First seen decides; the rest of the key only breaks ties stably."""
    (name, state, served, version, fields), record = item
    return (record["first_seen"], name, state, served or "",
            -1 if version is None else version, fields or "")


def _timeline_ledger(tally: dict) -> list[dict]:
    """Every combo ever seen in the window, in the order it first appeared.

    Self rows are kept: a self combo whose sig_fields change is the earliest
    warning that the capture itself is drifting.
    """
    records = sorted(tally["ledger"].items(), key=_ledger_order)
    return [
        {
            "model": r["model"], "state": r["state"],
            "served_model": r["served_model"],
            "sig_version": r["sig_version"], "sig_fields": r["sig_fields"],
            "first_seen": r["first_seen"], "last_seen": r["last_seen"],
            "blocks": r["blocks"], "sessions": len(r["sessions"]),
            "machines": sorted(r["machines"]),
            "first_session": r["first_session"],
            "first_machine": r["first_machine"],
        }
        for _, r in records
    ]


def served_model_timeline(conn, pred: str, days: int) -> dict:
    """The whole timeline payload for a `days`-long window. See the route."""
    since_day = _since_day(days)
    bin_seconds = _bin_seconds(days)
    tally = _tally(_timeline_rows(conn, pred, since_day), bin_seconds)
    models = _timeline_models(tally)
    return {
        "days": days,
        "bin_seconds": bin_seconds,
        "since_epoch": _since_epoch(since_day),
        "models": models,
        "bins": _timeline_bins(tally, models),
        "sessions": _timeline_sessions(tally),
        "ledger": _timeline_ledger(tally),
        "latest": {name: tally["latest"][name] for name in models},
    }


# ── routes ────────────────────────────────────────────────────────────────


def _personal_predicate(scope: Optional[str]) -> str:
    """Scope gate for both routes; returns the SQL predicate to query with.

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
    # effective can only be "personal" past the gate above.
    return cfg.scope_predicate(effective)


@router.get("/api/served-models",
            dependencies=[Depends(require_dashboard_auth)])
async def served_models(days: int = Query(default=DAYS_DEFAULT),
                        scope: Optional[str] = Query(default=None)):
    """Grouped served-model capture for the last `days` local days."""
    pred = _personal_predicate(scope)
    days = max(1, min(DAYS_MAX, days))  # clamp, never error
    rows = served_model_rows(get_conn(), pred, _since_day(days))
    return {"days": days, "rows": rows}


@router.get("/api/served-models/timeline",
            dependencies=[Depends(require_dashboard_auth)])
async def served_models_timeline(days: int = Query(default=DAYS_DEFAULT),
                                 scope: Optional[str] = Query(default=None)):
    """When each model was served by something other than itself.

    `models` are the models with anything to report, busiest first; `bins` are
    their binned cells; `sessions` are per-session runs for the ones that saw a
    reroute; `ledger` is every (model, served, header) combo including self,
    oldest first; `latest` is the state each reported model is in now.
    """
    pred = _personal_predicate(scope)
    days = max(1, min(DAYS_MAX, days))  # clamp, never error
    return served_model_timeline(get_conn(), pred, days)
