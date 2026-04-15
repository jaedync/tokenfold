# HA Metrics Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `GET /api/ha` — a flat, scrubbed JSON endpoint that Home Assistant's REST platform can fan out into Long-Term Statistics sensors, primarily to historize the *implied* 5-hour and weekly dollar limits derived from Anthropic's OAuth `utilization` percentages.

**Architecture:** New file `app/ha.py` holds the endpoint. A shared helper `compute_window_cost()` extracted into a new `app/cost_windows.py` is used by both the new endpoint and the existing `/api/rate-limits`. `meta.oauth_usage` (populated every 10 min by `app/usage_fetcher.py`) is the source of truth for `pct_used` and `resets_at`; `aggregator.build_dashboard_data()` supplies today/total costs. All timestamps truncated to minute precision to avoid account-fingerprinting.

**Tech Stack:** FastAPI, SQLite (WAL), existing `app/pricing.py` helpers (`compute_cost`, `display_model`), `zoneinfo`, `datetime`.

**Reference spec:** `docs/superpowers/specs/2026-04-15-ha-metrics-endpoint-design.md`

**Testing note:** This project has no automated test suite (see `CLAUDE.md`). Verification is via `curl` + `jq` against a running container. Each task includes concrete manual verification steps.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `app/cost_windows.py` | Create | Pure helper: dedup-by-`request_id` `MAX()`-then-`SUM` cost computation over an arbitrary `[start_epoch, end_epoch)` event window. |
| `app/api.py` | Modify | Replace the inline `week_cost` SQL (lines 90–108) with a call to `compute_window_cost()`. No behavior change. |
| `app/ha.py` | Create | `GET /api/ha` — flattens oauth_usage + aggregator + two `compute_window_cost()` calls into the HA-friendly shape. |
| `app/main.py` | Modify | Register `ha_router`. |
| `scripts/ha-smoke.sh` | Create | One-shot curl + `jq` smoke check. Optional but cheap. |

---

## Task 1: Extract `compute_window_cost()` helper

**Files:**
- Create: `app/cost_windows.py`
- Modify: `app/api.py:83-108`

The existing `/api/rate-limits` endpoint inlines a dedup-by-`request_id` cost query for the weekly window. Task 2 needs the identical pattern for the 5-hour window. Extract once, reuse twice.

- [ ] **Step 1: Create `app/cost_windows.py` with the helper**

```python
"""Shared cost accounting for bounded event windows.

Deduplicates streaming-API token repeats by taking MAX() per request_id,
then sums costs across models using pricing.compute_cost().
"""

import sqlite3

from .pricing import compute_cost, display_model


def compute_window_cost(
    conn: sqlite3.Connection,
    start_epoch: float,
    end_epoch: float,
) -> float:
    """Sum assistant-event cost over [start_epoch, end_epoch).

    Streaming API chunks repeat token counts on every message; we dedupe
    with MAX(tokens) per (model, request_id), then sum costs per model.
    Synthetic events and rows missing model/request_id are excluded.

    Returns 0.0 if the window is empty. Does not round — callers round
    to their preferred precision.
    """
    total = 0.0
    for r in conn.execute(
        "SELECT model, SUM(inp) as inp, SUM(outp) as outp, "
        "SUM(cc) as cc, SUM(cr) as cr "
        "FROM ("
        "  SELECT model, request_id, "
        "  MAX(input_tokens) as inp, MAX(output_tokens) as outp, "
        "  MAX(cache_creation_tokens) as cc, MAX(cache_read_tokens) as cr "
        "  FROM events WHERE type='assistant' AND model IS NOT NULL "
        "  AND model != '<synthetic>' AND request_id IS NOT NULL "
        "  AND ts_epoch >= ? AND ts_epoch < ? "
        "  GROUP BY model, request_id"
        ") GROUP BY model",
        (start_epoch, end_epoch),
    ):
        dm = display_model(r["model"])
        total += compute_cost(
            dm,
            r["inp"] or 0,
            r["outp"] or 0,
            r["cc"] or 0,
            r["cr"] or 0,
        )
    return total
```

- [ ] **Step 2: Replace the inline block in `app/api.py`**

Open `app/api.py`. Locate the block starting at `# Cost: deduplicate by request_id, filter by epoch window` (around line 89) and ending just before `# Active time: sum gaps within window` (around line 109). Replace with a single call.

Before (lines 89–108):
```python
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
```

After:
```python
            # Cost: delegated to cost_windows.compute_window_cost()
            week_cost = compute_window_cost(conn, week_start_epoch, reset_epoch)
```

Also add the import near the other `from .pricing import ...` line:
```python
from .cost_windows import compute_window_cost
```

- [ ] **Step 3: Rebuild and restart the container**

Run:
```bash
cd /home/jaedy/tokenfold && docker compose up -d --build tokenfold
```
Expected: container rebuilds, starts cleanly. Tail logs briefly:
```bash
docker compose logs --tail=30 tokenfold
```
Expected: no `ImportError`, `uvicorn running on http://0.0.0.0:5000` appears.

- [ ] **Step 4: Verify `/api/rate-limits` still returns the same `week_cost`**

Run:
```bash
curl -s http://localhost:5000/api/rate-limits | jq '.weekly_budget.week_cost'
```
Expected: a float (e.g. `63.2`) that matches the value the dashboard showed before the refactor. Confirm it is non-null and non-zero assuming the user has any weekly activity.

- [ ] **Step 5: Commit**

```bash
git add app/cost_windows.py app/api.py
git commit -m "refactor: extract compute_window_cost() for reuse"
```

---

## Task 2: Create `GET /api/ha` endpoint

**Files:**
- Create: `app/ha.py`
- Modify: `app/main.py:62-72`

- [ ] **Step 1: Create `app/ha.py`**

```python
"""GET /api/ha — flat, scrubbed metrics for Home Assistant REST sensors.

Designed for HA's `rest:` platform: one HTTP fetch, many sensors via
value_template. All timestamps truncated to minute precision so the
microsecond offsets of OAuth resets_at cannot fingerprint the account.
"""

import json
import time
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from .aggregator import build_dashboard_data
from .cost_windows import compute_window_cost
from .db import get_conn

router = APIRouter()

FIVE_HOURS_S = 5 * 3600
SEVEN_DAYS_S = 7 * 24 * 3600
MIN_PCT_FOR_IMPLIED_LIMIT = 5.0  # below this, API rounding noise swamps signal


def _truncate_to_minute(iso_str: str) -> tuple[str, float] | tuple[None, None]:
    """Parse an ISO-8601 timestamp, truncate to whole UTC minute.

    Returns (iso_string_with_00_seconds_and_no_microseconds, epoch_float).
    Returns (None, None) if the input is empty or malformed.
    """
    if not iso_str:
        return None, None
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None, None
    dt = dt.astimezone(timezone.utc).replace(second=0, microsecond=0)
    return dt.isoformat(), dt.timestamp()


def _implied_limit(spend_usd: float, pct_used: float) -> float | None:
    """spend / (pct/100), or None when utilization is too low to be trustworthy."""
    if pct_used is None or pct_used < MIN_PCT_FOR_IMPLIED_LIMIT:
        return None
    return round(spend_usd / (pct_used / 100.0), 2)


def _window_block(
    conn,
    raw_resets_at: str,
    pct_used: float | None,
    window_seconds: int,
    now_epoch: float,
) -> dict | None:
    """Build a five_hour / weekly sub-object, or None if resets_at is missing."""
    resets_iso, resets_epoch = _truncate_to_minute(raw_resets_at)
    if resets_iso is None:
        return None
    start_epoch = resets_epoch - window_seconds
    spend = round(compute_window_cost(conn, start_epoch, resets_epoch), 2)
    return {
        "pct_used": pct_used if pct_used is not None else 0.0,
        "spend_usd": spend,
        "implied_limit_usd": _implied_limit(spend, pct_used or 0.0),
        "resets_at": resets_iso,
        "resets_in_s": max(0, int(resets_epoch - now_epoch)),
    }


@router.get("/api/ha")
async def ha_metrics():
    """Flat metrics feed for Home Assistant REST sensors."""
    conn = get_conn()
    now_epoch = time.time()

    # Cost totals — always populated from the aggregator cache.
    dash = build_dashboard_data()
    cost_today = round((dash.get("today") or {}).get("cost", 0.0) or 0.0, 2)
    cost_total = round(dash.get("total_cost", 0.0) or 0.0, 2)

    # OAuth usage — may be absent (e.g. fresh install, pre-fetch).
    row = conn.execute(
        "SELECT value FROM meta WHERE key='oauth_usage'"
    ).fetchone()

    five_hour_block = None
    weekly_block = None
    updated_at_epoch = None

    if row:
        try:
            stored = json.loads(row["value"])
        except (ValueError, TypeError):
            stored = None

        if stored:
            usage = stored.get("data") or {}
            fh = usage.get("five_hour") or {}
            sd = usage.get("seven_day") or {}

            five_hour_block = _window_block(
                conn,
                fh.get("resets_at", ""),
                fh.get("utilization"),
                FIVE_HOURS_S,
                now_epoch,
            )
            weekly_block = _window_block(
                conn,
                sd.get("resets_at", ""),
                sd.get("utilization"),
                SEVEN_DAYS_S,
                now_epoch,
            )

            updated_at_iso = stored.get("updated_at", "")
            if updated_at_iso:
                try:
                    ua = datetime.fromisoformat(updated_at_iso).timestamp()
                    # Round to nearest 10 seconds for symmetry with resets_at.
                    updated_at_epoch = int(round(ua / 10.0) * 10)
                except (ValueError, TypeError):
                    pass

    return JSONResponse(content={
        "cost_today_usd": cost_today,
        "cost_total_usd": cost_total,
        "five_hour": five_hour_block,
        "weekly": weekly_block,
        "updated_at_epoch": updated_at_epoch,
    })
```

- [ ] **Step 2: Register the router in `app/main.py`**

In `app/main.py`, after the existing router imports (around line 66), add:

```python
from .ha import router as ha_router
```

And after the existing `app.include_router(...)` calls (around line 72), add:

```python
app.include_router(ha_router)
```

- [ ] **Step 3: Rebuild and restart the container**

Run:
```bash
cd /home/jaedy/tokenfold && docker compose up -d --build tokenfold
```
Expected: clean rebuild. Tail logs:
```bash
docker compose logs --tail=30 tokenfold
```
Expected: no import errors, no tracebacks, `Application startup complete.`

- [ ] **Step 4: Fetch the endpoint and confirm the shape**

Run:
```bash
curl -s http://localhost:5000/api/ha | jq
```
Expected: an object with exactly these top-level keys: `cost_today_usd`, `cost_total_usd`, `five_hour`, `weekly`, `updated_at_epoch`. `cost_today_usd` and `cost_total_usd` are floats. If oauth usage has been fetched in this environment, `five_hour` and `weekly` are objects with keys `pct_used`, `spend_usd`, `implied_limit_usd`, `resets_at`, `resets_in_s`; otherwise both are `null` and that is correct.

- [ ] **Step 5: Confirm the fingerprint scrub**

Run:
```bash
curl -s http://localhost:5000/api/ha | jq -r '.five_hour.resets_at, .weekly.resets_at'
```
Expected: each value ends with `:00+00:00` (zero seconds, no microseconds). If either shows a non-zero second or a `.` before the `+`, the truncation is broken.

Also:
```bash
curl -s http://localhost:5000/api/ha | jq '.updated_at_epoch % 10'
```
Expected: `0`. A non-zero remainder means the 10-second rounding is broken.

- [ ] **Step 6: Spot-check the implied-limit math**

Run:
```bash
curl -s http://localhost:5000/api/ha | jq '.weekly'
```
Expected: if `pct_used >= 5.0`, then `implied_limit_usd` is approximately `spend_usd * 100 / pct_used`. Verify mentally:

```bash
curl -s http://localhost:5000/api/ha | jq '.weekly | if .pct_used >= 5.0 then (.spend_usd * 100 / .pct_used) else "below-cutoff" end'
```
The printed number should match `.weekly.implied_limit_usd` to within 0.01 USD rounding.

If `pct_used < 5.0`, `implied_limit_usd` must be `null`:
```bash
curl -s http://localhost:5000/api/ha | jq '.weekly | {pct_used, implied_limit_usd}'
```
Expected: `implied_limit_usd: null` when `pct_used < 5.0`; otherwise a number.

- [ ] **Step 7: Confirm `cost_today_usd` matches the dashboard**

Run:
```bash
curl -s http://localhost:5000/api/ha | jq '.cost_today_usd'
curl -s http://localhost:5000/api/stats | jq '.today.cost'
```
Expected: identical values. A mismatch means the wrong key was pulled from the aggregator dict.

- [ ] **Step 8: Commit**

```bash
git add app/ha.py app/main.py
git commit -m "feat: add /api/ha endpoint for Home Assistant sensors"
```

---

## Task 3: Smoke script (optional but cheap)

**Files:**
- Create: `scripts/ha-smoke.sh`

A standalone script that future-proofs against accidental field renames. Fails loudly on any missing key.

- [ ] **Step 1: Check if `scripts/` exists, create if not**

Run:
```bash
ls /home/jaedy/tokenfold/scripts/ 2>/dev/null || mkdir -p /home/jaedy/tokenfold/scripts
```

- [ ] **Step 2: Write the script**

Create `scripts/ha-smoke.sh`:

```bash
#!/usr/bin/env bash
# Smoke-check /api/ha. Exits non-zero if any required key is missing or malformed.
#
# Usage: scripts/ha-smoke.sh [base_url]
#   base_url defaults to http://localhost:5000
set -euo pipefail

BASE="${1:-http://localhost:5000}"
URL="$BASE/api/ha"

body=$(curl -sf "$URL") || { echo "FAIL: could not fetch $URL"; exit 1; }

check() {
  local expr="$1" desc="$2"
  if ! echo "$body" | jq -e "$expr" >/dev/null; then
    echo "FAIL: $desc"
    echo "Response was:"
    echo "$body" | jq .
    exit 1
  fi
}

check '.cost_today_usd | type == "number"' 'cost_today_usd must be a number'
check '.cost_total_usd | type == "number"' 'cost_total_usd must be a number'
check 'has("five_hour")'                    'five_hour key must be present'
check 'has("weekly")'                       'weekly key must be present'
check 'has("updated_at_epoch")'             'updated_at_epoch key must be present'

# If the window blocks are present, their sub-shape must be complete.
for win in five_hour weekly; do
  if [ "$(echo "$body" | jq ".$win")" != "null" ]; then
    for sub in pct_used spend_usd implied_limit_usd resets_at resets_in_s; do
      check ".$win | has(\"$sub\")" "$win.$sub must be present when $win is not null"
    done
    check ".$win.resets_at | endswith(\":00+00:00\")" \
      "$win.resets_at must be minute-truncated"
  fi
done

# updated_at_epoch, if present, must be divisible by 10.
if [ "$(echo "$body" | jq '.updated_at_epoch')" != "null" ]; then
  check '.updated_at_epoch % 10 == 0' 'updated_at_epoch must be divisible by 10'
fi

echo "OK: /api/ha shape is valid"
```

Make it executable:
```bash
chmod +x /home/jaedy/tokenfold/scripts/ha-smoke.sh
```

- [ ] **Step 3: Run it against the live container**

Run:
```bash
/home/jaedy/tokenfold/scripts/ha-smoke.sh
```
Expected: `OK: /api/ha shape is valid` — exit code 0. Any `FAIL:` line means a shape regression; read the printed response and fix before committing.

- [ ] **Step 4: Commit**

```bash
git add scripts/ha-smoke.sh
git commit -m "chore: add ha endpoint smoke script"
```

---

## Task 4: Document the endpoint in CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` — add a row to the "Key Modules" table.

- [ ] **Step 1: Add the row**

In `CLAUDE.md`, find the "Key Modules" table. Add a new row after the `app/notify.py` row:

```markdown
| `app/ha.py` | GET /api/ha - flat scrubbed metrics for Home Assistant REST sensors |
| `app/cost_windows.py` | Shared helper: dedup-by-request_id cost over an event window |
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: note /api/ha and cost_windows in module list"
```

---

## Post-Task: HA Configuration (user-side, not part of the PR)

Not part of the tokenfold repo — this is what the user adds to their Home Assistant `configuration.yaml` once the endpoint is live. Copied here for reference so the implementer can hand it over:

```yaml
rest:
  - resource: http://tokenfold:5000/api/ha
    scan_interval: 60
    sensor:
      - name: claude_cost_today
        value_template: "{{ value_json.cost_today_usd }}"
        unit_of_measurement: USD
        state_class: total_increasing
      - name: claude_cost_total
        value_template: "{{ value_json.cost_total_usd }}"
        unit_of_measurement: USD
        state_class: total_increasing
      - name: claude_5h_pct
        value_template: "{{ value_json.five_hour.pct_used }}"
        unit_of_measurement: "%"
        state_class: measurement
      - name: claude_5h_implied_limit
        value_template: "{{ value_json.five_hour.implied_limit_usd }}"
        unit_of_measurement: USD
        state_class: measurement
      - name: claude_weekly_pct
        value_template: "{{ value_json.weekly.pct_used }}"
        unit_of_measurement: "%"
        state_class: measurement
      - name: claude_weekly_implied_limit
        value_template: "{{ value_json.weekly.implied_limit_usd }}"
        unit_of_measurement: USD
        state_class: measurement
```
