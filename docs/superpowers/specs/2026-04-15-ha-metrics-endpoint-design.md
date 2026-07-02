# Home Assistant Metrics Endpoint — Design

**Status:** Approved, ready for implementation
**Date:** 2026-04-15

## Problem

Tokenfold already stores the richest possible Claude Code usage data: per-request tokens, model, cost, and — via `app/usage_fetcher.py` — OAuth `utilization` percentages for the 5-hour and weekly windows. Today this data is visible only in the HTML dashboard and `/api/stats`, which are shaped for browser rendering.

The user wants to historize a small set of metrics in Home Assistant's Long-Term Statistics so that changes over time (especially Anthropic silently adjusting usage limits) become visible on a durable time-series chart. HA's REST sensor platform fetches JSON and extracts scalars via `value_template`; the existing endpoints are too nested for that pattern and leak fields that aren't needed.

## Goal

Expose a single `GET /api/ha` endpoint returning a flat, minimal JSON document that HA's `rest:` platform can fan out into Long-Term Statistics sensors. The primary analytical goal is historization of **implied usage limits** — derived server-side as `spend_in_window ÷ (pct_used ÷ 100)` — so that future Anthropic policy changes become visible as step changes in HA graphs.

## Non-Goals

- Per-machine or per-model breakdowns. Deliberately deferred; adds noise before confirming the core signal is useful.
- Backfill of historical implied-limit values. LTS will build up from the moment the sensor starts reporting.
- Authentication. Same posture as the dashboard; see §Auth.
- Test suite. No test infrastructure exists in this project (per `CLAUDE.md`); a manual smoke script is sufficient.

## Architecture

New file `app/ha.py`, registered in `app/main.py` alongside the other routers. Single route: `GET /api/ha`. No state of its own. Depends on:

- `meta.oauth_usage` row (populated by `usage_fetcher.py` every 10 min) — source for `pct_used` and `resets_at`.
- `app/aggregator.build_dashboard_data()` — source for `cost_today_usd` and `cost_total_usd`. Aggregator cache already memoizes and invalidates on ingest.
- A small helper `_window_cost(conn, start_epoch, end_epoch) -> float` that replays the dedup-by-`request_id` `MAX()`-then-`SUM` pattern currently inlined in `app/api.py:90–108`. Factoring it out lets both `/api/rate-limits` and `/api/ha` share the same accounting and removes duplicated SQL.

Separate file rather than growing `api.py` because `api.py` already holds both `/api/stats` and `/api/rate-limits` and the cost accounting logic. Keeping `ha.py` single-purpose fits the project's "many small files, high cohesion" convention.

## Response Shape

```json
{
  "cost_today_usd": 4.52,
  "cost_total_usd": 892.30,
  "five_hour": {
    "pct_used": 42.0,
    "spend_usd": 8.40,
    "implied_limit_usd": 20.00,
    "resets_at": "2026-04-15T23:00:00+00:00",
    "resets_in_s": 8421
  },
  "weekly": {
    "pct_used": 79.0,
    "spend_usd": 63.20,
    "implied_limit_usd": 80.00,
    "resets_at": "2026-04-17T04:00:00+00:00",
    "resets_in_s": 140040
  },
  "updated_at_epoch": 1760555040
}
```

All fields are scalars or one-level-deep objects — no arrays, no nested lists. The two windows share an identical sub-shape so HA templates read uniformly: `{{ value_json.five_hour.implied_limit_usd }}`.

### Field definitions

| Field | Type | Source / derivation |
|---|---|---|
| `cost_today_usd` | float, 2dp | `aggregator.build_dashboard_data()["today"]["cost"]` |
| `cost_total_usd` | float, 2dp | `aggregator.build_dashboard_data()["total_cost"]` |
| `five_hour.pct_used` | float | `meta.oauth_usage.data.five_hour.utilization` |
| `five_hour.spend_usd` | float, 2dp | `_window_cost(resets_at − 5h, resets_at)` |
| `five_hour.implied_limit_usd` | float, 2dp \| null | `spend_usd ÷ (pct_used ÷ 100)`, rounded to 2dp. **Null if `pct_used < 5.0`** (API precision noise floor — below 5% the signal is too noisy to historize). |
| `five_hour.resets_at` | ISO 8601 UTC | `meta.oauth_usage.data.five_hour.resets_at`, **truncated to the nearest minute** (zero seconds + subseconds) |
| `five_hour.resets_in_s` | int ≥ 0 | `max(0, int(resets_at_epoch − now))` using the truncated `resets_at` |
| `weekly.*` | — | Same as `five_hour.*` but using `seven_day` and a 7-day window |
| `updated_at_epoch` | int | `datetime.fromisoformat(meta.oauth_usage.updated_at).timestamp()`, **rounded to the nearest 10 seconds** |

## Information Leakage Posture

No authentication. Same posture as `GET /` and `GET /api/stats`. The response deliberately omits every field that could identify or cluster the operator:

- No machine names, project dirs, session IDs, model names, user IDs, file paths.
- `resets_at` microseconds truncated to a whole minute — the raw API returns per-account-unique offsets (e.g. `.533049+00:00`) that would otherwise serve as an account fingerprint across two requests.
- `updated_at_epoch` rounded to 10 seconds for symmetry.

Only account-aggregate dollar totals, percentages, and minute-precision window boundaries escape. If defense-in-depth is later desired, a Caddy IP allowlist or the same `X-API-Key` as `/api/ingest` can be added without a response-shape change.

## Error Handling

| Condition | Behavior |
|---|---|
| `meta.oauth_usage` row missing or unparseable | Return the shape with `five_hour` and `weekly` set to `null`, plus `cost_today_usd` and `cost_total_usd` populated. HA REST sensors tolerate null values — they show `unavailable`, which is correct. |
| `pct_used` is `0` or `< 5.0` | `spend_usd`, `resets_at`, `resets_in_s` still returned; `implied_limit_usd` is `null`. |
| `resets_at` missing or malformed | That window's entire object is `null`. |
| Aggregator raises | Propagate as HTTP 500 — matches existing `/api/stats` behavior. |

No custom exception types, no retry logic, no caching. The aggregator's own cache (invalidated on ingest) covers the cost accounting; `meta.oauth_usage` is a single indexed row read.

## HA Configuration (illustrative)

```yaml
# configuration.yaml
rest:
  - resource: http://tokenfold:5000/api/ha
    scan_interval: 60
    sensor:
      - name: claude_cost_today
        value_template: "{{ value_json.cost_today_usd }}"
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

Five sensors, one HTTP call per minute. `state_class: total_increasing` marks `cost_today` as a monotonically-growing counter (daily reset is handled by HA); `measurement` marks the rest as gauges for LTS.

## Testing

No automated tests — no test harness exists in the project.

**Manual verification plan:**

1. `curl http://localhost:5000/api/ha | jq` — verify all keys present and types correct.
2. Spot-check invariant: `round(spend_usd × 100 / pct_used, 2) == implied_limit_usd` for each window when `pct_used ≥ 5.0`.
3. Verify `resets_at` contains `:00+00:00` — minute-truncated, no subseconds.
4. Confirm `updated_at_epoch` is divisible by 10.
5. Remove the `meta.oauth_usage` row temporarily; verify both window objects come back `null` and `cost_today_usd` / `cost_total_usd` remain populated.

**Optional smoke script:** `scripts/ha-smoke.sh` — a one-liner that curls the endpoint and fails loudly on any missing key. Guards against accidental field renames.

## Known Caveat

The Anthropic OAuth usage API returns `utilization` rounded to approximately 0.1%. At low utilization the implied-limit calculation is numerically unstable — a 0.1% rounding error translates to a 2% error at 5% utilization, 1% at 10%, and 0.2% at 50%. The 5% cutoff on `implied_limit_usd` trades coverage (gaps at the start of every 5-hour window) for signal quality (only historize numbers that are within ~2% of truth). Inherent to the upstream data source; not fixable in this endpoint.

## Out of Scope (deferred)

- Per-model implied limits (Opus-only, Sonnet-only) — data is present in `seven_day_sonnet` / `seven_day_opus` but not needed yet. *Update 2026-07-01:* per-model **utilization** now ships as top-level `model_buckets` (`{<slug>: {pct_used, resets_at, resets_in_s}}`, minute-truncated, enterprise-gated like `five_hour`/`weekly`, sourced from `data.limits[]` via `app/usage_buckets.py`); per-model implied **dollar** limits remain deferred.
- `extra_usage` (monthly overage credits) — also present in `meta.oauth_usage` but deferred until the primary implied-limit signal is validated.
- Per-machine cost breakdowns.
- Tokens-today / cache hit ratio.

Each can be added as additional top-level keys without breaking the existing HA config.
