# CLAUDE.md

This file provides guidance to Claude Code when working on Tokenfold.

## Overview

Tokenfold is a FastAPI analytics dashboard that ingests Claude Code session events and privacy-scrubbed normalized Pi Agent events from multiple machines. It provides real-time usage statistics, source/provider-aware cost tracking, and productivity metrics. Claude events are pushed by a cron client (`client/claude-stats-push.py`); Pi clients POST typed batches to `/api/ingest/pi`. Both are stored in SQLite and displayed via a Bauhaus-styled HTML dashboard.

## Commands

```bash
# Build and run
docker compose up -d --build

# View logs
docker compose logs -f tokenfold

# Local dev (no Docker)
pip install -r requirements.txt
STATS_API_KEY=test uvicorn app.main:app --reload --host 127.0.0.1 --port 5000

# Run legacy data migration
docker compose exec tokenfold python -m migrate.import_jsonl

# Test ingest endpoint
curl -X POST http://localhost:5000/api/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $STATS_API_KEY" \
  -d '{"machine":"test","project_dir":"test","session_file":"test.jsonl","cursor":{"last_line_num":0},"events":[]}'
```

## Architecture

### Data Flow
```
Claude Code CLI -> ~/.claude/projects/**/*.jsonl
    -> client/claude-stats-push.py (cron every 5min, stdlib-only)
    -> POST /api/ingest (X-API-Key auth, Claude JSONL batches)
    -> POST /api/ingest/pi (X-API-Key auth, typed Pi normalized batches)
    -> POST /api/provider-usage (X-API-Key auth, best-effort provider quota snapshots)
    -> app/ingest.py (privacy-safe normalization, source/provider metadata, namespaced IDs, dedup)
    -> SQLite events + tool_uses tables (WAL mode; Pi reported costs remain distinct from Claude pricing)
    -> app/aggregator.py (thread-safe cached rebuild on invalidation)
    -> GET / (dashboard: Runway, Ledger, History, Lab; alias /next), GET /classic (old dashboard) or GET /api/stats (JSON)
```

`/api/ingest/pi` requires `X-API-Key`, a dotfleet `work`/`personal` account class, and bounded, typed, privacy-scrubbed events. The class maps to separate synthetic rollup identities so work and personal Pi usage cannot merge. Pi costs use reported component/total fields and group by provider/model. `/api/usage` remains Anthropic OAuth quota ingestion only; Pi data never enters that path.

### Key Modules

| Module | Purpose |
|--------|---------|
| `app/main.py` | FastAPI app + lifespan (DB init, pricing load) |
| `app/ingest.py` | POST /api/ingest (Claude), POST /api/ingest/pi (Pi Agent), and POST /api/provider-usage - typed normalization, namespaced dedup, tool extraction |
| `app/aggregator.py` | Core stats engine - session-by-session SQL aggregation with in-memory cache |
| `app/pricing.py` | Model pricing (static + dynamic from LiteLLM GitHub, 24h DB cache) |
| `app/next_page.py` | GET / and its alias GET /next - the dashboard shell (`templates/next.html`, `static/next/`, see `static/next/README.md`), CSP, hashed static path |
| `app/dashboard.py` | GET /classic - the old Jinja2 dashboard with number formatting |
| `app/api.py` | GET /api/stats - JSON passthrough of aggregator output |
| `app/db.py` | SQLite schema, WAL pragmas, connection management |
| `app/notify.py` | POST /api/notify - notification relay to Home Assistant |
| `app/ha.py` | GET /api/ha - flat scrubbed metrics for Home Assistant REST sensors |
| `app/cost_windows.py` | Shared helper: dedup-by-request_id cost over an event window |
| `app/meter_gap.py` | Work billing meter per local day against recorded Claude spend over the same reading window (`meter_days`); one source for the History meter chart (`daily_meter_deltas`) and the Ledger `meter_gaps` line |
| `app/limit_windows.py` | Shared helpers, no routes: `_iso_to_epoch`, `_scrub_to_minute_or_none` (fail-closed minute scrub) and `_bucket_window_start` (limit window start, moved to the latest granted reset) |
| `app/meridian_quota.py` | Meridian `/v1/usage/quota` -> `/api/usage/claude` payload; port of dotfleet `parseClaudeUsage`, parity-tested on the shared fixture |
| `app/meridian_collector.py` | `quota-collector` compose service (profile `meridian`): polls host-loopback Meridian every 120s (OAuth usage endpoint 429s faster polls), posts through `/api/usage/claude` |
| `app/limit_readings.py` | Quota history writer (managed flat runs compacted; burn, resets and boundaries unchanged at the latest observation), reset detection, `/api/limit-history` |
| `app/config.py` | Environment variable config |
| `app/models.py` | Pydantic request/response schemas |

### Database (SQLite)

Four tables: `events` (~50 columns, UUID PK), `tool_uses` (extracted from assistant content blocks), `sync_cursors` (ingest progress per machine/project/file), `meta` (key-value cache).

Indexes on: `session_id+type+ts_epoch`, `day`, `request_id`, `model`, `source_machine`, `project_dir`, `is_human_prompt`.

### Important Patterns

- **Token deduplication**: Streaming API repeats token counts per chunk. Aggregator uses `MAX()` per `request_id` before summing.
- **Pi copies**: A Pi session file copied to another machine or path (or a Pi fork) is reported again under a new namespace. Ingest stores a location-independent `events.pi_copy_key` (`event_dedupe.pi_copy_key`: the Anthropic `msg_*` ID, else the entry content) and flags later-ingested rows with the key of an earlier row in another session of the same work/personal class (`events.pi_copy`, `tool_uses.pi_copy`). Each start keys and flags the Pi rows that have no key (`mark_pi_copies`; code without the key wrote them) and queues only the changed days in `meta.daily_summary_rebuild_days` for the startup re-roll. `canonical_assistant_predicate` and `counts_as_activity` exclude flagged rows; a raw Pi query must add `not_pi_copy()`.
- **Main vs. subagent**: Events with `agent_id` are subagent invocations; tracked separately for cost attribution.
- **Active time**: Gaps between user/assistant events < 300s (IDLE_THRESHOLD_S). Gaps after tool_use->tool_result = tool execution time; others = thinking time.
- **Background threads and the shared connection**: closing a sqlite3 connection under a running statement crashes the process. Background work (cache rebuild, prewarm, limit-rate fit, `BackgroundCache` builds, `load_pricing`) reads on its own `read_conn()` and writes only through `db.background_write()`, which skips the write after `db.begin_shutdown()` and never opens the database. The lifespan shutdown signals and joins every background thread (`main._stop_background`) before `close_conn()`; after the shutdown starts `get_conn()` raises `DatabaseClosed` instead of reopening (`open_conn()` at startup clears it). `TOKENFOLD_PREWARM=0` turns the startup prewarm off; the test support sets it.
- **Cache invalidation**: `aggregator.invalidate_cache()` called after successful ingest; dashboard rebuilt lazily on next request.
- **Content stripping**: Client strips large message bodies before sending (privacy + bandwidth). Metadata (sizes, types) preserved.
- **Pricing fallback chain**: LiteLLM GitHub -> DB cache -> static hardcoded prices.
- **Claude quota freshness**: The Pi reporter posts only on Pi session events. The optional `quota-collector` service polls Meridian on a fixed cadence, so the snapshot stays within about two minutes when Pi is idle. Both post through `/api/usage/claude`, which rejects observations that are not newer.
- **OAuth login is read-only here**: the server-side usage fallback (`app/usage_fetcher.py`) runs only while no Meridian source owns the quota, reads the mounted credentials file and never refreshes it. A refresh-token grant rotates the token server-side and the mount is read-only, so a refresh would spend the token Claude Code and Meridian on the host need. Startup deletes the `meta.oauth_credentials` token pair that older code cached. The client (`client/claude-stats-push.py`) also only reads the file: with an expired token it skips the usage fetch. `client/refresh-oauth-token.sh` is a retired no-op.
- **Provider quota reporting**: Pi sends short-timeout Codex/OpenCode Go snapshots; observed Zen fallback spend is derived from reclassified Pi events. Missing feeds never fabricate gauges.

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `STATS_API_KEY` | (required) | API key for ingest auth |
| `DB_PATH` | `/app/data/tokenfold.db` | SQLite path |
| `TZ` | `America/Chicago` | Timezone for daily bucketing |
| `STATS_OWNER` | (empty) | Display name shown on dashboard |
| `NOTIFY_TOKEN` | (empty) | Auth token for `/api/notify` (falls back to `STATS_API_KEY`) |
| `HA_URL` | (empty) | Home Assistant URL for notification relay |
| `HA_TOKEN` | (empty) | Home Assistant long-lived access token |
| `HA_DEVICES` | (empty) | Comma-separated HA notify targets |

## Client (claude-stats-push.py)

Standalone stdlib-only Python script. Runs via launchd on macOS or cron on Linux. Scans `~/.claude/projects/**/*.jsonl` for event JSONL, tracks cursor in `~/.tokenfold-cursor.json`, strips content, POSTs batches to `/api/ingest`. On macOS also scans `~/Library/Application Support/Claude/claude-code-sessions/*/*/local_*.json` for Claude Desktop session metadata (titles, MCP toggles, lifecycle timestamps) and POSTs to `/api/desktop-metadata`. Uses `from __future__ import annotations` for Python 3.9 compatibility (stock macOS Python). Config via `TOKENFOLD_URL`, `TOKENFOLD_API_KEY`, `TOKENFOLD_MACHINE` env vars. Legacy `CLAUDE_STATS_*` vars are supported as fallbacks.

## Test Suite

Tests live under `app/tests/` (server) and `client/test_desktop_metadata.py` (client). Both use stdlib `unittest` — no pytest, no new dependencies. Run:

- Server: `.venv/bin/python -m unittest app.tests.test_desktop_sessions -v`
- Client: `.venv/bin/python client/test_desktop_metadata.py -v`

The service runs on port 5000 internally. Manual testing via curl or browser for anything not covered by the unit tests.
