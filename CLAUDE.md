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
    -> app/ingest.py (privacy-safe normalization, source/provider metadata, namespaced IDs, dedup)
    -> SQLite events + tool_uses tables (WAL mode; Pi reported costs remain distinct from Claude pricing)
    -> app/aggregator.py (thread-safe cached rebuild on invalidation)
    -> GET / (HTML dashboard) or GET /api/stats (JSON)
```

`/api/ingest/pi` requires `X-API-Key`, a dotfleet `work`/`personal` account class, and bounded, typed, privacy-scrubbed events. The class maps to separate synthetic rollup identities so work and personal Pi usage cannot merge. Pi costs use reported component/total fields and group by provider/model. `/api/usage` remains Anthropic OAuth quota ingestion only; Pi data never enters that path.

### Key Modules

| Module | Purpose |
|--------|---------|
| `app/main.py` | FastAPI app + lifespan (DB init, pricing load) |
| `app/ingest.py` | POST /api/ingest (Claude) and POST /api/ingest/pi (Pi Agent) - typed normalization, namespaced dedup, tool extraction |
| `app/aggregator.py` | Core stats engine - session-by-session SQL aggregation with in-memory cache |
| `app/pricing.py` | Model pricing (static + dynamic from LiteLLM GitHub, 24h DB cache) |
| `app/dashboard.py` | GET / - Jinja2 HTML rendering with number formatting |
| `app/api.py` | GET /api/stats - JSON passthrough of aggregator output |
| `app/db.py` | SQLite schema, WAL pragmas, connection management |
| `app/notify.py` | POST /api/notify - notification relay to Home Assistant |
| `app/ha.py` | GET /api/ha - flat scrubbed metrics for Home Assistant REST sensors |
| `app/cost_windows.py` | Shared helper: dedup-by-request_id cost over an event window |
| `app/config.py` | Environment variable config |
| `app/models.py` | Pydantic request/response schemas |

### Database (SQLite)

Four tables: `events` (~50 columns, UUID PK), `tool_uses` (extracted from assistant content blocks), `sync_cursors` (ingest progress per machine/project/file), `meta` (key-value cache).

Indexes on: `session_id+type+ts_epoch`, `day`, `request_id`, `model`, `source_machine`, `project_dir`, `is_human_prompt`.

### Important Patterns

- **Token deduplication**: Streaming API repeats token counts per chunk. Aggregator uses `MAX()` per `request_id` before summing.
- **Main vs. subagent**: Events with `agent_id` are subagent invocations; tracked separately for cost attribution.
- **Active time**: Gaps between user/assistant events < 300s (IDLE_THRESHOLD_S). Gaps after tool_use->tool_result = tool execution time; others = thinking time.
- **Cache invalidation**: `aggregator.invalidate_cache()` called after successful ingest; dashboard rebuilt lazily on next request.
- **Content stripping**: Client strips large message bodies before sending (privacy + bandwidth). Metadata (sizes, types) preserved.
- **Pricing fallback chain**: LiteLLM GitHub -> DB cache -> static hardcoded prices.

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
