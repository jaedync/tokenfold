# CLAUDE.md

This file provides guidance to Claude Code when working on Tokenfold.

## Overview

Tokenfold is a FastAPI analytics dashboard that ingests Claude Code session events from multiple machines and provides real-time usage statistics, cost tracking, and productivity metrics. Events are pushed by a cron client (`client/claude-stats-push.py`), stored in SQLite, and displayed via a Bauhaus-styled HTML dashboard.

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
    -> POST /api/ingest (X-API-Key auth, batches of 2000)
    -> app/ingest.py (parse, deduplicate by UUID, INSERT OR IGNORE)
    -> SQLite events + tool_uses tables (WAL mode)
    -> app/aggregator.py (thread-safe cached rebuild on invalidation)
    -> GET / (HTML dashboard) or GET /api/stats (JSON)
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `app/main.py` | FastAPI app + lifespan (DB init, pricing load) |
| `app/ingest.py` | POST /api/ingest - event parsing, normalization, dedup, tool extraction |
| `app/aggregator.py` | Core stats engine - session-by-session SQL aggregation with in-memory cache |
| `app/pricing.py` | Model pricing (static + dynamic from LiteLLM GitHub, 24h DB cache) |
| `app/dashboard.py` | GET / - Jinja2 HTML rendering with number formatting |
| `app/api.py` | GET /api/stats - JSON passthrough of aggregator output |
| `app/db.py` | SQLite schema, WAL pragmas, connection management |
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

## Client (claude-stats-push.py)

Standalone stdlib-only Python script. Runs via cron (Linux) or launchd (macOS). Scans `~/.claude/projects/**/*.jsonl`, tracks cursor in `~/.tokenfold-cursor.json`, strips content, POSTs batches to server. Config via `TOKENFOLD_URL`, `TOKENFOLD_API_KEY`, `TOKENFOLD_MACHINE` env vars. Legacy `CLAUDE_STATS_*` vars are supported as fallbacks.

## No Test Suite

There are currently no automated tests. The service runs on port 5000 internally. Manual testing via curl or browser.
