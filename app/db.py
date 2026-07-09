import sqlite3
from pathlib import Path

from .config import DB_PATH

_conn: sqlite3.Connection | None = None

_DAILY_SUMMARY_DDL = """
CREATE TABLE IF NOT EXISTS daily_summary (
    day               TEXT NOT NULL,
    account_email     TEXT NOT NULL,
    plan              TEXT,
    org_name          TEXT,
    org_type          TEXT,
    org_uuid          TEXT,
    sessions          INTEGER DEFAULT 0,
    human_prompts     INTEGER DEFAULT 0,
    tool_calls        INTEGER DEFAULT 0,
    input_tokens      INTEGER DEFAULT 0,
    output_tokens     INTEGER DEFAULT 0,
    cache_creation_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    active_s          REAL DEFAULT 0,
    thinking_s        REAL DEFAULT 0,
    tool_exec_s       REAL DEFAULT 0,
    subagent_s        REAL DEFAULT 0,
    agent_runs        INTEGER DEFAULT 0,
    cost              REAL DEFAULT 0,
    model_json        TEXT DEFAULT '{}',
    project_json      TEXT DEFAULT '{}',
    machine_json      TEXT DEFAULT '{}',
    tool_json         TEXT DEFAULT '{}',
    prompt_model_json TEXT DEFAULT '{}',
    gen_json          TEXT DEFAULT '{}',
    updated_at        TEXT NOT NULL,
    PRIMARY KEY (day, account_email)
);
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    uuid              TEXT PRIMARY KEY,
    type              TEXT NOT NULL,
    subtype           TEXT,
    timestamp         TEXT NOT NULL,
    ts_epoch          REAL NOT NULL,
    day               TEXT NOT NULL,

    session_id        TEXT,
    parent_uuid       TEXT,
    is_sidechain      INTEGER DEFAULT 0,
    user_type         TEXT,
    cwd               TEXT,
    git_branch        TEXT,
    version           TEXT,
    slug              TEXT,
    agent_id          TEXT,
    permission_mode   TEXT,

    source_machine    TEXT NOT NULL,
    project_dir       TEXT,

    model             TEXT,
    message_id        TEXT,
    request_id        TEXT,
    stop_reason       TEXT,
    api_error         TEXT,
    is_api_error      INTEGER DEFAULT 0,

    input_tokens          INTEGER DEFAULT 0,
    output_tokens         INTEGER DEFAULT 0,
    cache_creation_tokens INTEGER DEFAULT 0,
    cache_read_tokens     INTEGER DEFAULT 0,
    cache_ephemeral_5m    INTEGER DEFAULT 0,
    cache_ephemeral_1h    INTEGER DEFAULT 0,
    web_search_requests   INTEGER DEFAULT 0,
    web_fetch_requests    INTEGER DEFAULT 0,
    service_tier          TEXT,
    speed             TEXT,
    inference_geo     TEXT,
    account_email     TEXT,
    org_name          TEXT,
    plan              TEXT,
    rate_limit_tier   TEXT,
    org_type          TEXT,
    org_uuid          TEXT,

    has_text          INTEGER DEFAULT 0,
    has_thinking      INTEGER DEFAULT 0,
    has_tool_use      INTEGER DEFAULT 0,
    has_tool_result   INTEGER DEFAULT 0,
    has_image         INTEGER DEFAULT 0,
    is_human_prompt   INTEGER DEFAULT 0,
    text_length       INTEGER DEFAULT 0,
    thinking_length   INTEGER DEFAULT 0,

    level             TEXT,
    duration_ms       INTEGER,
    error_status      INTEGER,
    retry_attempt     INTEGER,
    max_retries       INTEGER,

    progress_type     TEXT,
    hook_event        TEXT,
    hook_name         TEXT,
    tool_use_id_ref   TEXT,

    file_op_type      TEXT,
    file_path         TEXT,

    queue_operation   TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_session_type_ts ON events(session_id, type, ts_epoch);
CREATE INDEX IF NOT EXISTS idx_events_day ON events(day);
CREATE INDEX IF NOT EXISTS idx_events_request ON events(request_id) WHERE request_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_events_model ON events(model) WHERE model IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_events_source ON events(source_machine);
CREATE INDEX IF NOT EXISTS idx_events_project ON events(project_dir);
CREATE INDEX IF NOT EXISTS idx_events_human ON events(is_human_prompt) WHERE is_human_prompt = 1;
CREATE INDEX IF NOT EXISTS idx_events_session_ts ON events(session_id, ts_epoch);
CREATE INDEX IF NOT EXISTS idx_events_billable_ts ON events(ts_epoch)
    WHERE type='assistant' AND request_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS tool_uses (
    tool_use_id       TEXT PRIMARY KEY,
    event_uuid        TEXT NOT NULL,
    session_id        TEXT,
    source_machine    TEXT NOT NULL,
    name              TEXT NOT NULL,
    timestamp         TEXT NOT NULL,
    ts_epoch          REAL NOT NULL,
    day               TEXT NOT NULL,
    result_event_uuid TEXT,
    is_error          INTEGER DEFAULT 0,
    duration_ms       REAL
);

CREATE INDEX IF NOT EXISTS idx_tool_uses_name ON tool_uses(name);
CREATE INDEX IF NOT EXISTS idx_tool_uses_day ON tool_uses(day);
CREATE INDEX IF NOT EXISTS idx_tool_uses_session ON tool_uses(session_id);

CREATE TABLE IF NOT EXISTS sync_cursors (
    machine         TEXT NOT NULL,
    project_dir     TEXT NOT NULL,
    session_file    TEXT NOT NULL,
    last_line_num   INTEGER DEFAULT 0,
    last_timestamp  TEXT,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (machine, project_dir, session_file)
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS session_titles (
    session_id  TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    updated_at  TEXT
);

-- Manually recorded Claude org-page billing figures. FIRST-CLASS data:
-- not derived from events, never rebuilt, included in DB backups.
CREATE TABLE IF NOT EXISTS billing_readings (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    scope          TEXT NOT NULL DEFAULT 'enterprise',
    amount_usd     REAL NOT NULL,
    measured_usd   REAL,
    month          TEXT NOT NULL,
    recorded_at    TEXT NOT NULL,
    recorded_epoch REAL NOT NULL,
    note           TEXT
);

CREATE INDEX IF NOT EXISTS idx_billing_readings_seq
    ON billing_readings(scope, month, recorded_epoch);

-- Anthropic's server-side billing meter (the oauth/usage extra_usage block,
-- US cents) captured from enterprise-account usage pushes. FIRST-CLASS data
-- like billing_readings: not derived from events, never rebuilt. Consecutive
-- identical (used,limit) readings are deduped at record time.
CREATE TABLE IF NOT EXISTS extra_usage_readings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_epoch REAL NOT NULL,
    machine       TEXT,
    used_cents    REAL NOT NULL,
    limit_cents   REAL,
    utilization   REAL
);

CREATE INDEX IF NOT EXISTS idx_extra_usage_readings_seq
    ON extra_usage_readings(fetched_epoch);

-- Append-only per-bucket OAuth usage-limit history (one row per bucket per
-- poll). No raw-JSON column: the per-bucket fields suffice, and the latest
-- full raw blob already lives in meta.oauth_usage. resets_at stores the RAW
-- pre-scrub string (minute-scrubbing happens at API boundaries only);
-- resets_at_epoch exists for query arithmetic.
CREATE TABLE IF NOT EXISTS limit_readings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_epoch   REAL NOT NULL,
    source          TEXT NOT NULL,
    bucket          TEXT NOT NULL,
    utilization     REAL NOT NULL,
    resets_at       TEXT,
    resets_at_epoch REAL
);

CREATE INDEX IF NOT EXISTS idx_limit_readings_seq
    ON limit_readings(bucket, fetched_epoch);

""" + _DAILY_SUMMARY_DDL + """
CREATE TABLE IF NOT EXISTS desktop_sessions (
    cli_session_id         TEXT PRIMARY KEY,
    desktop_session_id     TEXT,
    source_machine         TEXT NOT NULL,
    title                  TEXT,
    model                  TEXT,
    effort                 TEXT,
    permission_mode        TEXT,
    completed_turns        INTEGER,
    is_archived            INTEGER DEFAULT 0,
    cwd                    TEXT,
    origin_cwd             TEXT,
    created_at_ms          INTEGER,
    last_activity_at_ms    INTEGER,
    enabled_mcp_tools      TEXT,
    remote_mcp_servers     TEXT,
    chrome_permission_mode TEXT,
    chrome_allowed_domains TEXT,
    updated_at_ms          INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_desktop_last_activity
    ON desktop_sessions(last_activity_at_ms);
CREATE INDEX IF NOT EXISTS idx_desktop_source
    ON desktop_sessions(source_machine);
"""

# Columns added after initial deploy, per table. CREATE TABLE IF NOT EXISTS won't add
# columns to a pre-existing table, so ALTER them in on connect. Each guarded by a check.
_ADDED_COLUMNS = {
    "events": {
        "speed": "TEXT", "inference_geo": "TEXT",
        "account_email": "TEXT", "org_name": "TEXT",
        "plan": "TEXT", "rate_limit_tier": "TEXT",
        "org_type": "TEXT", "org_uuid": "TEXT",
        "web_search_requests": "INTEGER DEFAULT 0",
        "web_fetch_requests": "INTEGER DEFAULT 0",
    },
}


def _migrate(conn) -> None:
    for table, cols in _ADDED_COLUMNS.items():
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        for col, decl in cols.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")

    # daily_summary is a derived rollup — if it exists in a stale shape (missing
    # account_email or missing org_type), drop and recreate from _DAILY_SUMMARY_DDL.
    # Both columns are load-bearing: account_email is the PK component; org_type is
    # required for the config-driven enterprise predicate. The table is fully derived
    # from events, so a drop+recreate is safe — startup backfill rebuilds it.
    ds_cols = {r[1] for r in conn.execute("PRAGMA table_info(daily_summary)")}
    if ds_cols and ("account_email" not in ds_cols or "org_type" not in ds_cols):
        conn.execute("DROP TABLE daily_summary")   # derived rollup — rebuilt from events
        conn.executescript(_DAILY_SUMMARY_DDL)

    conn.commit()


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.execute("PRAGMA busy_timeout=60000")
        _conn.row_factory = sqlite3.Row
        _conn.executescript(SCHEMA)
        _migrate(_conn)
    return _conn


def close_conn():
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None
