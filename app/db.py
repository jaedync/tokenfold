import sqlite3
from pathlib import Path

from .config import DB_PATH

_conn: sqlite3.Connection | None = None

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
    service_tier          TEXT,

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
"""


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.execute("PRAGMA busy_timeout=5000")
        _conn.row_factory = sqlite3.Row
        _conn.executescript(SCHEMA)
    return _conn


def close_conn():
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None
