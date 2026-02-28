#!/usr/bin/env python3
"""One-time migration: import existing JSONL session data.

Run inside the container:
    docker compose exec tokenfold python -m migrate.import_jsonl

Or locally:
    DB_PATH=./data/tokenfold.db python -m migrate.import_jsonl
"""

import json
import os
import socket
import sys
import time
from pathlib import Path

# Allow running as module from /app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import get_conn
from app.ingest import _extract_event, _extract_tool_uses, EVENT_COLS, TOOL_COLS, _EVENT_SQL, _TOOL_SQL

SESSIONS_DIR = Path("/sessions")  # mounted read-only in container
SOURCE_MACHINE = os.environ.get("SOURCE_MACHINE", socket.gethostname())
BATCH_SIZE = 10000


def main():
    if not SESSIONS_DIR.exists():
        print(f"Sessions dir {SESSIONS_DIR} not found. Mount it in docker-compose.yml.")
        sys.exit(1)

    conn = get_conn()
    total_events = 0
    total_tools = 0
    total_files = 0
    total_lines = 0
    skipped = 0

    jsonl_files = sorted(SESSIONS_DIR.rglob("*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files to import")

    event_batch = []
    tool_batch = []

    for fpath in jsonl_files:
        # Derive project_dir from parent dir name
        try:
            rel = fpath.relative_to(SESSIONS_DIR)
            project_dir = rel.parts[0] if rel.parts else "unknown"
        except ValueError:
            project_dir = "unknown"

        total_files += 1
        if total_files % 100 == 0:
            print(f"  Processing file {total_files}/{len(jsonl_files)} ...")

        try:
            with open(fpath) as f:
                for line in f:
                    total_lines += 1
                    try:
                        rec = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        skipped += 1
                        continue

                    row = _extract_event(rec, SOURCE_MACHINE, project_dir)
                    if row is None:
                        skipped += 1
                        continue

                    event_batch.append(tuple(row[c] for c in EVENT_COLS))

                    if row["type"] == "assistant" and row["has_tool_use"]:
                        tools = _extract_tool_uses(
                            rec, row["uuid"], SOURCE_MACHINE,
                            row["session_id"], row["timestamp"],
                            row["ts_epoch"], row["day"],
                        )
                        for t in tools:
                            tool_batch.append(tuple(t[c] for c in TOOL_COLS))

                    # Flush batch
                    if len(event_batch) >= BATCH_SIZE:
                        _flush(conn, event_batch, tool_batch)
                        total_events += len(event_batch)
                        total_tools += len(tool_batch)
                        event_batch = []
                        tool_batch = []
        except OSError as e:
            print(f"  Error reading {fpath}: {e}", file=sys.stderr)
            continue

    # Final flush
    if event_batch:
        _flush(conn, event_batch, tool_batch)
        total_events += len(event_batch)
        total_tools += len(tool_batch)

    print(f"\nMigration complete:")
    print(f"  Files processed: {total_files}")
    print(f"  Lines read:      {total_lines}")
    print(f"  Events inserted: {total_events}")
    print(f"  Tool uses:       {total_tools}")
    print(f"  Skipped:         {skipped}")

    # Verify counts
    row = conn.execute("SELECT COUNT(*) as cnt FROM events").fetchone()
    print(f"  DB event count:  {row['cnt']}")
    row = conn.execute("SELECT COUNT(*) as cnt FROM tool_uses").fetchone()
    print(f"  DB tool count:   {row['cnt']}")


def _flush(conn, event_batch, tool_batch):
    cur = conn.cursor()
    for erow in event_batch:
        try:
            cur.execute(_EVENT_SQL, erow)
        except Exception:
            pass
    for trow in tool_batch:
        try:
            cur.execute(_TOOL_SQL, trow)
        except Exception:
            pass
    conn.commit()


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"  Elapsed: {elapsed:.1f}s")
