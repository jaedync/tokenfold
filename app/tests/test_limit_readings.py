"""Workstream C unit tests: limit_readings schema + writer (C1), reset
detection (C2), retention prune (C5).

The append-only limit_readings table historizes normalized OAuth usage
buckets every poll so burn-rate math can interpolate integer step-crossings;
these tests pin the writer contract (never raises, raw resets_at, duplicates
allowed) and the derived-on-read reset heuristic.
"""

import asyncio
import json
import sqlite3
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from app.tests._support import TempDBTestCase

FIXTURE_PATH = (Path(__file__).resolve().parent
                / "fixtures" / "oauth_usage_live_2026-07-01.json")


def _fixture():
    return json.loads(FIXTURE_PATH.read_text())


def _epoch(iso_str):
    return datetime.fromisoformat(iso_str.replace("Z", "+00:00")).timestamp()


# ---------------------------------------------------------------------------
# C1 — schema
# ---------------------------------------------------------------------------

class LimitReadingsSchemaTest(TempDBTestCase):
    def test_fresh_schema_has_limit_readings_table(self):
        cols = {r[1] for r in self.conn.execute(
            "PRAGMA table_info(limit_readings)")}
        self.assertTrue(
            {"id", "fetched_epoch", "source", "bucket", "utilization",
             "resets_at", "resets_at_epoch"} <= cols, cols)

    def test_seq_index_exists(self):
        names = {r[0] for r in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'")}
        self.assertIn("idx_limit_readings_seq", names)


# ---------------------------------------------------------------------------
# C1 — writer
# ---------------------------------------------------------------------------

class RecordLimitReadingsTest(TempDBTestCase):

    def test_live_fixture_writes_three_rows(self):
        from app.limit_readings import record_limit_readings
        record_limit_readings(self.conn, _fixture(), 1751000000.0, "server")
        rows = self.conn.execute(
            "SELECT * FROM limit_readings ORDER BY bucket").fetchall()
        self.assertEqual(len(rows), 3)
        by_bucket = {r["bucket"]: r for r in rows}
        self.assertEqual(set(by_bucket),
                         {"five_hour", "scoped:fable", "seven_day"})
        self.assertEqual(by_bucket["five_hour"]["utilization"], 1.0)
        self.assertEqual(by_bucket["seven_day"]["utilization"], 20.0)
        self.assertEqual(by_bucket["scoped:fable"]["utilization"], 34.0)
        for r in rows:
            self.assertEqual(r["source"], "server")
            self.assertEqual(r["fetched_epoch"], 1751000000.0)
        # resets_at stored RAW (pre-scrub) with epoch parsed alongside.
        self.assertEqual(by_bucket["seven_day"]["resets_at"],
                         "2026-07-02T08:00:00+00:00")
        self.assertAlmostEqual(by_bucket["seven_day"]["resets_at_epoch"],
                               _epoch("2026-07-02T08:00:00+00:00"), places=3)

    def test_append_only_twice_yields_six_rows(self):
        """Every-poll semantics: duplicates allowed, no dedupe-on-change."""
        from app.limit_readings import record_limit_readings
        record_limit_readings(self.conn, _fixture(), 1751000000.0, "server")
        record_limit_readings(self.conn, _fixture(), 1751000600.0, "server")
        n = self.conn.execute(
            "SELECT COUNT(*) c FROM limit_readings").fetchone()["c"]
        self.assertEqual(n, 6)

    def test_raw_resets_at_stored_unscrubbed(self):
        """Sub-second precision persists in the table; scrubbing is an API
        boundary concern, never a storage concern."""
        from app.limit_readings import record_limit_readings
        raw = "2026-07-02T08:00:12.345678+00:00"
        usage = {"seven_day": {"utilization": 63.0, "resets_at": raw}}
        record_limit_readings(self.conn, usage, 1751000000.0, "client")
        row = self.conn.execute("SELECT * FROM limit_readings").fetchone()
        self.assertEqual(row["resets_at"], raw)
        self.assertAlmostEqual(row["resets_at_epoch"], _epoch(raw), places=3)

    def test_unparseable_resets_at_gives_null_epoch(self):
        from app.limit_readings import record_limit_readings
        usage = {"seven_day": {"utilization": 63.0, "resets_at": "soon"}}
        record_limit_readings(self.conn, usage, 1751000000.0, "client")
        row = self.conn.execute("SELECT * FROM limit_readings").fetchone()
        self.assertEqual(row["resets_at"], "soon")
        self.assertIsNone(row["resets_at_epoch"])

    def test_invalid_buckets_skipped_valid_recorded(self):
        from app.limit_readings import record_limit_readings
        usage = {
            "five_hour": {"utilization": 42.0,
                          "resets_at": "2026-07-02T07:40:00+00:00"},
            "seven_day": {"utilization": "ninety"},  # non-numeric garbage
        }
        record_limit_readings(self.conn, usage, 1751000000.0, "client")
        rows = self.conn.execute("SELECT * FROM limit_readings").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["bucket"], "five_hour")
        self.assertEqual(rows[0]["utilization"], 42.0)

    def test_never_raises_on_garbage_usage(self):
        from app.limit_readings import record_limit_readings
        record_limit_readings(self.conn, "not a dict", 1751000000.0, "client")
        record_limit_readings(self.conn, None, 1751000000.0, "client")
        n = self.conn.execute(
            "SELECT COUNT(*) c FROM limit_readings").fetchone()["c"]
        self.assertEqual(n, 0)

    def test_never_raises_on_broken_connection(self):
        from app.limit_readings import record_limit_readings
        bare = sqlite3.connect(":memory:")  # no limit_readings table
        try:
            record_limit_readings(bare, _fixture(), 1751000000.0, "server")
        finally:
            bare.close()


class ServerWriterWiringTest(TempDBTestCase):
    """usage_fetcher._fetch_usage must historize each successful poll."""

    def test_fetch_usage_records_server_rows(self):
        from app import usage_fetcher

        fixture = _fixture()

        class _FakeResp:
            status_code = 200

            def json(self):
                return fixture

            def raise_for_status(self):
                pass

        class _FakeClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, *a, **k):
                return _FakeResp()

        saved_backoff = usage_fetcher._backoff_until
        usage_fetcher._backoff_until = 0.0
        try:
            with patch.object(usage_fetcher, "_get_access_token",
                              return_value="tok"), \
                 patch.object(usage_fetcher.httpx, "AsyncClient", _FakeClient):
                asyncio.run(usage_fetcher._fetch_usage())
        finally:
            usage_fetcher._backoff_until = saved_backoff

        rows = self.conn.execute(
            "SELECT bucket, source FROM limit_readings").fetchall()
        self.assertEqual(len(rows), 3)
        self.assertEqual({r["source"] for r in rows}, {"server"})
        # ...and the meta snapshot write is unchanged.
        meta = self.conn.execute(
            "SELECT value FROM meta WHERE key='oauth_usage'").fetchone()
        self.assertIsNotNone(meta)


# ---------------------------------------------------------------------------
# C2 — reset detection (pure function, plain dict rows)
# ---------------------------------------------------------------------------

T0 = 1751000000.0
POLL = 600.0


def _row(t, pct, resets_epoch, bucket="seven_day"):
    return {"bucket": bucket, "fetched_epoch": t, "utilization": pct,
            "resets_at_epoch": resets_epoch}


class DetectResetsTest(unittest.TestCase):

    def test_natural_expiry_is_not_a_reset(self):
        """prev resets_at already in the past — utilization returning to ~0
        is the window rolling over naturally, not an account reset."""
        from app.limit_readings import detect_resets
        rows = [_row(T0, 63.0, T0 - 100),
                _row(T0 + POLL, 0.0, T0 + 5 * 3600)]
        self.assertEqual(detect_resets(rows), [])

    def test_mid_window_drop_is_a_reset(self):
        from app.limit_readings import detect_resets
        rows = [_row(T0, 63.0, T0 + 3 * 3600),
                _row(T0 + POLL, 2.0, T0 + 7 * 86400)]
        events = detect_resets(rows)
        self.assertEqual(len(events), 1)
        e = events[0]
        self.assertEqual(e["bucket"], "seven_day")
        self.assertEqual(e["at_epoch"], T0 + POLL)
        self.assertEqual(e["utilization_before"], 63.0)
        self.assertEqual(e["utilization_after"], 2.0)
        # Minute-floored (privacy): T0 itself isn't minute-aligned, so the
        # raw values (T0 + 3*3600 / T0 + 7*86400) are NOT the expectation.
        self.assertEqual(e["resets_at_epoch_before"],
                         ((T0 + 3 * 3600) // 60) * 60.0)
        self.assertEqual(e["resets_at_epoch_after"],
                         ((T0 + 7 * 86400) // 60) * 60.0)
        self.assertEqual(e["resets_at_epoch_before"] % 60, 0)
        self.assertEqual(e["resets_at_epoch_after"] % 60, 0)

    def test_reset_event_epochs_are_minute_floored(self):
        """Privacy: a sub-minute resets_at_epoch offset can fingerprint the
        account across responses, so both epoch fields must be floored to
        whole minutes (None passes through unchanged elsewhere)."""
        from app.limit_readings import detect_resets
        sub_minute = T0 + 3 * 3600 + 12.345678  # deliberately not :00
        rows = [_row(T0, 63.0, sub_minute),
                _row(T0 + POLL, 2.0, sub_minute + 7 * 86400)]
        events = detect_resets(rows)
        self.assertEqual(len(events), 1)
        e = events[0]
        self.assertEqual(e["resets_at_epoch_before"] % 60, 0)
        self.assertEqual(e["resets_at_epoch_after"] % 60, 0)
        self.assertLess(e["resets_at_epoch_before"], sub_minute)
        self.assertLess(e["resets_at_epoch_after"], sub_minute + 7 * 86400)

    def test_resets_at_jump_with_flat_utilization_is_a_reset(self):
        from app.limit_readings import detect_resets
        rows = [_row(T0, 63.0, T0 + 3600),
                _row(T0 + POLL, 63.0, T0 + 3600 + 26 * 3600)]
        self.assertEqual(len(detect_resets(rows)), 1)

    def test_small_jitter_is_not_a_reset(self):
        from app.limit_readings import detect_resets
        rows = [_row(T0, 63.0, T0 + 3 * 3600),
                _row(T0 + POLL, 62.0, T0 + 3 * 3600)]
        self.assertEqual(detect_resets(rows), [])

    def test_weekly_rollover_at_expiry_is_not_a_reset(self):
        """resets_at advances ~7d AND utilization drops, but prev's window
        had already ended (resets_at_epoch <= fetched_epoch) — rollover."""
        from app.limit_readings import detect_resets
        rows = [_row(T0, 63.0, T0),  # resets exactly at the reading time
                _row(T0 + POLL, 1.0, T0 + 7 * 86400)]
        self.assertEqual(detect_resets(rows), [])

    def test_missing_resets_at_epoch_does_not_crash(self):
        from app.limit_readings import detect_resets
        rows = [_row(T0, 63.0, None),
                _row(T0 + POLL, 2.0, None),
                _row(T0 + 2 * POLL, 50.0, T0 + 3 * 3600),
                _row(T0 + 3 * POLL, 1.0, None)]
        # pair 1: prev epoch None -> no event; pair 3: drop >= 10 mid-window
        # with cur epoch None -> still an event via the drop condition.
        events = detect_resets(rows)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["at_epoch"], T0 + 3 * POLL)
        self.assertIsNone(events[0]["resets_at_epoch_after"])

    def test_empty_and_single_row_yield_nothing(self):
        from app.limit_readings import detect_resets
        self.assertEqual(detect_resets([]), [])
        self.assertEqual(detect_resets([_row(T0, 63.0, T0 + 3600)]), [])


# ---------------------------------------------------------------------------
# C5 — retention
# ---------------------------------------------------------------------------

class RetentionTest(TempDBTestCase):

    def _seed(self, fetched_epoch):
        self.conn.execute(
            "INSERT INTO limit_readings(fetched_epoch, source, bucket, "
            "utilization) VALUES(?, 'server', 'seven_day', 50.0)",
            (fetched_epoch,))
        self.conn.commit()

    def test_old_rows_pruned_recent_survive(self):
        from app.limit_readings import prune_limit_readings
        now = time.time()
        self._seed(now - 91 * 86400)
        self._seed(now - 89 * 86400)
        prune_limit_readings(self.conn, now_epoch=now)
        rows = self.conn.execute(
            "SELECT fetched_epoch FROM limit_readings").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["fetched_epoch"], now - 89 * 86400,
                               places=2)

    def test_empty_table_noop(self):
        from app.limit_readings import prune_limit_readings
        prune_limit_readings(self.conn)  # must not raise
        n = self.conn.execute(
            "SELECT COUNT(*) c FROM limit_readings").fetchone()["c"]
        self.assertEqual(n, 0)

    def test_fetcher_prune_hook_runs_at_most_daily(self):
        from app import usage_fetcher
        now = time.time()
        self._seed(now - 91 * 86400)
        saved = usage_fetcher._last_prune_epoch
        usage_fetcher._last_prune_epoch = 0.0
        try:
            self.assertTrue(usage_fetcher._maybe_prune_limit_readings(now))
            # second call inside the same day: bookkeeping short-circuits
            self.assertFalse(usage_fetcher._maybe_prune_limit_readings(now + 60))
        finally:
            usage_fetcher._last_prune_epoch = saved
        n = self.conn.execute(
            "SELECT COUNT(*) c FROM limit_readings").fetchone()["c"]
        self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main()
