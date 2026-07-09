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

    # ── drop-to-zero rule (2026-07-09 incident: account-wide grant zeroed
    #    a bucket sitting at 9% — below RESET_DROP_PTS, invisible to the
    #    magnitude rule) ──────────────────────────────────────────────────

    def test_low_util_drop_to_zero_mid_window_is_a_reset(self):
        """Utilization is a monotonic meter within a window: 9 -> 0 while
        the window is still active (anchor past BOTH poll times) can only
        be a grant, no matter how small the drop."""
        from app.limit_readings import detect_resets
        anchor = T0 + 5 * 86400
        rows = [_row(T0, 9.0, anchor),
                _row(T0 + POLL, 0.0, anchor)]
        events = detect_resets(rows)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["at_epoch"], T0 + POLL)
        self.assertEqual(events[0]["utilization_before"], 9.0)
        self.assertEqual(events[0]["utilization_after"], 0.0)

    def test_drop_to_zero_after_expiry_is_not_a_reset(self):
        """Anchor passed BETWEEN the two polls (stale anchor not yet
        refreshed): the meter returning to 0 is natural expiry — the
        zero rule requires the window still active at the LATER poll."""
        from app.limit_readings import detect_resets
        anchor = T0 + 300  # after prev poll, before cur poll
        rows = [_row(T0, 9.0, anchor),
                _row(T0 + POLL, 0.0, anchor)]
        self.assertEqual(detect_resets(rows), [])

    def test_small_drop_not_to_zero_is_not_a_reset(self):
        """9 -> 5 mid-window: under RESET_DROP_PTS and not a wipe — stays
        invisible to per-bucket detection (cross-bucket corroboration can
        still catch it at the window-anchor layer)."""
        from app.limit_readings import detect_resets
        anchor = T0 + 5 * 86400
        rows = [_row(T0, 9.0, anchor),
                _row(T0 + POLL, 5.0, anchor)]
        self.assertEqual(detect_resets(rows), [])

    def test_zero_to_zero_is_not_a_reset(self):
        from app.limit_readings import detect_resets
        anchor = T0 + 5 * 86400
        rows = [_row(T0, 0.0, anchor),
                _row(T0 + POLL, 0.0, anchor)]
        self.assertEqual(detect_resets(rows), [])


# ---------------------------------------------------------------------------
# persistent_resets — stale-replay filter over detect_resets (review M1)
# ---------------------------------------------------------------------------

class PersistentResetsTest(unittest.TestCase):

    def _rows(self, triples, anchor):
        return [{"bucket": "seven_day", "fetched_epoch": t,
                 "utilization": u, "resets_at_epoch": anchor}
                for t, u in triples]

    def test_real_grant_kept(self):
        from app.limit_readings import persistent_resets
        rows = self._rows([(1000.0, 55.0), (1600.0, 3.0), (2200.0, 4.0)],
                          anchor=900000.0)
        self.assertEqual(len(persistent_resets(rows)), 1)

    def test_stale_replay_dropped(self):
        """One-row dip that recovers on the next reading = out-of-order
        client snapshot, not a grant."""
        from app.limit_readings import detect_resets, persistent_resets
        rows = self._rows([(1000.0, 55.0), (1600.0, 40.0), (2200.0, 55.0)],
                          anchor=900000.0)
        self.assertEqual(len(detect_resets(rows)), 1)  # raw fires
        self.assertEqual(persistent_resets(rows), [])  # filter drops

    def test_trailing_event_kept_provisionally(self):
        """No subsequent reading yet: keep the event (self-corrects on
        the next poll)."""
        from app.limit_readings import persistent_resets
        rows = self._rows([(1000.0, 55.0), (1600.0, 40.0)],
                          anchor=900000.0)
        self.assertEqual(len(persistent_resets(rows)), 1)

    # ── proportional recovery test (2026-07-09 incident: the old
    #    "within RESET_DROP_PTS of before" rule is vacuously true whenever
    #    utilization_before < 10, so low-utilization grants could NEVER
    #    survive the filter) ──────────────────────────────────────────────

    def test_low_util_real_grant_kept(self):
        """9 -> 0 with the meter staying near zero afterwards: a real
        account-level grant observed at low utilization must survive."""
        from app.limit_readings import detect_resets, persistent_resets
        rows = self._rows([(1000.0, 9.0), (1600.0, 0.0), (2200.0, 1.0)],
                          anchor=900000.0)
        self.assertEqual(len(detect_resets(rows)), 1)
        self.assertEqual(len(persistent_resets(rows)), 1)

    def test_low_util_stale_replay_dropped(self):
        """9 -> 0 -> 9: the meter snapping back to its pre-event level is
        the replay fingerprint (a real grant restarts near zero)."""
        from app.limit_readings import detect_resets, persistent_resets
        rows = self._rows([(1000.0, 9.0), (1600.0, 0.0), (2200.0, 9.0)],
                          anchor=900000.0)
        self.assertEqual(len(detect_resets(rows)), 1)
        self.assertEqual(persistent_resets(rows), [])

    def test_recovery_threshold_is_proportional(self):
        """Recovery is judged against a FRACTION of the pre-event level,
        not a fixed point offset: 80 -> 0 recovering to 70 (87.5%) is a
        replay; recovering only to 50 (62.5%) is a kept grant."""
        from app.limit_readings import persistent_resets
        replay = self._rows([(1000.0, 80.0), (1600.0, 0.0), (2200.0, 70.0)],
                            anchor=900000.0)
        self.assertEqual(persistent_resets(replay), [])
        grant = self._rows([(1000.0, 80.0), (1600.0, 0.0), (2200.0, 50.0)],
                           anchor=900000.0)
        self.assertEqual(len(persistent_resets(grant)), 1)


# ---------------------------------------------------------------------------
# corroborated_resets — cross-bucket account-level reset corroboration
# ---------------------------------------------------------------------------

CT0 = 1751000400.0  # minute-aligned base for corroboration fixtures


class CorroboratedResetsTest(TempDBTestCase):
    """An account-wide grant zeroes every bucket in the same poll; a bucket
    whose own decrease is too small for detect_resets (e.g. 9 -> 1 with
    usage resuming inside the poll gap) borrows the event from a sibling
    that DID clear detection."""

    def _ins(self, bucket, fetched, pct, resets_epoch):
        self.conn.execute(
            "INSERT INTO limit_readings(fetched_epoch, source, bucket, "
            "utilization, resets_at, resets_at_epoch) "
            "VALUES(?, 'server', ?, ?, NULL, ?)",
            (fetched, bucket, pct, resets_epoch))
        self.conn.commit()

    def _seed_sibling_grant(self, at=CT0 + 1200):
        """scoped:fable 90 -> 0 mid-window at `at` — a persistent reset."""
        anchor = CT0 + 5 * 86400
        self._ins("scoped:fable", at - 600, 90.0, anchor)
        self._ins("scoped:fable", at, 0.0, anchor)
        self._ins("scoped:fable", at + 600, 1.0, anchor)

    def test_sibling_grant_corroborates_own_small_decrease(self):
        from app.limit_readings import corroborated_resets
        anchor = CT0 + 5 * 86400
        self._seed_sibling_grant()
        # Own bucket: 9 -> 1 straddling the sibling event — a decrease the
        # per-bucket rules can't see (not >=10pts, not to zero).
        self._ins("seven_day", CT0 + 600, 9.0, anchor)
        self._ins("seven_day", CT0 + 1200, 1.0, anchor)
        self._ins("seven_day", CT0 + 1800, 1.0, anchor)
        events = corroborated_resets(self.conn, "seven_day", CT0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["at_epoch"], CT0 + 1200)
        self.assertEqual(events[0]["bucket"], "seven_day")
        self.assertEqual(events[0]["corroborated_by"], "scoped:fable")

    def test_no_own_decrease_no_corroborated_event(self):
        from app.limit_readings import corroborated_resets
        anchor = CT0 + 5 * 86400
        self._seed_sibling_grant()
        self._ins("seven_day", CT0 + 600, 9.0, anchor)
        self._ins("seven_day", CT0 + 1200, 9.0, anchor)  # flat
        self.assertEqual(
            corroborated_resets(self.conn, "seven_day", CT0), [])

    def test_sibling_natural_expiry_does_not_corroborate(self):
        """A sibling event observed AFTER its own window's scheduled end
        (resets_at_epoch_before <= at_epoch) is an expiry rollover, not an
        account grant — it must not cut other buckets' windows."""
        from app.limit_readings import corroborated_resets
        anchor = CT0 + 5 * 86400
        # five_hour window expires at CT0+900, between its two polls: the
        # 39 -> 0 pair clears detect_resets' magnitude rule but is natural.
        self._ins("five_hour", CT0 + 600, 39.0, CT0 + 900)
        self._ins("five_hour", CT0 + 1200, 0.0, CT0 + 900 + 5 * 3600)
        self._ins("five_hour", CT0 + 1800, 0.0, CT0 + 900 + 5 * 3600)
        # Own bucket shows a 1-pt down-wobble straddling the same moment.
        self._ins("seven_day", CT0 + 600, 9.0, anchor)
        self._ins("seven_day", CT0 + 1200, 8.0, anchor)
        self.assertEqual(
            corroborated_resets(self.conn, "seven_day", CT0), [])

    def test_own_persistent_event_not_duplicated(self):
        """When the bucket's own detection already fired for the same
        real-world reset, the sibling event adds nothing."""
        from app.limit_readings import corroborated_resets
        anchor = CT0 + 5 * 86400
        self._seed_sibling_grant()
        self._ins("seven_day", CT0 + 600, 9.0, anchor)
        self._ins("seven_day", CT0 + 1200, 0.0, anchor)  # own zero-drop
        self._ins("seven_day", CT0 + 1800, 0.0, anchor)
        events = corroborated_resets(self.conn, "seven_day", CT0)
        self.assertEqual(len(events), 1)
        self.assertNotIn("corroborated_by", events[0])

    def test_own_events_sorted_and_shape_matches_detect(self):
        """corroborated_resets with no siblings degrades to
        persistent_resets: same events, same dict shape."""
        from app.limit_readings import corroborated_resets, persistent_resets
        anchor = CT0 + 5 * 86400
        self._ins("seven_day", CT0 + 600, 55.0, anchor)
        self._ins("seven_day", CT0 + 1200, 3.0, anchor)
        self._ins("seven_day", CT0 + 1800, 4.0, anchor)
        rows = self.conn.execute(
            "SELECT bucket, fetched_epoch, utilization, resets_at_epoch "
            "FROM limit_readings WHERE bucket='seven_day' "
            "ORDER BY fetched_epoch ASC").fetchall()
        self.assertEqual(corroborated_resets(self.conn, "seven_day", CT0),
                         persistent_resets(rows))


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
        from app.limit_readings import RETENTION_DAYS, prune_limit_readings
        now = time.time()
        keep_t = now - (RETENTION_DAYS - 1) * 86400
        self._seed(now - (RETENTION_DAYS + 1) * 86400)
        self._seed(keep_t)
        prune_limit_readings(self.conn, now_epoch=now)
        rows = self.conn.execute(
            "SELECT fetched_epoch FROM limit_readings").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["fetched_epoch"], keep_t, places=2)

    def test_empty_table_noop(self):
        from app.limit_readings import prune_limit_readings
        prune_limit_readings(self.conn)  # must not raise
        n = self.conn.execute(
            "SELECT COUNT(*) c FROM limit_readings").fetchone()["c"]
        self.assertEqual(n, 0)

    def test_fetcher_prune_hook_runs_at_most_daily(self):
        from app import usage_fetcher
        from app.limit_readings import RETENTION_DAYS
        now = time.time()
        self._seed(now - (RETENTION_DAYS + 1) * 86400)
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
