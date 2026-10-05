"""Run compaction of limit_readings: bounded volume, no derived-value loss.

A fixed-cadence collector polls all day, so most readings repeat the previous
value. Each flat run keeps its first reading, heartbeat rows, and a tail row
that moves forward to the latest poll. These tests pin that shape and prove
the derived consumers (burn at the provider observation, reset detection)
are identical to storing every poll.
"""
from datetime import datetime, timezone
from unittest.mock import patch

from app.limit_readings import (HISTORY_HEARTBEAT_S, RESET_JUMP_S, corroborated_resets,
                                detect_resets, persistent_resets, record_limit_readings)
from app.limit_trends import compute_burn
from app.claude_usage import MANAGED_SOURCE
from app.db import read_conn, write_txn
from app.spend_history import weekly_window_segments
from app.tests._support import TempDBTestCase

T0 = 1751000000.0
RESET = T0 + 7200


def _iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


class CompactionShapeTest(TempDBTestCase):
    def poll(self, t, pct, *, source=MANAGED_SOURCE, reset=RESET):
        record_limit_readings(self.conn, {"five_hour": {
            "utilization": pct, "resets_at": _iso(reset)}}, t, source, strict=True)

    def rows(self, source=MANAGED_SOURCE):
        return [(r["fetched_epoch"], r["utilization"]) for r in self.conn.execute(
            "SELECT fetched_epoch, utilization FROM limit_readings "
            "WHERE bucket='five_hour' AND source=? ORDER BY fetched_epoch", (source,))]

    def test_flat_run_keeps_first_heartbeats_and_latest_poll(self):
        for i in range(31):
            self.poll(T0 + 60 * i, 10)
        self.assertEqual([t - T0 for t, _ in self.rows()], [0, 600, 1200, 1800])
        self.assertEqual(HISTORY_HEARTBEAT_S, 600)

    def test_change_keeps_the_last_unchanged_poll(self):
        for i in range(16):
            self.poll(T0 + 60 * i, 10)
        self.poll(T0 + 960, 11)
        self.assertEqual(self.rows()[-2:], [(T0 + 900, 10), (T0 + 960, 11)])

    def test_anchor_jitter_compacts_but_a_real_jump_inserts(self):
        for i, jitter in enumerate((0, 0.4, -0.3, 0.2)):
            self.poll(T0 + 60 * i, 10, reset=RESET + jitter)
        self.assertEqual(len(self.rows()), 2)
        tail = self.conn.execute(
            "SELECT resets_at, resets_at_epoch FROM limit_readings "
            "ORDER BY fetched_epoch DESC LIMIT 1").fetchone()
        self.assertEqual((tail["resets_at"], tail["resets_at_epoch"]),
                         (_iso(RESET + 0.2), RESET + 0.2))
        self.poll(T0 + 240, 10, reset=RESET + RESET_JUMP_S + 1)
        self.assertEqual(len(self.rows()), 3)

    def test_sources_never_merge(self):
        for i in range(5):
            self.poll(T0 + 30 * i, 10, source=MANAGED_SOURCE if i % 2 == 0 else "client")
        self.assertEqual([t - T0 for t, _ in self.rows()], [0, 120])
        self.assertEqual([t - T0 for t, _ in self.rows("client")], [30, 90])

    def test_legacy_sources_are_never_compacted(self):
        # Legacy server and client rows are read merged with each other.
        for i in range(5):
            self.poll(T0 + 60 * i, 10, source="client")
        self.assertEqual(len(self.rows("client")), 5)

    def test_outage_longer_than_a_heartbeat_keeps_both_sides(self):
        for t in (0, 60, 1000):
            self.poll(T0 + t, 10)
        self.assertEqual([t - T0 for t, _ in self.rows()], [0, 60, 1000])

    def test_run_without_a_reset_anchor_compacts(self):
        for i in range(4):
            record_limit_readings(self.conn, {"five_hour": {"utilization": 10,
                                  "resets_at": "garbage"}}, T0 + 60 * i, MANAGED_SOURCE,
                                  strict=True)
        self.assertEqual([t - T0 for t, _ in self.rows()], [0, 180])
        self.poll(T0 + 240, 10)  # an anchor appears: not the same reading
        self.assertEqual(len(self.rows()), 3)

    def test_older_reading_inserts_instead_of_moving_the_tail_back(self):
        for t in (0, 60, 120):
            self.poll(T0 + t, 10)
        self.poll(T0 + 90, 10)
        self.assertEqual([t - T0 for t, _ in self.rows()], [0, 90, 120])

    def test_buckets_compact_independently(self):
        for i in range(4):
            record_limit_readings(self.conn, {
                "five_hour": {"utilization": 10 + i, "resets_at": _iso(RESET)},
                "seven_day": {"utilization": 3, "resets_at": _iso(RESET + 86400)},
            }, T0 + 60 * i, MANAGED_SOURCE, strict=True)
        counts = dict(self.conn.execute(
            "SELECT bucket, count(*) FROM limit_readings GROUP BY bucket").fetchall())
        self.assertEqual(counts, {"five_hour": 4, "seven_day": 2})


def _scenario():
    """8 h of 60 s polls: flat, rising, natural expiry, grant, anchor jitter."""
    out = []
    for i in range(480):
        t = T0 + 60 * i
        minute = i
        if minute < 120:                      # window 1, ends at T0+2h
            reset = RESET
            pct = 5 if minute < 30 else min(25, 5 + (minute - 30) // 3)
        else:                                 # window 2 after natural expiry
            reset = T0 + 7 * 3600
            if minute < 150:
                pct = 0
            elif minute < 210:
                pct = (minute - 150) // 2
            elif minute < 300:
                pct = 30
            elif minute < 330:
                pct = 0                       # grant: wiped while window active
            else:
                pct = (minute - 330) // 4
        out.append((t, float(pct), reset + (i % 3) * 0.4))
    return out


class CompactionLosslessTest(TempDBTestCase):
    """The production contract evaluates at the latest observation."""

    def test_burn_and_resets_match_storing_every_poll(self):
        full = []
        for t, pct, reset in _scenario():
            self.conn.execute(
                "INSERT INTO limit_readings(fetched_epoch, source, bucket, utilization, "
                "resets_at, resets_at_epoch) VALUES(?,?,?,?,?,?)",
                (t, "full", "five_hour", pct, _iso(reset), reset))
            self.conn.commit()
            full.append({"bucket": "five_hour", "fetched_epoch": t,
                         "utilization": pct, "resets_at_epoch": reset})
            record_limit_readings(self.conn, {"five_hour": {
                "utilization": pct, "resets_at": _iso(reset)}}, t, MANAGED_SOURCE,
                strict=True)
            newest = self.conn.execute(
                "SELECT max(fetched_epoch) FROM limit_readings WHERE source=?",
                (MANAGED_SOURCE,)).fetchone()[0]
            self.assertEqual(newest, t)  # the tail always holds the latest poll
            for window in (3600, 21600):
                want = compute_burn(self.conn, "five_hour", t, window, source=None)
                got = compute_burn(self.conn, "five_hour", t, window, source=MANAGED_SOURCE)
                self.assertEqual(got["resets_in_window"], want["resets_in_window"], t)
                if want["pct_per_hr"] is None:
                    self.assertIsNone(got["pct_per_hr"], t)
                else:
                    self.assertAlmostEqual(got["pct_per_hr"], want["pct_per_hr"], 9, t)

        compacted = [dict(r) for r in self.conn.execute(
            "SELECT bucket, fetched_epoch, utilization, resets_at_epoch FROM limit_readings "
            "WHERE source=? ORDER BY fetched_epoch", (MANAGED_SOURCE,))]
        self.assertLess(len(compacted), len(full) // 2)

        def key(events):
            return [(e["at_epoch"], e["utilization_before"], e["utilization_after"],
                     e["resets_at_epoch_before"], e["resets_at_epoch_after"])
                    for e in events]
        self.assertEqual(len(detect_resets(full)), 2)
        self.assertEqual(key(detect_resets(compacted)), key(detect_resets(full)))
        self.assertEqual(key(persistent_resets(compacted)), key(persistent_resets(full)))


def _weekly_scenario():
    """10 days of 60 s polls: two natural rollovers and one grant."""
    out = []
    week = 7 * 86400
    for i in range(0, 10 * 1440):
        t = T0 + 60 * i
        day = (t - T0) / 86400
        if day < 3:
            reset, pct = T0 + 3 * 86400, min(80, int(day * 20))
        elif day < 6:
            reset, pct = T0 + 3 * 86400 + week, int((day - 3) * 7)
        elif day < 6.5:
            reset, pct = T0 + 3 * 86400 + week, 0          # grant at day 6
        else:
            reset, pct = T0 + 3 * 86400 + week, int((day - 6.5) * 5)
        out.append((t, float(pct), reset + (i % 3) * 0.4))
    return out


class CompactionWeeklyBoundaryTest(TempDBTestCase):
    def test_window_segments_and_corroborated_resets_match_every_poll(self):
        for t, pct, reset in _weekly_scenario():
            five = float(min(100, pct // 2))
            for bucket, value, anchor in (("seven_day", pct, reset),
                                          ("five_hour", five, t + 3600)):
                self.conn.execute(
                    "INSERT INTO limit_readings(fetched_epoch, source, bucket, "
                    "utilization, resets_at, resets_at_epoch) VALUES(?,?,?,?,?,?)",
                    (t, "full", bucket, value, _iso(anchor), anchor))
            record_limit_readings(self.conn, {
                "seven_day": {"utilization": pct, "resets_at": _iso(reset)},
                "five_hour": {"utilization": five, "resets_at": _iso(t + 3600)},
            }, t, MANAGED_SOURCE, strict=True)
        self.conn.commit()
        now = T0 + 10 * 86400
        seven = self.conn.execute(
            "SELECT count(*) FROM limit_readings WHERE source=? AND bucket='seven_day'",
            (MANAGED_SOURCE,)).fetchone()[0]
        self.assertLess(seven, 10 * 1440 // 5)

        def shape(segments):
            return [(g["start_epoch"], g["end_epoch"], g["end_kind"], g["peak_pct"],
                     g["inferred"]) for g in segments]
        want = shape(weekly_window_segments(self.conn, "personal", now=now, source=None))
        got = shape(weekly_window_segments(self.conn, "personal", now=now,
                                           source=MANAGED_SOURCE))
        self.assertEqual(got, want)
        self.assertIn("granted", [g[2] for g in want])
        for bucket in ("seven_day", "five_hour"):
            self.assertEqual(
                corroborated_resets(self.conn, bucket, T0, until_epoch=now,
                                    source=MANAGED_SOURCE),
                corroborated_resets(self.conn, bucket, T0, until_epoch=now, source=None))


class ReadSnapshotTest(TempDBTestCase):
    def test_read_conn_sees_one_snapshot_for_the_whole_block(self):
        def count(conn):
            return conn.execute("SELECT count(*) FROM limit_readings").fetchone()[0]
        record_limit_readings(self.conn, {"five_hour": {
            "utilization": 1, "resets_at": _iso(RESET)}}, T0, MANAGED_SOURCE, strict=True)
        with read_conn(snapshot=True) as reader:
            before = count(reader)
            with write_txn(self.conn) as conn:
                conn.execute(
                    "UPDATE limit_readings SET fetched_epoch=fetched_epoch+60")
                conn.execute(
                    "INSERT INTO limit_readings(fetched_epoch, source, bucket, utilization) "
                    "VALUES(?,?,?,?)", (T0 + 120, MANAGED_SOURCE, "five_hour", 2))
            self.assertEqual(count(reader), before)
            self.assertEqual(reader.execute(
                "SELECT fetched_epoch FROM limit_readings").fetchone()[0], T0)
        with read_conn() as reader:
            self.assertEqual(count(reader), before + 1)

    def test_rate_limits_route_reads_in_one_snapshot(self):
        import app.api
        calls = []
        original = app.api.read_conn

        def traced(**kwargs):
            calls.append(kwargs)
            return original(**kwargs)
        with patch("app.api.read_conn", traced):
            self.client().get("/api/rate-limits?scope=personal")
        self.assertEqual(calls, [{"snapshot": True}])
