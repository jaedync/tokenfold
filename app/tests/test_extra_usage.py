"""Enterprise extra_usage meter: historized captures of Anthropic's
server-side billing meter, the authoritative monthly gauge, per-day spend
increases, and unaccounted usage (meter delta minus measured event cost)."""

import json
import unittest
from datetime import datetime, timezone

from app.tests._support import TempDBTestCase


def _ins_event(conn, uuid, ts_epoch, inp=1_000_000, day="2026-07-09"):
    """Enterprise Opus 4.8 event: 1M input = $5 at static rates."""
    conn.execute(
        "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,request_id,"
        "source_machine,project_dir,model,is_sidechain,agent_id,input_tokens,"
        "output_tokens,cache_creation_tokens,cache_read_tokens,account_email,plan,"
        "org_name,is_human_prompt) "
        "VALUES(?,'assistant',?,?,?,'s1',?,'m1','proj','claude-opus-4-8',0,NULL,"
        "?,0,0,0,'jchilton@vertech.com','enterprise','Vertech',0)",
        (uuid, day + "T12:00:00Z", ts_epoch, day, "r-" + uuid, inp))
    conn.commit()


def _epoch(iso):
    return datetime.fromisoformat(iso).replace(
        tzinfo=timezone.utc).timestamp()


class RecordMeterReadingTest(TempDBTestCase):

    def _record(self, used, limit=100000, epoch=1000.0, machine="vm-a",
                utilization=None):
        from app.extra_usage import record_meter_reading
        extra = {"is_enabled": True, "monthly_limit": limit,
                 "used_credits": used, "utilization": utilization}
        return record_meter_reading(self.conn, machine, extra, epoch)

    def _rows(self):
        return self.conn.execute(
            "SELECT * FROM extra_usage_readings ORDER BY fetched_epoch"
        ).fetchall()

    def test_records_reading(self):
        self.assertTrue(self._record(21794.0, epoch=1000.0))
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["used_cents"], 21794.0)
        self.assertEqual(rows[0]["limit_cents"], 100000)
        self.assertEqual(rows[0]["machine"], "vm-a")
        self.assertEqual(rows[0]["fetched_epoch"], 1000.0)

    def test_unchanged_meter_deduped(self):
        """Pushes arrive every ~5min from idle-ish machines; identical
        used+limit adds no information — keep the table lean."""
        self.assertTrue(self._record(100.0, epoch=1000.0))
        self.assertFalse(self._record(100.0, epoch=1300.0))
        self.assertTrue(self._record(150.0, epoch=1600.0))
        self.assertEqual(len(self._rows()), 2)

    def test_limit_change_alone_is_recorded(self):
        """Admin raising the cap is a real meter event even if spend paused."""
        self.assertTrue(self._record(100.0, limit=100000, epoch=1000.0))
        self.assertTrue(self._record(100.0, limit=200000, epoch=1300.0))
        self.assertEqual(len(self._rows()), 2)

    def test_garbage_used_credits_skipped_never_raises(self):
        from app.extra_usage import record_meter_reading
        for bad in ("ninety", None, float("nan"), float("inf"), -5, True):
            self.assertFalse(record_meter_reading(
                self.conn, "vm-a",
                {"used_credits": bad, "monthly_limit": 100000}, 1000.0))
        self.assertEqual(len(self._rows()), 0)

    def test_garbage_limit_stored_as_null(self):
        """used_credits is the meter; a bad/missing limit degrades to NULL
        rather than losing the reading."""
        from app.extra_usage import record_meter_reading
        self.assertTrue(record_meter_reading(
            self.conn, "vm-a", {"used_credits": 500.0,
                                "monthly_limit": "lots"}, 1000.0))
        self.assertIsNone(self._rows()[0]["limit_cents"])


class LatestMeterTest(TempDBTestCase):

    def test_none_when_empty(self):
        from app.extra_usage import latest_meter
        self.assertIsNone(latest_meter(self.conn))

    def test_latest_in_dollars(self):
        from app.extra_usage import latest_meter, record_meter_reading
        record_meter_reading(self.conn, "vm-a",
                             {"used_credits": 1000.0,
                              "monthly_limit": 100000}, 1000.0)
        record_meter_reading(self.conn, "vm-b",
                             {"used_credits": 21794.0,
                              "monthly_limit": 100000,
                              "utilization": 21.794}, 2000.0)
        m = latest_meter(self.conn)
        self.assertEqual(m["used_usd"], 217.94)
        self.assertEqual(m["limit_usd"], 1000.00)
        self.assertEqual(m["utilization"], 21.794)
        self.assertEqual(m["fetched_epoch"], 2000.0)
        self.assertEqual(m["machine"], "vm-b")


class DailyMeterDeltasTest(TempDBTestCase):

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def _record(self, used, iso, machine="vm-a"):
        from app.extra_usage import record_meter_reading
        record_meter_reading(self.conn, machine,
                             {"used_credits": used,
                              "monthly_limit": 100000}, _epoch(iso))

    def test_deltas_bucketed_by_utc_day(self):
        from app.extra_usage import daily_meter_deltas
        self._record(1000.0, "2026-07-08T10:00:00")
        self._record(3000.0, "2026-07-08T18:00:00")   # +$20 on 07-08
        self._record(3500.0, "2026-07-09T09:00:00")   # +$5 on 07-09
        days = daily_meter_deltas(self.conn, days=30,
                                  now=_epoch("2026-07-09T12:00:00"))
        by_day = {d["day"]: d for d in days}
        self.assertEqual(by_day["2026-07-08"]["official_usd"], 20.0)
        self.assertEqual(by_day["2026-07-09"]["official_usd"], 5.0)

    def test_cycle_reset_counts_from_zero(self):
        """A used_credits DROP is the billing-cycle rollover: the new
        reading's absolute value is the spend since the cycle started."""
        from app.extra_usage import daily_meter_deltas
        self._record(90000.0, "2026-07-31T20:00:00")
        self._record(1200.0, "2026-08-01T08:00:00")   # reset -> +$12
        days = daily_meter_deltas(self.conn, days=30,
                                  now=_epoch("2026-08-01T12:00:00"))
        by_day = {d["day"]: d for d in days}
        self.assertEqual(by_day["2026-08-01"]["official_usd"], 12.0)

    def test_unaccounted_is_official_minus_measured(self):
        """$20 official increase, $5 of recorded events in the same window:
        $15 was spent somewhere Tokenfold can't see (claude.ai web etc.)."""
        from app.extra_usage import daily_meter_deltas
        self._record(1000.0, "2026-07-09T08:00:00")
        _ins_event(self.conn, "e1", _epoch("2026-07-09T10:00:00"))  # $5
        self._record(3000.0, "2026-07-09T12:00:00")
        days = daily_meter_deltas(self.conn, days=30,
                                  now=_epoch("2026-07-09T13:00:00"))
        d = {x["day"]: x for x in days}["2026-07-09"]
        self.assertEqual(d["official_usd"], 20.0)
        self.assertEqual(d["measured_usd"], 5.0)
        self.assertEqual(d["unaccounted_usd"], 15.0)

    def test_single_reading_yields_no_delta(self):
        from app.extra_usage import daily_meter_deltas
        self._record(1000.0, "2026-07-09T08:00:00")
        self.assertEqual(daily_meter_deltas(
            self.conn, days=30, now=_epoch("2026-07-09T12:00:00")), [])


class BuildMeterPayloadTest(TempDBTestCase):

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_personal_scope_gets_none(self):
        from app.extra_usage import build_meter_payload, record_meter_reading
        record_meter_reading(self.conn, "vm-a",
                             {"used_credits": 100.0,
                              "monthly_limit": 100000}, 1000.0)
        self.assertIsNone(build_meter_payload(self.conn, "personal"))

    def test_none_when_no_captures(self):
        from app.extra_usage import build_meter_payload
        self.assertIsNone(build_meter_payload(self.conn, "enterprise"))

    def test_payload_carries_meter_freshness_and_daily(self):
        from app.extra_usage import build_meter_payload, record_meter_reading
        now = _epoch("2026-07-09T12:00:00")
        record_meter_reading(self.conn, "vm-a",
                             {"used_credits": 1000.0,
                              "monthly_limit": 100000}, now - 7200)
        record_meter_reading(self.conn, "vm-a",
                             {"used_credits": 3000.0,
                              "monthly_limit": 100000}, now - 3600)
        p = build_meter_payload(self.conn, "enterprise", now=now)
        self.assertEqual(p["used_usd"], 30.0)
        self.assertEqual(p["limit_usd"], 1000.0)
        self.assertTrue(p["fresh"])
        self.assertEqual(len(p["daily"]), 1)
        self.assertEqual(p["daily"][0]["official_usd"], 20.0)


class IngestRecordsMeterTest(TempDBTestCase):
    """The stomp-guard capture path historizes into extra_usage_readings."""

    ENTERPRISE_SHAPED = {
        "five_hour": {"utilization": None},
        "seven_day": {"utilization": None},
        "extra_usage": {"is_enabled": True, "monthly_limit": 100000,
                        "used_credits": 21794.0, "utilization": 21.794},
    }

    def test_ignored_push_lands_meter_row(self):
        r = self.client().post(
            "/api/usage",
            json={"machine": "Z000012-Mantle-VM-Dev01",
                  "usage": self.ENTERPRISE_SHAPED},
            headers={"X-API-Key": self.api_key})
        self.assertEqual(r.status_code, 200, r.text)
        rows = self.conn.execute(
            "SELECT machine, used_cents FROM extra_usage_readings").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["machine"], "Z000012-Mantle-VM-Dev01")
        self.assertEqual(rows[0]["used_cents"], 21794.0)


if __name__ == "__main__":
    unittest.main()
