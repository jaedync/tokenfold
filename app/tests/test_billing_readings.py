"""Enterprise billing readings: historize the Claude org-page MTD figure and
compare consecutive deltas against our measured cost over the same window."""

import unittest
from datetime import datetime, timedelta, timezone

from app.tests._support import TempDBTestCase


def _ins_event(conn, uuid, ts_epoch, inp=1_000_000, day="2026-06-09"):
    """Enterprise Opus 4.8 event: 1M input = $5 at static rates."""
    conn.execute(
        "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,request_id,"
        "source_machine,project_dir,model,is_sidechain,agent_id,input_tokens,"
        "output_tokens,cache_creation_tokens,cache_read_tokens,account_email,plan,"
        "org_name,is_human_prompt) "
        "VALUES(?,'assistant',?,?,?,'s1',?,'m1','proj','claude-opus-4-8',0,NULL,"
        "?,0,0,0,'jaedyn@acme.io','enterprise','Acme',0)",
        (uuid, day + "T12:00:00Z", ts_epoch, day, "r-" + uuid, inp))
    conn.commit()


class _ReadingsBase(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        self._saved_pw = self._config.DASHBOARD_PASSWORD
        self._saved_user = self._config.DASHBOARD_USER
        self._config.DASHBOARD_PASSWORD = "pw"
        self._config.DASHBOARD_USER = "jaedyn"
        self.addCleanup(self._restore_auth)

    def _restore_auth(self):
        self._config.DASHBOARD_PASSWORD = self._saved_pw
        self._config.DASHBOARD_USER = self._saved_user

    def _post(self, payload, auth=("jaedyn", "pw")):
        c = self.client()
        return c.post("/api/billing-readings", json=payload, auth=auth)

    def _delete(self, rid, auth=("jaedyn", "pw")):
        c = self.client()
        return c.delete(f"/api/billing-readings/{rid}", auth=auth)


class ReadingsAuthTest(_ReadingsBase):

    def test_post_requires_basic_auth(self):
        c = self.client()
        r = c.post("/api/billing-readings", json={"amount_usd": 100.0})
        self.assertEqual(r.status_code, 401)

    def test_wrong_password_rejected(self):
        r = self._post({"amount_usd": 100.0}, auth=("jaedyn", "nope"))
        self.assertEqual(r.status_code, 401)

    def test_fail_closed_without_dashboard_password(self):
        """require_dashboard_auth is OPEN when no password is set — writes
        must still be rejected (same policy as the ingest-key reveal)."""
        self._config.DASHBOARD_PASSWORD = ""
        c = self.client()
        r = c.post("/api/billing-readings", json={"amount_usd": 100.0})
        self.assertEqual(r.status_code, 403)


class ReadingsCrudTest(_ReadingsBase):

    def test_record_stamps_month_and_measured_snapshot(self):
        now = datetime.now(timezone.utc)
        # one enterprise event 1h ago, inside the current UTC month when possible
        ev_ts = max(now - timedelta(hours=1),
                    now.replace(day=1, hour=0, minute=0, second=0, microsecond=0))
        _ins_event(self.conn, "e1", ev_ts.timestamp() + 1,
                   day=ev_ts.strftime("%Y-%m-%d"))

        r = self._post({"amount_usd": 412.62, "note": "org page"})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["month"], now.strftime("%Y-%m"))
        self.assertAlmostEqual(body["amount_usd"], 412.62, places=2)
        self.assertAlmostEqual(body["measured_usd"], 5.0, places=2)

        row = self.conn.execute("SELECT * FROM billing_readings").fetchone()
        self.assertEqual(row["scope"], "enterprise")
        self.assertEqual(row["note"], "org page")

    def test_validation_rejects_bad_amounts(self):
        import json as _json
        c = self.client()
        for bad in ("-1", "100000000", "NaN", "Infinity"):
            # raw body: httpx's json= refuses to serialize NaN/inf client-side,
            # but a hostile/buggy client can still send them — python's
            # json.loads accepts these literals, so the server must reject.
            r = c.post("/api/billing-readings",
                       content='{"amount_usd": %s}' % bad,
                       headers={"Content-Type": "application/json"},
                       auth=("jaedyn", "pw"))
            self.assertIn(r.status_code, (400, 422),
                          f"amount {bad!r} must be rejected")
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) c FROM billing_readings")
            .fetchone()["c"], 0)

    def test_note_is_truncated_and_optional(self):
        r = self._post({"amount_usd": 1.0, "note": "x" * 1000})
        self.assertEqual(r.status_code, 200)
        row = self.conn.execute("SELECT note FROM billing_readings").fetchone()
        self.assertLessEqual(len(row["note"]), 256)

    def test_delete_and_404(self):
        rid = self._post({"amount_usd": 5.0}).json()["id"]
        self.assertEqual(self._delete(rid).status_code, 200)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) c FROM billing_readings")
            .fetchone()["c"], 0)
        self.assertEqual(self._delete(rid).status_code, 404)


class ReadingsPayloadTest(_ReadingsBase):

    def _seed_reading(self, amount, epoch, month=None, measured=0.0):
        month = month or datetime.fromtimestamp(
            epoch, tz=timezone.utc).strftime("%Y-%m")
        self.conn.execute(
            "INSERT INTO billing_readings(scope, amount_usd, measured_usd, month, "
            "recorded_at, recorded_epoch) VALUES('enterprise',?,?,?,?,?)",
            (amount, measured,
             month, datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat(),
             epoch))
        self.conn.commit()

    def test_interval_deltas_and_coverage(self):
        now = datetime.now(timezone.utc)
        t1 = (now - timedelta(hours=3)).timestamp()
        t2 = (now - timedelta(hours=1)).timestamp()
        month = now.strftime("%Y-%m")
        # $5 measured between the two readings
        _ins_event(self.conn, "e1", t1 + 600,
                   day=datetime.fromtimestamp(t1 + 600).strftime("%Y-%m-%d"))
        self._seed_reading(100.0, t1, month)
        self._seed_reading(110.0, t2, month)

        import app.aggregator as agg
        agg._cached_data.clear()
        d = agg.build_dashboard_data("enterprise")
        readings = d["billing_readings"]
        self.assertEqual(len(readings), 2)
        newest = readings[0]  # newest first
        self.assertAlmostEqual(newest["delta_official"], 10.0, places=2)
        self.assertAlmostEqual(newest["delta_measured"], 5.0, places=2)
        self.assertAlmostEqual(newest["coverage_pct"], 50.0, places=1)
        self.assertIn("measured_since", newest)

    def test_cross_month_pairs_get_no_interval(self):
        now = datetime.now(timezone.utc)
        t2 = (now - timedelta(hours=1)).timestamp()
        self._seed_reading(900.0, t2 - 7200, month="2026-05")
        self._seed_reading(50.0, t2, month=now.strftime("%Y-%m"))

        import app.aggregator as agg
        agg._cached_data.clear()
        d = agg.build_dashboard_data("enterprise")
        newest = d["billing_readings"][0]
        self.assertIsNone(newest.get("delta_official"))
        self.assertIsNone(newest.get("coverage_pct"))

    def test_zero_official_delta_means_no_coverage(self):
        now = datetime.now(timezone.utc)
        month = now.strftime("%Y-%m")
        self._seed_reading(100.0, (now - timedelta(hours=2)).timestamp(), month)
        self._seed_reading(100.0, (now - timedelta(hours=1)).timestamp(), month)

        import app.aggregator as agg
        agg._cached_data.clear()
        newest = agg.build_dashboard_data("enterprise")["billing_readings"][0]
        self.assertAlmostEqual(newest["delta_official"], 0.0, places=2)
        self.assertIsNone(newest["coverage_pct"])

    def test_personal_scope_has_no_readings(self):
        self._seed_reading(100.0, datetime.now(timezone.utc).timestamp() - 60)
        import app.aggregator as agg
        agg._cached_data.clear()
        d = agg.build_dashboard_data("personal")
        self.assertEqual(d.get("billing_readings", []), [])


class ReadingsTemplateTest(unittest.TestCase):
    """Source-level UI wiring, like test_dashboard_template."""

    def setUp(self):
        from pathlib import Path
        self.html = (Path(__file__).resolve().parents[2]
                     / "templates" / "dashboard.html").read_text()

    def test_section_and_renderer_present(self):
        self.assertIn("billingReadings", self.html)
        self.assertIn("billing_readings", self.html)
        self.assertIn("readings_writable", self.html)

    def test_note_rendered_through_esc(self):
        # the renderer must escape user-entered note text
        self.assertIn("esc(", self.html)
        import re
        renderer = re.search(r"function renderBillingReadings[\s\S]{0,3000}",
                             self.html)
        self.assertIsNotNone(renderer, "renderBillingReadings missing")
        self.assertIn("esc(", renderer.group(0))


if __name__ == "__main__":
    unittest.main()
