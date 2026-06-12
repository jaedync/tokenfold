"""US-residency assumption for enterprise usage.

Claude Code transcripts stamp inference_geo='not_available' on ALL
subscription traffic, so a US-pinned enterprise workspace (billed 1.1x on
every token category, Opus/Sonnet 4.6+) is invisible to us. When
TOKENFOLD_ENTERPRISE_GEO=us, enterprise-classified usage is billed at the
US rate at COMPUTE TIME — raw events are never modified; flipping the env
back (plus a day re-roll for stored rollups) fully reverts.
"""

import unittest
from datetime import datetime, timezone

from app.tests._support import TempDBTestCase

NOW = 1781000000.0
DAY = "2026-06-09"


class _GeoBase(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        self._saved_geo = getattr(self._config, "ENTERPRISE_ASSUME_GEO", "")
        self.addCleanup(self._restore_geo)

    def _restore_geo(self):
        self._config.ENTERPRISE_ASSUME_GEO = self._saved_geo

    def _set_assume(self, val):
        self._config.ENTERPRISE_ASSUME_GEO = val

    def _ins(self, uuid, ts=NOW, enterprise=False, geo=None, day=DAY):
        """Opus 4.8 event, 1M input = $5.00 at static rates ($5.50 at US 1.1x)."""
        acct = "j@acme.io" if enterprise else None
        org_type = "claude_enterprise" if enterprise else None
        self.conn.execute(
            "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,"
            "request_id,source_machine,project_dir,model,is_sidechain,agent_id,"
            "input_tokens,output_tokens,cache_creation_tokens,cache_read_tokens,"
            "inference_geo,account_email,org_type,is_human_prompt) "
            "VALUES(?,'assistant',?,?,?,?,?,'m1','proj','claude-opus-4-8',0,NULL,"
            "1000000,0,0,0,?,?,?,0)",
            (uuid, day + "T12:00:00Z", ts, day, "sess-" + uuid, "r-" + uuid,
             geo, acct, org_type))
        self.conn.commit()


class WindowCostGeoTest(_GeoBase):
    def _window(self, scope):
        from app.cost_windows import compute_window_cost
        return compute_window_cost(self.conn, NOW - 60, NOW + 60, scope)

    def test_flag_applies_1_1x_to_enterprise_scope_only(self):
        self._ins("e1", enterprise=True)
        self._ins("p1", enterprise=False)
        self._set_assume("us")
        self.assertAlmostEqual(self._window("enterprise"), 5.5, places=4)
        self.assertAlmostEqual(self._window("personal"), 5.0, places=4)

    def test_flag_off_is_noop(self):
        self._ins("e1", enterprise=True)
        self._set_assume("")
        self.assertAlmostEqual(self._window("enterprise"), 5.0, places=4)

    def test_recorded_us_geo_not_double_charged(self):
        self._ins("e1", enterprise=True, geo="us")
        self._set_assume("us")
        self.assertAlmostEqual(self._window("enterprise"), 5.5, places=4)


class SummarizerGeoTest(_GeoBase):
    def _day_cost(self, account):
        row = self.conn.execute(
            "SELECT cost FROM daily_summary WHERE day=? AND account_email=?",
            (DAY, account)).fetchone()
        return row["cost"] if row else None

    def test_rollup_stores_assumed_cost_and_reroll_reverts(self):
        from app.summarizer import summarize_days
        self._ins("e1", enterprise=True)
        self._ins("p1", enterprise=False)

        self._set_assume("us")
        summarize_days([DAY])
        self.assertAlmostEqual(self._day_cost("j@acme.io"), 5.5, places=4)
        self.assertAlmostEqual(self._day_cost("unknown"), 5.0, places=4)

        # reversibility: flag off + re-roll restores standard pricing
        self._set_assume("")
        summarize_days([DAY])
        self.assertAlmostEqual(self._day_cost("j@acme.io"), 5.0, places=4)


class DashboardGeoTest(_GeoBase):
    def _build(self, scope, days=None):
        import app.aggregator as agg
        from app.summarizer import summarize_days
        summarize_days(days or [DAY])
        agg._cached_data.clear()
        return agg.build_dashboard_data(scope)

    def test_sessions_and_flag_in_payload(self):
        now = datetime.now(timezone.utc).timestamp()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._ins("e1", ts=now, enterprise=True, day=today)
        self._set_assume("us")
        d = self._build("enterprise", days=[today])
        self.assertTrue(d["geo_assumed"])
        sess = d["recent_sessions"][0]
        self.assertAlmostEqual(sess["cost"], 5.5, places=2)

    def test_personal_payload_never_assumed(self):
        self._ins("p1", enterprise=False)
        self._set_assume("us")
        d = self._build("personal")
        self.assertFalse(d["geo_assumed"])


class TemplateGeoTest(unittest.TestCase):
    def test_assumption_is_disclosed_in_ui(self):
        from pathlib import Path
        html = (Path(__file__).resolve().parents[2]
                / "templates" / "dashboard.html").read_text()
        self.assertIn("geo_assumed", html)
        self.assertIn("1.1x", html)


if __name__ == "__main__":
    unittest.main()
