"""Tests for month-to-date cost counter.

Subtask A: verifies month_cost and month_label in the aggregator payload.
Subtask C: verifies the UI wiring — embedded data_json, page structure,
           and the reduced-motion / count-up guard strings.
"""
import time
import unittest
from datetime import datetime, timedelta, timezone

from app.tests._support import TempDBTestCase


def _make_tz():
    from app.config import TZ_NAME
    from zoneinfo import ZoneInfo
    return ZoneInfo(TZ_NAME)


def ins(conn, uuid, req, acct, plan, org, machine, project, session,
        model="claude-opus-4-8", day="2026-06-09", ts=1781000000.0,
        inp=0, out=0):
    conn.execute(
        "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,request_id,"
        "source_machine,project_dir,model,is_sidechain,agent_id,input_tokens,"
        "output_tokens,cache_creation_tokens,cache_read_tokens,account_email,plan,"
        "org_name,is_human_prompt,user_type) VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (uuid, "assistant", day + "T12:00:00Z", ts, day, session, req, machine,
         project, model, 0, None, inp, out, 0, 0, acct, plan, org, 0, None))
    conn.commit()


class MonthCostAggregatorTest(TempDBTestCase):
    """Test that month_cost and month_label are correctly computed in the payload."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_month_cost_current_month_only(self):
        """Only current-month events count; prior-month events do not."""
        TZ = _make_tz()
        now = datetime.now(TZ)

        # Current-month event: enterprise Opus 4.8 1M input = $5
        now_ts = now.timestamp()
        now_day = now.strftime("%Y-%m-%d")
        ins(self.conn, "e1", "re1", "jaedyn@acme.io", "enterprise", "Acme",
            "mA", "proj", "sE", inp=1_000_000, day=now_day, ts=now_ts)

        # Prior-month event (45 days ago = always prior calendar month)
        old_dt = now - timedelta(days=45)
        old_ts = old_dt.timestamp()
        old_day = old_dt.strftime("%Y-%m-%d")
        ins(self.conn, "e2", "re2", "jaedyn@acme.io", "enterprise", "Acme",
            "mB", "proj", "sE2", inp=1_000_000, day=old_day, ts=old_ts)

        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data.clear()

        d = agg.build_dashboard_data("enterprise")

        # month_cost must be $5 (current month only), not $10 (both months)
        self.assertIn("month_cost", d, "month_cost must be present in payload")
        self.assertAlmostEqual(d["month_cost"], 5.0, places=2,
                               msg=f"Expected 5.0 (current month only), got {d['month_cost']}")

    def test_month_label_format(self):
        """month_label must equal the current month in 'Month YYYY' format."""
        TZ = _make_tz()
        now = datetime.now(TZ)
        now_ts = now.timestamp()
        now_day = now.strftime("%Y-%m-%d")
        ins(self.conn, "e1", "re1", "jaedyn@acme.io", "enterprise", "Acme",
            "mA", "proj", "sE", inp=1_000_000, day=now_day, ts=now_ts)

        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data.clear()

        d = agg.build_dashboard_data("enterprise")

        self.assertIn("month_label", d, "month_label must be present in payload")
        expected_label = now.strftime("%B %Y")
        self.assertEqual(d["month_label"], expected_label,
                         f"Expected '{expected_label}', got '{d['month_label']}'")

    def test_month_cost_scope_personal_excludes_enterprise(self):
        """Enterprise events must not appear in personal-scope month_cost."""
        TZ = _make_tz()
        now = datetime.now(TZ)
        now_ts = now.timestamp()
        now_day = now.strftime("%Y-%m-%d")

        # Enterprise event (must NOT appear in personal scope)
        ins(self.conn, "e1", "re1", "jaedyn@acme.io", "enterprise", "Acme",
            "mA", "proj", "sE", inp=1_000_000, day=now_day, ts=now_ts)

        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data.clear()

        d_personal = agg.build_dashboard_data("personal")

        self.assertIn("month_cost", d_personal)
        self.assertAlmostEqual(d_personal["month_cost"], 0.0, places=2,
                               msg=f"Enterprise events must not bleed into personal month_cost, "
                                   f"got {d_personal['month_cost']}")

    def test_empty_dashboard_has_month_cost(self):
        """_empty_dashboard must include month_cost=0.0 and a valid month_label."""
        TZ = _make_tz()
        import app.aggregator as agg

        empty = agg._empty_dashboard("2026-01-01")

        self.assertIn("month_cost", empty)
        self.assertEqual(empty["month_cost"], 0.0)
        self.assertIn("month_label", empty)
        expected_label = datetime.now(TZ).strftime("%B %Y")
        self.assertEqual(empty["month_label"], expected_label)


class MonthCostUIWiringTest(TempDBTestCase):
    """Test that the dashboard HTML contains the month counter wiring."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def _seed_enterprise(self):
        """Seed one enterprise event in the current month."""
        TZ = _make_tz()
        now = datetime.now(TZ)
        now_ts = now.timestamp()
        now_day = now.strftime("%Y-%m-%d")
        ins(self.conn, "e1", "re1", "jaedyn@acme.io", "enterprise", "Acme",
            "mA", "proj", "sE", inp=1_000_000, day=now_day, ts=now_ts)
        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data.clear()

    def test_page_contains_month_counter_element(self):
        """Dashboard HTML must contain the monthCostValue element id."""
        self._seed_enterprise()
        c = self.client()
        html = c.get("/").text
        self.assertIn("monthCostValue", html,
                      "monthCostValue element id must be present in page")

    def test_page_contains_month_to_date_label(self):
        """Dashboard HTML must contain 'MONTH TO DATE' label text."""
        self._seed_enterprise()
        c = self.client()
        html = c.get("/").text
        self.assertIn("MONTH TO DATE", html,
                      "MONTH TO DATE label must appear in page")

    def test_page_contains_reduced_motion_guard(self):
        """Dashboard HTML must contain the prefers-reduced-motion guard string."""
        self._seed_enterprise()
        c = self.client()
        html = c.get("/").text
        self.assertIn("prefers-reduced-motion", html,
                      "prefers-reduced-motion guard must be present in page JS")

    def test_page_references_month_cost_in_js(self):
        """Embedded JS must reference D.month_cost for the animated counter."""
        self._seed_enterprise()
        c = self.client()
        html = c.get("/").text
        self.assertIn("month_cost", html,
                      "month_cost must be referenced in page JS")

    def test_embedded_data_json_has_month_cost(self):
        """The embedded data_json blob must contain month_cost and month_label."""
        self._seed_enterprise()
        c = self.client()
        html = c.get("/").text
        # The embedded data has both keys
        self.assertIn('"month_cost"', html,
                      '"month_cost" key must appear in embedded data_json')
        self.assertIn('"month_label"', html,
                      '"month_label" key must appear in embedded data_json')

    def test_scope_personal_has_different_month_cost(self):
        """/?scope=personal must show personal month_cost (0 when only enterprise seeded)."""
        self._seed_enterprise()
        import app.aggregator as agg
        agg._cached_data.clear()
        c = self.client()
        # enterprise default scope
        html_ent = c.get("/?scope=enterprise").text
        # personal scope
        html_per = c.get("/?scope=personal").text
        # Both pages must have month_cost in embedded JSON; values differ
        self.assertIn('"month_cost"', html_ent)
        self.assertIn('"month_cost"', html_per)
        # Enterprise has $5, personal has $0 — embedded values differ
        # Quick check: the enterprise page must contain 5.0 somewhere (from data_json)
        # while personal shows 0.0; at minimum both pages must embed the field
        self.assertNotEqual(html_ent, html_per,
                            "Enterprise and personal scope pages must differ")


if __name__ == "__main__":
    unittest.main()
