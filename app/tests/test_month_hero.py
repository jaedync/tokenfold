"""Tests for the month-to-date HERO figure selection.

Regression origin (2026-07-24): the dashboard hero read D.month_cost (the
event-derived ESTIMATE) while the monthly-budget gauge right below it read
Anthropic's billing meter. On the live enterprise instance the hero showed
$926.85 while the meter said $999.48 against a $1,000 limit. The headline
number under-reported real billing by the unaccounted gap ($72.63) and the
month limit was hit with the hero still showing ~$73 of apparent headroom.

The hero must show the AUTHORITATIVE billed figure whenever a fresh meter
supplies one, and fall back to the estimate only when it does not.
"""
import unittest
from datetime import datetime, timezone

from app.month_hero import month_hero_block
from app.tests._support import TempDBTestCase


def meter(used=999.48, limit=1000.0, fresh=True, utilization=99.948):
    return {
        "used_usd": used,
        "limit_usd": limit,
        "utilization": utilization,
        "fresh": fresh,
        "fetched_epoch": 1784915060.0,
        "machine": "VERTECH-HPC1T14",
    }


class MonthHeroSourceTest(unittest.TestCase):
    """Which number the hero shows, and where it comes from."""

    def test_no_meter_falls_back_to_estimate(self):
        b = month_hero_block(926.85, None)
        self.assertEqual(b["source"], "estimate")
        self.assertAlmostEqual(b["value"], 926.85)
        self.assertIsNone(b["unaccounted_usd"])
        self.assertIsNone(b["limit_usd"])

    def test_stale_meter_falls_back_to_estimate(self):
        """A stale meter is not authoritative: the estimate is fresher."""
        b = month_hero_block(926.85, meter(fresh=False))
        self.assertEqual(b["source"], "estimate")
        self.assertAlmostEqual(b["value"], 926.85)

    def test_fresh_meter_wins_over_estimate(self):
        b = month_hero_block(926.85, meter())
        self.assertEqual(b["source"], "meter")
        self.assertAlmostEqual(b["value"], 999.48)

    def test_live_regression_case(self):
        """The exact numbers that caused the overrun."""
        b = month_hero_block(926.85, meter(used=999.48, limit=1000.0))
        self.assertAlmostEqual(b["value"], 999.48)
        self.assertAlmostEqual(b["measured_usd"], 926.85)
        self.assertAlmostEqual(b["unaccounted_usd"], 72.63)
        self.assertAlmostEqual(b["limit_usd"], 1000.0)
        self.assertAlmostEqual(b["remaining_usd"], 0.52)
        self.assertAlmostEqual(b["utilization"], 99.95, places=2)

    def test_measured_always_reported(self):
        """The estimate stays available so the UI can show the gap."""
        b = month_hero_block(926.85, meter())
        self.assertAlmostEqual(b["measured_usd"], 926.85)
        b2 = month_hero_block(926.85, None)
        self.assertAlmostEqual(b2["measured_usd"], 926.85)


class MonthHeroValidationTest(unittest.TestCase):
    """Bad payloads must degrade to the estimate, never crash or show junk."""

    def test_meter_without_used_usd_falls_back(self):
        b = month_hero_block(926.85, meter(used=None))
        self.assertEqual(b["source"], "estimate")
        self.assertAlmostEqual(b["value"], 926.85)

    def test_meter_with_nan_used_falls_back(self):
        b = month_hero_block(926.85, meter(used=float("nan")))
        self.assertEqual(b["source"], "estimate")

    def test_meter_with_infinite_used_falls_back(self):
        b = month_hero_block(926.85, meter(used=float("inf")))
        self.assertEqual(b["source"], "estimate")

    def test_bad_month_cost_becomes_zero(self):
        b = month_hero_block(None, None)
        self.assertEqual(b["source"], "estimate")
        self.assertAlmostEqual(b["value"], 0.0)
        self.assertAlmostEqual(b["measured_usd"], 0.0)

    def test_non_dict_meter_falls_back(self):
        b = month_hero_block(926.85, "nonsense")
        self.assertEqual(b["source"], "estimate")


class MonthHeroLimitTest(unittest.TestCase):
    """Headroom math: the number that would have prevented the overrun."""

    def test_missing_limit_yields_no_headroom(self):
        b = month_hero_block(926.85, meter(limit=None, utilization=None))
        self.assertEqual(b["source"], "meter")
        self.assertIsNone(b["limit_usd"])
        self.assertIsNone(b["remaining_usd"])
        self.assertIsNone(b["utilization"])

    def test_zero_limit_yields_no_headroom(self):
        """Divide-by-zero guard: a 0 limit is not a usable denominator."""
        b = month_hero_block(926.85, meter(limit=0))
        self.assertIsNone(b["remaining_usd"])
        self.assertIsNone(b["utilization"])

    def test_remaining_floors_at_zero_when_over_limit(self):
        b = month_hero_block(900.0, meter(used=1200.0, limit=1000.0))
        self.assertAlmostEqual(b["remaining_usd"], 0.0)
        self.assertAlmostEqual(b["utilization"], 120.0)

    def test_utilization_derived_from_shown_value(self):
        """Utilization must agree with the number the hero displays, not with
        whatever Anthropic's coarse utilization field happened to say."""
        b = month_hero_block(100.0, meter(used=500.0, limit=1000.0,
                                          utilization=1.0))
        self.assertAlmostEqual(b["utilization"], 50.0)


class MonthHeroGapTest(unittest.TestCase):
    """The unaccounted gap is signed; tokenfold can over- or under-measure."""

    def test_positive_gap_when_billed_exceeds_measured(self):
        b = month_hero_block(926.85, meter(used=999.48))
        self.assertAlmostEqual(b["unaccounted_usd"], 72.63)

    def test_negative_gap_preserved_when_measured_exceeds_billed(self):
        b = month_hero_block(1000.0, meter(used=900.0))
        self.assertAlmostEqual(b["unaccounted_usd"], -100.0)

    def test_no_gap_reported_without_a_meter(self):
        self.assertIsNone(month_hero_block(926.85, None)["unaccounted_usd"])


class MonthHeroPayloadTest(TempDBTestCase):
    """The aggregator must ship the block, and the page must render from it."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def _seed(self, meter_used_cents=99948, limit_cents=100000,
              meter_age_s=3600.0):
        """One enterprise event this UTC month + a meter reading."""
        now = datetime.now(timezone.utc)
        ts = now.timestamp()
        day = now.strftime("%Y-%m-%d")
        self.conn.execute(
            "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,"
            "request_id,source_machine,project_dir,model,is_sidechain,agent_id,"
            "input_tokens,output_tokens,cache_creation_tokens,cache_read_tokens,"
            "account_email,plan,org_name,is_human_prompt) "
            "VALUES('e1','assistant',?,?,?,'s1','r1','m1','proj',"
            "'claude-opus-4-8',0,NULL,1000000,0,0,0,"
            "'jchilton@vertech.com','enterprise','Vertech',0)",
            (day + "T12:00:00Z", ts, day))
        self.conn.commit()
        if meter_used_cents is not None:
            from app.extra_usage import record_meter_reading
            record_meter_reading(
                self.conn, "vm-a",
                {"is_enabled": True, "monthly_limit": limit_cents,
                 "used_credits": meter_used_cents, "utilization": None},
                ts - meter_age_s)
        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data.clear()

    def test_payload_hero_prefers_fresh_meter(self):
        self._seed()
        import app.aggregator as agg
        d = agg.build_dashboard_data("enterprise")
        self.assertEqual(d["month_hero"]["source"], "meter")
        self.assertAlmostEqual(d["month_hero"]["value"], 999.48)
        # The estimate is still carried, just no longer the headline.
        self.assertAlmostEqual(d["month_hero"]["measured_usd"],
                               d["month_cost"], places=2)

    def test_payload_hero_exposes_headroom(self):
        self._seed()
        import app.aggregator as agg
        d = agg.build_dashboard_data("enterprise")
        self.assertAlmostEqual(d["month_hero"]["limit_usd"], 1000.0)
        self.assertAlmostEqual(d["month_hero"]["remaining_usd"], 0.52)

    def test_payload_hero_falls_back_without_meter(self):
        self._seed(meter_used_cents=None)
        import app.aggregator as agg
        d = agg.build_dashboard_data("enterprise")
        self.assertEqual(d["month_hero"]["source"], "estimate")
        self.assertAlmostEqual(d["month_hero"]["value"], d["month_cost"],
                               places=2)

    def test_payload_hero_falls_back_on_stale_meter(self):
        """Older than METER_STALE_S (48h) is no longer authoritative."""
        self._seed(meter_age_s=60 * 60 * 72)
        import app.aggregator as agg
        d = agg.build_dashboard_data("enterprise")
        self.assertEqual(d["month_hero"]["source"], "estimate")

    def test_personal_scope_never_gets_a_meter_hero(self):
        """Personal scope has no billing meter, so the estimate is authoritative
        there by definition, and no billed figure may be implied."""
        self._seed()
        import app.aggregator as agg
        agg._cached_data.clear()
        d = agg.build_dashboard_data("personal")
        self.assertEqual(d["month_hero"]["source"], "estimate")
        self.assertIsNone(d["month_hero"]["limit_usd"])

    def test_page_renders_hero_from_month_hero(self):
        """The hero JS must read D.month_hero, not D.month_cost."""
        self._seed()
        html = self.client().get("/").text
        self.assertIn("D.month_hero", html)
        self.assertIn('"month_hero"', html)

    def test_page_has_headroom_element(self):
        self._seed()
        html = self.client().get("/").text
        self.assertIn("monthCostHeadroom", html)

    def test_page_labels_the_estimate_as_estimated(self):
        """An unbacked figure must say so. That is the whole point of the fix."""
        self._seed()
        html = self.client().get("/").text
        self.assertIn("estimated · raw API pricing", html)


if __name__ == "__main__":
    unittest.main()
