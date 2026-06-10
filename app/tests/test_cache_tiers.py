"""1h vs 5m cache-write pricing.

Anthropic bills 5-minute cache writes at 1.25x base input and 1-HOUR cache
writes at 2x base input. Claude Code logs the split in the transcript as a
NESTED object: usage.cache_creation.{ephemeral_5m_input_tokens,
ephemeral_1h_input_tokens}. The extractor previously read flat keys that
don't exist, so every write was billed at the 5m rate (1.6x undercount on
1h-heavy sessions — interactive Claude Code uses 1h caching extensively).
"""

import unittest
from unittest.mock import patch

import app.pricing as pricing
from app.ingest import _extract_event, EVENT_COLS
from app.tests._support import TempDBTestCase


def _arec(**usage):
    u = {"input_tokens": 10, "output_tokens": 20}
    u.update(usage)
    return {
        "uuid": "u1", "type": "assistant", "timestamp": "2026-06-09T12:00:00Z",
        "sessionId": "s1", "requestId": "r1",
        "message": {"model": "claude-opus-4-8", "id": "m1", "usage": u},
    }


class ExtractCacheTiersTest(unittest.TestCase):
    """The nested cache_creation object is the real transcript shape."""

    def test_nested_cache_creation_captured(self):
        row = _extract_event(_arec(
            cache_creation_input_tokens=9567,
            cache_creation={"ephemeral_5m_input_tokens": 1200,
                            "ephemeral_1h_input_tokens": 8367},
        ), "mach", "proj")
        self.assertEqual(row["cache_ephemeral_5m"], 1200)
        self.assertEqual(row["cache_ephemeral_1h"], 8367)

    def test_absent_split_defaults_zero(self):
        row = _extract_event(_arec(cache_creation_input_tokens=500), "mach", "proj")
        self.assertEqual(row["cache_ephemeral_5m"], 0)
        self.assertEqual(row["cache_ephemeral_1h"], 0)

    def test_malformed_nested_object_safe(self):
        row = _extract_event(_arec(cache_creation="garbage"), "mach", "proj")
        self.assertEqual(row["cache_ephemeral_5m"], 0)
        self.assertEqual(row["cache_ephemeral_1h"], 0)

    def test_event_cols_wired(self):
        self.assertIn("cache_ephemeral_5m", EVENT_COLS)
        self.assertIn("cache_ephemeral_1h", EVENT_COLS)


class ComputeCostCacheTiersTest(unittest.TestCase):
    """Opus 4.8: base $5 -> 5m write $6.25, 1h write $10 per MTok."""

    def setUp(self):
        p = patch.dict(pricing._dynamic_pricing, {}, clear=True)
        p.start()
        self.addCleanup(p.stop)

    def test_legacy_unsplit_bills_5m(self):
        c = pricing.compute_cost("Opus 4.8", 0, 0, 1_000_000, 0)
        self.assertAlmostEqual(c, 6.25)

    def test_all_1h_bills_2x_base(self):
        c = pricing.compute_cost("Opus 4.8", 0, 0, 1_000_000, 0,
                                 cw_5m=0, cw_1h=1_000_000)
        self.assertAlmostEqual(c, 10.00)

    def test_mixed_split(self):
        c = pricing.compute_cost("Opus 4.8", 0, 0, 1_000_000, 0,
                                 cw_5m=500_000, cw_1h=500_000)
        self.assertAlmostEqual(c, 6.25 / 2 + 10.00 / 2)

    def test_partial_split_remainder_bills_5m(self):
        """Split covering only part of cw: remainder stays at the 5m rate."""
        c = pricing.compute_cost("Opus 4.8", 0, 0, 1_000_000, 0,
                                 cw_5m=0, cw_1h=400_000)
        self.assertAlmostEqual(c, 6.25 * 0.6 + 10.00 * 0.4)

    def test_corrupt_split_exceeding_total_ignored(self):
        """Split larger than the total is untrusted input — fall back to 5m."""
        c = pricing.compute_cost("Opus 4.8", 0, 0, 100, 0,
                                 cw_5m=0, cw_1h=1_000_000)
        self.assertAlmostEqual(c, 100 * 6.25 / 1e6)

    def test_fast_mode_stacks_with_1h(self):
        """Fast Opus 4.8 base $10 -> 1h write $20 per MTok."""
        c = pricing.compute_cost("Opus 4.8", 0, 0, 1_000_000, 0,
                                 speed="fast", cw_1h=1_000_000)
        self.assertAlmostEqual(c, 20.00)

    def test_geo_us_stacks_with_1h(self):
        c = pricing.compute_cost("Opus 4.8", 0, 0, 1_000_000, 0,
                                 inference_geo="us", cw_1h=1_000_000)
        self.assertAlmostEqual(c, 11.00)


class SummarizerCacheTierTest(TempDBTestCase):
    """End-to-end: a stored event with a 1h split rolls up at the 2x rate."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def _ins(self, uuid, cw, c5m, c1h):
        self.conn.execute(
            "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,request_id,"
            "source_machine,project_dir,model,is_sidechain,agent_id,input_tokens,"
            "output_tokens,cache_creation_tokens,cache_read_tokens,cache_ephemeral_5m,"
            "cache_ephemeral_1h,is_human_prompt) "
            "VALUES(?,'assistant','2026-06-09T12:00:00Z',1781000000.0,'2026-06-09',"
            "'s1',?,'m1','proj','claude-opus-4-8',0,NULL,0,0,?,0,?,?,0)",
            (uuid, "r-" + uuid, cw, c5m, c1h),
        )
        self.conn.commit()

    def test_1h_split_rolls_up_at_2x(self):
        from app.summarizer import summarize_days
        self._ins("u1", 1_000_000, 0, 1_000_000)
        summarize_days(["2026-06-09"])
        row = self.conn.execute(
            "SELECT cost FROM daily_summary WHERE day='2026-06-09'").fetchone()
        self.assertAlmostEqual(row["cost"], 10.00, places=2)


if __name__ == "__main__":
    unittest.main()


class CacheTierDisplayTest(TempDBTestCase):
    """The dashboard differentiates 5m vs 1h cache writes everywhere it shows
    pricing or cache-write usage: model_pricing carries cache_write_1h (2x
    base), per-model breakdowns carry tier token counts, and the per-model
    cache-write cost component prices each tier at its real rate."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def _ins(self, uuid, cw, c5m, c1h):
        self.conn.execute(
            "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,request_id,"
            "source_machine,project_dir,model,is_sidechain,agent_id,input_tokens,"
            "output_tokens,cache_creation_tokens,cache_read_tokens,cache_ephemeral_5m,"
            "cache_ephemeral_1h,is_human_prompt) "
            "VALUES(?,'assistant','2026-06-09T12:00:00Z',1781000000.0,'2026-06-09',"
            "'s1',?,'m1','proj','claude-opus-4-8',0,NULL,0,0,?,0,?,?,0)",
            (uuid, "r-" + uuid, cw, c5m, c1h),
        )
        self.conn.commit()

    def _build(self):
        from app.summarizer import summarize_days
        from app.aggregator import build_dashboard_data
        summarize_days(["2026-06-09"])
        return build_dashboard_data("personal")

    def test_model_pricing_includes_1h_rate(self):
        self._ins("u1", 1_000_000, 0, 1_000_000)
        data = self._build()
        p = data["model_pricing"]["Opus 4.8"]
        self.assertEqual(p["cache_write"], 6.25)
        self.assertEqual(p["cache_write_1h"], 10.00)

    def test_model_breakdown_carries_tier_tokens(self):
        self._ins("u1", 1_000_000, 250_000, 750_000)
        data = self._build()
        m = [x for x in data["model_breakdown"] if x["model"] == "Opus 4.8"][0]
        self.assertEqual(m["cache_5m"], 250_000)
        self.assertEqual(m["cache_1h"], 750_000)

    def test_cost_cache_write_component_tiered(self):
        # 1 MTok all-1h: component must be $10, not $6.25
        self._ins("u1", 1_000_000, 0, 1_000_000)
        data = self._build()
        m = [x for x in data["model_breakdown"] if x["model"] == "Opus 4.8"][0]
        self.assertAlmostEqual(m["cost_cache_write"], 10.00, places=2)

    def test_template_shows_both_tiers(self):
        from pathlib import Path
        tpl = (Path(__file__).resolve().parents[2] / "templates" / "dashboard.html").read_text()
        self.assertIn("cache_write_1h", tpl)
        self.assertNotIn("with 5-min prompt caching", tpl)
