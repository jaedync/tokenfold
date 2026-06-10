"""No-silent-fallback pricing policy + Fable 5 rates.

A model without confirmed pricing must contribute $0 everywhere (never a
made-up Sonnet-rate number) and be flagged `unpriced` so the UI shows an
em-dash instead of a fabricated dollar figure. Encountering an unknown
model triggers a forced (TTL-bypassing) LiteLLM refresh, rate-limited, so
brand-new models get real pricing within minutes of first use instead of
up to 24h later.
"""

import unittest
from unittest.mock import patch

import app.pricing as pricing
from app.tests._support import TempDBTestCase


class FablePricingTest(unittest.TestCase):
    """Fable 5 must have confirmed static pricing (docs.claude.com 2026-06):
    $10 in / $50 out / $12.50 5m-cache-write / $1 cache-read per MTok."""

    def test_fable_display_name(self):
        self.assertEqual(pricing.display_model("claude-fable-5"), "Fable 5")

    def test_fable_static_pricing(self):
        with patch.dict(pricing._dynamic_pricing, {}, clear=True):
            self.assertEqual(pricing.get_pricing("Fable 5"), (10.00, 50.00, 12.50, 1.00))

    def test_fable_sorts_first(self):
        self.assertEqual(pricing.MODEL_ORDER[0], "Fable 5")


class NoSilentFallbackTest(unittest.TestCase):
    """Unknown models: zero cost + unpriced flag, never Sonnet-rate guesses."""

    def setUp(self):
        # Make the refresh path a no-op and non-rate-limited for these tests.
        self._lp = patch.object(pricing, "load_pricing", lambda force=False: None)
        self._lp.start()
        self.addCleanup(self._lp.stop)
        pricing._unknown_refresh_ts = 0.0

    def test_fallback_constant_removed(self):
        self.assertFalse(hasattr(pricing, "FALLBACK_PRICING"),
                         "silent fallback pricing must not exist")

    def test_unknown_model_priced_zero(self):
        with patch.dict(pricing._dynamic_pricing, {}, clear=True):
            self.assertEqual(pricing.get_pricing("Frobnicator 9"), (0.0, 0.0, 0.0, 0.0))

    def test_unknown_model_costs_nothing(self):
        with patch.dict(pricing._dynamic_pricing, {}, clear=True):
            c = pricing.compute_cost("Frobnicator 9", 1_000_000, 1_000_000, 0, 0)
        self.assertEqual(c, 0.0)

    def test_is_priced(self):
        with patch.dict(pricing._dynamic_pricing, {}, clear=True):
            self.assertTrue(pricing.is_priced("Fable 5"))
            self.assertTrue(pricing.is_priced("Opus 4.8"))
            self.assertFalse(pricing.is_priced("Frobnicator 9"))


class UnknownTriggersRefreshTest(unittest.TestCase):
    """First unknown-model lookup forces a TTL-bypassing refresh; subsequent
    lookups inside the rate-limit window do not re-fetch."""

    def test_refresh_called_once_with_force(self):
        calls = []
        with patch.object(pricing, "load_pricing",
                          lambda force=False: calls.append(force)):
            pricing._unknown_refresh_ts = 0.0
            with patch.dict(pricing._dynamic_pricing, {}, clear=True):
                pricing.get_pricing("Frobnicator 9")
                pricing.get_pricing("Frobnicator 9")  # within rate-limit window
        self.assertEqual(calls, [True])

    def test_refresh_can_rescue_unknown(self):
        """If the forced refresh learns the model, the SAME call returns real rates."""
        def fake_load(force=False):
            pricing._dynamic_pricing["Frobnicator 9"] = (7.0, 21.0, 8.75, 0.7)
        with patch.object(pricing, "load_pricing", fake_load):
            pricing._unknown_refresh_ts = 0.0
            with patch.dict(pricing._dynamic_pricing, {}, clear=True):
                got = pricing.get_pricing("Frobnicator 9")
        self.assertEqual(got, (7.0, 21.0, 8.75, 0.7))


class AggregatorUnpricedFlagTest(TempDBTestCase):
    """model_breakdown entries carry unpriced=True (and cost 0) for unknown
    models so the dashboard can render an em-dash + badge."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        pricing._unknown_refresh_ts = 0.0

    def _ins(self, uuid, model, inp=1_000_000):
        self.conn.execute(
            "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,request_id,"
            "source_machine,project_dir,model,is_sidechain,agent_id,input_tokens,"
            "output_tokens,cache_creation_tokens,cache_read_tokens,is_human_prompt) "
            "VALUES(?,'assistant','2026-06-09T12:00:00Z',1781000000.0,'2026-06-09',"
            "'s1',?,'m1','proj',?,0,NULL,?,1000,0,0,0)",
            (uuid, "r-" + uuid, model, inp),
        )
        self.conn.commit()

    def test_unknown_model_flagged_and_zero_cost(self):
        from app.summarizer import summarize_days
        from app.aggregator import build_dashboard_data
        self._ins("u1", "claude-opus-4-8")
        self._ins("u2", "claude-wibble-7")
        summarize_days(["2026-06-09"])
        data = build_dashboard_data("personal")
        by_model = {m["model"]: m for m in data["model_breakdown"]}
        self.assertIn("Wibble 7", by_model)
        self.assertTrue(by_model["Wibble 7"]["unpriced"])
        self.assertEqual(by_model["Wibble 7"]["cost"], 0.0)
        self.assertFalse(by_model["Opus 4.8"]["unpriced"])
        self.assertGreater(by_model["Opus 4.8"]["cost"], 0)
        self.assertIn("Wibble 7", data["unpriced_models"])

    def test_priced_models_only_no_unpriced_list(self):
        from app.summarizer import summarize_days
        from app.aggregator import build_dashboard_data
        self._ins("u1", "claude-opus-4-8")
        summarize_days(["2026-06-09"])
        data = build_dashboard_data("personal")
        self.assertEqual(data["unpriced_models"], [])


class TemplateUnpricedUITest(unittest.TestCase):
    """Source-level: the model table must not drop zero-cost unpriced models
    (the 14d/all filter is cost-per-hour-based) and must badge them."""

    def test_template_handles_unpriced(self):
        from pathlib import Path
        tpl = (Path(__file__).resolve().parents[2] / "templates" / "dashboard.html").read_text()
        self.assertIn("m.unpriced", tpl)
        self.assertIn("unpriced-badge", tpl)


if __name__ == "__main__":
    unittest.main()
