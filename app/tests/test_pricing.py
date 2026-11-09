import unittest
from unittest.mock import patch

from app.pricing import compute_cost

M = 1_000_000


class PricingModifiersTest(unittest.TestCase):
    def test_fast_opus_doubles_base(self):
        self.assertAlmostEqual(compute_cost("Opus 4.8", 1_000_000, 0, 0, 0), 5.0, places=4)
        self.assertAlmostEqual(
            compute_cost("Opus 4.8", 1_000_000, 0, 0, 0, speed="fast"), 10.0, places=4)

    def test_fast_recomputes_cache_from_base(self):
        self.assertAlmostEqual(
            compute_cost("Opus 4.8", 0, 0, 1_000_000, 0, speed="fast"), 12.5, places=4)
        self.assertAlmostEqual(
            compute_cost("Opus 4.8", 0, 0, 0, 1_000_000, speed="fast"), 1.0, places=4)

    def test_us_geo_scales_all(self):
        self.assertAlmostEqual(
            compute_cost("Sonnet 4.6", 1_000_000, 0, 0, 0, inference_geo="us"), 3.3, places=4)

    def test_fast_and_geo_compose(self):
        self.assertAlmostEqual(
            compute_cost("Opus 4.8", 1_000_000, 0, 0, 0, speed="fast", inference_geo="us"),
            11.0, places=4)

    def test_defaults_unchanged(self):
        self.assertAlmostEqual(compute_cost("Haiku 4.5", 1_000_000, 0, 0, 0), 1.0, places=4)

    def test_fast_is_listed_models_only(self):
        self.assertAlmostEqual(
            compute_cost("Sonnet 4.6", 1_000_000, 0, 0, 0, speed="fast"), 3.0, places=4)


class Fable52PricingTest(unittest.TestCase):
    """Fable 5.2 (launched 2026-11-09): $8 in / $40 out / $10 5m write /
    $0.20 cache read; fast mode $16 / $80 with the same cache multipliers
    (platform.claude.com/docs/en/about-claude/pricing, read 2026-11-09).
    LiteLLM had no claude-fable-5-2 entry at launch, so the static rate is
    what keeps these rows from pricing at $0."""

    def setUp(self):
        import app.tests._support  # noqa: F401  (offline pricing)
        import app.pricing as pricing
        self._dyn = patch.dict(pricing._dynamic_pricing, {}, clear=True)
        self._dyn.start()
        self.addCleanup(self._dyn.stop)

    def test_display_name(self):
        from app.pricing import display_model
        self.assertEqual(display_model("claude-fable-5-2"), "Fable 5.2")

    def test_standard_rates(self):
        from app.pricing import get_pricing
        self.assertEqual(get_pricing("Fable 5.2"), (8.00, 40.00, 10.00, 0.20))
        self.assertAlmostEqual(compute_cost("Fable 5.2", M, M, M, M), 58.20, places=4)

    def test_fast_doubles_base(self):
        self.assertAlmostEqual(compute_cost("Fable 5.2", M, 0, 0, 0, speed="fast"), 16.0, places=4)
        self.assertAlmostEqual(compute_cost("Fable 5.2", 0, M, 0, 0, speed="fast"), 80.0, places=4)

    def test_fast_keeps_the_model_cache_multipliers(self):
        # Fable cache reads are 0.025x input, not the 0.1x default.
        self.assertAlmostEqual(compute_cost("Fable 5.2", 0, 0, 0, M, speed="fast"), 0.40, places=4)
        self.assertAlmostEqual(compute_cost("Fable 5.2", 0, 0, M, 0, speed="fast"), 20.0, places=4)
        # 1h writes are 2x the fast input rate.
        self.assertAlmostEqual(
            compute_cost("Fable 5.2", 0, 0, M, 0, speed="fast", cw_1h=M), 32.0, places=4)


class Opus5FastTest(unittest.TestCase):
    """Opus 5.5 and Opus 5 base rates come from LiteLLM; fast mode is
    $8/$40 and $10/$50 with each model's own cache multipliers."""

    def test_opus_55_fast(self):
        import app.pricing as pricing
        with patch.dict(pricing._dynamic_pricing,
                        {"Opus 5.5": (4.0, 20.0, 5.0, 0.20)}, clear=True):
            self.assertAlmostEqual(compute_cost("Opus 5.5", M, M, 0, 0, speed="fast"), 48.0, places=4)
            # Opus 5.5 cache reads are 0.05x input.
            self.assertAlmostEqual(compute_cost("Opus 5.5", 0, 0, 0, M, speed="fast"), 0.40, places=4)
            self.assertAlmostEqual(compute_cost("Opus 5.5", M, 0, 0, 0), 4.0, places=4)

    def test_opus_5_fast(self):
        import app.pricing as pricing
        with patch.dict(pricing._dynamic_pricing,
                        {"Opus 5": (5.0, 25.0, 6.25, 0.50)}, clear=True):
            self.assertAlmostEqual(compute_cost("Opus 5", M, M, 0, M, speed="fast"), 61.0, places=4)

    def test_unpriced_model_in_fast_mode_stays_unpriced(self):
        import app.pricing as pricing
        with patch.dict(pricing._dynamic_pricing, {}, clear=True), \
                patch.object(pricing, "load_pricing", lambda force=False: None), \
                patch.dict(pricing.FAST_BASE, {"Nope 1": (1.0, 2.0)}):
            self.assertEqual(compute_cost("Nope 1", M, M, M, M, speed="fast"), 0.0)


if __name__ == "__main__":
    unittest.main()


class PricingFetchSwitchTest(unittest.TestCase):
    """The suite runs offline: TOKENFOLD_PRICING_FETCH=0 (set in _support)
    keeps load_pricing off the network when the cache is empty."""

    def test_no_fetch_when_the_switch_is_off(self):
        from unittest.mock import patch
        import app.pricing as pricing
        import app.tests._support  # noqa: F401  (sets the test environment)
        with patch.object(pricing, "_cached_row", return_value=None), \
                patch("urllib.request.urlopen") as urlopen:
            pricing.load_pricing()
        urlopen.assert_not_called()
