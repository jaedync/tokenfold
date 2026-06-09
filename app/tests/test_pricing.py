import unittest

from app.pricing import compute_cost


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

    def test_fast_is_opus_only(self):
        self.assertAlmostEqual(
            compute_cost("Sonnet 4.6", 1_000_000, 0, 0, 0, speed="fast"), 3.0, places=4)


if __name__ == "__main__":
    unittest.main()
