"""Date-effective pricing eras + Sonnet 5 rates (Workstream A).

Sonnet 5 launched at INTRO rates ($2/$10/$2.50/$0.20 per MTok) and flips to
standard rates at 2026-09-01T00:00:00Z (billing cutover timezone assumed UTC).
Standard-period cache rates $3.75/$0.30 are ASSUMED 1.25x/0.1x of the $3 base
— Anthropic only published intro cache rates. Era selection keys on the EVENT
timestamp so the daily re-summarize sweeps can never silently reprice August
history at standard rates after the boundary passes.
"""

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import app.pricing as pricing
from app.tests._support import TempDBTestCase
from app.tests.test_summarizer_pricing import insert_assistant


def _pricing_clock(ts):
    """Freeze ONLY pricing's module-level `time` name. Patching
    app.pricing.time.time would mutate the shared stdlib module and leak a
    frozen clock into every thread in the test process (background fetchers
    spawned by other tests' TestClients included)."""
    return patch.object(pricing, "time", SimpleNamespace(time=lambda: ts))

# One second before / exactly at the era flip.
AUG_TS = datetime(2026, 8, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp()
SEP_TS = datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()
INTRO = (2.00, 10.00, 2.50, 0.20)
STANDARD = (3.00, 15.00, 3.75, 0.30)  # cache rates ASSUMED 1.25x/0.1x of base


def _tag_enterprise(conn):
    """compute_window_cost is fail-closed to verified-enterprise usage."""
    conn.execute(
        "UPDATE events SET plan='enterprise', org_name='Acme', "
        "account_email='test@acme.io'"
    )
    conn.commit()


class EraResolutionTest(unittest.TestCase):
    def setUp(self):
        p = patch.dict(pricing._dynamic_pricing, {}, clear=True)
        p.start()
        self.addCleanup(p.stop)

    def test_get_pricing_flips_at_boundary(self):
        self.assertEqual(pricing.get_pricing("Sonnet 5", AUG_TS), INTRO)
        self.assertEqual(pricing.get_pricing("Sonnet 5", SEP_TS), STANDARD)

    def test_input_flips(self):
        self.assertAlmostEqual(
            pricing.compute_cost("Sonnet 5", 1_000_000, 0, 0, 0, ts_epoch=AUG_TS),
            2.00, places=4)
        self.assertAlmostEqual(
            pricing.compute_cost("Sonnet 5", 1_000_000, 0, 0, 0, ts_epoch=SEP_TS),
            3.00, places=4)

    def test_output_flips(self):
        self.assertAlmostEqual(
            pricing.compute_cost("Sonnet 5", 0, 1_000_000, 0, 0, ts_epoch=AUG_TS),
            10.00, places=4)
        self.assertAlmostEqual(
            pricing.compute_cost("Sonnet 5", 0, 1_000_000, 0, 0, ts_epoch=SEP_TS),
            15.00, places=4)

    def test_cache_write_flips(self):
        self.assertAlmostEqual(
            pricing.compute_cost("Sonnet 5", 0, 0, 1_000_000, 0, ts_epoch=AUG_TS),
            2.50, places=4)
        # $3.75 is the ASSUMED 1.25x standard-period 5m rate
        self.assertAlmostEqual(
            pricing.compute_cost("Sonnet 5", 0, 0, 1_000_000, 0, ts_epoch=SEP_TS),
            3.75, places=4)

    def test_cache_read_flips(self):
        self.assertAlmostEqual(
            pricing.compute_cost("Sonnet 5", 0, 0, 0, 1_000_000, ts_epoch=AUG_TS),
            0.20, places=4)
        # $0.30 is the ASSUMED 0.1x standard-period cache-read rate
        self.assertAlmostEqual(
            pricing.compute_cost("Sonnet 5", 0, 0, 0, 1_000_000, ts_epoch=SEP_TS),
            0.30, places=4)

    def test_one_hour_cache_premium_uses_era_base(self):
        # 1h writes bill 2x the ERA's base input: $4 intro vs $6 standard.
        self.assertAlmostEqual(
            pricing.compute_cost("Sonnet 5", 0, 0, 1_000_000, 0,
                                 cw_1h=1_000_000, ts_epoch=AUG_TS),
            4.00, places=4)
        self.assertAlmostEqual(
            pricing.compute_cost("Sonnet 5", 0, 0, 1_000_000, 0,
                                 cw_1h=1_000_000, ts_epoch=SEP_TS),
            6.00, places=4)

    def test_geo_us_multiplies_era_rates(self):
        self.assertAlmostEqual(
            pricing.compute_cost("Sonnet 5", 1_000_000, 0, 0, 0,
                                 inference_geo="us", ts_epoch=AUG_TS),
            2.20, places=4)
        self.assertAlmostEqual(
            pricing.compute_cost("Sonnet 5", 1_000_000, 0, 0, 0,
                                 inference_geo="us", ts_epoch=SEP_TS),
            3.30, places=4)

    def test_constant_tuple_model_identical_with_and_without_ts(self):
        for ts in (None, AUG_TS, SEP_TS):
            self.assertEqual(pricing.get_pricing("Opus 4.8", ts),
                             (5.00, 25.00, 6.25, 0.50))
        self.assertAlmostEqual(
            pricing.compute_cost("Opus 4.8", 1_000_000, 0, 0, 0),
            pricing.compute_cost("Opus 4.8", 1_000_000, 0, 0, 0, ts_epoch=AUG_TS),
            places=6)

    def test_default_ts_is_wall_clock(self):
        with _pricing_clock(AUG_TS):
            self.assertEqual(pricing.get_pricing("Sonnet 5"), INTRO)
        with _pricing_clock(SEP_TS):
            self.assertEqual(pricing.get_pricing("Sonnet 5"), STANDARD)

    def test_era_boundaries_helper(self):
        bounds = pricing.era_boundaries()
        self.assertEqual(bounds, [SEP_TS])
        self.assertTrue(all(b > 0 for b in bounds))


class StaticEraPrecedenceTest(unittest.TestCase):
    """Era-LISTED static entries win over LiteLLM; plain tuples stay
    dynamic-first so LiteLLM can still correct drift for the other models."""

    def test_poisoned_dynamic_sonnet5_still_intro_for_august(self):
        # The live pricing_cache is poisoned with standard-rate Sonnet 5 —
        # no cache invalidation needed, static era list must win.
        with patch.dict(pricing._dynamic_pricing, {"Sonnet 5": STANDARD}, clear=True):
            self.assertEqual(pricing.get_pricing("Sonnet 5", AUG_TS), INTRO)
            self.assertEqual(pricing.get_pricing("Sonnet 5", SEP_TS), STANDARD)

    def test_dynamic_still_overrides_plain_tuple_static(self):
        with patch.dict(pricing._dynamic_pricing,
                        {"Opus 4.8": (9.0, 9.0, 9.0, 9.0)}, clear=True):
            self.assertEqual(pricing.get_pricing("Opus 4.8"), (9.0, 9.0, 9.0, 9.0))
            self.assertEqual(pricing.get_pricing("Opus 4.8", AUG_TS),
                             (9.0, 9.0, 9.0, 9.0))


class Sonnet5NamingAndOrderTest(unittest.TestCase):
    def test_display_names(self):
        self.assertIn("claude-sonnet-5", pricing.MODEL_DISPLAY)
        self.assertEqual(pricing.display_model("claude-sonnet-5"), "Sonnet 5")
        self.assertEqual(pricing.display_model("claude-sonnet-5-20260930"), "Sonnet 5")

    def test_model_order_after_opus_before_sonnet46(self):
        order = pricing.MODEL_ORDER
        i5 = order.index("Sonnet 5")
        for name in order:
            if name.startswith("Opus") or name == "Fable 5":
                self.assertLess(order.index(name), i5)
        self.assertLess(i5, order.index("Sonnet 4.6"))

    def test_is_priced_with_dynamic_cleared(self):
        with patch.dict(pricing._dynamic_pricing, {}, clear=True):
            self.assertTrue(pricing.is_priced("Sonnet 5"))

    def test_water_energy_entry(self):
        from app.water import MODEL_ENERGY_WH_PER_MTOK
        self.assertIn("Sonnet 5", MODEL_ENERGY_WH_PER_MTOK)
        self.assertEqual(MODEL_ENERGY_WH_PER_MTOK["Sonnet 5"],
                         MODEL_ENERGY_WH_PER_MTOK["Sonnet 4.6"])


class UnknownFallbackWithTsTest(unittest.TestCase):
    """The no-silent-fallback contract is unchanged when ts_epoch is passed:
    $0 + unpriced flag + one rate-limited forced LiteLLM refresh, and the
    model-independent web-search fee still bills."""

    def setUp(self):
        self._lp = patch.object(pricing, "load_pricing", lambda force=False: None)
        self._lp.start()
        self.addCleanup(self._lp.stop)
        pricing._unknown_refresh_ts = 0.0

    def test_unknown_returns_unpriced_with_ts(self):
        with patch.dict(pricing._dynamic_pricing, {}, clear=True):
            self.assertEqual(pricing.get_pricing("Frobnicator 9", AUG_TS),
                             (0.0, 0.0, 0.0, 0.0))
            self.assertFalse(pricing.is_priced("Frobnicator 9"))

    def test_forced_refresh_once_with_ts(self):
        calls = []
        with patch.object(pricing, "load_pricing",
                          lambda force=False: calls.append(force)):
            pricing._unknown_refresh_ts = 0.0
            with patch.dict(pricing._dynamic_pricing, {}, clear=True):
                pricing.get_pricing("Frobnicator 9", AUG_TS)
                pricing.get_pricing("Frobnicator 9", SEP_TS)  # rate-limited
        self.assertEqual(calls, [True])

    def test_refresh_can_rescue_unknown_with_ts(self):
        def fake_load(force=False):
            pricing._dynamic_pricing["Frobnicator 9"] = (7.0, 21.0, 8.75, 0.7)
        with patch.object(pricing, "load_pricing", fake_load):
            pricing._unknown_refresh_ts = 0.0
            with patch.dict(pricing._dynamic_pricing, {}, clear=True):
                got = pricing.get_pricing("Frobnicator 9", AUG_TS)
        self.assertEqual(got, (7.0, 21.0, 8.75, 0.7))

    def test_web_search_fee_still_charged_for_unpriced(self):
        with patch.dict(pricing._dynamic_pricing, {}, clear=True):
            c = pricing.compute_cost("Frobnicator 9", 1_000_000, 0, 0, 0,
                                     web_search=1000, ts_epoch=AUG_TS)
        self.assertAlmostEqual(c, 10.0, places=4)


class WindowCostEraStraddleTest(TempDBTestCase):
    """A window straddling the era flip must price each side at its own era:
    exactly 2.00*M_aug + 3.00*M_sep input dollars, never one blended rate."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_straddling_window_sums_both_eras(self):
        from app.cost_windows import compute_window_cost
        aug = datetime(2026, 8, 31, 23, 0, tzinfo=timezone.utc).timestamp()
        sep = datetime(2026, 9, 1, 1, 0, tzinfo=timezone.utc).timestamp()
        insert_assistant(self.conn, "u1", "r1", model="claude-sonnet-5",
                         day="2026-08-31", ts=aug, inp=1_000_000)
        insert_assistant(self.conn, "u2", "r2", model="claude-sonnet-5",
                         day="2026-08-31", ts=sep, inp=1_000_000)
        _tag_enterprise(self.conn)
        got = compute_window_cost(self.conn, aug - 10, sep + 10)
        # $2 intro + $3 standard — a single-era group would give $4 or $6.
        self.assertAlmostEqual(got, 5.00, places=4)


class WindowCostNoErasTest(TempDBTestCase):
    """The bounds-empty query branch (no era-listed models anywhere) must stay
    correct: it becomes the hot path again if Sonnet 5's era list is ever
    collapsed to a plain tuple after the flip."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_window_cost_with_no_era_models(self):
        from app.cost_windows import compute_window_cost
        flat = {k: v for k, v in pricing.MODEL_PRICING.items()
                if isinstance(v, tuple)}
        with patch.dict(pricing.MODEL_PRICING, flat, clear=True):
            self.assertEqual(pricing.era_boundaries(), [])
            ts = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc).timestamp()
            insert_assistant(self.conn, "u1", "r1", model="claude-opus-4-8",
                             day="2026-08-15", ts=ts, inp=1_000_000)
            _tag_enterprise(self.conn)
            got = compute_window_cost(self.conn, ts - 10, ts + 10)
            self.assertAlmostEqual(got, 5.00, places=4)


class SummarizerEraTest(TempDBTestCase):
    """summarize_days keys pricing on EVENT time: an August day re-summarized
    after the boundary (wall clock in September) still stores intro cost —
    this is what stops the daily full sweep from corrupting August history."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_august_day_resummarized_in_september_keeps_intro(self):
        from app.summarizer import summarize_days
        aug_ts = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc).timestamp()
        insert_assistant(self.conn, "u1", "r1", model="claude-sonnet-5",
                         day="2026-08-15", ts=aug_ts, inp=1_000_000)
        with _pricing_clock(SEP_TS + 86400):
            summarize_days(["2026-08-15"])
        row = self.conn.execute(
            "SELECT cost FROM daily_summary WHERE day='2026-08-15'").fetchone()
        self.assertAlmostEqual(row["cost"], 2.00, places=4)


class AggregatorSonnet5RowTest(TempDBTestCase):
    """A8: the model table renders a Sonnet 5 row without KeyError, and the
    dollar-parts decomposition era-splits (per-day) so displayed historical
    costs never move when the wall clock crosses the boundary."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_sonnet5_row_renders_and_parts_era_split(self):
        from app.aggregator import build_dashboard_data
        from app.summarizer import summarize_days
        aug_ts = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc).timestamp()
        sep_ts = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc).timestamp()
        insert_assistant(self.conn, "u1", "r1", model="claude-sonnet-5",
                         day="2026-08-15", ts=aug_ts, inp=1_000_000)
        insert_assistant(self.conn, "u2", "r2", model="claude-sonnet-5",
                         day="2026-09-02", ts=sep_ts, inp=1_000_000)
        summarize_days(["2026-08-15", "2026-09-02"])
        data = build_dashboard_data("personal")
        rows = [m for m in data["model_breakdown"] if m["model"] == "Sonnet 5"]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertFalse(row["unpriced"])
        self.assertAlmostEqual(row["cost"], 5.00, places=2)
        # cost_input decomposition must blend eras ($2 + $3), not price the
        # whole 2M-token aggregate at either era's single rate ($4 / $6).
        self.assertAlmostEqual(row["cost_input"], 5.00, places=2)
        self.assertIn("Sonnet 5", data["model_pricing"])
        self.assertIn("Sonnet 5", data["output_pricing"])


if __name__ == "__main__":
    unittest.main()
