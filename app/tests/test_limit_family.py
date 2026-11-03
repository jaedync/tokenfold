"""Family weights with a base rate per 7-day window (app/limit_family.py),
on a synthetic world with a known answer."""

import random
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from app import limit_family as lfam
from app import limit_fit as lf
from app import limit_rate_analysis as lra

PT = ZoneInfo("America/Los_Angeles")
HOUR = 3600.0
DAY = 86400.0
T0 = datetime(2026, 9, 28, tzinfo=PT).timestamp()  # a Monday
POLL = 120.0
# Per million tokens: input, output, cache write (1 h), cache read.
OPUS = (4.0, 20.0, 8.0, 0.2)
SONNET = (2.0, 10.0, 4.0, 0.1)
# Limit weight per dollar of each token type (cache reads count at 0.4).
TYPE_W = (1.0, 1.0, 0.6, 0.4)
SONNET_WEIGHT = 1.3          # per effective dollar, relative to Opus
DRIFT = (1.0, 1.0, 1.12, 1.25)  # points per dollar grow in weeks 3 and 4


def _is_peak(t):
    d = datetime.fromtimestamp(t, PT)
    return d.weekday() < 5 and 5 <= d.hour < 11


class _DriftWorld:
    """Opus and Sonnet requests; Sonnet use grows in the weeks where the
    base rate also grows, and Sonnet reads less cache. A fit with one base
    rate and raw dollars calls Sonnet heavier than it is."""

    def __init__(self, seed=5):
        rng = random.Random(seed)
        self.requests = []
        self.readings = {"seven_day": [], "five_hour": []}
        acc7 = acc5 = 0.0
        w7_end = T0 + 7 * DAY
        w5_end = None
        t, end, block = T0, T0 + 28 * DAY, None
        eff = {"opus": 0.0, "sonnet": 0.0}
        raw = {"opus": 0.0, "sonnet": 0.0}
        while t < end:
            week = int((t - T0) // (7 * DAY))
            if block != int(t // (3 * HOUR)):
                block = int(t // (3 * HOUR))
                share = rng.choice((0.0, 0.0, 0.05) if week < 2 else (0.1, 0.3, 0.6))
                load = rng.choice([0.2, 1.0, 2.0, 3.0])
                mix = [rng.uniform(0.5, 1.5) for _ in range(4)]
            fam = "sonnet" if rng.random() < share else "opus"
            busy = load * (0.5 + rng.random())
            tok = [2000 * busy * mix[0], 15_000 * busy * mix[1], 60_000 * busy * mix[2],
                   1_500_000 * busy * mix[3] * (0.4 if fam == "sonnet" else 1.0)]
            price = SONNET if fam == "sonnet" else OPUS
            usd = [tk * p / 1e6 for tk, p in zip(tok, price)]
            t_req = t + rng.uniform(0, POLL)
            self.requests.append((t_req, fam, usd, tok))
            e = sum(w * u for w, u in zip(TYPE_W, usd))
            eff[fam] += e
            raw[fam] += sum(usd)
            limit = e * (SONNET_WEIGHT if fam == "sonnet" else 1.0) * DRIFT[week]
            if t >= w7_end:
                acc7, w7_end = 0.0, w7_end + 7 * DAY
            if w5_end is None or t >= w5_end:
                acc5, w5_end = 0.0, t + 5 * HOUR
            acc7 += limit / 24.0
            acc5 += limit * (1.4 if _is_peak(t_req) else 1.0) / 3.6
            t += POLL
            self.readings["seven_day"].append((t, float(int(acc7)), w7_end))
            self.readings["five_hour"].append((t, float(int(acc5)), w5_end))
        self.requests.sort(key=lambda r: r[0])
        self.end = end
        self.e = {f: eff[f] / raw[f] for f in eff}

    def run(self, with_tokens=True):
        reqs = self.requests if with_tokens else [r[:3] for r in self.requests]
        return lra.analyze(self.readings, reqs, [], now=self.end)


class DriftTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.world = _DriftWorld()
        cls.out = cls.world.run()
        cls.fams = {f["family"]: f for f in cls.out["families"]}

    def test_per_token_weight_ignores_drift_and_mix(self):
        # Same tokens: Sonnet's price is 0.5 of Opus, its limit weight 1.3.
        truth = 0.5 * SONNET_WEIGHT
        pt = self.fams["sonnet"]["per_token"]
        self.assertEqual(self.out["family_main"], "opus")
        for bucket in ("seven_day", "five_hour"):
            self.assertAlmostEqual(pt[bucket]["value"], truth, delta=0.06, msg=bucket)
            self.assertLessEqual(pt[bucket]["lo"], pt[bucket]["value"])
            self.assertGreaterEqual(pt[bucket]["hi"], pt[bucket]["value"])
        self.assertEqual(self.fams["opus"]["per_token"]["seven_day"]["value"], 1.0)

    def test_dollars_per_point_ratio(self):
        # Limit per own dollar, Sonnet over Opus: the weight times how much
        # of each family's dollars count (Sonnet reads less cache).
        truth = SONNET_WEIGHT * self.world.e["sonnet"] / self.world.e["opus"]
        got = self.fams["opus"]["usd_per_point"] / self.fams["sonnet"]["usd_per_point"]
        self.assertAlmostEqual(got, truth, delta=truth * 0.07)

    def test_fit_quality(self):
        self.assertGreater(self.out["fit_quality"]["r2"], 0.9)

    def test_without_tokens_no_per_token(self):
        out = self.world.run(with_tokens=False)
        fams = {f["family"]: f for f in out["families"]}
        self.assertIsNone(fams["sonnet"]["per_token"]["seven_day"]["value"])
        self.assertIsNotNone(fams["sonnet"]["usd_per_point"])


class HelperTest(unittest.TestCase):

    def test_window_ends_merge_jitter(self):
        r = [(1.0, 0.0, 1000.0), (2.0, 1.0, 1300.0), (3.0, 0.0, 1000.0 + 7 * DAY), (4.0, 1.0, None)]
        self.assertEqual(lfam.window_ends(r), [1000.0, 1000.0 + 7 * DAY])

    def test_block_ids_compact(self):
        ivs = [lf.Interval(0, 10, 1), lf.Interval(2000, 2100, 1), lf.Interval(5000, 5100, 1)]
        blocks, nb = lfam.block_ids(ivs, [1, 2], [1000.0, 3000.0, 9000.0])
        self.assertEqual((blocks, nb), ({1: 0, 2: 1}, 2))

    def test_type_weights_fallback(self):
        self.assertEqual(lfam.type_weights(None), [1.0] * 4)
        self.assertEqual(lfam.type_weights({0: 0.5, 2: 0.5}), [1.0] * 4)  # no output
        self.assertEqual(lfam.type_weights({1: 1.0, 3: 0.0}), [1.0] * 4)  # zero weight
        self.assertEqual(lfam.type_weights({1: 1.0, 2: 5.0}), [1.0] * 4)  # implausible
        # Input left out of the fit keeps its price weight.
        self.assertEqual(lfam.type_weights({1: 1.0, 2: 0.6, 3: 0.4}), [1.0, 1.0, 0.6, 0.4])

    def test_token_ratio_uses_shared_types(self):
        main_usd, main_tok = [0, 20, 8, 2], [0, 1e6, 1e6, 1e7]
        usd, tok = [0, 10, 4, 1], [0, 1e6, 1e6, 1e7]
        self.assertAlmostEqual(lfam.token_ratio(usd, tok, main_usd, main_tok, [1, 1, 1, 0.4]), 0.5)
        self.assertIsNone(lfam.token_ratio(usd, [0] * 4, main_usd, main_tok, [1] * 4))

    def test_als_recovers_weight_on_exact_data(self):
        # Two windows with base rates 1 and 2, family weight 1.5, no noise.
        series = lra.Series(
            [lf.Interval(i * 3600.0, i * 3600.0 + 3000, 0.0) for i in range(8)], [], [])
        feats, ivs = [], []
        for i, iv in enumerate(series.intervals):
            a, b = 1.0 + i % 3, 0.5 + (i * 7) % 4
            k = 1.0 if i < 4 else 2.0
            feats.append([0, a, 0, 0, 0, b, 0, 0])
            ivs.append(lf.Interval(iv.ta, iv.tb, k * (a + 1.5 * b)))
        s = lra.Series(ivs, feats, [sum(f) for f in feats])
        blocks = {i: (0 if i < 4 else 1) for i in range(8)}
        res = lfam.fit(s, list(range(8)), lambda i: [feats[i][1], feats[i][5]], 2, 0,
                       blocks, 2, 20, 1)
        self.assertAlmostEqual(res["m"][1], 1.5, places=4)
        self.assertAlmostEqual(res["k"][0], 1.0, places=4)
        self.assertAlmostEqual(res["k"][1], 2.0, places=4)


if __name__ == "__main__":
    unittest.main()
