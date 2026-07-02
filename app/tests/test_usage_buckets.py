"""Unit tests for app.usage_buckets.normalize_usage_buckets (B1).

Ground truth: app/tests/fixtures/oauth_usage_live_2026-07-01.json — the EXACT
live production API response shape (minute-scrubbed timestamps). Contract:
limits[] is the PRIMARY source, legacy dict buckets are the FALLBACK, and
null/unknown/garbage shapes are silently skipped — the payload is fluid and
the normalizer must never raise.
"""

import json
import unittest
from pathlib import Path

from app.usage_buckets import normalize_usage_buckets

FIXTURE = (Path(__file__).resolve().parent
           / "fixtures" / "oauth_usage_live_2026-07-01.json")


def load_fixture():
    with open(FIXTURE) as f:
        return json.load(f)


class FixtureNormalizationTest(unittest.TestCase):
    """The live prod payload must normalize to exactly three buckets."""

    def test_exactly_three_buckets_in_render_order(self):
        buckets = normalize_usage_buckets(load_fixture())
        self.assertEqual([b["key"] for b in buckets],
                         ["five_hour", "seven_day", "scoped:fable"])

    def test_five_hour_from_limits_session_entry(self):
        b = normalize_usage_buckets(load_fixture())[0]
        self.assertEqual(b["key"], "five_hour")
        self.assertEqual(b["label"], "5-Hour")
        self.assertEqual(b["utilization"], 1.0)
        self.assertEqual(b["resets_at"], "2026-07-02T07:40:00+00:00")

    def test_seven_day_from_limits_weekly_all_entry(self):
        b = normalize_usage_buckets(load_fixture())[1]
        self.assertEqual(b["key"], "seven_day")
        self.assertEqual(b["label"], "7-Day")
        self.assertEqual(b["utilization"], 20.0)
        self.assertEqual(b["resets_at"], "2026-07-02T08:00:00+00:00")

    def test_scoped_fable_from_weekly_scoped_entry(self):
        b = normalize_usage_buckets(load_fixture())[2]
        self.assertEqual(b["key"], "scoped:fable")
        self.assertEqual(b["label"], "Fable")  # display_name verbatim
        self.assertEqual(b["utilization"], 34.0)
        # resets_at RAW here — scrubbing happens at API boundaries.
        self.assertEqual(b["resets_at"], "2026-07-02T08:00:00+00:00")

    def test_entry_shape(self):
        for b in normalize_usage_buckets(load_fixture()):
            self.assertEqual(set(b), {"key", "label", "utilization", "resets_at"})
            self.assertIsInstance(b["utilization"], float)


class LegacyFallbackTest(unittest.TestCase):
    """Legacy dict buckets still normalize when limits[] is absent."""

    def test_legacy_only_payload_normalizes(self):
        usage = {
            "five_hour": {"utilization": 51.0,
                          "resets_at": "2026-06-09T12:00:00+00:00"},
            "seven_day": {"utilization": 11.0,
                          "resets_at": "2026-06-10T12:00:00+00:00"},
            "seven_day_opus": {"utilization": 62.0,
                               "resets_at": "2026-06-10T12:00:00+00:00"},
        }
        buckets = normalize_usage_buckets(usage)
        self.assertEqual([b["key"] for b in buckets],
                         ["five_hour", "seven_day", "scoped:opus"])
        opus = buckets[2]
        self.assertEqual(opus["label"], "Opus")  # title-cased from suffix
        self.assertEqual(opus["utilization"], 62.0)
        self.assertEqual(opus["resets_at"], "2026-06-10T12:00:00+00:00")

    def test_limits_wins_over_legacy_for_same_key(self):
        usage = {
            "seven_day": {"utilization": 11.0,
                          "resets_at": "2026-06-10T12:00:00+00:00"},
            "limits": [
                {"kind": "weekly_all", "group": "weekly", "percent": 20,
                 "resets_at": "2026-07-02T08:00:00+00:00", "scope": None,
                 "severity": "normal", "is_active": False},
            ],
        }
        buckets = normalize_usage_buckets(usage)
        sevens = [b for b in buckets if b["key"] == "seven_day"]
        self.assertEqual(len(sevens), 1, "must dedupe to ONE seven_day entry")
        self.assertEqual(sevens[0]["utilization"], 20.0)
        self.assertEqual(sevens[0]["resets_at"], "2026-07-02T08:00:00+00:00")

    def test_legacy_bool_utilization_rejected(self):
        self.assertEqual(
            normalize_usage_buckets({"seven_day": {"utilization": True}}), [])

    def test_ignored_keys_never_become_buckets(self):
        usage = {
            "spend": {"utilization": 40.0},
            "extra_usage": {"utilization": 12.0, "is_enabled": True},
            "member_dashboard_available": True,
        }
        self.assertEqual(normalize_usage_buckets(usage), [])


class LimitsParsingTest(unittest.TestCase):
    def test_display_name_slugging(self):
        usage = {"limits": [
            {"kind": "weekly_scoped", "percent": 7,
             "resets_at": "2026-07-02T08:00:00+00:00",
             "scope": {"model": {"id": None, "display_name": "Nova 9"},
                       "surface": None}},
        ]}
        buckets = normalize_usage_buckets(usage)
        self.assertEqual(len(buckets), 1)
        self.assertEqual(buckets[0]["key"], "scoped:nova_9")
        self.assertEqual(buckets[0]["label"], "Nova 9")
        self.assertEqual(buckets[0]["utilization"], 7.0)

    def test_non_null_model_id_tolerated(self):
        usage = {"limits": [
            {"kind": "weekly_scoped", "percent": 12,
             "resets_at": "2026-07-02T08:00:00+00:00",
             "scope": {"model": {"id": "claude-fable-5",
                                 "display_name": "Fable"},
                       "surface": None}},
        ]}
        buckets = normalize_usage_buckets(usage)
        self.assertEqual([b["key"] for b in buckets], ["scoped:fable"])

    def test_weekly_scoped_without_display_name_skipped(self):
        usage = {"limits": [
            {"kind": "weekly_scoped", "percent": 12, "scope": {}},
            {"kind": "weekly_scoped", "percent": 12, "scope": None},
            {"kind": "weekly_scoped", "percent": 12,
             "scope": {"model": {"id": None, "display_name": None}}},
        ]}
        self.assertEqual(normalize_usage_buckets(usage), [])

    def test_scoped_buckets_sorted_alphabetically(self):
        usage = {"limits": [
            {"kind": "weekly_scoped", "percent": 1,
             "scope": {"model": {"display_name": "Zeta"}}},
            {"kind": "weekly_scoped", "percent": 2,
             "scope": {"model": {"display_name": "Alpha"}}},
            {"kind": "weekly_all", "percent": 3},
            {"kind": "session", "percent": 4},
        ]}
        buckets = normalize_usage_buckets(usage)
        self.assertEqual([b["key"] for b in buckets],
                         ["five_hour", "seven_day", "scoped:alpha",
                          "scoped:zeta"])


class MergeAndClampTest(unittest.TestCase):
    """Field-level merge (limits[] wins, legacy fills resets_at gaps) and
    utilization clamping at bucket construction."""

    def test_primary_null_resets_at_filled_from_legacy(self):
        """limits[] wins per field, but a null resets_at is a data gap — the
        legacy bucket's resets_at fills it."""
        usage = {
            "seven_day": {"utilization": 11.0,
                          "resets_at": "2026-06-10T12:00:00+00:00"},
            "limits": [
                {"kind": "weekly_all", "percent": 20, "resets_at": None},
            ],
        }
        buckets = normalize_usage_buckets(usage)
        sevens = [b for b in buckets if b["key"] == "seven_day"]
        self.assertEqual(len(sevens), 1)
        self.assertEqual(sevens[0]["utilization"], 20.0)  # limits still wins
        self.assertEqual(sevens[0]["resets_at"], "2026-06-10T12:00:00+00:00")

    def test_merge_does_not_mutate_inputs(self):
        import copy
        usage = {
            "seven_day": {"utilization": 11.0,
                          "resets_at": "2026-06-10T12:00:00+00:00"},
            "limits": [
                {"kind": "weekly_all", "percent": 20, "resets_at": None},
            ],
        }
        snapshot = copy.deepcopy(usage)
        normalize_usage_buckets(usage)
        self.assertEqual(usage, snapshot, "inputs must never be mutated")

    def test_negative_limits_percent_clamped_to_zero(self):
        usage = {"limits": [
            {"kind": "weekly_all", "percent": -5,
             "resets_at": "2026-07-02T08:00:00+00:00"},
        ]}
        buckets = normalize_usage_buckets(usage)
        self.assertEqual(len(buckets), 1)
        self.assertEqual(buckets[0]["utilization"], 0.0)

    def test_negative_legacy_utilization_clamped_to_zero(self):
        usage = {"seven_day": {"utilization": -5,
                               "resets_at": "2026-06-10T12:00:00+00:00"}}
        buckets = normalize_usage_buckets(usage)
        self.assertEqual(len(buckets), 1)
        self.assertEqual(buckets[0]["utilization"], 0.0)


class GarbageToleranceTest(unittest.TestCase):
    """Unknown shapes never raise — they are silently skipped."""

    def test_garbage_everywhere_yields_empty(self):
        usage = {
            "five_hour": {"utilization": "high"},             # non-numeric
            "seven_day": {"resets_at": "x"},                  # no utilization
            "seven_day_opus": {"utilization": float("nan")},  # non-finite
            "tangelo": None,
            "iguana_necktie": "noise",
            "amber_ladder": 42,
            "limits": [
                {"kind": "session", "percent": True},         # bool rejected
                {"kind": "weekly_all"},                       # no percent
                {"kind": "weekly_scoped", "percent": 10, "scope": {}},
                {"kind": "mystery_kind", "percent": 10},
                {"percent": 10},                              # no kind
                "not-a-dict",
                None,
                17,
            ],
        }
        self.assertEqual(normalize_usage_buckets(usage), [])

    def test_limits_not_a_list_falls_back_to_legacy(self):
        usage = {"limits": "nope",
                 "seven_day": {"utilization": 5,
                               "resets_at": "2026-06-10T12:00:00+00:00"}}
        buckets = normalize_usage_buckets(usage)
        self.assertEqual([b["key"] for b in buckets], ["seven_day"])
        self.assertEqual(buckets[0]["utilization"], 5.0)

    def test_non_string_resets_at_becomes_none(self):
        usage = {"limits": [
            {"kind": "weekly_all", "percent": 3, "resets_at": 12345},
        ]}
        buckets = normalize_usage_buckets(usage)
        self.assertEqual(len(buckets), 1)
        self.assertIsNone(buckets[0]["resets_at"])

    def test_empty_dict(self):
        self.assertEqual(normalize_usage_buckets({}), [])

    def test_non_dict_input(self):
        self.assertEqual(normalize_usage_buckets(None), [])
        self.assertEqual(normalize_usage_buckets([]), [])
        self.assertEqual(normalize_usage_buckets("nope"), [])


if __name__ == "__main__":
    unittest.main()
