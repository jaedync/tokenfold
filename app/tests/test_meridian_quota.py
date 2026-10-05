"""Parity of the server-side Meridian quota parser with the dotfleet Pi reporter.

Mirrors dotfleet pi/tests/claude-usage.test.ts case for case, against the same
client fixture, so the collector and the Pi reporter send identical payloads.
"""
import copy
import json
import math
import unittest
from pathlib import Path

from app.meridian_quota import parse_meridian_quota
from app.models import ClaudeUsageRequest

FIXTURE = json.loads((Path(__file__).parent / "fixtures"
                      / "claude_usage_metadata.json").read_text())
OBSERVED_MS = FIXTURE["observed_at_epoch"] * 1000
MACHINE = "test-personal"


def quota():
    """The Meridian /v1/usage/quota body that serializes to FIXTURE."""
    return {
        "profile": None,
        "sources": {"oauth": {"fetchedAt": OBSERVED_MS}, "sdk": {"entryCount": 1}},
        "buckets": [{"type": b["key"].replace("scoped:", "seven_day_"),
                     "utilization": b["pct"] / 100,
                     "resetsAt": b["resets_at_epoch"] * 1000,
                     "observedAt": OBSERVED_MS} for b in FIXTURE["buckets"]],
        "extraUsage": {"isEnabled": False, "monthlyLimit": 0, "usedCredits": 0,
                       "currency": "USD"},
        "auth": "never-forward", "transcript": "never-forward",
    }


def parse(body, machine=MACHINE, now_ms=OBSERVED_MS):
    return parse_meridian_quota(body, machine, now_ms / 1000)


class SerializerTest(unittest.TestCase):
    def test_fixture_exact_fractions_ms_to_seconds_once_and_whitelist(self):
        self.assertEqual(parse(quota()), FIXTURE)
        body = quota()
        body["buckets"][0]["utilization"] = 1
        self.assertEqual(parse(body)["buckets"][0]["pct"], 100)
        body["extraUsage"]["isEnabled"] = True
        self.assertNotIn("extra_usage", parse(body))

    def test_output_is_a_valid_ingest_request(self):
        ClaudeUsageRequest.model_validate(parse(quota()))

    def test_input_is_never_mutated(self):
        body = quota()
        snapshot = copy.deepcopy(body)
        parse(body)
        self.assertEqual(body, snapshot)


class RejectionTest(unittest.TestCase):
    def test_reject_profile_time_and_required_bucket_faults(self):
        def drop(key):
            return lambda b: b.pop(key)
        mutations = [
            lambda b: b.update(profile="work"),
            drop("profile"),
            lambda b: b["sources"].pop("oauth"),
            lambda b: b["sources"]["oauth"].update(fetchedAt=OBSERVED_MS - 86400001),
            lambda b: b["sources"]["oauth"].update(fetchedAt=OBSERVED_MS + 300001),
            lambda b: b["buckets"][0].update(observedAt=OBSERVED_MS + 1),
            lambda b: b["buckets"][0].update(source="sdk"),
            lambda b: b["buckets"].append(b["buckets"][0]),
            lambda b: b["buckets"][0].update(utilization=77),
            lambda b: b["buckets"][0].update(resetsAt=math.inf),
            lambda b: b.update(buckets=[b["buckets"][0]] * 17),
        ]
        for i, mutate in enumerate(mutations):
            body = quota()
            mutate(body)
            self.assertIsNone(parse(body), i)

    def test_reject_non_numeric_fetched_at(self):
        for value in (math.nan, math.inf, -1, 0, "1788706007457", True, None, 10 ** 400):
            body = quota()
            body["sources"]["oauth"]["fetchedAt"] = value
            self.assertIsNone(parse(body), value)

    def test_reject_bad_machine_and_non_object_bodies(self):
        self.assertIsNone(parse(quota(), machine="x" * 129))
        self.assertIsNone(parse(quota(), machine="  "))
        self.assertIsNone(parse(quota(), machine="a\nb"))
        self.assertIsNone(parse(quota(), machine="\ufeff "))
        for body in (None, [], "quota", 7):
            self.assertIsNone(parse(body))


class OptionalBucketTest(unittest.TestCase):
    def test_skip_optional_mixed_time_or_source(self):
        for patch in ({"observedAt": OBSERVED_MS - 1}, {"source": "sdk"}):
            body = quota()
            body["buckets"][2].update(patch)
            self.assertEqual(len(parse(body)["buckets"]), 2, patch)

    def test_allowed_clock_skew_keeps_the_observation_time(self):
        self.assertEqual(parse(quota(), now_ms=OBSERVED_MS - 300000)["observed_at_epoch"],
                         FIXTURE["observed_at_epoch"])
        self.assertEqual(parse(quota(), now_ms=OBSERVED_MS + 86400000)["observed_at_epoch"],
                         FIXTURE["observed_at_epoch"])

    def test_malformed_optional_buckets_are_skipped(self):
        for patch in ({"utilization": math.nan}, {"utilization": 2}, {"resetsAt": -1},
                      {"resetsAt": 1e15}, {"utilization": True}, {"resetsAt": 10 ** 400},
                      {"utilization": 10 ** 400}):
            body = quota()
            body["buckets"][2].update(patch)
            self.assertEqual(len(parse(body)["buckets"]), 2, patch)

    def test_scoped_identities_use_canonical_slugs(self):
        for slug in ("_fable", "fable_", "a__b", "_", "Fable", "a-b", "a" * 58, "fable\n"):
            body = quota()
            body["buckets"][2]["type"] = f"seven_day_{slug}"
            self.assertEqual(len(parse(body)["buckets"]), 2, slug)
        for slug in ("a", "a_0", "a" * 57):
            body = quota()
            body["buckets"][2]["type"] = f"seven_day_{slug}"
            self.assertEqual(parse(body)["buckets"][2]["key"], f"scoped:{slug}")
        body = quota()
        body["buckets"][2]["type"] = "seven_day_extra_fast"
        self.assertEqual(parse(body)["buckets"][2]["label"], "Extra fast")


if __name__ == "__main__":
    unittest.main()
