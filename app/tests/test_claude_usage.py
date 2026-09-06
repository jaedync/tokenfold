"""Metadata-only personal Claude observations, ordering and source ownership."""
import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import app.config
from app.tests._support import TempDBTestCase


class ClaudeUsageTest(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.now = time.time()
        self.payload = {
            "machine": "test-mac", "account_class": "personal",
            "source": "meridian-oauth", "source_profile": "default",
            "observed_at_epoch": self.now - 600,
            "buckets": [
                {"key": "five_hour", "label": "5-Hour", "pct": 0,
                 "resets_at_epoch": self.now + 3600},
                {"key": "seven_day", "label": "7-Day", "pct": 7,
                 "resets_at_epoch": self.now + 86400},
                {"key": "scoped:fable", "label": "Custom Fable", "pct": 13,
                 "resets_at_epoch": self.now + 86400}],
            "extra_usage": {"enabled": False},
        }
        self.rebuild = patch("app.aggregator.trigger_eager_rebuild").start()
        self.addCleanup(patch.stopall)

    def post(self, payload=None):
        return self.client().post("/api/usage/claude", json=payload or self.payload,
                                  headers={"X-API-Key": self.api_key})

    def stored(self):
        row = self.conn.execute("SELECT value FROM meta WHERE key='oauth_usage'").fetchone()
        return json.loads(row[0]) if row else None

    def test_auth_and_scope_fail_closed_without_writes(self):
        self.assertEqual(self.client().post("/api/usage/claude", json=self.payload).status_code, 401)
        for account in ("work", "enterprise"):
            self.assertEqual(self.post({**self.payload, "account_class": account}).status_code, 422)
        with patch.object(app.config, "LOCKED_SCOPE", "enterprise"):
            self.assertEqual(self.post().status_code, 403)
        self.assertEqual(self.conn.execute("SELECT count(*) FROM meta").fetchone()[0], 0)
        self.assertEqual(self.conn.execute("SELECT count(*) FROM limit_readings").fetchone()[0], 0)

    def test_metadata_and_history_keep_observation_and_source(self):
        self.assertEqual(self.post().status_code, 200)
        stored = self.stored()
        self.assertEqual(stored["observed_at_epoch"], self.payload["observed_at_epoch"])
        self.assertEqual(stored["source"], "meridian-oauth")
        self.assertEqual(stored["source_profile"], "default")
        rows = self.conn.execute("SELECT * FROM limit_readings").fetchall()
        self.assertEqual(len(rows), 3)
        self.assertEqual({r["fetched_epoch"] for r in rows}, {self.payload["observed_at_epoch"]})
        self.assertEqual({r["source"] for r in rows}, {"meridian-oauth"})
        oauth = self.client().get("/api/rate-limit-snapshots?scope=personal").json()["weekly_budget"]["oauth"]
        self.assertEqual(oauth["updated_at_epoch"], self.payload["observed_at_epoch"])
        self.assertEqual(oauth["source"], "meridian-oauth")
        self.assertEqual(oauth["five_hour_pct"], 0)
        self.assertEqual(oauth["buckets"][2]["label"], "Custom Fable")

    def test_strict_bounded_body(self):
        bad = []
        for key, value in (("source_profile", "work"), ("machine", " "),
                           ("source", "sdk"), ("observed_at_epoch", "123"),
                           ("observed_at_epoch", True), ("observed_at_epoch", self.now + 301),
                           ("observed_at_epoch", self.now - 86401), ("token", "secret")):
            bad.append({**self.payload, key: value})
        for key, value in (("pct", True), ("pct", "7"), ("pct", 101),
                           ("resets_at_epoch", 0), ("resets_at_epoch", 10**12),
                           ("key", "scoped:Not-Safe"), ("raw", "private")):
            p = copy.deepcopy(self.payload)
            p["buckets"][0][key] = value
            bad.append(p)
        bad += [{**self.payload, "buckets": self.payload["buckets"][:1]},
                {**self.payload, "buckets": self.payload["buckets"] * 2},
                {**self.payload, "extra_usage": {"enabled": False, "used_cents": -1}},
                {**self.payload, "extra_usage": {"enabled": 0}}]
        for payload in bad:
            with self.subTest(payload=payload):
                self.assertEqual(self.post(payload).status_code, 422)
        self.assertIsNone(self.stored())

    def test_nonfinite_numbers_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            p = {**self.payload, "observed_at_epoch": value}
            r = self.client().post("/api/usage/claude", content=json.dumps(p),
                                   headers={"X-API-Key": self.api_key, "Content-Type": "application/json"})
            self.assertEqual(r.status_code, 422)

    def test_skew_clamped_but_original_time_retained(self):
        original = self.now + 100
        with patch("app.claude_usage.time.time", return_value=self.now):
            self.assertEqual(self.post({**self.payload, "observed_at_epoch": original}).status_code, 200)
        self.assertEqual(self.stored()["observed_at_epoch"], self.now)
        self.assertEqual(self.stored()["original_observed_at_epoch"], original)

    def test_atomic_monotonic_and_duplicate_history(self):
        self.assertEqual(self.post().status_code, 200)
        self.post()
        self.post({**self.payload, "observed_at_epoch": self.now - 700})
        self.assertEqual(self.conn.execute("SELECT count(*) FROM limit_readings").fetchone()[0], 3)
        payloads = [{**self.payload, "observed_at_epoch": self.now - n} for n in range(1, 20)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertTrue(all(r.status_code == 200 for r in pool.map(self.post, payloads)))
        self.assertEqual(self.stored()["observed_at_epoch"], self.now - 1)

    def test_legacy_cannot_rejuvenate_managed_source_but_enterprise_meter_survives(self):
        self.post()
        before = self.stored()
        legacy = {"usage": {"five_hour": {"utilization": 82, "resets_at": "2026-01-01T00:00:00Z"}}}
        r = self.client().post("/api/usage", json=legacy, headers={"X-API-Key": self.api_key})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.stored(), before)
        self.assertEqual(self.conn.execute("SELECT count(*) FROM limit_readings").fetchone()[0], 3)
        r = self.client().post("/api/usage", json={"machine": "work", "usage": {
            "extra_usage": {"is_enabled": True, "used_credits": 100}}},
            headers={"X-API-Key": self.api_key})
        self.assertTrue(r.json()["captured_extra_usage"])
        self.assertIsNotNone(self.conn.execute("SELECT value FROM meta WHERE key='oauth_usage_enterprise'").fetchone())

    def test_first_transfer_ignores_newer_legacy_receipt_stamp(self):
        from app.claude_usage import store_snapshot
        store_snapshot({"five_hour": {"utilization": 82}}, self.now, "client")
        self.assertEqual(self.post().json()["status"], "ok")
        self.assertEqual(self.stored()["source"], "meridian-oauth")
        self.assertEqual(self.stored()["observed_at_epoch"], self.payload["observed_at_epoch"])

    def test_only_canonical_scoped_slugs_are_accepted(self):
        for slug in ("_foo", "foo_", "foo__bar", "_", "x" * 58):
            p = copy.deepcopy(self.payload)
            p["buckets"][2]["key"] = "scoped:" + slug
            with self.subTest(slug=slug):
                self.assertEqual(self.post(p).status_code, 422)
        p["buckets"][2]["key"] = "scoped:" + "x" * 57
        self.assertEqual(self.post(p).status_code, 200)

    def test_client_fixture_parity_expired_window_and_absent_amounts(self):
        fixture = json.loads((Path(__file__).parent / "fixtures" / "claude_usage_metadata.json").read_text())
        with patch("app.claude_usage.time.time", return_value=fixture["observed_at_epoch"] + 60):
            self.assertEqual(self.post(fixture).status_code, 200)
            b = self.client().get("/api/rate-limits?scope=personal").json()["weekly_budget"]["oauth"]
        self.assertNotIn("five_hour_window", b)
        self.assertEqual(b["weekly_pct"], 0)
        self.assertEqual(b["source"], "meridian-oauth")
        self.assertIsNone(b["extra_usage"]["used_cents"])
        self.assertNotIn("used_credits", self.stored()["data"]["extra_usage"])

    def test_replayed_skewed_observation_cannot_advance_at_receipt(self):
        p = {**self.payload, "observed_at_epoch": self.now + 100}
        with patch("app.claude_usage.time.time", return_value=self.now):
            self.post(p)
        with patch("app.claude_usage.time.time", return_value=self.now + 10):
            self.assertEqual(self.post(p).json()["status"], "ignored_stale")
        self.assertEqual(self.stored()["observed_at_epoch"], self.now)
        self.assertEqual(self.conn.execute("SELECT count(*) FROM limit_readings").fetchone()[0], 3)
