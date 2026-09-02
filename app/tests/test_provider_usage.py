"""Provider quota snapshots reported by the Pi dotfleet extension."""
import time
from datetime import datetime, timezone

from app.tests._support import TempDBTestCase


class ProviderUsageTest(TempDBTestCase):
    def post(self, limits):
        return self.client().post(
            "/api/provider-usage",
            json={"machine": "laptop", "limits": limits},
            headers={"X-API-Key": self.api_key},
        )

    def test_requires_api_key(self):
        response = self.client().post("/api/provider-usage", json={})
        self.assertEqual(response.status_code, 401)

    def test_merges_provider_snapshots_and_exposes_only_personal_scope(self):
        now = time.time()
        codex = {
            "provider": "codex",
            "windows": [{
                "key": "primary",
                "label": "5-hour limit",
                "pct": 31.5,
                "resets_at_epoch": now + 3600,
                "window_seconds": 5 * 3600,
            }],
        }
        go = {
            "provider": "opencode-go",
            "windows": [{
                "key": "weekly",
                "label": "Weekly limit",
                "pct": 12,
                "resets_at_epoch": now + 86400,
                "window_seconds": 7 * 86400,
            }],
        }
        self.assertEqual(self.post([codex]).status_code, 200)
        self.assertEqual(self.post([go]).status_code, 200)

        personal = self.client().get(
            "/api/rate-limits?scope=personal").json()["weekly_budget"]
        self.assertEqual(set(personal["providers"]), {"codex", "opencode-go"})
        self.assertEqual(personal["providers"]["codex"]["windows"][0]["pct"], 31.5)
        self.assertNotIn("machine", personal["providers"]["codex"])
        self.assertRegex(
            personal["providers"]["codex"]["windows"][0]["resets_at"],
            r"Z$",
        )

        enterprise = self.client().get(
            "/api/rate-limits?scope=enterprise").json()["weekly_budget"]
        self.assertNotIn("providers", enterprise)

    def test_rejects_unknown_providers_and_invalid_windows(self):
        unknown = self.post([{"provider": "other", "windows": []}])
        self.assertEqual(unknown.status_code, 422)
        invalid = self.post([{
            "provider": "codex",
            "windows": [{
                "key": "primary", "label": "Primary", "pct": -1,
                "resets_at_epoch": time.time(), "window_seconds": 1,
            }],
        }])
        self.assertEqual(invalid.status_code, 422)
        over_limit = self.post([{
            "provider": "codex",
            "windows": [{"key": "primary", "label": "Primary", "pct": 101,
                         "resets_at_epoch": time.time(), "window_seconds": 18000}],
        }])
        self.assertEqual(over_limit.status_code, 422)

    def test_zen_usage_comes_from_reported_pi_cost_without_fake_quota(self):
        now = time.time()
        timestamp = datetime.fromtimestamp(
            now, timezone.utc).isoformat().replace("+00:00", "Z")
        response = self.client().post(
            "/api/ingest/pi",
            json={
                "machine": "laptop", "account_class": "personal",
                "project_dir": "/project", "session_file": "pi.jsonl",
                "cursor": {"last_line_num": 0},
                "events": [{
                    "event_id": "zen-event", "timestamp": timestamp,
                    "session_id": "session", "kind": "assistant",
                    "provider": "opencode", "model": "glm",
                    "request_id": "zen-request",
                    "usage": {"input": 1, "cost_total": 1.25},
                }],
            },
            headers={"X-API-Key": self.api_key},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = self.client().get(
            "/api/rate-limits?scope=personal").json()["weekly_budget"]
        self.assertEqual(body["providers"]["opencode-zen"]["month_cost"], 1.25)
        self.assertEqual(body["providers"]["opencode-zen"]["windows"], [])

    def test_stale_snapshot_does_not_replace_newer_data(self):
        now = time.time()
        newer = {
            "provider": "codex", "observed_at_epoch": now,
            "windows": [{"key": "primary", "label": "Primary", "pct": 50,
                         "resets_at_epoch": now + 1000, "window_seconds": 18000}],
        }
        older = {
            "provider": "codex", "observed_at_epoch": now - 60,
            "windows": [{"key": "primary", "label": "Primary", "pct": 10,
                         "resets_at_epoch": now + 1000, "window_seconds": 18000}],
        }
        self.post([newer]); self.post([older])
        body = self.client().get(
            "/api/rate-limits?scope=personal").json()["weekly_budget"]
        self.assertEqual(body["providers"]["codex"]["windows"][0]["pct"], 50)
