"""Provider quota snapshots reported by the Pi dotfleet extension.

Snapshots are keyed by account class (personal vs work) so an enterprise
Codex account on a work box can never stomp the personal Codex snapshot,
and the enterprise dashboard shows enterprise limits only.
"""
import json
import time
from datetime import datetime, timezone

from app.tests._support import TempDBTestCase


def _codex(pct, now, plan=None):
    snapshot = {
        "provider": "codex",
        "windows": [{
            "key": "primary",
            "label": "5-hour limit",
            "pct": pct,
            "resets_at_epoch": now + 3600,
            "window_seconds": 5 * 3600,
        }],
    }
    if plan is not None:
        snapshot["plan"] = plan
    return snapshot


class ProviderUsageTest(TempDBTestCase):
    def post(self, limits, account_class="personal", machine="laptop"):
        body = {"machine": machine, "limits": limits}
        if account_class is not None:
            body["account_class"] = account_class
        return self.client().post(
            "/api/provider-usage", json=body,
            headers={"X-API-Key": self.api_key},
        )

    def budget(self, scope):
        return self.client().get(
            f"/api/rate-limits?scope={scope}").json()["weekly_budget"]

    def test_requires_api_key(self):
        response = self.client().post("/api/provider-usage", json={})
        self.assertEqual(response.status_code, 401)

    def test_requires_account_class(self):
        # An unclassified snapshot is exactly the stomp this endpoint exists
        # to prevent, so pre-scope clients fail closed until they sync.
        response = self.post([_codex(10, time.time())], account_class=None)
        self.assertEqual(response.status_code, 422)
        unknown = self.post([_codex(10, time.time())], account_class="other")
        self.assertEqual(unknown.status_code, 422)

    def test_merges_provider_snapshots_for_the_personal_scope(self):
        now = time.time()
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
        self.assertEqual(self.post([_codex(31.5, now)]).status_code, 200)
        self.assertEqual(self.post([go]).status_code, 200)

        personal = self.budget("personal")
        self.assertEqual(set(personal["providers"]), {"codex", "opencode-go"})
        self.assertEqual(personal["providers"]["codex"]["windows"][0]["pct"], 31.5)
        self.assertNotIn("machine", personal["providers"]["codex"])
        self.assertRegex(
            personal["providers"]["codex"]["windows"][0]["resets_at"],
            r"Z$",
        )
        # Nothing reported from a work box: the enterprise view has no block.
        self.assertNotIn("providers", self.budget("enterprise"))

    def test_work_snapshots_never_stomp_personal_and_vice_versa(self):
        now = time.time()
        self.assertEqual(
            self.post([_codex(31.5, now)], machine="laptop").status_code, 200)
        # A newer report from a work box lands in the enterprise scope only.
        work = self.post([{**_codex(80, now + 5), "observed_at_epoch": now + 5}],
                         account_class="work", machine="work-vm")
        self.assertEqual(work.status_code, 200)

        personal = self.budget("personal")["providers"]["codex"]
        self.assertEqual(personal["windows"][0]["pct"], 31.5)
        enterprise = self.budget("enterprise")["providers"]["codex"]
        self.assertEqual(enterprise["windows"][0]["pct"], 80)
        self.assertNotIn("machine", enterprise)

        # And a later personal report does not touch the enterprise snapshot.
        self.post([{**_codex(40, now + 10), "observed_at_epoch": now + 10}])
        self.assertEqual(self.budget("personal")["providers"]["codex"]
                         ["windows"][0]["pct"], 40)
        self.assertEqual(self.budget("enterprise")["providers"]["codex"]
                         ["windows"][0]["pct"], 80)

    def test_plan_type_is_exposed_as_a_label(self):
        now = time.time()
        self.assertEqual(self.post([_codex(5, now, plan="plus")]).status_code, 200)
        self.assertEqual(self.budget("personal")["providers"]["codex"]["plan"],
                         "plus")
        bad_plan = self.post([_codex(5, now, plan="not a plan!")])
        self.assertEqual(bad_plan.status_code, 422)

    def test_legacy_unscoped_snapshots_are_discarded(self):
        # Pre-scope servers stored a flat providers dict. Its snapshots have
        # no account class, so they are the stomp bug in stored form.
        now = time.time()
        legacy = {"providers": {"codex": {
            **_codex(66, now), "observed_at_epoch": now, "machine": "unknown"}}}
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
            ("provider_usage", json.dumps(legacy)))
        self.conn.commit()
        self.assertNotIn("providers", self.budget("personal"))
        self.assertNotIn("providers", self.budget("enterprise"))
        # A fresh scoped report replaces the legacy blob outright.
        self.assertEqual(self.post([_codex(12, now)]).status_code, 200)
        stored = json.loads(self.conn.execute(
            "SELECT value FROM meta WHERE key='provider_usage'").fetchone()["value"])
        self.assertNotIn("providers", stored)
        self.assertEqual(set(stored["scopes"]), {"personal"})

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

    def _ingest_zen_event(self, account_class, cost):
        timestamp = datetime.fromtimestamp(
            time.time(), timezone.utc).isoformat().replace("+00:00", "Z")
        response = self.client().post(
            "/api/ingest/pi",
            json={
                "machine": "laptop", "account_class": account_class,
                "project_dir": "/project", "session_file": "pi.jsonl",
                "cursor": {"last_line_num": 0},
                "events": [{
                    "event_id": f"zen-event-{account_class}",
                    "timestamp": timestamp,
                    "session_id": "session", "kind": "assistant",
                    "provider": "opencode", "model": "glm",
                    "request_id": f"zen-request-{account_class}",
                    "usage": {"input": 1, "cost_total": cost},
                }],
            },
            headers={"X-API-Key": self.api_key},
        )
        self.assertEqual(response.status_code, 200, response.text)

    def test_zen_usage_comes_from_reported_pi_cost_without_fake_quota(self):
        self._ingest_zen_event("personal", 1.25)
        body = self.budget("personal")
        self.assertEqual(body["providers"]["opencode-zen"]["month_cost"], 1.25)
        self.assertEqual(body["providers"]["opencode-zen"]["windows"], [])

    def test_reported_costs_follow_the_requested_scope(self):
        self._ingest_zen_event("personal", 1.25)
        self._ingest_zen_event("work", 9.5)
        self.assertEqual(self.budget("personal")["providers"]["opencode-zen"]
                         ["month_cost"], 1.25)
        self.assertEqual(self.budget("enterprise")["providers"]["opencode-zen"]
                         ["month_cost"], 9.5)

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
        body = self.budget("personal")
        self.assertEqual(body["providers"]["codex"]["windows"][0]["pct"], 50)
