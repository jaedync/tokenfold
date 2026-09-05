"""Fast quota snapshots and observation-aligned Codex capacity regressions."""
import asyncio
import json
import time
from unittest.mock import patch

from app.tests._support import TempDBTestCase


class ProviderLimitLoadingTest(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.now = time.time()
        self.start = self.now - 86400
        self.snapshot = {
            "provider": "codex", "observed_at_epoch": self.now - 60,
            "windows": [{"key": "primary", "label": "5-hour limit", "pct": 10,
                         "window_seconds": 604800,
                         "resets_at_epoch": self.start + 604800}],
        }
        self.store()

    def store(self):
        self.conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (
            "provider_usage", json.dumps({"scopes": {"personal": {
                "codex": self.snapshot}}})))
        self.conn.commit()

    def event(self, uid, cost, ts=None, provider="openai-codex", plan="max", req=None):
        self.conn.execute(
            "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,source_machine,"
            "source_client,provider,model,request_id,reported_cost_total,plan,account_email) "
            "VALUES (?,'assistant','2026-09-04',?,'2026-09-04','test',"
            "'pi-agent',?,'gpt',?,?,?,'test@example.org')",
            (uid, ts if ts is not None else self.now - 120, provider, req or uid, cost, plan))
        self.conn.commit()

    def test_stored_primary_weekly_duration_overrides_label(self):
        from app.provider_usage import provider_usage_block
        win = provider_usage_block("personal", self.now)["codex"]["windows"][0]
        self.assertEqual(win["label"], "7-day limit")
        self.snapshot["windows"][0]["window_seconds"] = None
        self.store()
        win = provider_usage_block("personal", self.now)["codex"]["windows"][0]
        self.assertEqual(win["label"], "Primary limit")

    def test_window_cost_is_deduped_scoped_provider_filtered_and_observation_aligned(self):
        from app.provider_usage import provider_usage_block
        self.event("a", 10, req="same")
        self.event("b", 10, req="same")
        self.event("other-provider", 50, provider="openrouter")
        self.event("work", 50, plan="enterprise")
        self.event("before", 50, ts=self.start - 1)
        self.event("after-observation", 50, ts=self.now - 30)
        win = provider_usage_block("personal", self.now)["codex"]["windows"][0]
        self.assertEqual(win["window_cost"], 10)
        self.assertEqual(win["estimated_capacity"], 100)
        self.assertEqual(win["estimated_remaining"], 90)

    def test_reported_components_and_explicit_zero_total_are_preserved(self):
        from app.provider_usage import provider_usage_block
        self.event("components", None)
        self.event("zero", 0)
        self.conn.execute("UPDATE events SET reported_cost_input=2, reported_cost_output=3")
        self.conn.commit()
        win = provider_usage_block("personal", self.now)["codex"]["windows"][0]
        self.assertEqual(win["window_cost"], 5)
        self.assertEqual(win["estimated_capacity"], 50)

    def test_snapshot_oauth_matches_enriched_and_is_never_enterprise(self):
        usage = {"seven_day": {"utilization": 17, "resets_at": "2026-10-01T12:34:56Z"},
                 "five_hour": {"utilization": 3, "resets_at": "2026-10-01T01:02:03Z"}}
        self.conn.execute("INSERT INTO meta VALUES (?,?)", ("oauth_usage", json.dumps({
            "data": usage, "updated_at": "2026-09-04T01:00:00Z"})))
        self.conn.commit()
        client = self.client()
        cheap = client.get("/api/rate-limit-snapshots?scope=personal").json()["weekly_budget"]
        enriched = client.get("/api/rate-limits?scope=personal").json()["weekly_budget"]
        for key in ("weekly_pct", "weekly_resets_at", "five_hour_pct", "buckets", "updated_at_epoch"):
            self.assertEqual(cheap["oauth"][key], enriched["oauth"][key])
        self.assertNotIn("oauth", client.get("/api/rate-limit-snapshots?scope=enterprise")
                         .json()["weekly_budget"])
        self.assertNotIn("machine", cheap["providers"]["codex"])
        self.assertNotIn("resets_at_epoch", cheap["providers"]["codex"]["windows"][0])

    def test_independent_reader_does_not_wait_for_shared_connection_statement(self):
        import threading
        entered, release = threading.Event(), threading.Event()
        self.conn.create_function("slow_quota_test", 0, lambda: (entered.set(), release.wait(2), 1)[2])
        worker = threading.Thread(target=lambda: self.conn.execute("SELECT slow_quota_test()").fetchone())
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            response = self.client().get("/api/rate-limit-snapshots?scope=personal")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(worker.is_alive(), "snapshot waited for the shared connection")
        finally:
            release.set()
            worker.join(3)

    def test_no_estimates_for_zero_low_pct_expired_invalid_or_stale_windows(self):
        from app.provider_usage import provider_usage_block
        self.event("a", 10)
        original = json.loads(json.dumps(self.snapshot))
        for change in ({"pct": 0}, {"pct": 4}, {"resets_at_epoch": self.now - 1},
                       {"window_seconds": None}, {"resets_at_epoch": self.now + 900000}):
            with self.subTest(change=change):
                self.snapshot = json.loads(json.dumps(original))
                self.snapshot["windows"][0].update(change)
                self.store()
                win = provider_usage_block("personal", self.now)["codex"]["windows"][0]
                self.assertNotIn("estimated_capacity", win)
        self.snapshot["observed_at_epoch"] = self.now - 90000
        self.store()
        self.assertEqual(provider_usage_block("personal", self.now)["codex"]["windows"], [])

    def test_snapshot_endpoint_never_queries_events_and_corrects_stored_label(self):
        import app.db as db
        queries = []
        original = db.read_conn
        from contextlib import contextmanager

        @contextmanager
        def traced():
            with original() as conn:
                conn.set_trace_callback(queries.append)
                yield conn
        with patch("app.api.read_conn", traced):
            response = self.client().get("/api/rate-limit-snapshots?scope=personal")
        self.assertEqual(response.status_code, 200)
        body = response.json()["weekly_budget"]
        self.assertEqual(body["providers"]["codex"]["windows"][0]["label"], "7-day limit")
        self.assertFalse(any("events" in q.lower() for q in queries), queries)
        self.assertNotIn("month_cost", body["providers"]["codex"])

    def test_snapshot_auth_scope_and_lock(self):
        with patch.object(self._config, "DASHBOARD_PASSWORD", "secret"):
            self.assertEqual(self.client().get("/api/rate-limit-snapshots").status_code, 401)
        self.assertEqual(self.client().get("/api/rate-limit-snapshots?scope=invalid").status_code, 400)
        self.assertEqual(self.client().get("/api/rate-limit-snapshots?scope=enterprise")
                         .json()["weekly_budget"]["providers"], {})
        with patch.object(self._config, "LOCKED_SCOPE", "enterprise"):
            self.assertEqual(self.client().get("/api/rate-limit-snapshots?scope=personal").status_code, 403)

    def test_slow_enrichment_runs_off_event_loop(self):
        from app.api import rate_limits
        import threading
        entered, release = threading.Event(), threading.Event()

        def slow(*args):
            entered.set()
            release.wait(2)
            return {}

        async def exercise():
            task = asyncio.create_task(rate_limits("personal"))
            try:
                for _ in range(100):
                    if entered.is_set():
                        break
                    await asyncio.sleep(.005)
                self.assertTrue(entered.is_set())
                self.assertFalse(task.done(), "blocking aggregation ran on the event loop")
            finally:
                release.set()
                await task
        with patch("app.api._build_rate_limits", slow):
            asyncio.run(exercise())
