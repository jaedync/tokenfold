"""Provider quota dollars read Pi rows through a small partial index.

Without it, every /api/rate-limits request scanned a month of all assistant
events (about 0.7 s on the production copy) to find a few thousand Codex
and OpenCode rows.
"""
import sqlite3
import time
from unittest.mock import patch

from app.tests._support import TempDBTestCase
from app.tests.test_bucket_windows import _ins_pi_event

INDEX = "idx_events_pi_provider_ts"


class ProviderIndexTest(TempDBTestCase):
    def _plans(self):
        """EXPLAIN QUERY PLAN of every SQL statement the provider block runs."""
        from app.provider_usage import (_month_reported_costs,
                                        _window_reported_cost)
        plans = []
        real = self.conn

        class Spy:
            def execute(self, sql, params=()):
                if sql.lstrip().upper().startswith("SELECT") and "events" in sql:
                    plans.append(" ".join(r[-1] for r in real.execute(
                        "EXPLAIN QUERY PLAN " + sql, params).fetchall()))
                return real.execute(sql, params)

        now = time.time()
        _month_reported_costs(Spy(), now, "personal")
        _window_reported_cost(Spy(), "personal", "codex", now - 86400, now)
        return plans

    def test_index_exists(self):
        names = {r[0] for r in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'")}
        self.assertIn(INDEX, names)

    def test_provider_queries_use_the_index(self):
        plans = self._plans()
        self.assertEqual(len(plans), 2)
        for plan in plans:
            self.assertIn(INDEX, plan)

    def test_month_cost_still_counts_only_quota_providers(self):
        from app.provider_usage import _month_reported_costs
        now = time.time()
        _ins_pi_event(self.conn, "a", "ra", now - 60, "openai-codex", "gpt-5", 3.0)
        _ins_pi_event(self.conn, "b", "rb", now - 60, "opencode", "glm", 2.0)
        _ins_pi_event(self.conn, "c", "rc", now - 60, "anthropic",
                      "claude-opus-4-8", 9.0)
        costs = _month_reported_costs(self.conn, now, "personal")
        self.assertEqual(costs, {"codex": 3.0, "opencode-go": 0,
                                 "opencode-zen": 2.0})
