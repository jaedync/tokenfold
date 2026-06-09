"""Tests for scope-aware routes: /api/stats, /api/rate-limits, and / (dashboard).

Covers:
- ?scope=personal returns personal totals
- ?scope=bogus -> 400
- With LOCKED_SCOPE='enterprise': ?scope=personal -> 403, unscoped -> enterprise
- Dashboard page: soft-fail (no 403 on bad scope), badge shows scope label not org name
"""

import json
import time
import unittest
from unittest.mock import patch

from app.tests._support import TempDBTestCase


def ins(conn, uuid, req, acct, plan, org, machine, project, session,
        model="claude-opus-4-8", day="2026-06-09", ts=1781000000.0, inp=0, out=0):
    conn.execute(
        "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,request_id,"
        "source_machine,project_dir,model,is_sidechain,agent_id,input_tokens,"
        "output_tokens,cache_creation_tokens,cache_read_tokens,account_email,plan,"
        "org_name,is_human_prompt,user_type) VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (uuid, "assistant", "2026-06-09T12:00:00Z", ts, day, session, req, machine,
         project, model, 0, None, inp, out, 0, 0, acct, plan, org, 0, None))
    conn.commit()


class ApiStatsScopeTest(TempDBTestCase):
    """GET /api/stats?scope= routing tests."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_api_stats_default_returns_enterprise(self):
        """Unscoped /api/stats returns enterprise data."""
        ins(self.conn, "eA", "rA", "a@acme.io", "enterprise", "Acme",
            "mA", "proj-a", "sA", inp=1_000_000, ts=1781000000.0)
        ins(self.conn, "pB", "rB", "b@personal.io", "max", None,
            "mB", "proj-b", "sB", inp=1_000_000, ts=1781000100.0)
        from app.summarizer import summarize_days
        summarize_days(None)
        c = self.client()
        resp = c.get("/api/stats")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["scope"], "enterprise")
        self.assertAlmostEqual(data["total_cost"], 5.0, places=2)

    def test_api_stats_personal_scope_returns_personal(self):
        """?scope=personal returns personal totals."""
        ins(self.conn, "eA", "rA", "a@acme.io", "enterprise", "Acme",
            "mA", "proj-a", "sA", inp=1_000_000, ts=1781000000.0)
        ins(self.conn, "pB", "rB", "b@personal.io", "max", None,
            "mB", "proj-b", "sB", model="claude-sonnet-4-6", inp=1_000_000, ts=1781000100.0)
        from app.summarizer import summarize_days
        summarize_days(None)
        c = self.client()
        resp = c.get("/api/stats?scope=personal")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["scope"], "personal")
        self.assertAlmostEqual(data["total_cost"], 3.0, places=2)

    def test_api_stats_bogus_scope_returns_400(self):
        """?scope=bogus must return 400."""
        c = self.client()
        resp = c.get("/api/stats?scope=bogus")
        self.assertEqual(resp.status_code, 400)

    def test_api_stats_locked_enterprise_blocks_personal(self):
        """With TOKENFOLD_SCOPE=enterprise locked, ?scope=personal must return 403."""
        import app.config as cfg
        with patch.object(cfg, "LOCKED_SCOPE", "enterprise"):
            c = self.client()
            resp = c.get("/api/stats?scope=personal")
            self.assertEqual(resp.status_code, 403)

    def test_api_stats_locked_enterprise_unscoped_returns_enterprise(self):
        """With TOKENFOLD_SCOPE=enterprise, unscoped request returns enterprise data."""
        ins(self.conn, "eA", "rA", "a@acme.io", "enterprise", "Acme",
            "mA", "proj-a", "sA", inp=1_000_000, ts=1781000000.0)
        from app.summarizer import summarize_days
        summarize_days(None)
        import app.config as cfg
        with patch.object(cfg, "LOCKED_SCOPE", "enterprise"):
            c = self.client()
            resp = c.get("/api/stats")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["scope"], "enterprise")


class ApiRateLimitsScopeTest(TempDBTestCase):
    """GET /api/rate-limits?scope= routing tests."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_rate_limits_bogus_scope_returns_400(self):
        """?scope=bogus on /api/rate-limits must return 400."""
        c = self.client()
        resp = c.get("/api/rate-limits?scope=bogus")
        self.assertEqual(resp.status_code, 400)

    def test_rate_limits_locked_blocks_wrong_scope(self):
        """Locked instance returns 403 when different scope requested."""
        import app.config as cfg
        with patch.object(cfg, "LOCKED_SCOPE", "enterprise"):
            c = self.client()
            resp = c.get("/api/rate-limits?scope=personal")
            self.assertEqual(resp.status_code, 403)

    def test_rate_limits_personal_scope_returns_personal_spend(self):
        """?scope=personal on /api/rate-limits returns personal-only week_cost."""
        now = time.time()
        ins(self.conn, "eA", "rA", "a@acme.io", "enterprise", "Acme",
            "mA", "proj-a", "sA", inp=1_000_000, ts=now - 3600)
        ins(self.conn, "pB", "rB", "b@personal.io", "max", None,
            "mB", "proj-b", "sB", model="claude-sonnet-4-6", inp=1_000_000, ts=now - 1800)
        c = self.client()
        rl = c.get("/api/rate-limits?scope=personal").json()["weekly_budget"]
        # Personal scope: only Sonnet 4.6 $3, not enterprise Opus $5
        self.assertAlmostEqual(rl["week_cost"], 3.0, places=2)


class DashboardScopeTest(TempDBTestCase):
    """GET / dashboard page scope tests."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def _seed_both(self, now):
        ins(self.conn, "eA", "rA", "a@acme.io", "enterprise", "Acme",
            "acme-hpc1", "acme-portal", "sA", inp=1_000_000, ts=now - 3600)
        ins(self.conn, "pB", "rB", "b@personal.io", "max", None,
            "personal-mbp", "home-proj", "sB", inp=1_000_000, ts=now - 1800)
        from app.summarizer import summarize_days
        summarize_days(None)

    def test_dashboard_default_shows_enterprise_badge(self):
        """Default / shows ENTERPRISE scope label in badge; no org name."""
        now = time.time()
        self._seed_both(now)
        c = self.client()
        html = c.get("/").text
        self.assertIn("header-enterprise-band", html)
        self.assertIn("ENTERPRISE", html)
        self.assertNotIn("Acme", html, "org name must not be rendered")

    def test_dashboard_personal_scope_shows_personal_badge(self):
        """?scope=personal shows PERSONAL scope label."""
        now = time.time()
        self._seed_both(now)
        c = self.client()
        html = c.get("/?scope=personal").text
        self.assertIn("PERSONAL", html)

    def test_dashboard_bad_scope_falls_back_no_403(self):
        """?scope=bogus (bad value) must NOT return 403 — soft-fail to default."""
        c = self.client()
        resp = c.get("/?scope=bogus")
        self.assertEqual(resp.status_code, 200)

    def test_dashboard_locked_enterprise_ignores_personal_param(self):
        """With lock=enterprise, /?scope=personal must still serve 200 with ENTERPRISE badge."""
        now = time.time()
        self._seed_both(now)
        import app.config as cfg
        with patch.object(cfg, "LOCKED_SCOPE", "enterprise"):
            c = self.client()
            resp = c.get("/?scope=personal")
            self.assertEqual(resp.status_code, 200)
            self.assertIn("ENTERPRISE", resp.text)


if __name__ == "__main__":
    unittest.main()
