"""Fix 1: /api/rate-limits week_cost and hourly_costs must honour fast/geo pricing."""

import time
import unittest

from app.tests._support import TempDBTestCase


def _ins_event(conn, uuid, req, acct, plan, org, machine, session,
               model="claude-opus-4-8", inp=0, out=0, speed=None, geo=None):
    """Insert a single assistant event with optional speed/geo."""
    ts = time.time() - 3600  # within rolling-7d window
    conn.execute(
        "INSERT INTO events("
        "uuid,type,timestamp,ts_epoch,day,session_id,request_id,"
        "source_machine,project_dir,model,is_sidechain,agent_id,"
        "input_tokens,output_tokens,cache_creation_tokens,cache_read_tokens,"
        "account_email,plan,org_name,speed,inference_geo"
        ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (uuid, "assistant", "2026-06-09T12:00:00Z", ts, "2026-06-09",
         session, req, machine, "proj", model, 0, None,
         inp, out, 0, 0, acct, plan, org, speed, geo),
    )
    conn.commit()


class RateLimitsFastPricingTest(TempDBTestCase):
    """week_cost and hourly_costs must apply fast-Opus multiplier."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_week_cost_fast_opus_is_10_not_5(self):
        """fast Opus 4.8, 1M input → $10, NOT $5 (the base rate)."""
        _ins_event(
            self.conn, "e1", "re1",
            "jaedyn@acme.io", "enterprise", "Acme",
            "hpc1", "sE",
            model="claude-opus-4-8",
            inp=1_000_000,
            speed="fast",
        )
        import app.aggregator as agg
        agg._cached_data.clear()
        c = self.client()
        rl = c.get("/api/rate-limits").json()["weekly_budget"]
        self.assertAlmostEqual(
            rl["week_cost"], 10.0, places=2,
            msg=f"expected 10.0 (fast-Opus rate), got {rl['week_cost']}")

    def test_hourly_costs_fast_opus_is_10_not_5(self):
        """hourly_costs must also use fast pricing; total across all buckets == 10."""
        _ins_event(
            self.conn, "e1", "re1",
            "jaedyn@acme.io", "enterprise", "Acme",
            "hpc1", "sE",
            model="claude-opus-4-8",
            inp=1_000_000,
            speed="fast",
        )
        c = self.client()
        rl = c.get("/api/rate-limits").json()["weekly_budget"]
        total_hourly = sum(h["c"] for h in rl["hourly_costs"])
        self.assertAlmostEqual(
            total_hourly, 10.0, places=2,
            msg=f"hourly sum expected 10.0 (fast-Opus), got {total_hourly}")


if __name__ == "__main__":
    unittest.main()
