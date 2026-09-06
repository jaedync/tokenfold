"""Fix 1: /api/rate-limits week_cost and hourly_costs must honour fast/geo pricing."""

import json
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


class RateLimitsClaudeWindowsExcludeOtherProvidersTest(TempDBTestCase):
    """Personal-scope Claude gauge windows must count Anthropic usage only.

    limit_window.cost / five_hour_window.cost (and the rolling week_cost
    fallback) previously summed every personal-scope assistant event,
    including pi-agent rows with reported costs from openai-codex,
    opencode-go, and openrouter — inflating "spent \u00b7 this window" on the
    Claude gauges.
    """

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def _ins(self, uuid, req, *, source_client="claude-code", provider=None,
             model="claude-opus-4-8", inp=1_000_000, reported=0.0, ts=None):
        if ts is None:
            ts = time.time() - 100  # inside rolling-7d and both limit windows
        self.conn.execute(
            "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,"
            "request_id,source_machine,project_dir,model,is_sidechain,agent_id,"
            "input_tokens,output_tokens,cache_creation_tokens,cache_read_tokens,"
            "speed,inference_geo,account_email,plan,org_name,source_client,"
            "provider,reported_cost_total) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (uuid, "assistant", "2026-09-02T12:00:00Z", ts, "2026-09-02",
             "s", req, "m", "proj", model, 0, None, inp, 0, 0, 0, None,
             None, "jaedyn@personal.io", "max", "Personal", source_client,
             provider, reported),
        )
        self.conn.commit()

    def _seed_oauth(self):
        now = time.time()
        payload = {
            "data": {
                "five_hour": {
                    "resets_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ",
                        time.gmtime(now + 3600)),
                    "utilization": 40.0,
                },
                "seven_day": {
                    "resets_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ",
                        time.gmtime(now + 3600)),
                    "utilization": 50.0,
                },
            },
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        }
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("oauth_usage", json.dumps(payload)))
        self.conn.commit()

    def test_claude_gauge_windows_exclude_other_providers(self):
        import app.aggregator as agg
        agg._cached_data.clear()
        self._seed_oauth()
        self._ins("e1", "re1")  # claude-code, $5 server-priced
        self._ins("e2", "re2", source_client="pi-agent",
                  provider="openai-codex", model="gpt-5-codex",
                  reported=3.0)
        self._ins("e3", "re3", source_client="pi-agent",
                  provider="opencode-go", reported=2.0)

        c = self.client()
        body = c.get("/api/rate-limits?scope=personal").json()
        wb = body["weekly_budget"]
        self.assertAlmostEqual(wb["week_cost"], 5.0, places=2,
                               msg="week_cost must be Claude-only")
        self.assertAlmostEqual(
            wb["oauth"]["limit_window"]["cost"], 5.0, places=2,
            msg="weekly 'spent this window' must be Claude-only")
        self.assertAlmostEqual(
            wb["oauth"]["five_hour_window"]["cost"], 5.0, places=2,
            msg="5h 'spent this window' must be Claude-only")
        total_hourly = round(sum(h["c"] for h in wb["hourly_costs"]), 2)
        self.assertAlmostEqual(
            total_hourly, 5.0, places=2,
            msg="hourly chart (weekly gauge sparkline) must be Claude-only")

    def test_claude_windows_still_count_pi_anthropic(self):
        import app.aggregator as agg
        agg._cached_data.clear()
        self._seed_oauth()
        self._ins("e1", "re1")  # claude-code, $5
        self._ins("e2", "re2", source_client="pi-agent",
                  provider="anthropic", reported=1.25)
        c = self.client()
        wb = c.get("/api/rate-limits?scope=personal").json()["weekly_budget"]
        self.assertAlmostEqual(wb["week_cost"], 6.25, places=2)
        self.assertAlmostEqual(
            wb["oauth"]["limit_window"]["cost"], 6.25, places=2)
        self.assertAlmostEqual(
            wb["oauth"]["five_hour_window"]["cost"], 6.25, places=2)


if __name__ == "__main__":
    unittest.main()
