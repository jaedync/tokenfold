import json
import time
from datetime import datetime, timezone

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


class EnterpriseOnlyGateTest(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_zero_consumer_bleedover(self):
        # Enterprise: 1M input Opus 4.8 = $5
        ins(self.conn, "e1", "re1", "jaedyn@acme.io", "enterprise", "Acme",
            "acme-hpc1", "acme-portal", "sE", inp=1_000_000)
        # Consumer: 2M input = $10 — must NEVER appear anywhere
        ins(self.conn, "c1", "rc1", "me@gmail.com", "max", None,
            "personal-mbp", "secret-side-project", "sC", inp=2_000_000,
            ts=1781000100.0)
        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data = None
        d = agg.build_dashboard_data()
        blob = json.dumps(d)
        # ZERO consumer bleedover: none of these strings anywhere in the payload
        self.assertNotIn("me@gmail.com", blob)
        self.assertNotIn("personal-mbp", blob)
        self.assertNotIn("secret-side-project", blob)
        self.assertNotIn('"max"', blob)  # consumer plan label shouldn't surface
        # Enterprise IS present, and totals are enterprise-only ($5, not blended $15)
        self.assertIn("acme-hpc1", blob)
        self.assertAlmostEqual(sum(m["cost"] for m in d["model_breakdown"]), 5.0, places=2)
        self.assertAlmostEqual(sum(x["cost"] for x in d["daily"]), 5.0, places=2)

    def test_no_enterprise_data_is_empty_not_crash(self):
        ins(self.conn, "c1", "rc1", "me@gmail.com", "max", None,
            "personal-mbp", "proj", "sC", inp=1_000_000)
        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data = None
        d = agg.build_dashboard_data()  # only consumer data -> enterprise view empty, no crash
        self.assertAlmostEqual(sum(x["cost"] for x in d["daily"]), 0.0, places=2)
        self.assertNotIn("me@gmail.com", json.dumps(d))

    def _seed_oauth_usage(self, resets_epoch):
        """Insert an oauth_usage meta row so /api/rate-limits and /api/ha
        populate their windowed spend (both need seven_day.resets_at)."""
        resets_iso = datetime.fromtimestamp(
            resets_epoch, tz=timezone.utc).isoformat()
        stored = {
            "data": {
                "seven_day": {"utilization": 50, "resets_at": resets_iso},
                "five_hour": {"utilization": 50, "resets_at": resets_iso},
            },
            "updated_at": resets_iso,
        }
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES('oauth_usage', ?)",
            (json.dumps(stored),))
        self.conn.commit()

    def test_routes_exclude_consumer_spend(self):
        # Events in the CURRENT week window (rolling-7d window; no oauth row needed).
        now = time.time()
        # enterprise: 1M input Opus 4.8 = $5
        ins(self.conn, "e1", "re1", "jaedyn@acme.io", "enterprise", "Acme",
            "acme-hpc1", "acme-portal", "sE", inp=1_000_000, ts=now - 3600)
        # consumer: 2M input = $10 — must NOT appear in any route's spend
        ins(self.conn, "c1", "rc1", "me@gmail.com", "max", None,
            "personal-mbp", "secret", "sC", inp=2_000_000, ts=now - 1800)
        # oauth row still needed for /api/ha (ha.py reads it directly)
        self._seed_oauth_usage(now + 3600)
        self.conn.commit()
        import app.aggregator as agg
        agg._cached_data = None
        c = self.client()  # NOTE: NO `with` — lifespan triggers usage_fetcher network/segfault
        rl = c.get("/api/rate-limits").json()["weekly_budget"]
        # week_cost must be enterprise-only $5, not blended $15
        self.assertAlmostEqual(rl["week_cost"], 5.0, places=2)
        ha = c.get("/api/ha").json()
        # five_hour + weekly spend must be enterprise-only too
        self.assertAlmostEqual(ha["weekly"]["spend_usd"], 5.0, places=2)
        self.assertAlmostEqual(ha["five_hour"]["spend_usd"], 5.0, places=2)

    def test_rate_limits_works_without_oauth_row(self):
        """Decoupling: /api/rate-limits must return enterprise week_cost
        even when no oauth_usage meta row exists at all."""
        now = time.time()
        ins(self.conn, "e1", "re1", "jaedyn@acme.io", "enterprise", "Acme",
            "acme-hpc1", "acme-portal", "sE", inp=1_000_000, ts=now - 3600)
        ins(self.conn, "c1", "rc1", "me@gmail.com", "max", None,
            "personal-mbp", "secret", "sC", inp=2_000_000, ts=now - 1800)
        # No _seed_oauth_usage here — that's the whole point
        self.conn.commit()
        import app.aggregator as agg
        agg._cached_data = None
        c = self.client()
        rl = c.get("/api/rate-limits").json()["weekly_budget"]
        self.assertIsNotNone(rl, "weekly_budget should not be None when no oauth row")
        self.assertAlmostEqual(rl["week_cost"], 5.0, places=2)

    def test_rate_limits_no_personal_gauge_fields(self):
        """Response must contain NONE of the personal-Max gauge keys."""
        now = time.time()
        ins(self.conn, "e1", "re1", "jaedyn@acme.io", "enterprise", "Acme",
            "acme-hpc1", "acme-portal", "sE", inp=1_000_000, ts=now - 3600)
        self.conn.commit()
        c = self.client()
        body = c.get("/api/rate-limits").json()
        wb = body.get("weekly_budget", {})
        for forbidden in ("weekly_pct", "five_hour_pct", "weekly_resets_at",
                          "five_hour_resets_at", "opus_pct", "sonnet_pct",
                          "extra_usage"):
            self.assertNotIn(forbidden, wb,
                             f"personal gauge field '{forbidden}' must not appear in response")

    def test_aggregator_exposes_org_name_and_plan_scope(self):
        """_build_dashboard_data_inner must include org_name and plan_scope."""
        ins(self.conn, "e1", "re1", "jaedyn@acme.io", "enterprise", "Acme",
            "acme-hpc1", "acme-portal", "sE", inp=1_000_000)
        ins(self.conn, "c1", "rc1", "me@gmail.com", "max", None,
            "personal-mbp", "secret-side-project", "sC", inp=2_000_000,
            ts=1781000100.0)
        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data = None
        d = agg.build_dashboard_data()
        self.assertEqual(d["org_name"], "Acme")
        self.assertEqual(d["plan_scope"], "enterprise")

    def test_aggregator_org_name_empty_when_consumer_only(self):
        """When only consumer data exists, org_name must be empty string."""
        ins(self.conn, "c1", "rc1", "me@gmail.com", "max", None,
            "personal-mbp", "proj", "sC", inp=1_000_000)
        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data = None
        d = agg.build_dashboard_data()
        self.assertEqual(d["org_name"], "")
        self.assertEqual(d["plan_scope"], "enterprise")

    def test_dashboard_html_enterprise_badge_no_consumer_strings(self):
        """Dashboard HTML must show ENTERPRISE + org name; no consumer strings;
        no personal-gauge JS field names."""
        now = time.time()
        ins(self.conn, "e1", "re1", "jaedyn@acme.io", "enterprise", "Acme",
            "acme-hpc1", "acme-portal", "sE", inp=1_000_000, ts=now - 3600)
        ins(self.conn, "c1", "rc1", "me@gmail.com", "max", None,
            "personal-mbp", "secret-side-project", "sC", inp=2_000_000,
            ts=now - 1800)
        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data = None
        c = self.client()
        html = c.get("/").text
        # Load-bearing badge assertions: the band markup and its rendered text.
        # Plain "ENTERPRISE"/"Acme" substrings could be satisfied by an HTML
        # comment or the embedded data_json even if the badge broke.
        self.assertIn("header-enterprise-band", html)
        self.assertIn("ENTERPRISE · Acme", html)
        self.assertNotIn("me@gmail.com", html)
        self.assertNotIn("personal-mbp", html)
        self.assertNotIn("secret-side-project", html)
        # personal-gauge field names must not appear in served JS
        self.assertNotIn("weekly_pct", html)
        self.assertNotIn("opus_pct", html)
