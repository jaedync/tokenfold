import json

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
