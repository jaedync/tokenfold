from app.tests._support import TempDBTestCase


def ins(conn, uuid, req, acct, plan, org, model="claude-opus-4-8", day="2026-06-09",
        ts=1781000000.0, inp=0, out=0, machine="m", session="s"):
    conn.execute(
        "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,request_id,"
        "source_machine,project_dir,model,is_sidechain,agent_id,input_tokens,"
        "output_tokens,cache_creation_tokens,cache_read_tokens,account_email,plan,org_name) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (uuid, "assistant", "2026-06-09T12:00:00Z", ts, day, session, req, machine,
         "proj", model, 0, None, inp, out, 0, 0, acct, plan, org))
    conn.commit()


class RollupSchemaTest(TempDBTestCase):
    def test_daily_summary_shape(self):
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(daily_summary)")}
        self.assertTrue({"day", "account_email", "plan", "org_name"} <= cols)
        self.assertNotIn("account_json", cols)


class RollupPerAccountTest(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_one_row_per_account_with_plan_and_cost(self):
        ins(self.conn, "u1", "r1", "jaedyn@acme.io", "enterprise", "Acme",
            inp=1_000_000, machine="hpc1", session="sE")
        ins(self.conn, "u2", "r2", "me@gmail.com", "max", None,
            inp=1_000_000, machine="mb", session="sC", ts=1781000100.0)
        from app.summarizer import summarize_days
        summarize_days(["2026-06-09"])
        rows = {r["account_email"]: r for r in self.conn.execute(
            "SELECT * FROM daily_summary WHERE day='2026-06-09'")}
        self.assertEqual(set(rows), {"jaedyn@acme.io", "me@gmail.com"})
        self.assertAlmostEqual(rows["jaedyn@acme.io"]["cost"], 5.0, places=2)
        self.assertAlmostEqual(rows["me@gmail.com"]["cost"], 5.0, places=2)
        self.assertEqual(rows["jaedyn@acme.io"]["plan"], "enterprise")
        self.assertEqual(rows["jaedyn@acme.io"]["org_name"], "Acme")
        self.assertEqual(rows["me@gmail.com"]["plan"], "max")

    def test_active_time_attributed_per_account(self):
        # enterprise session: two main events 120s apart -> active_s ~= 120
        ins(self.conn, "e1", "re1", "ent@x", "enterprise", "Org", inp=100,
            machine="hpc1", session="sE", ts=1781000000.0)
        ins(self.conn, "e2", "re2", "ent@x", "enterprise", "Org", inp=100,
            machine="hpc1", session="sE", ts=1781000120.0)
        # consumer session: two main events 60s apart -> active_s ~= 60
        ins(self.conn, "c1", "rc1", "con@x", "max", None, inp=100,
            machine="mb", session="sC", ts=1781000000.0)
        ins(self.conn, "c2", "rc2", "con@x", "max", None, inp=100,
            machine="mb", session="sC", ts=1781000060.0)
        from app.summarizer import summarize_days
        summarize_days(["2026-06-09"])
        rows = {r["account_email"]: r for r in self.conn.execute(
            "SELECT account_email, active_s FROM daily_summary WHERE day='2026-06-09'")}
        self.assertAlmostEqual(rows["ent@x"]["active_s"], 120.0, delta=1.0)
        self.assertAlmostEqual(rows["con@x"]["active_s"], 60.0, delta=1.0)

    def test_no_prev_day_bleedover_across_accounts(self):
        # Consumer event on the PREVIOUS day sharing a session_id with an enterprise
        # session on the target day must NOT inflate the enterprise active-time.
        ins(self.conn, "c0", "rc0", "con@x", "max", None, inp=100,
            machine="mb", session="shared", day="2026-06-08", ts=1780999880.0)
        ins(self.conn, "e1", "re1", "ent@x", "enterprise", "Org", inp=100,
            machine="hpc1", session="shared", day="2026-06-09", ts=1781000000.0)
        ins(self.conn, "e2", "re2", "ent@x", "enterprise", "Org", inp=100,
            machine="hpc1", session="shared", day="2026-06-09", ts=1781000060.0)
        from app.summarizer import summarize_days
        summarize_days(["2026-06-09"])  # prev_day = 2026-06-08
        row = self.conn.execute(
            "SELECT active_s FROM daily_summary "
            "WHERE day='2026-06-09' AND account_email='ent@x'").fetchone()
        # Only the legit e1->e2 gap (60s); the consumer's prev-day event must not leak in.
        self.assertAlmostEqual(row["active_s"], 60.0, delta=1.0)
