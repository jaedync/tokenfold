import json

from app.tests._support import TempDBTestCase


def insert_assistant(conn, uuid, req, model="claude-opus-4-8", day="2026-06-09",
                     ts=1781000000.0, inp=0, out=0, cc=0, cr=0, speed=None, geo=None,
                     machine="m", sidechain=0, agent_id=None):
    conn.execute(
        "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,request_id,"
        "source_machine,project_dir,model,is_sidechain,agent_id,"
        "input_tokens,output_tokens,cache_creation_tokens,cache_read_tokens,"
        "speed,inference_geo) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (uuid, "assistant", "2026-06-09T12:00:00Z", ts, day, "s", req, machine,
         "proj", model, sidechain, agent_id, inp, out, cc, cr, speed, geo),
    )
    conn.commit()


class SummarizerFastGeoTest(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_prices_fast_per_request_and_stores_model_cost(self):
        # Opus 4.8, 1M input each: normal $5 + fast $10 = $15 day total.
        insert_assistant(self.conn, "u1", "r1", inp=1_000_000)
        insert_assistant(self.conn, "u2", "r2", inp=1_000_000, speed="fast", ts=1781000100.0)
        from app.summarizer import summarize_days
        summarize_days(["2026-06-09"])
        row = self.conn.execute(
            "SELECT cost, model_json FROM daily_summary WHERE day=?",
            ("2026-06-09",)).fetchone()
        self.assertAlmostEqual(row["cost"], 15.0, places=2)
        md = json.loads(row["model_json"])["Opus 4.8"]
        self.assertAlmostEqual(md["cost"], 15.0, places=2)
