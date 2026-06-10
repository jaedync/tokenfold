"""Per-session burn-rate rollup for the dashboard.

build_dashboard_data exposes `recent_sessions`: the last RECENCY_DAYS of
sessions with cost, tokens, wall duration, burn rate ($/hr), and a human
title joined from desktop_sessions (Claude Desktop pushes those); CLI
sessions fall back to their project name. Scope-partitioned like
everything else — an enterprise session must never appear in personal.
"""

import unittest

from app.tests._support import TempDBTestCase


NOW = 1781000000.0  # within the recent window when paired with day below


class SessionBurnTest(TempDBTestCase):

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def _ins(self, uuid, session, ts, model="claude-opus-4-8", inp=1_000_000,
             acct=None, org_type=None, machine="mac1", project="projA"):
        self.conn.execute(
            "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,request_id,"
            "source_machine,project_dir,model,is_sidechain,agent_id,input_tokens,"
            "output_tokens,cache_creation_tokens,cache_read_tokens,is_human_prompt,"
            "account_email,org_type) "
            "VALUES(?,'assistant','2026-06-09T12:00:00Z',?,'2026-06-09',?,?,?,?,?,0,"
            "NULL,?,1000,0,0,0,?,?)",
            (uuid, ts, session, "r-" + uuid, machine, project, model, inp,
             acct, org_type),
        )
        self.conn.commit()

    def _title(self, session, title):
        self.conn.execute(
            "INSERT INTO desktop_sessions(cli_session_id, source_machine, title, "
            "updated_at_ms) VALUES(?, 'mac1', ?, 0)", (session, title))
        self.conn.commit()

    def _build(self, scope="personal"):
        from app.summarizer import summarize_days
        from app.aggregator import build_dashboard_data
        summarize_days(["2026-06-09"])
        return build_dashboard_data(scope)

    def test_session_rollup_with_burn_rate(self):
        # two requests 30 min apart -> 0.5h wall duration
        self._ins("u1", "sessA", NOW)
        self._ins("u2", "sessA", NOW + 1800)
        data = self._build()
        sess = {s["session_id"]: s for s in data["recent_sessions"]}
        self.assertIn("sessA", sess)
        s = sess["sessA"]
        self.assertGreater(s["cost"], 0)
        self.assertEqual(s["machine"], "mac1")
        self.assertEqual(s["project"], "projA")
        self.assertGreater(s["total_tokens"], 0)
        self.assertAlmostEqual(s["duration_s"], 1800, delta=1)
        # burn = cost / wall hours (0.5h here)
        self.assertAlmostEqual(s["burn_per_hr"], s["cost"] / 0.5, places=2)

    def test_title_joined_from_desktop_sessions(self):
        self._ins("u1", "sessA", NOW)
        self._title("sessA", "Fix the frobnicator")
        data = self._build()
        s = [x for x in data["recent_sessions"] if x["session_id"] == "sessA"][0]
        self.assertEqual(s["title"], "Fix the frobnicator")

    def test_untitled_session_has_null_title(self):
        self._ins("u1", "sessA", NOW)
        data = self._build()
        s = [x for x in data["recent_sessions"] if x["session_id"] == "sessA"][0]
        self.assertIsNone(s["title"])

    def test_scope_partition_respected(self):
        self._ins("u1", "persSess", NOW)
        self._ins("u2", "entSess", NOW, acct="who@corp.com", org_type="claude_enterprise")
        per = self._build("personal")
        ent = self._build("enterprise")
        per_ids = {s["session_id"] for s in per["recent_sessions"]}
        ent_ids = {s["session_id"] for s in ent["recent_sessions"]}
        self.assertIn("persSess", per_ids)
        self.assertNotIn("entSess", per_ids)
        self.assertIn("entSess", ent_ids)
        self.assertNotIn("persSess", ent_ids)

    def test_sorted_by_last_activity_desc(self):
        self._ins("u1", "older", NOW - 7200)
        self._ins("u2", "newer", NOW)
        data = self._build()
        ids = [s["session_id"] for s in data["recent_sessions"]]
        self.assertLess(ids.index("newer"), ids.index("older"))


class TemplateSessionsUITest(unittest.TestCase):
    def test_template_renders_sessions_section(self):
        from pathlib import Path
        tpl = (Path(__file__).resolve().parents[2] / "templates" / "dashboard.html").read_text()
        self.assertIn("recent_sessions", tpl)
        self.assertIn("sessionsBody", tpl)
        self.assertIn("burn_per_hr", tpl)


if __name__ == "__main__":
    unittest.main()


class AiTitleCaptureTest(TempDBTestCase):
    """Claude Code writes {"type":"ai-title","aiTitle":...,"sessionId":...}
    transcript records (no uuid/timestamp — the event extractor skips them).
    Ingest must upsert them into session_titles, and the sessions rollup must
    use them, with explicit desktop_sessions titles taking precedence."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def _ingest(self, events):
        c = self.client()
        r = c.post("/api/ingest", json={
            "machine": "mac1", "project_dir": "projA", "session_file": "s.jsonl",
            "cursor": {"last_line_num": 0}, "events": events,
        }, headers={"X-API-Key": self.api_key})
        self.assertEqual(r.status_code, 200)

    def _assistant(self, uuid, session):
        return {
            "uuid": uuid, "type": "assistant", "timestamp": "2026-06-09T12:00:00Z",
            "sessionId": session, "requestId": "r-" + uuid,
            "message": {"model": "claude-opus-4-8", "id": "m1",
                        "usage": {"input_tokens": 1000000, "output_tokens": 1000}},
        }

    def test_ai_title_upserted_and_used(self):
        self._ingest([
            self._assistant("u1", "sessA"),
            {"type": "ai-title", "aiTitle": "Fix the frobnicator", "sessionId": "sessA"},
        ])
        row = self.conn.execute(
            "SELECT title FROM session_titles WHERE session_id='sessA'").fetchone()
        self.assertEqual(row["title"], "Fix the frobnicator")
        from app.summarizer import summarize_days
        from app.aggregator import build_dashboard_data
        summarize_days(["2026-06-09"])
        s = [x for x in build_dashboard_data("personal")["recent_sessions"]
             if x["session_id"] == "sessA"][0]
        self.assertEqual(s["title"], "Fix the frobnicator")

    def test_later_ai_title_wins(self):
        self._ingest([
            self._assistant("u1", "sessA"),
            {"type": "ai-title", "aiTitle": "old name", "sessionId": "sessA"},
            {"type": "ai-title", "aiTitle": "new name", "sessionId": "sessA"},
        ])
        row = self.conn.execute(
            "SELECT title FROM session_titles WHERE session_id='sessA'").fetchone()
        self.assertEqual(row["title"], "new name")

    def test_desktop_title_beats_ai_title(self):
        self._ingest([
            self._assistant("u1", "sessA"),
            {"type": "ai-title", "aiTitle": "ai name", "sessionId": "sessA"},
        ])
        self.conn.execute(
            "INSERT INTO desktop_sessions(cli_session_id, source_machine, title, "
            "updated_at_ms) VALUES('sessA', 'mac1', 'human name', 0)")
        self.conn.commit()
        from app.summarizer import summarize_days
        from app.aggregator import build_dashboard_data
        import app.aggregator as _agg
        summarize_days(["2026-06-09"])
        _agg._cached_data.clear()
        s = [x for x in build_dashboard_data("personal")["recent_sessions"]
             if x["session_id"] == "sessA"][0]
        self.assertEqual(s["title"], "human name")

    def test_malformed_ai_title_ignored(self):
        self._ingest([
            self._assistant("u1", "sessA"),
            {"type": "ai-title", "aiTitle": {"nested": "junk"}, "sessionId": "sessA"},
            {"type": "ai-title", "aiTitle": "ok", "sessionId": None},
        ])
        n = self.conn.execute("SELECT COUNT(*) c FROM session_titles").fetchone()["c"]
        self.assertEqual(n, 0)
