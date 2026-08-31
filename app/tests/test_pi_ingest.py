"""Pi Agent normalized ingest contract and aggregation tests."""
import json
import unittest
from datetime import datetime, timezone

from app.tests._support import TempDBTestCase


class PiIngestTest(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def event(self, eid="e1", sid="session", **extra):
        out = {
            "event_id": eid, "timestamp": "2026-06-09T12:00:00Z",
            "session_id": sid, "kind": "assistant", "provider": "openai",
            "api": "responses", "model": "gpt-4o", "request_id": "req",
            "usage": {"input": 10, "output": 20, "cache_read": 2,
                       "cache_write": 3, "reasoning": 4,
                       "cost_input": 1.0, "cost_output": 2.0,
                       "cost_cache_read": .1, "cost_cache_write": .2,
                       "cost_total": 3.3},
        }
        out.update(extra)
        return out

    def post(self, events, **envelope):
        body = {"machine": "m", "account_class": "personal", "project_dir": "/p",
                "session_file": "pi.jsonl", "cursor": {"last_line_num": 0},
                "events": events}
        body.update(envelope)
        return self.client().post("/api/ingest/pi", json=body,
                                  headers={"X-API-Key": self.api_key})

    def test_auth(self):
        r = self.client().post("/api/ingest/pi", json={})
        self.assertEqual(r.status_code, 401)

    def test_mapping_and_namespace(self):
        events = [self.event(tools=[{"tool_use_id": "tool", "name": "shell"}]),
                  {"event_id": "u", "timestamp": "2026-06-09T12:01:00Z",
                   "session_id": "session", "kind": "user", "has_text": True,
                   "text_length": 4},
                  {"event_id": "tr", "timestamp": "2026-06-09T12:02:00Z",
                   "session_id": "session", "kind": "tool_result",
                   "has_tool_result": True}]
        r = self.post(events)
        self.assertEqual(r.status_code, 200, r.text)
        rows = self.conn.execute("SELECT * FROM events ORDER BY timestamp").fetchall()
        self.assertEqual([x["type"] for x in rows], ["assistant", "user", "user"])
        self.assertTrue(rows[0]["uuid"].startswith("pi:"))
        self.assertEqual(rows[0]["source_client"], "pi-agent")
        self.assertEqual(rows[0]["account_email"], "pi-personal@dotfleet.local")
        self.assertEqual(rows[0]["usage_kind"], "assistant")
        self.assertEqual(rows[0]["reasoning_tokens"], 4)
        self.assertEqual(rows[2]["has_tool_result"], 1)
        tool = self.conn.execute("SELECT * FROM tool_uses").fetchone()
        self.assertTrue(tool["tool_use_id"].startswith("pi:"))

    def test_replay_and_fallback_request_id(self):
        e = self.event(request_id=None)
        first = self.post([e]); second = self.post([e])
        self.assertEqual(first.json()["accepted"], 1)
        self.assertEqual(second.json()["duplicates"], 1)
        row = self.conn.execute("SELECT request_id FROM events").fetchone()
        self.assertIn("event:e1", row["request_id"])

    def test_native_ids_are_collision_safe(self):
        e1, e2 = self.event(), self.event()
        a = self.post([e1], machine="m1", session_file="a")
        b = self.post([e2], machine="m2", session_file="b")
        self.assertEqual(a.json()["accepted"], 1)
        self.assertEqual(b.json()["accepted"], 1)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0], 2)

    def test_provider_models_and_reported_zero_cost(self):
        a = self.event(eid="a", provider="openai", model="same", request_id="a",
                       usage={"input": 1, "output": 1, "cost_total": 0})
        b = self.event(eid="b", sid="s-google", provider="google", model="same", request_id="b",
                       usage={"input": 1, "output": 1, "cost_total": 4})
        follow = {"event_id": "follow", "timestamp": "2026-06-09T12:01:00Z",
                  "session_id": "session", "kind": "user"}
        self.assertEqual(self.post([a, b, follow]).status_code, 200)
        from app.summarizer import summarize_days
        summarize_days(["2026-06-09"])
        row = self.conn.execute("SELECT cost, model_json FROM daily_summary").fetchone()
        self.assertEqual(row["cost"], 4)
        models = json.loads(row["model_json"])
        self.assertEqual(set(models), {"openai/Same", "google/Same"})
        self.assertGreater(models["openai/Same"]["active_s"], 0)
        self.assertNotIn("Same", models)

    def test_work_and_personal_scope_partition(self):
        personal = self.event(eid="personal", sid="personal", request_id="personal")
        work = self.event(eid="work", sid="work", request_id="work")
        self.assertEqual(self.post([personal]).status_code, 200)
        self.assertEqual(self.post([work], account_class="work").status_code, 200)
        from app.config import ENTERPRISE_PRED, PERSONAL_PRED
        row = self.conn.execute(
            f"SELECT SUM({ENTERPRISE_PRED}) AS enterprise, "
            f"SUM({PERSONAL_PRED}) AS personal FROM events").fetchone()
        self.assertEqual((row["enterprise"], row["personal"]), (1, 1))

    def test_bounds_and_malformed(self):
        self.assertEqual(self.post([self.event(kind="bad")]).status_code, 422)
        self.assertEqual(self.post([self.event()], account_class="unknown").status_code, 422)
        extra = self.event(); extra["unexpected_content"] = "must reject"
        self.assertEqual(self.post([extra]).status_code, 422)
        self.assertEqual(self.post([self.event(usage={"cost_total": -1})]).status_code, 422)
        self.assertEqual(self.post([self.event()] * 5001).status_code, 422)


if __name__ == "__main__":
    unittest.main()
