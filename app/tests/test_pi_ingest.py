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
        self.assertEqual(set(models), {"OpenAI / Same", "Google / Same"})
        self.assertGreater(models["OpenAI / Same"]["active_s"], 0)
        self.assertNotIn("Same", models)
        from app.aggregator import _build_dashboard_data_inner
        dashboard = _build_dashboard_data_inner("personal")
        display_names = {item["model"] for item in dashboard["model_breakdown"]}
        self.assertEqual(display_names, {"OpenAI / Same", "Google / Same"})

    def test_pi_model_display_normalization(self):
        from app.pricing import display_model_for_row
        cases = {
            ("openai-codex", "gpt-5.6-sol"): "OpenAI Codex / GPT-5.6 Sol",
            ("openai-codex", "openai-codex/gpt-5.6-sol"): "OpenAI Codex / GPT-5.6 Sol",
            ("anthropic", "claude-opus-4-8"): "Anthropic / Opus 4.8",
            ("openrouter", "z-ai/glm-5.3"): "OpenRouter / GLM-5.3",
            ("openrouter", "moonshotai/kimi-k3"): "OpenRouter / Kimi K3",
            ("huggingface", "zai-org/GLM-5.3-Flash"): "Hugging Face / GLM-5.3 Flash",
        }
        for (provider, model), expected in cases.items():
            with self.subTest(provider=provider, model=model):
                self.assertEqual(display_model_for_row(model, provider, "pi-agent"), expected)

    def test_provider_is_hidden_unless_normalized_model_names_collide(self):
        from app.aggregator import _conditional_model_names, _summary_model_identity
        opus_plain = _summary_model_identity("Opus 4.8")
        opus_pi = _summary_model_identity("Anthropic / Opus 4.8")
        codex = _summary_model_identity("OpenAI Codex / GPT-5.6 Luna")
        router = _summary_model_identity("OpenRouter / GPT-5.6 Luna")
        glm = _summary_model_identity("OpenCode Go / GLM-5.3")
        self.assertEqual(opus_plain, opus_pi)
        names = _conditional_model_names({opus_plain, codex, router, glm})
        self.assertEqual(names[opus_plain], "Opus 4.8")
        self.assertEqual(names[glm], "GLM-5.3")
        self.assertEqual(names[codex], "OpenAI Codex / GPT-5.6 Luna")
        self.assertEqual(names[router], "OpenRouter / GPT-5.6 Luna")

    def test_today_breakdown_qualifies_only_colliding_providers(self):
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        codex = self.event(
            eid="codex", sid="codex", timestamp=timestamp,
            provider="openai-codex", model="gpt-5.6-luna",
            request_id="codex", usage={"input": 1, "cost_total": 1})
        router = self.event(
            eid="router", sid="router", timestamp=timestamp,
            provider="openrouter", model="openai/gpt-5.6-luna",
            request_id="router", usage={"input": 1, "cost_total": 2})
        sol = self.event(
            eid="sol", sid="sol", timestamp=timestamp,
            provider="openai-codex", model="gpt-5.6-sol",
            request_id="sol", usage={"input": 1, "cost_total": 3})
        self.assertEqual(self.post([codex, router, sol]).status_code, 200)
        from app.aggregator import _build_dashboard_data_inner
        today = _build_dashboard_data_inner("personal")["today"]
        models = {item["model"]: item for item in today["model_breakdown"]}
        self.assertEqual(
            set(models),
            {"OpenAI Codex / GPT-5.6 Luna", "OpenRouter / GPT-5.6 Luna",
             "GPT-5.6 Sol"})
        self.assertEqual(models["GPT-5.6 Sol"]["cost"], 3)

    def test_reported_cost_components_populate_model_breakdown(self):
        component = self.event(
            eid="component", provider="openai-codex", model="gpt-5.6-sol",
            request_id="component")
        residual = self.event(
            eid="residual", sid="router", provider="openrouter",
            model="z-ai/glm-5.3", request_id="residual",
            usage={"input": 1, "cost_total": 4})
        tiny = self.event(
            eid="tiny", sid="tiny", provider="huggingface",
            model="zai-org/GLM-5.3-Flash", request_id="tiny",
            usage={"input": 1, "cost_input": .001,
                   "cost_output": .002, "cost_cache_read": .000461,
                   "cost_total": .003461})
        self.assertEqual(self.post([component, residual, tiny]).status_code, 200)
        from app.aggregator import _build_dashboard_data_inner
        dashboard = _build_dashboard_data_inner("personal")
        models = {item["model"]: item for item in dashboard["model_breakdown"]}
        codex = models["GPT-5.6 Sol"]
        self.assertEqual(codex["cost_input"], 1)
        self.assertEqual(codex["cost_output"], 2)
        self.assertEqual(codex["cost_cache_read"], .1)
        self.assertEqual(codex["cost_cache_write_reported"], .2)
        self.assertEqual(codex["cost_other"], 0)
        self.assertFalse(codex["unpriced"])
        self.assertTrue(codex["has_reported_cost"])
        router = models["GLM-5.3"]
        self.assertEqual(router["cost"], 4)
        self.assertEqual(router["cost_other"], 4)
        flash = models["GLM-5.3 Flash"]
        self.assertEqual(flash["cost"], .0035)
        self.assertEqual(flash["cost_input"], .001)

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

    def test_identical_native_ids_are_namespaced_by_scope(self):
        event = self.event()
        personal = self.post([event], account_class="personal")
        work = self.post([event], account_class="work")
        self.assertEqual(personal.json()["accepted"], 1)
        self.assertEqual(work.json()["accepted"], 1)
        rows = self.conn.execute(
            "SELECT uuid, session_id, account_email FROM events ORDER BY account_email"
        ).fetchall()
        self.assertEqual(len({row["uuid"] for row in rows}), 2)
        self.assertEqual(len({row["session_id"] for row in rows}), 2)
        self.assertEqual({row["account_email"] for row in rows},
                         {"pi-personal@dotfleet.local", "pi-work@dotfleet.local"})
        cursors = self.conn.execute(
            "SELECT machine FROM sync_cursors ORDER BY machine").fetchall()
        self.assertEqual([row["machine"] for row in cursors],
                         ["pi:personal:m", "pi:work:m"])

    def test_mixed_reported_cost_shapes_are_summed_per_request(self):
        total = self.event(eid="total", request_id="total",
                           usage={"input": 1, "cost_total": 4})
        components = self.event(
            eid="components", request_id="components",
            usage={"input": 1, "cost_input": 1, "cost_output": 2})
        self.assertEqual(self.post([total, components]).status_code, 200)
        from app.cost_windows import compute_window_cost
        cost = compute_window_cost(self.conn, 1_781_000_000, 1_782_000_000,
                                   scope="personal")
        self.assertEqual(cost, 7)
        from app.summarizer import summarize_days
        summarize_days(["2026-06-09"])
        summary = self.conn.execute(
            "SELECT cost FROM daily_summary WHERE account_email=?",
            ("pi-personal@dotfleet.local",)).fetchone()
        self.assertEqual(summary["cost"], 7)

    def test_usage_requires_provider_and_model(self):
        self.assertEqual(self.post([self.event(provider=None)]).status_code, 422)
        self.assertEqual(self.post([self.event(model=None)]).status_code, 422)
        no_cost = self.event(usage={"input": 1, "output": 1})
        self.assertEqual(self.post([no_cost]).status_code, 200)

    def test_bounds_and_malformed(self):
        self.assertEqual(self.post([self.event(kind="bad")]).status_code, 422)
        self.assertEqual(self.post([self.event()], account_class="unknown").status_code, 422)
        extra = self.event(); extra["unexpected_content"] = "must reject"
        self.assertEqual(self.post([extra]).status_code, 422)
        self.assertEqual(self.post([self.event(usage={"cost_total": -1})]).status_code, 422)
        self.assertEqual(self.post([self.event()] * 5001).status_code, 422)


if __name__ == "__main__":
    unittest.main()
