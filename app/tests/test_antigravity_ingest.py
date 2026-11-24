"""Synthetic Antigravity transcript through the client and server."""
import importlib.util
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from app.tests._support import TempDBTestCase

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).with_name("fixtures") / "antigravity-transcript.jsonl"
CID = "fbdad862-c5cd-4255-9849-ef2bcfec60b9"
EVENT = {
    "conversation_id": CID, "step_index": 1, "timestamp": "2026-11-24T05:58:00Z",
    "model": "gemini-4-neon-xhigh", "input_tokens": 100_000,
    "cache_read_tokens": 2_000_000, "output_tokens": 20_000,
}


class AntigravityIngestTest(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        self.headers = {"X-API-Key": self.api_key}

    def post(self, events, machine="test-host", headers=None):
        return self.client().post("/api/ingest/antigravity", json={
            "machine": machine, "events": events,
        }, headers=self.headers if headers is None else headers)

    def test_pricing_has_only_confirmed_models_and_efforts(self):
        from app import pricing
        for suffix in ("", "-xhigh", "-high", "-low"):
            with self.subTest(suffix=suffix):
                name = pricing.display_model_for_row(
                    "gemini-4-neon" + suffix, "antigravity", "antigravity")
                self.assertTrue(pricing.is_priced(name))
                # $0.30 fresh + $0.30 cached + $0.30 output.
                self.assertAlmostEqual(pricing.compute_cost(name, 100_000, 20_000, 0, 2_000_000), 0.90)
        for model in ("gemini-4-neon-unverified", "gemini-4-hydrogen-high", "claude-opus-4-6"):
            with self.subTest(model=model):
                name = pricing.display_model_for_row(model, "antigravity", "antigravity")
                with patch.dict(pricing._dynamic_pricing, {name: (5, 6, 7, 8)}):
                    self.assertFalse(pricing.is_priced(name))
                    self.assertEqual(pricing.compute_cost(name, 100_000, 20_000, 0, 2_000_000), 0)

    def test_duplicate_identity_is_conversation_and_step_not_machine(self):
        r = self.post([EVENT, EVENT])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json(), {"accepted": 1, "duplicates": 1})
        r = self.post([dict(EVENT, conversation_id=CID.upper()),
                       dict(EVENT, step_index=2),
                       dict(EVENT, conversation_id="b948ea48-ec36-4ac1-80f2-b2156c58ae2b")],
                      machine="other-host")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json(), {"accepted": 2, "duplicates": 1})
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0], 3)

    def test_key_and_allowlist_validation(self):
        self.assertEqual(self.post([EVENT], headers={}).status_code, 401)
        for overrides in (
            {"content": "reject extra keys"}, {"cost": 20}, {"input_tokens": True},
            {"output_tokens": -1}, {"model": "a model with spaces"},
            {"timestamp": "2026-11-24"}, {"timestamp": "2026-11-24T05:58:00"},
            {"timestamp": "0001-01-01T00:00:00+15:00"},
            {"conversation_id": "not a UUID"}, {"step_index": "1"},
        ):
            with self.subTest(overrides=overrides):
                self.assertEqual(self.post([dict(EVENT, **overrides)]).status_code, 422)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0], 0)

    def test_transcript_to_daily_summary_ledger_and_quota_isolation(self):
        from app.aggregator import _build_dashboard_data_inner
        from app.cost_windows import compute_window_cost

        spec = importlib.util.spec_from_file_location("agy_push", ROOT / "client/antigravity-push.py")
        client = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(client)
        bodies = []

        def send(endpoint, body):
            self.assertEqual(endpoint, "/api/ingest/antigravity")
            bodies.append(body)
            r = self.client().post(endpoint, json=body, headers=self.headers)
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.json(), {"accepted": 3, "duplicates": 0})
            return r.json()

        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            path = root / CID / client.TRANSCRIPT
            path.parent.mkdir(parents=True)
            path.write_bytes(FIXTURE.read_bytes())
            for _ in range(2):
                client.sync_spend(root, root / "cursor.json", send, "test-host")
        self.assertEqual(len(bodies), 1)
        self.assertEqual(bodies[0]["events"][0], EVENT)
        self.assertNotIn("synthetic", json.dumps(bodies))

        raw = self.conn.execute("SELECT * FROM events ORDER BY ts_epoch").fetchall()
        self.assertEqual({e["source_client"] for e in raw}, {"antigravity"})
        self.assertEqual(raw[0]["input_tokens"], 100_000)
        self.assertEqual(raw[0]["cache_read_tokens"], 2_000_000)
        self.assertIsNone(raw[0]["reported_cost_total"])
        self.assertTrue(all(e["has_text"] == 0 and e["has_thinking"] == 0 for e in raw))
        self.assertTrue(all(e["cwd"] is None for e in raw))
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM tool_uses").fetchone()[0], 0)

        tz = ZoneInfo(self._config.TZ_NAME)
        days = {}
        for e, cost in zip(raw, (0.9, 3.0, 0.0)):
            day = datetime.fromtimestamp(e["ts_epoch"], tz).date().isoformat()
            days.setdefault(day, {})["Antigravity / " + e["model"]] = cost
        summary = self.conn.execute("SELECT day, model_json, active_s FROM daily_summary").fetchall()
        self.assertEqual({r["day"] for r in summary}, days.keys())
        for row in summary:
            self.assertEqual(row["active_s"], 0)
            models = json.loads(row["model_json"])
            self.assertEqual(models.keys(), days[row["day"]].keys())
            for name, cost in days[row["day"]].items():
                self.assertAlmostEqual(models[name]["cost"], cost)

        view = _build_dashboard_data_inner("personal")
        breakdown = {m["model"]: m for m in view["model_breakdown"]}
        self.assertFalse(breakdown["Antigravity / gemini-4-neon-xhigh"]["unpriced"])
        unknown = breakdown["Antigravity / claude-opus-4-6"]
        self.assertTrue(unknown["unpriced"])
        self.assertEqual(unknown["total_tokens"], 200_090)
        prices = view["model_pricing"]
        self.assertEqual(prices["Antigravity / gemini-4-neon-xhigh"], {
            "input": 3.0, "output": 15.0, "cache_read": 0.15,
            "cache_write": None, "cache_write_1h": None,
        })
        self.assertTrue(all(rate is None for rate in prices["Antigravity / claude-opus-4-6"].values()))

        day = datetime(2026, 11, 24, tzinfo=timezone.utc).timestamp()
        bounds = (day, day + 86400)
        ledger = self.client().get("/api/ledger", params={
            "scope": "personal", "from": bounds[0], "to": bounds[1],
        })
        self.assertEqual(ledger.status_code, 200, ledger.text)
        ledger = ledger.json()
        self.assertAlmostEqual(ledger["totals"]["cost_usd"], 3.9)
        self.assertEqual(ledger["totals"]["active_s"], 0)
        self.assertEqual(ledger["model_providers"]["Antigravity / gemini-4-neon-xhigh"], "Antigravity")
        daily = {d["day"]: d["by_model"] for d in ledger["daily"] if d["by_model"]}
        self.assertEqual(daily.keys(), days.keys())
        for d, models in daily.items():
            self.assertEqual(models.keys(), {m for m, cost in days[d].items() if cost > 0})
            for model, cost in models.items():
                self.assertAlmostEqual(cost, days[d][model])
        self.assertEqual(compute_window_cost(self.conn, *bounds, scope="enterprise"), 0)
        self.assertEqual(compute_window_cost(
            self.conn, *bounds, scope="personal", anthropic_only=True), 0)
