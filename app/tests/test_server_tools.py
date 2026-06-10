"""Server-tool (web search / web fetch) billing.

Anthropic bills web search at $10 per 1,000 requests on top of token costs;
web fetch is free (it only bills the fetched-content tokens, already counted).
Claude Code logs per-request counts in the transcript as a nested object:
usage.server_tool_use = {web_search_requests, web_fetch_requests}.

These counts are captured into events, billed through every cost surface
(summarizer, window costs), and backfillable from transcripts — the same
repair pattern as the cache-tier split.
"""

import time
import unittest
from unittest.mock import patch

import app.pricing as pricing
from app.ingest import _extract_event, EVENT_COLS
from app.tests._support import TempDBTestCase


def _arec(uuid="u1", req="r1", ts="2026-06-09T12:00:00Z", **usage):
    u = {"input_tokens": 10, "output_tokens": 20}
    u.update(usage)
    return {
        "uuid": uuid, "type": "assistant", "timestamp": ts,
        "sessionId": "s1", "requestId": req,
        "message": {"model": "claude-opus-4-8", "id": "m-" + uuid, "usage": u},
    }


class ExtractServerToolsTest(unittest.TestCase):
    """usage.server_tool_use is the real transcript shape (verified live)."""

    def test_counts_captured(self):
        row = _extract_event(_arec(
            server_tool_use={"web_search_requests": 3, "web_fetch_requests": 5},
        ), "mach", "proj")
        self.assertEqual(row["web_search_requests"], 3)
        self.assertEqual(row["web_fetch_requests"], 5)

    def test_absent_defaults_zero(self):
        row = _extract_event(_arec(), "mach", "proj")
        self.assertEqual(row["web_search_requests"], 0)
        self.assertEqual(row["web_fetch_requests"], 0)

    def test_malformed_object_safe(self):
        row = _extract_event(_arec(server_tool_use="garbage"), "mach", "proj")
        self.assertEqual(row["web_search_requests"], 0)
        self.assertEqual(row["web_fetch_requests"], 0)

    def test_non_int_counts_zeroed(self):
        """Untrusted transcript JSON: a dict/str count would raise at the
        sqlite bind and 500 the whole ingest batch — coerce to 0 instead."""
        row = _extract_event(_arec(
            server_tool_use={"web_search_requests": "9",
                             "web_fetch_requests": {"nested": 1}},
        ), "mach", "proj")
        self.assertEqual(row["web_search_requests"], 0)
        self.assertEqual(row["web_fetch_requests"], 0)

    def test_event_cols_wired(self):
        self.assertIn("web_search_requests", EVENT_COLS)
        self.assertIn("web_fetch_requests", EVENT_COLS)


class ComputeCostWebSearchTest(unittest.TestCase):
    """$10 per 1,000 web searches, flat — independent of model and modifiers."""

    def setUp(self):
        p = patch.dict(pricing._dynamic_pricing, {}, clear=True)
        p.start()
        self.addCleanup(p.stop)

    def test_thousand_searches_cost_ten_dollars(self):
        c = pricing.compute_cost("Sonnet 4.6", 0, 0, 0, 0, web_search=1000)
        self.assertAlmostEqual(c, 10.00)

    def test_default_zero_reproduces_prior_pricing(self):
        c = pricing.compute_cost("Opus 4.8", 1_000_000, 0, 0, 0)
        self.assertAlmostEqual(c, 5.00)

    def test_fee_stacks_on_token_cost(self):
        c = pricing.compute_cost("Opus 4.8", 1_000_000, 0, 0, 0, web_search=100)
        self.assertAlmostEqual(c, 5.00 + 1.00)

    def test_fee_not_geo_multiplied(self):
        """The US-geo 1.1x multiplier applies to token rates only — the
        search fee is a flat per-request charge."""
        c = pricing.compute_cost("Sonnet 4.6", 0, 0, 0, 0,
                                 inference_geo="us", web_search=1000)
        self.assertAlmostEqual(c, 10.00)

    def test_fee_applies_to_unpriced_models(self):
        """Unpriced models contribute $0 in token cost (no fabricated rates),
        but the search fee is confirmed, model-independent pricing."""
        with patch.object(pricing, "_unknown_refresh_ts", time.time()):
            c = pricing.compute_cost("Mystery 9", 1_000_000, 0, 0, 0,
                                     web_search=100)
        self.assertAlmostEqual(c, 1.00)


class IngestServerToolsTest(TempDBTestCase):
    """End-to-end: ingest captures counts, daily_summary bills the fee."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def _post_ingest(self, events):
        # No `with` — TestClient shutdown would close the shared DB conn
        # before this test's assertions run (matches test_backfill.py).
        c = self.client()
        return c.post("/api/ingest", json={
            "machine": "mach", "project_dir": "proj",
            "session_file": "s1.jsonl",
            "cursor": {"last_line_num": 0}, "events": events,
        }, headers={"X-API-Key": self.api_key})

    def test_ingest_stores_counts_and_bills_fee(self):
        r = self._post_ingest([_arec(
            input_tokens=1_000_000, output_tokens=0,
            server_tool_use={"web_search_requests": 4, "web_fetch_requests": 7},
        )])
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["accepted"], 1)

        row = self.conn.execute(
            "SELECT web_search_requests ws, web_fetch_requests wf "
            "FROM events WHERE uuid='u1'").fetchone()
        self.assertEqual((row["ws"], row["wf"]), (4, 7))

        ds = self.conn.execute(
            "SELECT cost, model_json FROM daily_summary "
            "WHERE day='2026-06-09'").fetchone()
        # Opus 4.8: 1M input = $5.00, plus 4 searches = $0.04. Fetch is free.
        self.assertAlmostEqual(ds["cost"], 5.04, places=4)

        import json
        ms = json.loads(ds["model_json"])["Opus 4.8"]
        self.assertEqual(ms["web_search"], 4)
        self.assertEqual(ms["web_fetch"], 7)

    def test_streaming_repeats_deduped_by_max(self):
        """Streaming chunks repeat cumulative usage per request_id — the fee
        must be charged once on the MAX, not once per chunk."""
        r = self._post_ingest([
            _arec(uuid="u1", req="r1", input_tokens=1_000_000, output_tokens=0,
                  server_tool_use={"web_search_requests": 1,
                                   "web_fetch_requests": 0}),
            _arec(uuid="u2", req="r1", input_tokens=1_000_000, output_tokens=0,
                  server_tool_use={"web_search_requests": 3,
                                   "web_fetch_requests": 0}),
        ])
        self.assertEqual(r.status_code, 200)
        ds = self.conn.execute(
            "SELECT cost FROM daily_summary WHERE day='2026-06-09'").fetchone()
        self.assertAlmostEqual(ds["cost"], 5.03, places=4)  # MAX(3) -> $0.03


class WindowCostServerToolsTest(TempDBTestCase):
    """compute_window_cost (rate-limit gauges, /api/ha, hourly) bills the fee."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_window_cost_includes_search_fee(self):
        self.conn.execute(
            "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,"
            "request_id,source_machine,project_dir,model,is_sidechain,agent_id,"
            "input_tokens,output_tokens,cache_creation_tokens,cache_read_tokens,"
            "web_search_requests,web_fetch_requests,is_human_prompt) "
            "VALUES('u1','assistant','2026-06-09T12:00:00Z',1781000000.0,"
            "'2026-06-09','s1','r1','m1','proj','claude-opus-4-8',0,NULL,"
            "1000000,0,0,0,10,3,0)")
        self.conn.commit()
        from app.cost_windows import compute_window_cost
        total = compute_window_cost(
            self.conn, 1780000000.0, 1782000000.0, scope="personal")
        self.assertAlmostEqual(total, 5.10, places=4)  # $5 tokens + 10 searches


class BackfillServerToolsTest(TempDBTestCase):
    """uuid -> [web_search, web_fetch] repairs, fill-only-unset like cache tiers."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def _ins(self, uuid, ws=0, wf=0, day="2026-06-09"):
        self.conn.execute(
            "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,"
            "request_id,source_machine,project_dir,model,is_sidechain,agent_id,"
            "input_tokens,output_tokens,cache_creation_tokens,cache_read_tokens,"
            "web_search_requests,web_fetch_requests,is_human_prompt) "
            "VALUES(?,'assistant',?,1781000000.0,?,'s1',?,'m1','proj',"
            "'claude-opus-4-8',0,NULL,1000000,0,0,0,?,?,0)",
            (uuid, day + "T12:00:00Z", day, "r-" + uuid, ws, wf))
        self.conn.commit()

    def _post(self, payload):
        c = self.client()
        return c.post("/api/backfill", json=payload,
                      headers={"X-API-Key": self.api_key})

    def test_fills_unset_counts_and_rerolls_day(self):
        from app.summarizer import summarize_days
        self._ins("u1")
        summarize_days(["2026-06-09"])
        before = self.conn.execute(
            "SELECT cost FROM daily_summary WHERE day='2026-06-09'").fetchone()["cost"]
        self.assertAlmostEqual(before, 5.00, places=4)

        r = self._post({"server_tools": {"u1": [6, 2]}})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["updated_server_tools"], 1)
        self.assertIn("2026-06-09", body["touched_days"])

        row = self.conn.execute(
            "SELECT web_search_requests ws, web_fetch_requests wf "
            "FROM events WHERE uuid='u1'").fetchone()
        self.assertEqual((row["ws"], row["wf"]), (6, 2))
        after = self.conn.execute(
            "SELECT cost FROM daily_summary WHERE day='2026-06-09'").fetchone()["cost"]
        self.assertAlmostEqual(after, 5.06, places=4)

    def test_never_clobbers_existing_counts(self):
        self._ins("u1", ws=2, wf=0)
        r = self._post({"server_tools": {"u1": [9, 9]}})
        self.assertEqual(r.json()["updated_server_tools"], 0)
        row = self.conn.execute(
            "SELECT web_search_requests ws, web_fetch_requests wf "
            "FROM events WHERE uuid='u1'").fetchone()
        self.assertEqual((row["ws"], row["wf"]), (2, 0))

    def test_unknown_uuid_and_zero_pairs_ignored(self):
        self._ins("u1")
        r = self._post({"server_tools": {"ghost": [1, 1], "u1": [0, 0]}})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["updated_server_tools"], 0)

    def test_invalid_pairs_rejected_quietly(self):
        self._ins("u1")
        r = self._post({"server_tools": {"u1": [-1, 2]}})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["updated_server_tools"], 0)


class NotifyCostWebSearchTest(unittest.TestCase):
    """The HA notify per-turn cost is computed straight from hook usage
    entries (transcript-shaped) — it must carry the search fee too."""

    def setUp(self):
        p = patch.dict(pricing._dynamic_pricing, {}, clear=True)
        p.start()
        self.addCleanup(p.stop)

    def test_notify_cost_includes_search_fee(self):
        from app.notify import _cost_from_usage
        c = _cost_from_usage({
            "model": "claude-opus-4-8",
            "input_tokens": 1_000_000, "output_tokens": 0,
            "server_tool_use": {"web_search_requests": 10,
                                "web_fetch_requests": 2},
        })
        self.assertAlmostEqual(c, 5.00 + 0.10, places=4)

    def test_notify_cost_safe_on_malformed_counts(self):
        from app.notify import _cost_from_usage
        c = _cost_from_usage({
            "model": "claude-opus-4-8",
            "input_tokens": 1_000_000, "output_tokens": 0,
            "server_tool_use": {"web_search_requests": "junk"},
        })
        self.assertAlmostEqual(c, 5.00, places=4)


class AggregatorWebSearchComponentTest(TempDBTestCase):
    """The Cost-by-Model chart stacks per-component costs — without a
    web-search component the stacked bars would not sum to the model's total
    cost (same regression class as the missed recent_cost_cache_write)."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_model_breakdown_carries_web_search_component(self):
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        c = self.client()
        r = c.post("/api/ingest", json={
            "machine": "mach", "project_dir": "proj",
            "session_file": "s1.jsonl",
            "cursor": {"last_line_num": 0},
            "events": [_arec(ts=now_iso, input_tokens=1_000_000,
                             output_tokens=0,
                             server_tool_use={"web_search_requests": 4,
                                              "web_fetch_requests": 1})],
        }, headers={"X-API-Key": self.api_key})
        self.assertEqual(r.status_code, 200)

        import app.aggregator as agg
        agg._cached_data.clear()
        d = agg.build_dashboard_data("personal")

        mb = {m["model"]: m for m in d["model_breakdown"]}
        self.assertIn("Opus 4.8", mb)
        m = mb["Opus 4.8"]
        self.assertEqual(m["web_search"], 4)
        self.assertAlmostEqual(m["cost_web_search"], 0.04, places=4)
        self.assertAlmostEqual(m["recent_cost_web_search"], 0.04, places=4)
        # total still reconciles: components + fee == cost
        self.assertAlmostEqual(m["cost"], 5.04, places=2)

        tb = {m["model"]: m for m in d["today"]["model_breakdown"]}
        self.assertIn("Opus 4.8", tb)
        self.assertAlmostEqual(tb["Opus 4.8"]["cost_web_search"], 0.04, places=4)

    def test_dashboard_template_stacks_web_search_dataset(self):
        from pathlib import Path
        html = (Path(__file__).resolve().parents[2]
                / "templates" / "dashboard.html").read_text()
        self.assertIn("recent_cost_web_search", html)
        self.assertIn("'Web Search'", html)


if __name__ == "__main__":
    unittest.main()
