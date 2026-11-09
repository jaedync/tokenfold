"""Pi-reported Anthropic costs against verified list prices
(app/pi_price_check.py): the rule, ingest, and the startup re-check."""

import json
import unittest
from datetime import datetime, timezone

from app import pi_price_check as ppc
from app.pricing import SONNET55_CR_CUT_UTC as CUT
from app.tests._support import TempDBTestCase

M = 1_000_000
AFTER = CUT + 86400   # Sonnet 5.5 cache reads at $0.10
BEFORE = CUT - 60     # ... at $0.20


def reported(inp=0.0, out=0.0, cr=0.0, cw=0.0, total="sum"):
    r = {"input": inp, "output": out, "cache_read": cr, "cache_write": cw}
    r["total"] = inp + out + cr + cw if total == "sum" else total
    return r


class RuleTest(unittest.TestCase):

    def test_sonnet_55_cache_read_overcharge_is_corrected(self):
        # Pi's registry priced Sonnet 5.5 cache reads at $0.20 until mid-Oct.
        new = ppc.corrected("claude-sonnet-5-5", AFTER, 0, M, 10 * M,
                            reported(out=10.0, cr=2.0, cw=4.0))
        self.assertAlmostEqual(new["cache_read"], 1.0)
        self.assertAlmostEqual(new["total"], 15.0)
        self.assertEqual((new["output"], new["cache_write"]), (10.0, 4.0))

    def test_sonnet_55_before_the_cut_was_right(self):
        # $0.20 was the list price until Anthropic cut it on 7 Oct 2026.
        self.assertIsNone(ppc.corrected("claude-sonnet-5-5", BEFORE, 0, M, 10 * M,
                                        reported(out=10.0, cr=2.0)))
        new = ppc.corrected("claude-sonnet-5-5", BEFORE, 0, M, 10 * M, reported(out=10.0, cr=1.0))
        self.assertAlmostEqual(new["cache_read"], 2.0)
        # An era model without a request time is not judged.
        self.assertIsNone(ppc.corrected("claude-sonnet-5-5", None, 0, M, 10 * M,
                                        reported(out=10.0, cr=2.0)))

    def test_fable_51_cache_read_and_input(self):
        new = ppc.corrected("claude-fable-5-1", AFTER, M, M, 4 * M,
                            reported(inp=9.0, out=50.0, cr=4.0))
        self.assertAlmostEqual(new["cache_read"], 1.0)
        self.assertAlmostEqual(new["input"], 10.0)
        self.assertAlmostEqual(new["total"], 61.0)

    def test_correct_rows_stay(self):
        self.assertIsNone(ppc.corrected("claude-opus-5-5", AFTER, 100, M, 10 * M,
                                        reported(inp=0.0004, out=20.0, cr=2.0, cw=8.0)))
        # Differences inside the tolerance are rounding, not a wrong price.
        self.assertIsNone(ppc.corrected("claude-opus-5-5", AFTER, 3, M, 10 * M,
                                        reported(inp=0.0, out=20.0, cr=2.0)))

    def test_unverified_or_unanchored_rows_stay(self):
        bad_cr = reported(out=10.0, cr=2.0)
        # Haiku 5.5 has prompt-size tiers; unknown models have no rate.
        self.assertIsNone(ppc.corrected("claude-haiku-5-5", AFTER, 0, M, 10 * M, bad_cr))
        self.assertIsNone(ppc.corrected("claude-mystery-9", AFTER, 0, M, 10 * M, bad_cr))
        self.assertIsNone(ppc.corrected(None, AFTER, 0, M, 10 * M, bad_cr))
        # Output off the standard rate (fast mode, US-only): not ours to judge.
        self.assertIsNone(ppc.corrected("claude-opus-5-5", AFTER, 0, M, 10 * M,
                                        reported(out=40.0, cr=4.0)))
        # No output tokens or no output component: no anchor.
        self.assertIsNone(ppc.corrected("claude-sonnet-5-5", AFTER, 0, 0, 10 * M, reported(cr=2.0)))
        self.assertIsNone(ppc.corrected("claude-sonnet-5-5", AFTER, 0, M, 10 * M,
                                        {"input": None, "output": None, "cache_read": 2.0,
                                         "cache_write": None, "total": 12.0}))

    def test_missing_component_or_total(self):
        new = ppc.corrected("claude-sonnet-5-5", AFTER, 0, M, 10 * M,
                            {"input": None, "output": 10.0, "cache_read": 2.0,
                             "cache_write": None, "total": None})
        self.assertAlmostEqual(new["cache_read"], 1.0)
        self.assertIsNone(new["total"])
        self.assertIsNone(new["input"])

    def test_unknown_token_count_keeps_the_component(self):
        self.assertIsNone(ppc.corrected("claude-opus-5-5", AFTER, None, M, None,
                                        reported(inp=0.002, out=20.0, cr=2.0)))

    def test_fable_52_rates(self):
        # $8 in, $40 out, $0.20 cache read (Fable 5.2 launch, 2026-11-09).
        self.assertIsNone(ppc.corrected("claude-fable-5-2", AFTER, M, M, 10 * M,
                                        reported(inp=8.0, out=40.0, cr=2.0)))
        # A cache read priced at the Fable 5.1 rate ($0.25) is corrected.
        new = ppc.corrected("claude-fable-5-2", AFTER, M, M, 10 * M,
                            reported(inp=8.0, out=40.0, cr=2.5))
        self.assertAlmostEqual(new["cache_read"], 2.0)
        self.assertAlmostEqual(new["total"], 50.0)
        # Output at the Fable 5.1 rate is not the standard tier: left alone.
        self.assertIsNone(ppc.corrected("claude-fable-5-2", AFTER, M, M, 10 * M,
                                        reported(inp=10.0, out=50.0, cr=2.5)))

    def test_dated_model_id(self):
        new = ppc.corrected("claude-sonnet-5-5-20260928", AFTER, 0, M, 10 * M,
                            reported(out=10.0, cr=2.0))
        self.assertAlmostEqual(new["cache_read"], 1.0)


def _pi_event(eid, model, usage):
    return {"event_id": eid, "timestamp": "2026-10-09T12:00:00Z", "session_id": "s",
            "kind": "assistant", "provider": "anthropic", "api": "anthropic-messages",
            "model": model, "request_id": f"msg_{eid}", "usage": usage}


class IngestTest(TempDBTestCase):

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def post(self, events):
        body = {"machine": "m", "account_class": "personal", "project_dir": "/p",
                "session_file": "pi.jsonl", "cursor": {"last_line_num": 0}, "events": events}
        r = self.client().post("/api/ingest/pi", json=body, headers={"X-API-Key": self.api_key})
        self.assertEqual(r.status_code, 200, r.text)

    def test_wrong_component_is_stored_corrected_with_the_original(self):
        usage = {"input": 0, "output": M, "cache_read": 10 * M, "cache_write": M,
                 "cost_input": 0.0, "cost_output": 10.0, "cost_cache_read": 2.0,
                 "cost_cache_write": 4.0, "cost_total": 16.0}
        ok = dict(usage, cost_cache_read=1.0, cost_total=15.0)
        self.post([_pi_event("a", "claude-sonnet-5-5", usage),
                   _pi_event("b", "claude-sonnet-5-5", ok)])
        rows = {r["message_id"]: r for r in self.conn.execute(
            "SELECT message_id, reported_cost_cache_read, reported_cost_total, "
            "reported_cost_pi_raw FROM events WHERE type='assistant'")}
        self.assertAlmostEqual(rows["msg_a"]["reported_cost_cache_read"], 1.0)
        self.assertAlmostEqual(rows["msg_a"]["reported_cost_total"], 15.0)
        raw = json.loads(rows["msg_a"]["reported_cost_pi_raw"])
        self.assertEqual((raw["cache_read"], raw["total"]), (2.0, 16.0))
        self.assertIsNone(rows["msg_b"]["reported_cost_pi_raw"])

    def test_other_providers_are_not_checked(self):
        usage = {"input": 0, "output": M, "cache_read": 10 * M, "cache_write": 0,
                 "cost_input": 0.0, "cost_output": 10.0, "cost_cache_read": 2.0,
                 "cost_cache_write": 0.0, "cost_total": 12.0}
        ev = _pi_event("c", "claude-sonnet-5-5", usage)
        ev["provider"] = "openrouter"
        self.post([ev])
        row = self.conn.execute("SELECT reported_cost_cache_read, reported_cost_pi_raw "
                                "FROM events WHERE type='assistant'").fetchone()
        self.assertEqual((row[0], row[1]), (2.0, None))


def _noon(day):
    return datetime.fromisoformat(day + "T12:00:00+00:00").timestamp()


class RecheckTest(TempDBTestCase):

    def _insert(self, uuid, model, day, cr_cost, raw=None, provider="anthropic"):
        self.conn.execute(
            "INSERT INTO events(uuid, type, timestamp, ts_epoch, day, source_machine, "
            "session_id, request_id, "
            "model, source_client, provider, input_tokens, output_tokens, cache_read_tokens, "
            "cache_creation_tokens, reported_cost_input, reported_cost_output, "
            "reported_cost_cache_read, reported_cost_cache_write, reported_cost_total, "
            "reported_cost_pi_raw) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (uuid, "assistant", day + "T12:00:00Z", _noon(day), day, "m", "s", "r-" + uuid, model,
             "pi-agent", provider, 0, M, 10 * M, 0, 0.0, 10.0, cr_cost, 0.0, 10.0 + cr_cost, raw))

    def _row(self, uuid):
        return self.conn.execute(
            "SELECT reported_cost_cache_read, reported_cost_total, reported_cost_pi_raw "
            "FROM events WHERE uuid=?", (uuid,)).fetchone()

    def _queued(self):
        from app import db
        return db.queued_rollup_days(self.conn)

    def test_startup_corrects_history_once_and_queues_days(self):
        from app import db
        self._insert("wrong", "claude-sonnet-5-5", "2026-10-09", 2.0)
        self._insert("right", "claude-sonnet-5-5", "2026-10-20", 1.0)
        self._insert("other", "claude-sonnet-5-5", "2026-10-10", 2.0, provider="openrouter")
        self.conn.execute("DELETE FROM meta WHERE key=?", (ppc.VERSION_KEY,))
        self.conn.commit()
        db._migrate(self.conn)
        cr, total, raw = self._row("wrong")
        self.assertAlmostEqual(cr, 1.0)
        self.assertAlmostEqual(total, 11.0)
        self.assertEqual(json.loads(raw)["cache_read"], 2.0)
        self.assertEqual(tuple(self._row("right")), (1.0, 11.0, None))
        self.assertEqual(tuple(self._row("other")), (2.0, 12.0, None))
        self.assertEqual(self._queued(), {"2026-10-09"})
        # The next start does nothing: the version is stored.
        self.conn.execute(f"DELETE FROM meta WHERE key='{db.ROLLUP_DAYS_KEY}'")
        self.conn.commit()
        db._migrate(self.conn)
        self.assertEqual(self._queued(), set())
        self.assertAlmostEqual(self._row("wrong")[0], 1.0)

    def test_new_version_reapplies_from_the_original(self):
        from app import db
        raw = json.dumps({"input": 0.0, "output": 10.0, "cache_read": 2.0,
                          "cache_write": 0.0, "total": 12.0})
        # Corrected by an older table to $0.05 per million; the current
        # table says $0.10, so the row is corrected again from Pi's $2.00.
        self._insert("old", "claude-sonnet-5-5", "2026-10-09", 0.5, raw=raw)
        # A row the current table no longer corrects goes back to Pi's values.
        self._insert("haiku", "claude-haiku-5-5", "2026-10-11", 0.1, raw=json.dumps(
            {"input": 0.0, "output": 10.0, "cache_read": 0.2, "cache_write": 0.0, "total": 10.2}))
        self.conn.execute("UPDATE meta SET value='0' WHERE key=?", (ppc.VERSION_KEY,))
        self.conn.commit()
        db._migrate(self.conn)
        cr, total, raw_after = self._row("old")
        self.assertAlmostEqual(cr, 1.0)
        self.assertAlmostEqual(total, 11.0)
        self.assertEqual(json.loads(raw_after)["cache_read"], 2.0)
        self.assertEqual(tuple(self._row("haiku")), (0.2, 10.2, None))
        self.assertEqual(self._queued(), {"2026-10-09", "2026-10-11"})

    def test_sdk_days_before_the_sonnet_55_cut_are_rerolled(self):
        from app import db
        self._insert("early", "claude-sonnet-5-5", "2026-10-02", 2.0)
        self.conn.execute("UPDATE events SET source_client='claude-code', provider=NULL, "
                          "reported_cost_total=NULL WHERE uuid='early'")
        self.conn.execute("DELETE FROM meta WHERE key=?", (ppc.VERSION_KEY,))
        self.conn.commit()
        db._migrate(self.conn)
        self.assertEqual(self._queued(), {"2026-10-02"})

    def test_rollup_cost_follows_the_correction(self):
        from app import db, main
        from app.summarizer import summarize_days
        self._insert("wrong", "claude-sonnet-5-5", "2026-10-09", 2.0)
        self.conn.execute("UPDATE events SET account_email='pi-personal@dotfleet.local' "
                          "WHERE uuid='wrong'")
        self.conn.commit()
        summarize_days(["2026-10-09"])
        before = self.conn.execute("SELECT SUM(cost) FROM daily_summary").fetchone()[0]
        self.conn.execute("DELETE FROM meta WHERE key=?", (ppc.VERSION_KEY,))
        self.conn.commit()
        db._migrate(self.conn)
        main._backfill_if_needed()
        after = self.conn.execute("SELECT SUM(cost) FROM daily_summary").fetchone()[0]
        self.assertAlmostEqual(before - after, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
