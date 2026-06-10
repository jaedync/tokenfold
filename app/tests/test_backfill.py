"""POST /api/backfill — repair historical rows from machines' local transcripts.

Pre-fix events have cache_ephemeral_5m/1h stored as 0 (the extractor read
nonexistent flat keys) and sessions ingested before ai-title capture have no
title. Transcripts still on disk carry both. The endpoint applies
uuid -> (c5m, c1h) updates (only where the split is currently unset — never
clobbers real data) and session-title upserts, then re-rolls the affected
days so stored costs correct themselves.
"""

import unittest

from app.tests._support import TempDBTestCase


class BackfillTest(TempDBTestCase):

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def _ins(self, uuid, cw=1_000_000, c5m=0, c1h=0, day="2026-06-09"):
        self.conn.execute(
            "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,request_id,"
            "source_machine,project_dir,model,is_sidechain,agent_id,input_tokens,"
            "output_tokens,cache_creation_tokens,cache_read_tokens,cache_ephemeral_5m,"
            "cache_ephemeral_1h,is_human_prompt) "
            "VALUES(?,'assistant',?,1781000000.0,?,'s1',?,'m1','proj',"
            "'claude-opus-4-8',0,NULL,0,0,?,0,?,?,0)",
            (uuid, day + "T12:00:00Z", day, "r-" + uuid, cw, c5m, c1h),
        )
        self.conn.commit()

    def _post(self, payload, key=None):
        c = self.client()
        return c.post("/api/backfill", json=payload,
                      headers={"X-API-Key": key or self.api_key})

    def test_requires_api_key(self):
        c = self.client()
        r = c.post("/api/backfill", json={"cache_tiers": {}, "titles": {}})
        self.assertEqual(r.status_code, 401)

    def test_cache_tiers_updated_and_day_rerolled(self):
        from app.summarizer import summarize_days
        self._ins("u1", cw=1_000_000)
        summarize_days(["2026-06-09"])
        before = self.conn.execute(
            "SELECT cost FROM daily_summary WHERE day='2026-06-09'").fetchone()["cost"]
        self.assertAlmostEqual(before, 6.25, places=2)  # billed at 5m rate

        r = self._post({"cache_tiers": {"u1": [0, 1_000_000]}, "titles": {}})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["updated_events"], 1)
        self.assertIn("2026-06-09", body["touched_days"])

        after = self.conn.execute(
            "SELECT cost FROM daily_summary WHERE day='2026-06-09'").fetchone()["cost"]
        self.assertAlmostEqual(after, 10.00, places=2)  # re-rolled at 1h rate

    def test_never_clobbers_existing_split(self):
        self._ins("u1", cw=1_000_000, c5m=400_000, c1h=600_000)
        r = self._post({"cache_tiers": {"u1": [1_000_000, 0]}, "titles": {}})
        self.assertEqual(r.json()["updated_events"], 0)
        row = self.conn.execute(
            "SELECT cache_ephemeral_5m c5, cache_ephemeral_1h c1 FROM events "
            "WHERE uuid='u1'").fetchone()
        self.assertEqual((row["c5"], row["c1"]), (400_000, 600_000))

    def test_unknown_uuid_ignored(self):
        r = self._post({"cache_tiers": {"ghost": [0, 5]}, "titles": {}})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["updated_events"], 0)

    def test_titles_upserted_not_overwriting_newer(self):
        r = self._post({"cache_tiers": {}, "titles": {"sessA": "from backfill"}})
        self.assertEqual(r.json()["updated_titles"], 1)
        row = self.conn.execute(
            "SELECT title FROM session_titles WHERE session_id='sessA'").fetchone()
        self.assertEqual(row["title"], "from backfill")
        # an existing title is NOT overwritten by backfill (live ingest is newer)
        r = self._post({"cache_tiers": {}, "titles": {"sessA": "stale older title"}})
        self.assertEqual(r.json()["updated_titles"], 0)
        row = self.conn.execute(
            "SELECT title FROM session_titles WHERE session_id='sessA'").fetchone()
        self.assertEqual(row["title"], "from backfill")

    def test_oversized_batch_rejected(self):
        tiers = {f"u{i}": [0, 1] for i in range(20_001)}
        r = self._post({"cache_tiers": tiers, "titles": {}})
        self.assertEqual(r.status_code, 422)


if __name__ == "__main__":
    unittest.main()
