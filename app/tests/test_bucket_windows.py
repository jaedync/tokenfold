"""Per-bucket dollar windows on /api/rate-limits: oauth.five_hour_window and
the window_cost / window_start_epoch fields on scoped:* bucket entries.

All windows share _bucket_window_start (resets_at - window, minute-floored,
pushed forward past the bucket's latest persistent granted reset), so the
granted/stale-replay semantics proven for limit_window in test_limit_trends
are re-proven here against the NEW consumers, per bucket kind.

'now' is derived at seed time and every window is seeded relative to it.
"""

import json
import time
import unittest
from datetime import datetime, timezone

from app.tests._support import TempDBTestCase


def _iso(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _ins_event(conn, uuid, req, ts, inp=0, model="claude-opus-4-8"):
    conn.execute(
        "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,"
        "request_id,source_machine,project_dir,model,is_sidechain,agent_id,"
        "input_tokens,output_tokens,cache_creation_tokens,cache_read_tokens,"
        "account_email,plan,org_name,is_human_prompt,user_type) VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (uuid, "assistant", "2026-07-01T12:00:00Z", ts, "2026-07-01", "s1",
         req, "personal-mbp", "proj", model, 0, None, inp, 0, 0, 0,
         "me@gmail.com", "max", None, 0, None))
    conn.commit()


def _ins_pi_event(conn, uuid, req, ts, provider, model, cost):
    """A Pi-agent row with a client-reported cost (Pi rows are never
    server-priced), attributed to the same personal account as _ins_event."""
    conn.execute(
        "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,"
        "request_id,source_machine,project_dir,model,is_sidechain,agent_id,"
        "input_tokens,output_tokens,cache_creation_tokens,cache_read_tokens,"
        "account_email,plan,org_name,is_human_prompt,user_type,"
        "source_client,provider,reported_cost_total) VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (uuid, "assistant", "2026-07-01T12:00:00Z", ts, "2026-07-01", "s1",
         req, "personal-mbp", "proj", model, 0, None, 1000, 0, 0, 0,
         "me@gmail.com", "max", None, 0, None,
         "pi-agent", provider, cost))
    conn.commit()


def _ins_reading(conn, bucket, fetched, pct, resets_epoch):
    conn.execute(
        "INSERT INTO limit_readings(fetched_epoch, source, bucket, "
        "utilization, resets_at, resets_at_epoch) "
        "VALUES(?, 'server', ?, ?, NULL, ?)",
        (fetched, bucket, pct, resets_epoch))
    conn.commit()


class BucketWindowsTest(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def _seed_oauth(self, seven_day_resets, five_hour_resets,
                    scoped=None):
        """Seed the oauth_usage meta blob. scoped: {model_slug: resets_epoch}
        via the legacy seven_day_<model> dict form (normalize_usage_buckets
        maps those to scoped:<model> — same output shape as limits[])."""
        data = {
            "seven_day": {"utilization": 40, "resets_at": _iso(seven_day_resets)},
            "five_hour": {"utilization": 20, "resets_at": _iso(five_hour_resets)},
        }
        for slug, resets in (scoped or {}).items():
            data["seven_day_" + slug] = {
                "utilization": 30, "resets_at": _iso(resets)}
        stored = {"data": data, "updated_at": _iso(time.time())}
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('oauth_usage', ?)",
            (json.dumps(stored),))
        self.conn.commit()

    def _oauth(self):
        return self.client().get("/api/rate-limits?scope=personal").json()[
            "weekly_budget"]["oauth"]

    # ── five_hour_window ────────────────────────────────────────────────

    def test_five_hour_window_present_and_consistent(self):
        now = time.time()
        fh_resets = now + 2 * 3600            # 5h window: [resets-5h, resets)
        self._seed_oauth(now + 3 * 86400, fh_resets)
        # $5 of Opus inside the current 5h window, $5 well before it (must
        # be excluded even though it's inside the WEEKLY window).
        _ins_event(self.conn, "f1", "r1", now - 1800, inp=1_000_000)
        _ins_event(self.conn, "f2", "r2", now - 8 * 3600, inp=1_000_000)
        oauth = self._oauth()
        self.assertIn("five_hour_window", oauth)
        fw = oauth["five_hour_window"]
        expect_start = ((fh_resets // 60) * 60.0) - 5 * 3600
        self.assertAlmostEqual(fw["start_epoch"], expect_start, places=2)
        self.assertEqual(fw["start_epoch"] % 60, 0)  # minute-floor invariant
        self.assertAlmostEqual(fw["cost"], 5.0, places=2)

    def test_five_hour_window_truncated_by_granted_reset(self):
        now = time.time()
        fh_resets = now + 4 * 3600
        grant_t = now - 1800
        self._seed_oauth(now + 3 * 86400, fh_resets)
        # 55 -> 3 -> 3 on the five_hour bucket while its window is active:
        # a persistent granted reset at grant_t.
        for t, pct in ((grant_t - 600, 55.0), (grant_t, 3.0),
                       (grant_t + 600, 3.0)):
            _ins_reading(self.conn, "five_hour", t, pct, fh_resets)
        # $5 before the grant (inside resets-5h), $5 after: only post-grant
        # spend belongs to the granted window.
        _ins_event(self.conn, "g1", "rg1", grant_t - 900, inp=1_000_000)
        _ins_event(self.conn, "g2", "rg2", now - 60, inp=1_000_000)
        fw = self._oauth()["five_hour_window"]
        self.assertAlmostEqual(fw["start_epoch"], (grant_t // 60) * 60.0,
                               places=2)
        self.assertAlmostEqual(fw["cost"], 5.0, places=2)

    def test_five_hour_window_absent_when_resets_stale(self):
        """A resets_at already in the past means the stored pct describes an
        ENDED window — pairing it with cost through now would mix windows,
        so the key is omitted (client D6 nulls its marker the same way)."""
        now = time.time()
        self._seed_oauth(now + 3 * 86400, now - 600)
        _ins_event(self.conn, "s1", "r1", now - 1800, inp=1_000_000)
        oauth = self._oauth()
        self.assertNotIn("five_hour_window", oauth)
        self.assertIn("limit_window", oauth)  # weekly unaffected

    def test_five_hour_window_absent_when_resets_unparseable(self):
        now = time.time()
        stored = {"data": {
            "seven_day": {"utilization": 40, "resets_at": _iso(now + 86400)},
            "five_hour": {"utilization": 20, "resets_at": "soon"},
        }, "updated_at": _iso(now)}
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('oauth_usage', ?)",
            (json.dumps(stored),))
        self.conn.commit()
        self.assertNotIn("five_hour_window", self._oauth())

    # ── scoped:* window_cost ────────────────────────────────────────────

    def test_scoped_window_cost_model_filtered(self):
        """Each scoped bucket's window_cost counts ONLY its model family:
        $5 Opus + $10 Fable in-window -> scoped:opus 5.0, scoped:fable 10.0,
        never each other's or the sum."""
        now = time.time()
        resets = now + 3 * 86400
        self._seed_oauth(resets, now + 2 * 3600,
                         scoped={"opus": resets, "fable": resets})
        _ins_event(self.conn, "m1", "r1", now - 3600, inp=1_000_000,
                   model="claude-opus-4-8")
        _ins_event(self.conn, "m2", "r2", now - 3600, inp=1_000_000,
                   model="claude-fable-5")
        by_key = {b["key"]: b for b in self._oauth()["buckets"]}
        opus, fable = by_key["scoped:opus"], by_key["scoped:fable"]
        self.assertAlmostEqual(opus["window_cost"], 5.0, places=2)
        self.assertAlmostEqual(fable["window_cost"], 10.0, places=2)
        expect_start = ((resets // 60) * 60.0) - 7 * 86400
        for bkt in (opus, fable):
            self.assertAlmostEqual(bkt["window_start_epoch"], expect_start,
                                   places=2)
            self.assertEqual(bkt["window_start_epoch"] % 60, 0)
        # Non-scoped buckets never grow the fields.
        self.assertNotIn("window_cost", by_key["seven_day"])
        self.assertNotIn("window_cost", by_key["five_hour"])

    def test_scoped_window_cost_is_anthropic_only(self):
        """A scoped gauge is a Claude-subscription limit, so its dollars must
        come only from rows that consumed that subscription: Claude Code CLI
        rows and Pi rows served by the Anthropic provider. A Pi row that ran
        the SAME model family through OpenRouter, or any Codex row, is billed
        elsewhere and must not count (the family-name match alone would let
        'OpenRouter / Fable 5.1' through)."""
        now = time.time()
        resets = now + 3 * 86400
        self._seed_oauth(resets, now + 2 * 3600, scoped={"fable": resets})
        _ins_event(self.conn, "a1", "r1", now - 3600, inp=1_000_000,
                   model="claude-fable-5")                          # $10
        _ins_pi_event(self.conn, "a2", "r2", now - 3600,
                      "anthropic", "claude-fable-5-1", 4.0)         # counts
        _ins_pi_event(self.conn, "a3", "r3", now - 3600,
                      "openrouter", "anthropic/claude-fable-5-1", 7.0)  # no
        _ins_pi_event(self.conn, "a4", "r4", now - 3600,
                      "openai-codex", "gpt-5.6-sol", 9.0)           # no
        oauth = self._oauth()
        by_key = {b["key"]: b for b in oauth["buckets"]}
        self.assertAlmostEqual(by_key["scoped:fable"]["window_cost"], 14.0,
                               places=2)
        # The weekly and 5h dollars already follow the same rule.
        self.assertAlmostEqual(oauth["limit_window"]["cost"], 14.0, places=2)
        self.assertAlmostEqual(oauth["five_hour_window"]["cost"], 14.0,
                               places=2)

    def test_scoped_window_cost_truncated_by_granted_reset(self):
        """Granted resets are tracked PER BUCKET: a grant on scoped:opus
        moves only that bucket's window start; scoped:fable keeps the full
        resets-7d window."""
        now = time.time()
        resets = now + 3 * 86400
        grant_t = now - 86400
        self._seed_oauth(resets, now + 2 * 3600,
                         scoped={"opus": resets, "fable": resets})
        for t, pct in ((grant_t - 600, 55.0), (grant_t, 3.0),
                       (grant_t + 600, 3.0)):
            _ins_reading(self.conn, "scoped:opus", t, pct, resets)
        # Opus: $5 pre-grant + $5 post-grant. Fable: $10 pre-grant only.
        _ins_event(self.conn, "t1", "r1", grant_t - 3600, inp=1_000_000,
                   model="claude-opus-4-8")
        _ins_event(self.conn, "t2", "r2", now - 60, inp=1_000_000,
                   model="claude-opus-4-8")
        _ins_event(self.conn, "t3", "r3", grant_t - 3600, inp=1_000_000,
                   model="claude-fable-5")
        by_key = {b["key"]: b for b in self._oauth()["buckets"]}
        self.assertAlmostEqual(by_key["scoped:opus"]["window_cost"], 5.0,
                               places=2)
        self.assertAlmostEqual(
            by_key["scoped:opus"]["window_start_epoch"],
            (grant_t // 60) * 60.0, places=2)
        self.assertAlmostEqual(by_key["scoped:fable"]["window_cost"], 10.0,
                               places=2)

    def test_scoped_window_cost_from_limits_versioned_display_name(self):
        """Review MEDIUM: the PRIMARY limits[] path with a versioned
        display_name ('Opus 4.8' -> key scoped:opus_4_8) must still match
        the family — the raw slug ('opus_4_8') never substring-matches any
        space-separated display name, so the family stem ('opus') is what
        the filter keys on. Both Opus eras count; Fable does not."""
        now = time.time()
        resets = now + 3 * 86400
        stored = {"data": {"limits": [
            {"kind": "weekly_all", "percent": 40,
             "resets_at": _iso(resets)},
            {"kind": "session", "percent": 20,
             "resets_at": _iso(now + 2 * 3600)},
            {"kind": "weekly_scoped", "percent": 30,
             "resets_at": _iso(resets),
             "scope": {"model": {"display_name": "Opus 4.8", "id": None}}},
        ]}, "updated_at": _iso(now)}
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('oauth_usage', ?)",
            (json.dumps(stored),))
        self.conn.commit()
        _ins_event(self.conn, "v1", "r1", now - 3600, inp=1_000_000,
                   model="claude-opus-4-8")
        _ins_event(self.conn, "v2", "r2", now - 3600, inp=1_000_000,
                   model="claude-opus-4-7")
        _ins_event(self.conn, "v3", "r3", now - 3600, inp=1_000_000,
                   model="claude-fable-5")
        by_key = {b["key"]: b for b in self._oauth()["buckets"]}
        self.assertIn("scoped:opus_4_8", by_key)
        self.assertAlmostEqual(by_key["scoped:opus_4_8"]["window_cost"],
                               10.0, places=2)  # 4.8 + 4.7, never Fable

    def test_scoped_window_cost_absent_when_resets_stale(self):
        now = time.time()
        self._seed_oauth(now + 3 * 86400, now + 2 * 3600,
                         scoped={"opus": now - 600})
        _ins_event(self.conn, "x1", "r1", now - 3600, inp=1_000_000)
        by_key = {b["key"]: b for b in self._oauth()["buckets"]}
        self.assertNotIn("window_cost", by_key["scoped:opus"])
        self.assertNotIn("window_start_epoch", by_key["scoped:opus"])

    # ── 2026-07-09 account-grant regression (low-utilization reset) ─────

    def test_limit_window_truncated_by_low_util_grant(self):
        """Prod shape from the 2026-07-09 mid-window account grant: the
        weekly meter wiped 9 -> 0 with the anchor unchanged. Below
        RESET_DROP_PTS the old heuristic missed it, leaving limit_window
        anchored at resets-7d and counting ~11h of forgiven spend."""
        now = time.time()
        sd_resets = now + 5 * 86400          # weekly window began 2d ago
        self._seed_oauth(sd_resets, now + 2 * 3600)
        grant_t = now - 3600
        for t, pct in ((grant_t - 600, 9.0), (grant_t, 0.0),
                       (grant_t + 600, 0.0)):
            _ins_reading(self.conn, "seven_day", t, pct, sd_resets)
        # $5 pre-grant (forgiven), $5 post-grant (the real window spend).
        _ins_event(self.conn, "z1", "rz1", grant_t - 7200, inp=1_000_000)
        _ins_event(self.conn, "z2", "rz2", now - 60, inp=1_000_000)
        lw = self._oauth()["limit_window"]
        self.assertEqual(lw["start_epoch"], (grant_t // 60) * 60.0)
        self.assertAlmostEqual(lw["cost"], 5.0, places=2)

    def test_limit_window_truncated_by_corroborated_sibling_grant(self):
        """Same incident, harder shape: the weekly meter shows only a
        DECREASE (9 -> 1, usage resumed inside the poll gap) while a
        sibling scoped bucket cleared detection outright — the sibling
        event corroborates the account-level reset for the weekly
        window."""
        now = time.time()
        sd_resets = now + 5 * 86400
        self._seed_oauth(sd_resets, now + 2 * 3600,
                         scoped={"fable": sd_resets})
        grant_t = now - 3600
        for t, pct in ((grant_t - 600, 90.0), (grant_t, 0.0),
                       (grant_t + 600, 1.0)):
            _ins_reading(self.conn, "scoped:fable", t, pct, sd_resets)
        for t, pct in ((grant_t - 600, 9.0), (grant_t, 1.0),
                       (grant_t + 600, 1.0)):
            _ins_reading(self.conn, "seven_day", t, pct, sd_resets)
        _ins_event(self.conn, "c1", "rc1", grant_t - 7200, inp=1_000_000)
        _ins_event(self.conn, "c2", "rc2", now - 60, inp=1_000_000)
        lw = self._oauth()["limit_window"]
        self.assertEqual(lw["start_epoch"], (grant_t // 60) * 60.0)
        self.assertAlmostEqual(lw["cost"], 5.0, places=2)

    # ── failure isolation ───────────────────────────────────────────────

    def test_window_cost_failure_drops_only_dollar_fields(self):
        """A bug in the shared window-start helper must drop limit_window,
        five_hour_window and the scoped dollar fields — never the gauges'
        pct/resets data or the rest of the oauth block. (Patching the cost
        query itself would also kill the UNGUARDED top-level week_cost and
        500 the endpoint — the helper is exactly the new consumers' shared
        surface.)"""
        from unittest.mock import patch
        now = time.time()
        self._seed_oauth(now + 3 * 86400, now + 2 * 3600,
                         scoped={"opus": now + 3 * 86400})
        with patch("app.api._bucket_window_start",
                   side_effect=RuntimeError("boom")):
            r = self.client().get("/api/rate-limits?scope=personal")
        self.assertEqual(r.status_code, 200, r.text)
        oauth = r.json()["weekly_budget"]["oauth"]
        self.assertNotIn("limit_window", oauth)
        self.assertNotIn("five_hour_window", oauth)
        by_key = {b["key"]: b for b in oauth["buckets"]}
        self.assertNotIn("window_cost", by_key["scoped:opus"])
        for k in ("weekly_pct", "five_hour_pct", "buckets"):
            self.assertIn(k, oauth)


if __name__ == "__main__":
    unittest.main()
