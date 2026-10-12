"""GET /api/ha five_hour / weekly window blocks: same window as /api/rate-limits.

The HA feed (and the Pi usage tool that reads it) divides window spend by the
OAuth utilization to get implied_limit_usd. Both numbers must describe the
SAME span: the current limit window, which starts at resets_at - window or at
a later persistent GRANTED reset (a grant voids pre-grant usage), and ends at
the observation the utilization came from. Claude-subscription spend only.

'now' is derived at seed time and every window is seeded relative to it.
"""

import json
import time
from datetime import datetime, timezone
from unittest.mock import patch

import app.config
from app.claude_usage import MANAGED_SOURCE
from app.tests._support import TempDBTestCase
from app.tests.test_bucket_windows import _ins_event, _ins_pi_event

WEEK_S = 7 * 86400
FIVE_H_S = 5 * 3600


def _iso(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _minute(epoch):
    return (epoch // 60) * 60.0


def _ins_reading(conn, bucket, fetched, pct, resets_epoch, source="server"):
    conn.execute(
        "INSERT INTO limit_readings(fetched_epoch, source, bucket, "
        "utilization, resets_at, resets_at_epoch) VALUES(?, ?, ?, ?, NULL, ?)",
        (fetched, source, bucket, pct, resets_epoch))
    conn.commit()


class HAWindowsTest(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        self.now = time.time()
        # Odd seconds + microseconds on purpose: the served values must be
        # minute-truncated (anti-fingerprinting invariant).
        self.sd_resets = self.now + 3 * 86400 + 37.123456
        self.fh_resets = self.now + 2 * 3600 + 41.654321

    def _seed_oauth(self, sd_pct=40.0, fh_pct=20.0, observed=None, source=None):
        observed = self.now - 30 if observed is None else observed
        stored = {"data": {
            "seven_day": {"utilization": sd_pct, "resets_at": _iso(self.sd_resets)},
            "five_hour": {"utilization": fh_pct, "resets_at": _iso(self.fh_resets)},
        }, "updated_at": _iso(observed)}
        if source is not None:
            stored["source"] = source
            stored["observed_at_epoch"] = observed
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('oauth_usage', ?)",
            (json.dumps(stored),))
        self.conn.commit()

    def _ha(self):
        with patch.object(app.config, "LOCKED_SCOPE", None):
            return self.client().get(
                "/api/ha", headers={"X-API-Key": self.api_key}).json()

    def _grant(self, bucket, grant_t, resets, before=55.0, after=3.0,
               source="server"):
        """before -> after -> after while the window is active: a persistent
        granted reset at grant_t (same resets_at, so not a natural rollover)."""
        for t, pct in ((grant_t - 600, before), (grant_t, after),
                       (grant_t + 600, after)):
            _ins_reading(self.conn, bucket, t, pct, resets, source)

    # ── no reset: unchanged start ───────────────────────────────────────

    def test_no_reset_keeps_resets_minus_window_start(self):
        self._seed_oauth()
        _ins_event(self.conn, "n1", "r1", self.now - 2 * 86400, inp=1_000_000)
        _ins_event(self.conn, "n2", "r2", self.now - 600, inp=1_000_000)
        body = self._ha()
        weekly, five = body["weekly"], body["five_hour"]
        self.assertEqual(weekly["window_start"],
                         _iso(_minute(self.sd_resets) - WEEK_S))
        self.assertEqual(five["window_start"],
                         _iso(_minute(self.fh_resets) - FIVE_H_S))
        self.assertAlmostEqual(weekly["spend_usd"], 10.0, places=2)
        self.assertAlmostEqual(weekly["implied_limit_usd"], 25.0, places=2)
        self.assertAlmostEqual(five["spend_usd"], 5.0, places=2)
        # Existing fields keep their meaning.
        self.assertEqual(weekly["resets_at"], _iso(_minute(self.sd_resets)))
        self.assertEqual(weekly["pct_used"], 40.0)

    # ── granted mid-window reset moves the start and the spend ──────────

    def test_weekly_granted_reset_moves_start_and_spend(self):
        grant_t = self.now - 86400 + 13.5
        self._seed_oauth(sd_pct=10.0)
        self._grant("seven_day", grant_t, self.sd_resets, before=97.0, after=0.0)
        _ins_event(self.conn, "w1", "r1", grant_t - 3600, inp=1_000_000)  # voided
        _ins_event(self.conn, "w2", "r2", self.now - 600, inp=1_000_000)  # counts
        weekly = self._ha()["weekly"]
        self.assertEqual(weekly["window_start"], _iso(_minute(grant_t)))
        self.assertAlmostEqual(weekly["spend_usd"], 5.0, places=2)
        self.assertAlmostEqual(weekly["implied_limit_usd"], 50.0, places=2)

    def test_five_hour_granted_reset_moves_start_and_spend(self):
        grant_t = self.now - 1800
        self._seed_oauth(fh_pct=50.0)
        self._grant("five_hour", grant_t, self.fh_resets)
        _ins_event(self.conn, "f1", "r1", grant_t - 900, inp=1_000_000)
        _ins_event(self.conn, "f2", "r2", self.now - 60, inp=1_000_000)
        five = self._ha()["five_hour"]
        self.assertEqual(five["window_start"], _iso(_minute(grant_t)))
        self.assertAlmostEqual(five["spend_usd"], 5.0, places=2)
        self.assertAlmostEqual(five["implied_limit_usd"], 10.0, places=2)

    def test_stale_replay_does_not_move_start(self):
        """60 -> 40 -> 62: the 40 is a stale out-of-order push, not a grant
        (the next reading recovered), so the window keeps its full span."""
        t0 = self.now - 86400
        self._seed_oauth()
        for t, pct in ((t0 - 600, 60.0), (t0, 40.0), (t0 + 600, 62.0)):
            _ins_reading(self.conn, "seven_day", t, pct, self.sd_resets)
        _ins_event(self.conn, "s1", "r1", t0 - 3600, inp=1_000_000)
        _ins_event(self.conn, "s2", "r2", self.now - 600, inp=1_000_000)
        weekly = self._ha()["weekly"]
        self.assertEqual(weekly["window_start"],
                         _iso(_minute(self.sd_resets) - WEEK_S))
        self.assertAlmostEqual(weekly["spend_usd"], 10.0, places=2)

    def test_grant_from_inactive_history_source_is_ignored(self):
        """The managed collector owns the snapshot, so a grant visible only
        in legacy readings must not move the window (same rule as
        /api/rate-limits)."""
        grant_t = self.now - 86400
        self._seed_oauth(source=MANAGED_SOURCE)
        self._grant("seven_day", grant_t, self.sd_resets, source="server")
        _ins_event(self.conn, "h1", "r1", grant_t - 3600, inp=1_000_000)
        weekly = self._ha()["weekly"]
        self.assertEqual(weekly["window_start"],
                         _iso(_minute(self.sd_resets) - WEEK_S))
        self.assertAlmostEqual(weekly["spend_usd"], 5.0, places=2)

    def test_grant_from_active_managed_source_moves_start(self):
        grant_t = self.now - 86400
        self._seed_oauth(source=MANAGED_SOURCE)
        self._grant("seven_day", grant_t, self.sd_resets, source=MANAGED_SOURCE)
        _ins_event(self.conn, "m1", "r1", grant_t - 3600, inp=1_000_000)
        weekly = self._ha()["weekly"]
        self.assertEqual(weekly["window_start"], _iso(_minute(grant_t)))
        self.assertAlmostEqual(weekly["spend_usd"], 0.0, places=2)

    # ── spend window ends at the observation ────────────────────────────

    def test_spend_ends_at_observation(self):
        """pct_used describes the observation, so spend made after it is not
        part of the ratio yet."""
        observed = self.now - 900
        self._seed_oauth(observed=observed, source=MANAGED_SOURCE)
        _ins_event(self.conn, "o1", "r1", observed - 600, inp=1_000_000)
        _ins_event(self.conn, "o2", "r2", observed + 300, inp=1_000_000)
        body = self._ha()
        self.assertAlmostEqual(body["weekly"]["spend_usd"], 5.0, places=2)
        self.assertAlmostEqual(body["five_hour"]["spend_usd"], 5.0, places=2)

    # ── Claude-subscription spend only ──────────────────────────────────

    def test_non_anthropic_pi_rows_excluded(self):
        self._seed_oauth()
        t = self.now - 600
        _ins_event(self.conn, "a1", "r1", t, inp=1_000_000)               # $5
        _ins_pi_event(self.conn, "a2", "r2", t, "anthropic",
                      "claude-opus-4-8", 4.0)                              # counts
        _ins_pi_event(self.conn, "a3", "r3", t, "openai-codex",
                      "gpt-5.6-sol", 9.0)                                  # no
        _ins_pi_event(self.conn, "a4", "r4", t, "opencode-go",
                      "kimi-k3", 7.0)                                      # no
        body = self._ha()
        self.assertAlmostEqual(body["weekly"]["spend_usd"], 9.0, places=2)
        self.assertAlmostEqual(body["five_hour"]["spend_usd"], 9.0, places=2)

    # ── anti-fingerprinting ─────────────────────────────────────────────

    def test_window_start_is_minute_truncated(self):
        grant_t = self.now - 1800 + 0.987654
        self._seed_oauth()
        self._grant("five_hour", grant_t, self.fh_resets)
        body = self._ha()
        for block in (body["weekly"], body["five_hour"]):
            start = block["window_start"]
            self.assertTrue(start.endswith(":00+00:00"), start)
            dt = datetime.fromisoformat(start)
            self.assertEqual((dt.second, dt.microsecond), (0, 0))
