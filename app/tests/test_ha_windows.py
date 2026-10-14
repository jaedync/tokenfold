"""GET /api/ha five_hour / weekly window blocks: same window as /api/rate-limits.

The HA feed (and the Pi usage tool that reads it) divides window spend by the
OAuth utilization to get implied_limit_usd. Both numbers must describe the
SAME span: the current limit window, which starts at resets_at - window or at
a later persistent GRANTED reset (a grant voids pre-grant usage), and ends at
the observation the utilization came from. Claude-subscription spend only.

'now' is derived at seed time and every window is seeded relative to it.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from unittest.mock import patch

import app.config
import app.ha
from app.aggregator import build_dashboard_data
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


class _HABase(TempDBTestCase):
    """Seeding helpers only; no tests, so subclasses do not rerun any."""

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


class HAWindowsTest(_HABase):
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

    def test_meridian_twin_pair_counts_once_in_window_spend(self):
        """Pi row with a positive reported cost + its SDK copy (same msg_ id)
        are one request: the window bills the Pi cost, not both."""
        self._seed_oauth(fh_pct=50.0)
        t = self.now - 600
        _ins_pi_event(self.conn, "pi1", "msg_twin", t, "anthropic",
                      "claude-opus-4-8", 3.0)
        _ins_event(self.conn, "sdk1", "req_sdk1", t + 5, inp=1_000_000)  # $5
        self.conn.execute("UPDATE events SET message_id='msg_twin'")
        self.conn.commit()
        body = self._ha()
        self.assertAlmostEqual(body["five_hour"]["spend_usd"], 3.0, places=2)
        self.assertAlmostEqual(body["weekly"]["spend_usd"], 3.0, places=2)

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


DOLLAR_KEYS = ("spend_usd", "implied_limit_usd", "window_start")


class HAWindowsSafetyTest(_HABase):
    """No pairing of a percent with spend from another time, no fallback to
    resets_at - window on error, and agreement with /api/rate-limits."""

    def _assert_no_dollars(self, block):
        for key in DOLLAR_KEYS:
            self.assertIn(key, block)
            self.assertIsNone(block[key], key)
        # The quota fields stay: HA history needs them even without dollars.
        self.assertIsNotNone(block["pct_used"])
        self.assertIsNotNone(block["resets_at"])
        self.assertIsNotNone(block["resets_in_s"])

    # ── parity with /api/rate-limits ────────────────────────────────────

    def test_matches_rate_limits_windows(self):
        grant_t = self.now - 86400
        self._seed_oauth(sd_pct=12.0, fh_pct=30.0, source=MANAGED_SOURCE)
        self._grant("seven_day", grant_t, self.sd_resets, source=MANAGED_SOURCE)
        _ins_event(self.conn, "p1", "r1", grant_t - 3600, inp=1_000_000)
        _ins_event(self.conn, "p2", "r2", self.now - 7200, inp=2_000_000)
        _ins_event(self.conn, "p3", "r3", self.now - 600, inp=1_000_000)
        _ins_pi_event(self.conn, "p4", "r4", self.now - 600, "openai-codex",
                      "gpt-5.6-sol", 9.0)
        ha = self._ha()
        oauth = self.client().get("/api/rate-limits?scope=personal").json()[
            "weekly_budget"]["oauth"]
        for block, window in ((ha["weekly"], oauth["limit_window"]),
                              (ha["five_hour"], oauth["five_hour_window"])):
            self.assertEqual(block["spend_usd"], window["cost"])
            self.assertEqual(datetime.fromisoformat(
                block["window_start"]).timestamp(), window["start_epoch"])
        self.assertAlmostEqual(ha["weekly"]["spend_usd"], 15.0, places=2)

    # ── error isolation ─────────────────────────────────────────────────

    def test_window_error_keeps_quota_and_other_fields(self):
        self._seed_oauth()
        _ins_event(self.conn, "e1", "r1", self.now - 600, inp=1_000_000)
        for target in ("app.ha._bucket_window_start", "app.ha.compute_window_cost"):
            with self.subTest(target=target), \
                    patch(target, side_effect=RuntimeError("boom")):
                resp = None
                with patch.object(app.config, "LOCKED_SCOPE", None):
                    resp = self.client().get(
                        "/api/ha", headers={"X-API-Key": self.api_key})
                self.assertEqual(resp.status_code, 200)
                body = resp.json()
                self.assertIn("cost_today_usd", body)
                self.assertIn("updated_at_epoch", body)
                self._assert_no_dollars(body["weekly"])
                self._assert_no_dollars(body["five_hour"])

    # ── no end: the percent cannot be paired with spend ─────────────────

    def test_no_dollars_when_observation_missing(self):
        stored = {"data": {
            "seven_day": {"utilization": 40.0, "resets_at": _iso(self.sd_resets)},
            "five_hour": {"utilization": 20.0, "resets_at": _iso(self.fh_resets)},
        }}
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('oauth_usage', ?)",
            (json.dumps(stored),))
        self.conn.commit()
        _ins_event(self.conn, "m1", "r1", self.now - 600, inp=1_000_000)
        body = self._ha()
        self._assert_no_dollars(body["weekly"])
        self._assert_no_dollars(body["five_hour"])

    def test_no_dollars_when_observation_in_future(self):
        self._seed_oauth(observed=self.now + 600)
        _ins_event(self.conn, "f1", "r1", self.now - 600, inp=1_000_000)
        body = self._ha()
        self._assert_no_dollars(body["weekly"])
        self._assert_no_dollars(body["five_hour"])

    def test_no_dollars_when_observation_before_start(self):
        # Before both starts: weekly starts ~4 days ago, five_hour ~3h ago.
        self._seed_oauth(observed=self.now - 5 * 86400)
        _ins_event(self.conn, "b1", "r1", self.now - 600, inp=1_000_000)
        body = self._ha()
        self._assert_no_dollars(body["weekly"])
        self._assert_no_dollars(body["five_hour"])

    # ── grant corroborated by a sibling bucket ──────────────────────────

    def test_sibling_corroborated_grant_moves_weekly_start(self):
        """An account-wide grant zeroed five_hour (55 -> 3) while seven_day
        only fell 9 -> 7 (usage resumed inside the poll gap): too small for
        its own rules, but the sibling's grant cuts its window too."""
        grant_t = self.now - 1800
        self._seed_oauth(sd_pct=8.0)
        self._grant("five_hour", grant_t, self.fh_resets)
        for t, pct in ((grant_t - 600, 9.0), (grant_t, 7.0),
                       (grant_t + 600, 8.0)):
            _ins_reading(self.conn, "seven_day", t, pct, self.sd_resets)
        _ins_event(self.conn, "c1", "r1", grant_t - 3600, inp=1_000_000)
        _ins_event(self.conn, "c2", "r2", self.now - 60, inp=1_000_000)
        weekly = self._ha()["weekly"]
        self.assertEqual(weekly["window_start"], _iso(_minute(grant_t)))
        self.assertAlmostEqual(weekly["spend_usd"], 5.0, places=2)

    def test_no_dollars_when_observation_after_reset(self):
        """A stale snapshot: its resets_at passed before the observation, so
        the percent describes an ended window (/api/rate-limits refuses the
        same case through quota_window_valid)."""
        self.fh_resets = self.now - 600.0
        self._seed_oauth(observed=self.now - 30)
        _ins_event(self.conn, "r1", "r1", self.now - 1200, inp=1_000_000)
        five = self._ha()["five_hour"]
        self._assert_no_dollars(five)
        self.assertEqual(five["resets_in_s"], 0)

    def test_snapshot_read_failure_keeps_cost_totals(self):
        self._seed_oauth()
        with patch("app.ha.read_conn", side_effect=RuntimeError("db gone")):
            with patch.object(app.config, "LOCKED_SCOPE", None):
                resp = self.client().get(
                    "/api/ha", headers={"X-API-Key": self.api_key})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIsNotNone(body["cost_today_usd"])
        self.assertIsNotNone(body["cost_total_usd"])
        for key in ("five_hour", "weekly", "model_buckets", "updated_at_epoch"):
            self.assertIn(key, body)
            self.assertIsNone(body[key], key)


class HAUpdatedAtTest(_HABase):
    """updated_at_epoch leaves the server, so it is minute-floored, and it
    comes from the observation clock the window spend ends at."""

    def _seed_stored(self, stored):
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('oauth_usage', ?)",
            (json.dumps(stored),))
        self.conn.commit()

    def _data(self):
        return {
            "seven_day": {"utilization": 40.0, "resets_at": _iso(self.sd_resets)},
            "five_hour": {"utilization": 20.0, "resets_at": _iso(self.fh_resets)},
        }

    def test_managed_observation_clock_floored_to_minute(self):
        # updated_at differs from the observation on purpose: the field must
        # follow observed_at_epoch, the end of the window spend.
        observed = _minute(self.now - 300) + 37.654321
        self._seed_stored({"data": self._data(), "source": MANAGED_SOURCE,
                           "observed_at_epoch": observed,
                           "updated_at": _iso(self.now - 20.5)})
        value = self._ha()["updated_at_epoch"]
        self.assertIsInstance(value, int)
        self.assertEqual(value, int(_minute(observed)))

    def test_legacy_updated_at_floored_to_minute(self):
        # 37 s into the minute: the old 10 s rounding gave minute + 40.
        observed = _minute(self.now - 300) + 37.25
        self._seed_stored({"data": self._data(), "updated_at": _iso(observed)})
        value = self._ha()["updated_at_epoch"]
        self.assertIsInstance(value, int)
        self.assertEqual(value, int(_minute(observed)))

    def test_none_when_no_clock_parses(self):
        self._seed_stored({"data": self._data(), "updated_at": "not-a-time"})
        self.assertIsNone(self._ha()["updated_at_epoch"])


def _loop_running():
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


class HAThreadpoolTest(_HABase):
    """The blocking SQLite work runs off the event loop, and the body is
    unchanged."""

    def test_blocking_work_runs_off_event_loop(self):
        self._seed_oauth()
        seen = []
        real_dash, real_blocks = app.ha.build_dashboard_data, app.ha._oauth_blocks

        def dash(scope):
            seen.append(("dash", _loop_running()))
            return real_dash(scope)

        def blocks(conn, now_epoch):
            seen.append(("blocks", _loop_running()))
            return real_blocks(conn, now_epoch)

        with patch("app.ha.build_dashboard_data", side_effect=dash), \
                patch("app.ha._oauth_blocks", side_effect=blocks):
            body = self._ha()
        self.assertEqual(seen, [("dash", False), ("blocks", False)])
        self.assertIsNotNone(body["weekly"])

    def test_body_unchanged(self):
        self._seed_oauth()
        _ins_event(self.conn, "b1", "r1", self.now - 600, inp=1_000_000)
        with patch.object(app.ha, "time") as clock:
            clock.time.return_value = self.now
            body = self._ha()
        # Cost totals come from the aggregator (day summaries), not from
        # this route; the route only rounds them.
        dash = build_dashboard_data("personal")

        def block(resets, window_s, pct, implied):
            return {
                "pct_used": pct, "spend_usd": 5.0, "implied_limit_usd": implied,
                "resets_at": _iso(_minute(resets)),
                "resets_in_s": int(_minute(resets) - self.now),
                "window_start": _iso(_minute(resets) - window_s),
            }

        self.assertEqual(body, {
            "cost_today_usd": round(
                (dash.get("today") or {}).get("cost", 0.0) or 0.0, 2),
            "cost_total_usd": round(dash.get("total_cost", 0.0) or 0.0, 2),
            "five_hour": block(self.fh_resets, FIVE_H_S, 20.0, 25.0),
            "weekly": block(self.sd_resets, WEEK_S, 40.0, 12.5),
            "model_buckets": None,
            "updated_at_epoch": int(_minute(self.now - 30)),
        })
