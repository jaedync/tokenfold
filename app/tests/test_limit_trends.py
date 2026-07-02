"""Workstream D unit + route tests: compute_burn (D1), the /api/rate-limits
oauth.trend block (D2), and the limit_window budget-window fields (D5 server
half).

'now' is always passed/derived explicitly and the DB is seeded relative to it,
so nothing here depends on wall-clock drift between seed and assertion.
"""

import json
import time
import unittest
from datetime import datetime, timezone

from app.tests._support import TempDBTestCase

T0 = 1751000000.0  # arbitrary frozen anchor; not minute-aligned matters only
                   # for series/eta flooring, never for compute_burn itself.


class _BurnBase(TempDBTestCase):
    def _row(self, bucket, t, pct, resets_at=None, resets_at_epoch=None,
             source="server"):
        self.conn.execute(
            "INSERT INTO limit_readings(fetched_epoch, source, bucket, "
            "utilization, resets_at, resets_at_epoch) VALUES(?,?,?,?,?,?)",
            (t, source, bucket, pct, resets_at, resets_at_epoch))
        self.conn.commit()


# ---------------------------------------------------------------------------
# D1 — compute_burn
# ---------------------------------------------------------------------------

class ComputeBurnTest(_BurnBase):

    def test_flat_series_is_zero(self):
        """Fix 4 does NOT change this expected value: points[0][0] == t0 ==
        boundary (now - window_s), so effective_s = now - max(boundary,
        points[0][0]) = now - t0 = window_s exactly (the straddler-clamps-
        to-boundary case, where Fix 4's docstring says 'nothing changes').
        delta is 0 either way, so pct_per_hr stays 0.0 regardless."""
        from app.limit_trends import compute_burn
        for t, pct in ((T0, 42.0), (T0 + 1800, 42.0), (T0 + 3600, 42.0)):
            self._row("seven_day", t, pct)
        r = compute_burn(self.conn, "seven_day", now=T0 + 3600, window_s=3600)
        self.assertEqual(r["pct_per_hr"], 0.0)
        self.assertEqual(r["samples"], 3)
        self.assertEqual(r["resets_in_window"], 0)

    def test_stepping_series_exact_value(self):
        """Readings 42,42,43,43,44 at 600s spacing, t0..t0+2400, now=t0+2400,
        window=3600s -> boundary = t0-1200.

        No straddler exists (nothing before t0-1200) and no reset, so the
        segment is all five points. û is clamped constant beyond the ends:
          û(now)        = û(t0+2400) = 44   (last point / right clamp)
          û(now-window) = û(t0-1200) = 42   (boundary is left of the first
                                             point t0 -> clamp to first = 42)
        delta = 44 - 42 = 2.

        Fix 4 hand-derivation: the segment's first point (t0) sits INSIDE the
        window (t0 > boundary = t0-1200), so effective_s is the OBSERVED span
        from t0, not the full requested window_s:
          effective_s = now - max(boundary, points[0][0])
                      = (t0+2400) - max(t0-1200, t0) = (t0+2400) - t0 = 2400s
          pct_per_hr = delta / (effective_s/3600) = 2 / (2400/3600) = 3.0
        (was 2.0 pre-Fix-4, when the denominator was always the full
        window_s=3600 regardless of how much of it the segment covered.)
        """
        from app.limit_trends import compute_burn
        for i, pct in enumerate((42.0, 42.0, 43.0, 43.0, 44.0)):
            self._row("seven_day", T0 + i * 600, pct)
        r = compute_burn(self.conn, "seven_day", now=T0 + 2400, window_s=3600)
        self.assertEqual(r["pct_per_hr"], 3.0)
        self.assertEqual(r["samples"], 5)
        self.assertEqual(r["resets_in_window"], 0)

    def test_mid_window_drop_uses_post_reset_segment_only(self):
        """55 -> 3 drop mid-window (prev resets_at 3h in the future) is a reset;
        burn must come from the post-drop segment (3,5,7,9) only."""
        from app.limit_trends import compute_burn
        active = T0 + 3 * 3600          # prev window still open -> real reset
        far = T0 + 7 * 86400
        self._row("seven_day", T0, 55.0, resets_at_epoch=active)
        self._row("seven_day", T0 + 600, 3.0, resets_at_epoch=far)
        self._row("seven_day", T0 + 1200, 5.0, resets_at_epoch=far)
        self._row("seven_day", T0 + 1800, 7.0, resets_at_epoch=far)
        self._row("seven_day", T0 + 2400, 9.0, resets_at_epoch=far)
        r = compute_burn(self.conn, "seven_day", now=T0 + 2400, window_s=3600)
        self.assertEqual(r["resets_in_window"], 1)
        self.assertEqual(r["samples"], 4)  # only the post-drop segment
        # segment (3@t0+600 .. 9@t0+2400): û(now)=9, û(t0-1200) clamps to the
        # segment's first reading = 3 (t0-1200 is left of t0+600). delta=6.
        #
        # Fix 4 hand-derivation: the reset trimmed the segment so its first
        # point (t0+600) sits INSIDE the window (t0+600 > boundary=t0-1200),
        # so effective_s is the OBSERVED span from t0+600, not the full
        # window_s=3600 the pre-Fix-4 code divided by:
        #   effective_s = now - max(boundary, points[0][0])
        #               = (t0+2400) - max(t0-1200, t0+600)
        #               = (t0+2400) - (t0+600) = 1800s  (30 min)
        #   pct_per_hr = delta / (effective_s/3600) = 6 / (1800/3600) = 12.0
        # (was 6.0 pre-Fix-4 — dividing the same 6-point rise by the full
        # 3600s window diluted a 30-min post-reset climb by 2x.)
        self.assertEqual(r["pct_per_hr"], 12.0)

    def test_single_reading_is_none(self):
        from app.limit_trends import compute_burn
        self._row("seven_day", T0, 42.0)
        r = compute_burn(self.conn, "seven_day", now=T0 + 3600, window_s=3600)
        self.assertIsNone(r["pct_per_hr"])
        self.assertEqual(r["samples"], 1)

    def test_two_readings_under_min_span_is_none(self):
        from app.limit_trends import compute_burn
        self._row("seven_day", T0, 42.0)
        self._row("seven_day", T0 + 600, 44.0)  # 600s span < 900s floor
        r = compute_burn(self.conn, "seven_day", now=T0 + 600, window_s=3600)
        self.assertIsNone(r["pct_per_hr"])
        self.assertEqual(r["samples"], 2)

    def test_duplicate_fetched_epoch_collapses_to_one_point(self):
        from app.limit_trends import compute_burn
        # Two rows land the same second (server + client push).
        self._row("seven_day", T0, 42.0, source="server")
        self._row("seven_day", T0, 42.0, source="client")
        self._row("seven_day", T0 + 1800, 42.0)
        self._row("seven_day", T0 + 3600, 42.0)
        r = compute_burn(self.conn, "seven_day", now=T0 + 3600, window_s=3600)
        self.assertEqual(r["samples"], 3)  # duplicate second counted once
        self.assertEqual(r["pct_per_hr"], 0.0)

    def test_resets_in_window_boundary_is_inclusive(self):
        """Fix 8: a reset landing exactly ON the window boundary counts as
        IN-window (>=), not excluded by a strict '>' comparison.

        Reset event fires at cur.fetched_epoch = t0+900 (55 -> 5 drop, prev
        window active). window_s=1000, now=t0+1900 -> boundary =
        now-window_s = t0+900, exactly the event's at_epoch."""
        from app.limit_trends import compute_burn
        active = T0 + 3600
        self._row("seven_day", T0, 55.0, resets_at_epoch=active)
        self._row("seven_day", T0 + 900, 5.0, resets_at_epoch=T0 + 7 * 86400)
        r = compute_burn(self.conn, "seven_day", now=T0 + 1900, window_s=1000)
        self.assertEqual(r["resets_in_window"], 1)

    def test_boundary_interpolates_between_straddler_and_first_in_window(self):
        """Fix 5(a): window boundary falls STRICTLY BETWEEN the straddler and
        the first in-window reading (with a straddler present) — û(boundary)
        must linearly interpolate rather than clamp.

        Rows: t0=10 (straddler), t0+1800=30, t0+3600=50, t0+5400=70 (=now).
        window_s=4500 -> boundary = now-4500 = t0+900, strictly inside
        (t0, t0+1800) -> straddler = t0(10), first-in-window = t0+1800(30).

        Hand-derivation:
          û(boundary) = 10 + (900-0)/(1800-0) * (30-10) = 10 + 0.5*20 = 20
          û(now) = 70 (last reading, clamp)
          delta = 70 - 20 = 50
          effective_s: points[0][0]=t0 <= boundary=t0+900, so the straddler
            interpolates within the segment -> effective_s == window_s ==
            4500s exactly (Fix 4's 'nothing changes' case).
          pct_per_hr = 50 / (4500/3600) = 50 / 1.25 = 40.0
        """
        from app.limit_trends import compute_burn
        self._row("seven_day", T0, 10.0)
        self._row("seven_day", T0 + 1800, 30.0)
        self._row("seven_day", T0 + 3600, 50.0)
        self._row("seven_day", T0 + 5400, 70.0)
        r = compute_burn(self.conn, "seven_day", now=T0 + 5400, window_s=4500)
        self.assertEqual(r["pct_per_hr"], 40.0)
        self.assertEqual(r["samples"], 4)


# ---------------------------------------------------------------------------
# D2 — bucket_trend (unit-level: direct calls, no HTTP/oauth meta needed)
# ---------------------------------------------------------------------------

class BucketTrendUnitTest(_BurnBase):

    def test_epoch_fields_are_minute_floored(self):
        """Fix 2: every epoch-valued field in the trend payload must be
        minute-floored — eta_100_epoch and series[i][0] already were;
        resets[].at_epoch (and resets_at_epoch_before/after, already floored
        inside detect_resets) is the field this fix adds flooring to.

        T0 % 60 == 20 (not minute-aligned) and +600/+1800/+3600 are all
        multiples of 60, so every raw fetched_epoch here keeps that same
        %60==20 remainder — a real regression (unfloored at_epoch) would fail
        the %60==0 assertion below, not pass it vacuously.

        Seed: t0=50 (resets_at_epoch active, T0+3600), t0+600=2 (drop >=10pt
        -> reset event at_epoch=t0+600), t0+1800=20, t0+3600=38 (=now,
        rising -> positive burn_6h -> eta_100_epoch gets computed, not null).
        """
        from app.limit_trends import bucket_trend
        self._row("seven_day", T0, 50.0, resets_at_epoch=T0 + 3600)
        self._row("seven_day", T0 + 600, 2.0, resets_at_epoch=T0 + 7 * 86400)
        self._row("seven_day", T0 + 1800, 20.0, resets_at_epoch=T0 + 7 * 86400)
        self._row("seven_day", T0 + 3600, 38.0, resets_at_epoch=T0 + 7 * 86400)
        now = T0 + 3600
        tr = bucket_trend(self.conn, "seven_day", now)

        self.assertEqual(len(tr["resets"]), 1)  # the 50->2 drop
        self.assertIsNotNone(tr["eta_100_epoch"])  # burn_6h > 0 here

        if tr["eta_100_epoch"] is not None:
            self.assertEqual(tr["eta_100_epoch"] % 60, 0)
        for epoch, _pct in tr["series"]:
            self.assertEqual(epoch % 60, 0)
        for evt in tr["resets"]:
            self.assertEqual(evt["at_epoch"] % 60, 0)
            for key in ("resets_at_epoch_before", "resets_at_epoch_after"):
                if evt[key] is not None:
                    self.assertEqual(evt[key] % 60, 0)

    def test_jitter_dip_yields_negative_burn_null_eta_under_pace(self):
        """Fix 5(b): a small dip (below the 10pt reset-drop threshold) is
        genuine jitter, not a reset — burn goes negative, eta_100_epoch is
        suppressed (only computed when the relevant burn is > 0), and pace
        reads 'under' (a negative burn is always < even_drain*0.9).

        Rows: t0=40, t0+1200=42, t0+2400=39 (=now; 3pt dip, no
        resets_at_epoch set anywhere -> detect_resets never fires).
        boundary(6h) = now-21600, well before t0 -> no straddler, clamps.
        û(now)=39 (last), û(boundary)=40 (clamp to first). delta=-1.
        effective_s = now - points[0][0] = (t0+2400)-t0 = 2400s.
        pct_per_hr = -1 / (2400/3600) = -1.5.
        """
        from app.limit_trends import bucket_trend
        self._row("seven_day", T0, 40.0)
        self._row("seven_day", T0 + 1200, 42.0)
        self._row("seven_day", T0 + 2400, 39.0)
        now = T0 + 2400
        tr = bucket_trend(self.conn, "seven_day", now)
        self.assertEqual(tr["burn_6h_pct_per_hr"], -1.5)
        self.assertIsNone(tr["eta_100_epoch"])
        self.assertEqual(tr["pace"], "under")

    def test_five_hour_bucket_selects_burn_1h_not_burn_6h(self):
        """Fix 5(d): the five_hour bucket's eta/pace must follow burn_1h, not
        burn_6h — seed windows with clearly different slopes and confirm the
        1h number (not the 6h number) drives the verdict.

        Rows (bucket=five_hour): t0=10 (flat for 5h) .. t0+18000=10, then a
        sharp climb in the final hour to t0+21600=40 (=now).
          burn_1h: boundary=now-3600=t0+18000 (coincides with the 2nd row).
            û(now)=40, û(boundary)=10 (exact point match). delta=30.
            effective_s = now-max(boundary,t0) = now-(t0+18000) = 3600s
            (== window_s, boundary sits at/after the segment start).
            pct_per_hr = 30/(3600/3600) = 30.0
          burn_6h: boundary=now-21600=t0. û(boundary)=10 (t<=points[0][0]).
            effective_s = now-t0 = 21600s (== window_s). delta=30.
            pct_per_hr = 30/(21600/3600) = 5.0
        even_drain (five_hour) = 100/5 = 20.0 exactly.
          burn_1h=30.0 > 20*1.1=22.0 -> pace 'over' if burn_1h is used.
          burn_6h=5.0  < 20*0.9=18.0 -> pace 'under' if burn_6h were used.
        eta uses relevant=burn_1h=30.0, current_pct=40 (latest reading):
          eta = floor_to_minute(now + (100-40)/30.0*3600) = now + 7200,
          floored to the minute = 1751028780.0 (verified by direct
          computation of the same formula bucket_trend uses).
        """
        from app.limit_trends import bucket_trend
        self._row("five_hour", T0, 10.0)
        self._row("five_hour", T0 + 18000, 10.0)
        self._row("five_hour", T0 + 21600, 40.0)
        now = T0 + 21600
        tr = bucket_trend(self.conn, "five_hour", now)
        self.assertEqual(tr["burn_1h_pct_per_hr"], 30.0)
        self.assertEqual(tr["burn_6h_pct_per_hr"], 5.0)
        # The verdict must follow the 1h number (over), not the 6h number
        # (which alone would read 'under').
        self.assertEqual(tr["pace"], "over")
        expect_eta = (((now + (100.0 - 40.0) / 30.0 * 3600.0) // 60) * 60.0)
        self.assertEqual(expect_eta, 1751028780.0)  # pinned, not just derived
        self.assertEqual(tr["eta_100_epoch"], expect_eta)

    # Fix 5(c): pace deadband is +/-10% of even_drain, INCLUSIVE at both
    # edges ('on' at exactly 0.9x/1.1x, per the code's strict < / >
    # comparisons — neither is strictly beyond the threshold, so both fall
    # through to the 'on' else-branch), 'under'/'over' clearly beyond.
    #
    # bucket_trend's is_five check is a literal string match on "five_hour",
    # so every case below MUST use that bucket name to get the clean
    # even_drain=100/5=20.0 (20.0*0.9==18.0 and 20.0*1.1==22.0 exactly in
    # this runtime, verified separately — no repeating-decimal edge like
    # 100/168 would have). Two points 3600s apart with boundary==
    # points[0][0] gives effective_s==window_s==3600 exactly, so
    # pct_per_hr == delta directly (no rounding beyond bucket_trend's own
    # 2dp round()).

    def test_pace_deadband_edge_on_low(self):
        """relevant == even_drain*0.9 == 18.0 exactly -> 'on' (inclusive)."""
        from app.limit_trends import bucket_trend
        self._row("five_hour", T0, 10.0)
        self._row("five_hour", T0 + 3600, 28.0)  # delta=18.0
        tr = bucket_trend(self.conn, "five_hour", T0 + 3600)
        self.assertEqual(tr["burn_1h_pct_per_hr"], 18.0)
        self.assertEqual(tr["pace"], "on")

    def test_pace_deadband_edge_on_high(self):
        """relevant == even_drain*1.1 == 22.0 exactly -> 'on' (inclusive)."""
        from app.limit_trends import bucket_trend
        self._row("five_hour", T0, 10.0)
        self._row("five_hour", T0 + 3600, 32.0)  # delta=22.0
        tr = bucket_trend(self.conn, "five_hour", T0 + 3600)
        self.assertEqual(tr["burn_1h_pct_per_hr"], 22.0)
        self.assertEqual(tr["pace"], "on")

    def test_pace_deadband_just_under(self):
        """relevant == 17.0, clearly below even_drain*0.9==18.0 -> 'under'."""
        from app.limit_trends import bucket_trend
        self._row("five_hour", T0, 10.0)
        self._row("five_hour", T0 + 3600, 27.0)  # delta=17.0
        tr = bucket_trend(self.conn, "five_hour", T0 + 3600)
        self.assertEqual(tr["burn_1h_pct_per_hr"], 17.0)
        self.assertEqual(tr["pace"], "under")

    def test_pace_deadband_just_over(self):
        """relevant == 23.0, clearly above even_drain*1.1==22.0 -> 'over'."""
        from app.limit_trends import bucket_trend
        self._row("five_hour", T0, 10.0)
        self._row("five_hour", T0 + 3600, 33.0)  # delta=23.0
        tr = bucket_trend(self.conn, "five_hour", T0 + 3600)
        self.assertEqual(tr["burn_1h_pct_per_hr"], 23.0)
        self.assertEqual(tr["pace"], "over")


# ---------------------------------------------------------------------------
# D2 + D5 — /api/rate-limits oauth.trend and oauth.limit_window
# ---------------------------------------------------------------------------

def _ins_event(conn, uuid, req, ts, inp=0, acct="me@gmail.com", plan="max"):
    conn.execute(
        "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,"
        "request_id,source_machine,project_dir,model,is_sidechain,agent_id,"
        "input_tokens,output_tokens,cache_creation_tokens,cache_read_tokens,"
        "account_email,plan,org_name,is_human_prompt,user_type) VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (uuid, "assistant", "2026-07-01T12:00:00Z", ts, "2026-07-01", "s1",
         req, "personal-mbp", "proj", "claude-opus-4-8", 0, None, inp, 0, 0, 0,
         acct, plan, None, 0, None))
    conn.commit()


class TrendEndpointTest(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def _seed_oauth(self, resets_epoch):
        resets_iso = datetime.fromtimestamp(
            resets_epoch, tz=timezone.utc).isoformat()
        stored = {
            "data": {
                "seven_day": {"utilization": 40, "resets_at": resets_iso},
                "five_hour": {"utilization": 20, "resets_at": resets_iso},
            },
            "updated_at": resets_iso,
        }
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('oauth_usage', ?)",
            (json.dumps(stored),))
        self.conn.commit()

    def _seed_readings(self, bucket, now, n=30, span_s=6 * 3600,
                       start_pct=10.0, end_pct=40.0, resets_epoch=None):
        for i in range(n):
            frac = i / (n - 1)
            t = now - span_s + frac * span_s
            pct = start_pct + frac * (end_pct - start_pct)
            self.conn.execute(
                "INSERT INTO limit_readings(fetched_epoch, source, bucket, "
                "utilization, resets_at, resets_at_epoch) "
                "VALUES(?, 'server', ?, ?, NULL, ?)",
                (t, bucket, round(pct), resets_epoch))
        self.conn.commit()

    def test_trend_block_present_with_burn_eta_pace_series(self):
        now = time.time()
        self._seed_oauth(now + 3 * 3600)
        self._seed_readings("seven_day", now, n=40, span_s=6 * 3600,
                            start_pct=10.0, end_pct=40.0)
        c = self.client()
        oauth = c.get("/api/rate-limits?scope=personal").json()[
            "weekly_budget"]["oauth"]
        self.assertIn("trend", oauth)
        self.assertIn("seven_day", oauth["trend"])
        tr = oauth["trend"]["seven_day"]
        # ~30pt rise over 6h ~= 5 pct/hr (2-dp float).
        self.assertIsNotNone(tr["burn_6h_pct_per_hr"])
        self.assertGreater(tr["burn_6h_pct_per_hr"], 0)
        self.assertEqual(round(tr["burn_6h_pct_per_hr"], 2),
                         tr["burn_6h_pct_per_hr"])
        # 5 pct/hr on a 7-day bucket (even-drain ~0.6/hr) is way over pace.
        self.assertEqual(tr["pace"], "over")
        # ETA is minute-floored and lies in the future.
        self.assertIsNotNone(tr["eta_100_epoch"])
        self.assertEqual(tr["eta_100_epoch"] % 60, 0)
        self.assertGreater(tr["eta_100_epoch"], now)
        # Series: bounded, minute-floored epochs, first/last preserved.
        self.assertLessEqual(len(tr["series"]), 200)
        for epoch, _pct in tr["series"]:
            self.assertEqual(epoch % 60, 0)

    def test_series_downsampled_to_200_max(self):
        now = time.time()
        self._seed_oauth(now + 3 * 3600)
        self._seed_readings("seven_day", now, n=500, span_s=6 * 3600)
        c = self.client()
        oauth = c.get("/api/rate-limits?scope=personal").json()[
            "weekly_budget"]["oauth"]
        series = oauth["trend"]["seven_day"]["series"]
        self.assertLessEqual(len(series), 200)
        self.assertGreater(len(series), 1)
        for epoch, _pct in series:
            self.assertEqual(epoch % 60, 0)

    def test_no_readings_no_trend_key_shape_unchanged(self):
        now = time.time()
        self._seed_oauth(now + 3 * 3600)  # oauth row but ZERO limit_readings
        c = self.client()
        oauth = c.get("/api/rate-limits?scope=personal").json()[
            "weekly_budget"]["oauth"]
        self.assertNotIn("trend", oauth)
        # Existing gauge shape is untouched.
        for k in ("weekly_pct", "five_hour_pct", "buckets"):
            self.assertIn(k, oauth)

    def test_limit_window_present_and_consistent(self):
        now = time.time()
        resets = now + 3 * 3600
        self._seed_oauth(resets)
        # personal event inside the limit window: 1M input Opus 4.8 = $5.
        _ins_event(self.conn, "c1", "r1", now - 3600, inp=1_000_000)
        c = self.client()
        oauth = c.get("/api/rate-limits?scope=personal").json()[
            "weekly_budget"]["oauth"]
        self.assertIn("limit_window", oauth)
        lw = oauth["limit_window"]
        # Window start = floor(resets)-7d.
        from app.cost_windows import compute_window_cost
        expect_start = ((resets // 60) * 60.0) - 7 * 86400
        self.assertAlmostEqual(lw["start_epoch"], expect_start, places=2)
        # cost is consistent with compute_window_cost over the SAME window
        # [start_epoch, now] at the same scope (the whole point of D5).
        self.assertAlmostEqual(
            lw["cost"],
            round(compute_window_cost(self.conn, lw["start_epoch"],
                                      now, scope="personal"), 2),
            places=2)
        self.assertGreaterEqual(lw["cost"], 4.99)  # the $5 event is inside
        self.assertGreaterEqual(lw["active_s"], 0)

    def test_limit_window_absent_when_resets_unparseable(self):
        now = time.time()
        stored = {
            "data": {
                "seven_day": {"utilization": 40, "resets_at": "soon"},
                "five_hour": {"utilization": 20, "resets_at": "soon"},
            },
            "updated_at": "2026-07-01T12:00:00+00:00",
        }
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('oauth_usage', ?)",
            (json.dumps(stored),))
        self.conn.commit()
        c = self.client()
        oauth = c.get("/api/rate-limits?scope=personal").json()[
            "weekly_budget"]["oauth"]
        self.assertNotIn("limit_window", oauth)

    def test_resets_appear_in_trend_when_inside_series_window(self):
        """Fix 5(e): a reset landing inside the served series window (168h
        for seven_day) must show up in trend[bucket].resets, not just
        influence the compute_burn segment trim."""
        now = time.time()
        self._seed_oauth(now + 3 * 3600)
        # Active window at high pct, then a genuine reset drop (>=10pt while
        # the previous reading's resets_at was still hours in the future).
        self.conn.execute(
            "INSERT INTO limit_readings(fetched_epoch, source, bucket, "
            "utilization, resets_at, resets_at_epoch) "
            "VALUES(?, 'server', 'seven_day', 55.0, NULL, ?)",
            (now - 3600, now + 3600))
        self.conn.execute(
            "INSERT INTO limit_readings(fetched_epoch, source, bucket, "
            "utilization, resets_at, resets_at_epoch) "
            "VALUES(?, 'server', 'seven_day', 3.0, NULL, ?)",
            (now - 3000, now + 7 * 86400))
        self.conn.commit()
        c = self.client()
        oauth = c.get("/api/rate-limits?scope=personal").json()[
            "weekly_budget"]["oauth"]
        tr = oauth["trend"]["seven_day"]
        self.assertIn("resets", tr)
        self.assertEqual(len(tr["resets"]), 1)
        self.assertEqual(tr["resets"][0]["bucket"], "seven_day")

    def test_trend_failure_does_not_wipe_oauth_block(self):
        """Fix 6: a bug in trend computation must only drop the 'trend' key
        — before this fix it sat inside the SAME broad except as
        weekly_pct/five_hour_pct/buckets/extra_usage, so an exception there
        would have silently deleted the entire oauth block."""
        from unittest.mock import patch
        now = time.time()
        self._seed_oauth(now + 3 * 3600)
        self._seed_readings("seven_day", now, n=40, span_s=6 * 3600)
        with patch("app.limit_trends.distinct_buckets",
                   side_effect=RuntimeError("boom")):
            c = self.client()
            r = c.get("/api/rate-limits?scope=personal")
        self.assertEqual(r.status_code, 200, r.text)
        oauth = r.json()["weekly_budget"]["oauth"]
        self.assertNotIn("trend", oauth)
        for k in ("weekly_pct", "five_hour_pct", "buckets"):
            self.assertIn(k, oauth)

    def test_limit_window_failure_does_not_wipe_oauth_block(self):
        """Fix 6: a bug in limit_window computation must only drop the
        'limit_window' key, never the rest of the oauth block (and must not
        prevent 'trend' from being computed either — the two try/excepts are
        independent)."""
        from unittest.mock import patch
        now = time.time()
        resets = now + 3 * 3600
        self._seed_oauth(resets)
        self._seed_readings("seven_day", now, n=40, span_s=6 * 3600)
        with patch("app.api._iso_to_epoch", side_effect=RuntimeError("boom")):
            c = self.client()
            r = c.get("/api/rate-limits?scope=personal")
        self.assertEqual(r.status_code, 200, r.text)
        oauth = r.json()["weekly_budget"]["oauth"]
        self.assertNotIn("limit_window", oauth)
        for k in ("weekly_pct", "five_hour_pct", "buckets"):
            self.assertIn(k, oauth)
        self.assertIn("trend", oauth)  # unaffected by the limit_window bug


if __name__ == "__main__":
    unittest.main()
