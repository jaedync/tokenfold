"""Workstream F tests: weekly window segmentation (F4), UTC monthly costs
(F5), the /api/spend-history endpoint + gating (F6), and the by-model cost
split (F3).

'now' is always explicit and the DB seeded relative to it — nothing here
depends on wall-clock drift. Opus 4.8 static pricing: 1M input = $5.00
under freeze_pricing().
"""

import json
import time
import unittest
from datetime import datetime, timezone

from app.tests._support import TempDBTestCase

WINDOW_S = 7 * 86400


def _ins_event(conn, uuid, req, ts, inp=0, model="claude-opus-4-8",
               acct="me@gmail.com", plan="max", org=None):
    day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    conn.execute(
        "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,"
        "request_id,source_machine,project_dir,model,is_sidechain,agent_id,"
        "input_tokens,output_tokens,cache_creation_tokens,cache_read_tokens,"
        "account_email,plan,org_name,is_human_prompt,user_type) VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (uuid, "assistant", "2026-07-01T12:00:00Z", ts, day, "s1",
         req, "personal-mbp", "proj", model, 0, None, inp, 0, 0, 0,
         acct, plan, org, 0, None))
    conn.commit()


class _SegBase(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def _row(self, t, pct, anchor):
        self.conn.execute(
            "INSERT INTO limit_readings(fetched_epoch, source, bucket, "
            "utilization, resets_at, resets_at_epoch) "
            "VALUES(?, 'server', 'seven_day', ?, NULL, ?)",
            (t, pct, anchor))
        self.conn.commit()

    def _segments(self, now, anchor=None):
        from app.spend_history import weekly_window_segments
        return weekly_window_segments(self.conn, "personal", now=now,
                                      anchor_epoch=anchor)


# ---------------------------------------------------------------------------
# F4 — weekly_window_segments
# ---------------------------------------------------------------------------

class SegmentationTest(_SegBase):

    def test_no_anchor_no_segments(self):
        self.assertEqual(self._segments(now=time.time()), [])

    def test_single_window_ongoing_only(self):
        """Historization started mid-window, no boundary observed yet:
        one ongoing segment [anchor-7d, now) projecting to the anchor."""
        now = 1751500800.0  # minute-aligned for exact comparisons
        anchor = now + 3 * 86400
        for i in range(5):
            self._row(now - 3600 + i * 600, 10.0 + i, anchor)
        _ins_event(self.conn, "e1", "r1", now - 1800, inp=1_000_000)
        segs = self._segments(now)
        self.assertEqual(segs[-1]["end_kind"], "ongoing")
        self.assertIsNone(segs[-1]["end_epoch"])
        self.assertEqual(segs[-1]["projected_end_epoch"],
                         (anchor // 60) * 60.0)
        self.assertEqual(segs[-1]["start_epoch"],
                         (anchor // 60) * 60.0 - WINDOW_S)
        self.assertAlmostEqual(segs[-1]["cost"], 5.0, places=2)
        self.assertEqual(segs[-1]["peak_pct"], 14.0)

    def test_natural_rollover_boundary_exact(self):
        """Anchor transition A -> B where A passed inside the poll gap:
        the boundary is exactly A (minute-floored), kind natural — even
        though no poll landed at the rollover moment."""
        now = 1751500800.0
        a = now - 2 * 86400 + 37          # deliberately not minute-aligned
        b = a + WINDOW_S
        self._row(a - 1200, 80.0, a)
        self._row(a - 600, 81.0, a)
        self._row(a + 600, 2.0, b)        # first post-rollover poll
        self._row(a + 1200, 3.0, b)
        segs = self._segments(now, anchor=b)
        # NB the segment still carries inferred=True — its END is the exact
        # observed expiry but its START (earliest_anchor - 7d) is assumed.
        naturals = [s for s in segs
                    if s["end_epoch"] == (a // 60) * 60.0]
        self.assertEqual(len(naturals), 1)
        self.assertEqual(naturals[0]["end_kind"], "natural")
        self.assertEqual(naturals[0]["end_epoch"] % 60, 0)
        self.assertEqual(naturals[0]["peak_pct"], 81.0)

    def test_natural_rollover_survives_downtime(self):
        """Server slept 6h across the rollover: boundary still lands at
        the old anchor's scheduled expiry, not at the next poll."""
        now = 1751500800.0
        a = now - 2 * 86400
        self._row(a - 5 * 3600, 70.0, a)      # last pre-sleep poll
        self._row(a + 3600, 1.0, a + WINDOW_S)  # first post-sleep poll
        segs = self._segments(now, anchor=a + WINDOW_S)
        naturals = [s for s in segs if s["end_epoch"] == a]
        self.assertEqual(len(naturals), 1)
        self.assertEqual(naturals[0]["end_kind"], "natural")

    def test_granted_reset_in_place_splits_window(self):
        """THE user-reported shape: a granted reset mid-window (meter wiped,
        anchor unchanged) produces MULTIPLE segments inside one week."""
        now = 1751500800.0
        anchor = now + 2 * 86400          # natural window [anchor-7d, anchor)
        grant_t = now - 86400
        self._row(grant_t - 600, 55.0, anchor)
        self._row(grant_t, 3.0, anchor)   # 52-pt plunge, anchor unchanged
        self._row(grant_t + 600, 4.0, anchor)
        # $5 before the grant, $5 after — must land in different segments.
        _ins_event(self.conn, "e1", "r1", grant_t - 7200, inp=1_000_000)
        _ins_event(self.conn, "e2", "r2", now - 3600, inp=1_000_000)
        segs = self._segments(now)
        granted = [s for s in segs if s["end_kind"] == "granted"]
        self.assertEqual(len(granted), 1)
        self.assertEqual(granted[0]["end_epoch"], (grant_t // 60) * 60.0)
        self.assertAlmostEqual(granted[0]["cost"], 5.0, places=2)
        self.assertEqual(granted[0]["peak_pct"], 55.0)
        ongoing = segs[-1]
        self.assertEqual(ongoing["end_kind"], "ongoing")
        self.assertEqual(ongoing["start_epoch"], (grant_t // 60) * 60.0)
        self.assertAlmostEqual(ongoing["cost"], 5.0, places=2)
        # Both segments fall within ONE natural window: multiple entries
        # per week, as designed.
        self.assertGreater(granted[0]["start_epoch"],
                           (anchor // 60) * 60.0 - WINDOW_S - 60)

    def test_anchor_jump_before_expiry_is_granted(self):
        """resets_at jumps forward while the old window is still live —
        a granted reset located at the observing poll."""
        now = 1751500800.0
        a = now + 86400                    # old anchor, still 1d in future
        c = now + 5 * 86400                # replacement anchor
        jump_t = now - 3600
        self._row(jump_t - 600, 60.0, a)
        self._row(jump_t, 58.0, c)         # anchor moved, meter NOT wiped
        self._row(jump_t + 600, 58.0, c)
        segs = self._segments(now, anchor=c)
        granted = [s for s in segs if s["end_kind"] == "granted"]
        self.assertEqual(len(granted), 1)
        self.assertEqual(granted[0]["end_epoch"], (jump_t // 60) * 60.0)

    def test_double_fire_merges_to_one_boundary(self):
        """Jump-rule fires on one pair, drop-rule on the next (classic
        detect-resets double-fire): one merged boundary, not two."""
        now = 1751500800.0
        a = now + 86400
        c = now + 5 * 86400
        t = now - 3600
        self._row(t - 600, 60.0, a)
        self._row(t, 59.0, c)              # jump fires
        self._row(t + 600, 2.0, c)         # 57-pt drop fires next pair
        segs = self._segments(now, anchor=c)
        granted = [s for s in segs if s["end_kind"] == "granted"]
        self.assertEqual(len(granted), 1)

    def test_inferred_backstep_covers_prehistory_events(self):
        """Events 3 weeks older than any reading: inferred 7d-cadence
        segments cover them, flagged inferred with peak_pct None."""
        now = 1751500800.0
        anchor = now + 2 * 86400
        self._row(now - 600, 10.0, anchor)
        old_ts = now - 23 * 86400
        _ins_event(self.conn, "e1", "r1", old_ts, inp=1_000_000)
        segs = self._segments(now)
        # Completed inferred segments (the ongoing one is also inferred but
        # legitimately contains today's readings — exclude it here).
        inferred = [s for s in segs
                    if s["inferred"] and s["end_epoch"] is not None]
        self.assertGreaterEqual(len(inferred), 3)
        for s in inferred:
            self.assertIsNone(s["peak_pct"])
            self.assertEqual(s["end_kind"], "natural")
            # cadence: every inferred cut sits k*7d behind the base
            off = ((anchor // 60) * 60.0 - WINDOW_S
                   - s["end_epoch"]) % WINDOW_S
            self.assertEqual(off, 0)
        # The $5 event is inside exactly one segment.
        owners = [s for s in segs
                  if s["start_epoch"] <= old_ts
                  and (s["end_epoch"] or now) > old_ts]
        self.assertEqual(len(owners), 1)
        self.assertAlmostEqual(owners[0]["cost"], 5.0, places=2)

    def test_ongoing_segment_never_inferred(self):
        """Review L1: fresh install, no boundary observed yet — the live
        ongoing segment must NOT be flagged inferred (the UI fades
        inferred bars, wrongly implying live spend is assumed)."""
        now = 1751500800.0
        anchor = now + 3 * 86400
        for i in range(5):
            self._row(now - 3600 + i * 600, 10.0 + i, anchor)
        segs = self._segments(now)
        self.assertEqual(segs[-1]["end_kind"], "ongoing")
        self.assertFalse(segs[-1]["inferred"])

    def test_stale_replay_is_not_a_granted_reset(self):
        """Review M1: a lagging client replays an older snapshot (55 ->
        40 -> 55). The one-row dip must not become a granted boundary —
        the meter 'recovering' instantly is impossible after a real
        grant."""
        now = 1751500800.0
        anchor = now + 2 * 86400
        t = now - 3600
        self._row(t - 600, 55.0, anchor)
        self._row(t, 40.0, anchor)        # stale snapshot, 15-pt "drop"
        self._row(t + 600, 55.0, anchor)  # fresh again — recovered
        segs = self._segments(now)
        self.assertEqual([s for s in segs if s["end_kind"] == "granted"],
                         [])

    def test_low_util_grant_to_zero_splits_window(self):
        """2026-07-09 incident: an account-wide grant zeroed the weekly
        meter at 9% — under RESET_DROP_PTS, so the magnitude rule missed
        it and the window chart drew straight through the reset. A wipe
        to zero while the window is active is a granted boundary at any
        magnitude."""
        now = 1751500800.0
        anchor = now + 2 * 86400
        grant_t = now - 86400
        self._row(grant_t - 600, 9.0, anchor)
        self._row(grant_t, 0.0, anchor)       # 9-pt wipe, anchor unchanged
        self._row(grant_t + 600, 1.0, anchor)
        _ins_event(self.conn, "e1", "r1", grant_t - 7200, inp=1_000_000)
        _ins_event(self.conn, "e2", "r2", now - 3600, inp=1_000_000)
        segs = self._segments(now)
        granted = [s for s in segs if s["end_kind"] == "granted"]
        self.assertEqual(len(granted), 1)
        self.assertEqual(granted[0]["end_epoch"], (grant_t // 60) * 60.0)
        self.assertAlmostEqual(granted[0]["cost"], 5.0, places=2)

    def test_low_util_replay_to_zero_not_granted(self):
        """9 -> 0 -> 9: recovery to the pre-event level is the replay
        fingerprint even below RESET_DROP_PTS (the old fixed-offset
        recovery test was vacuously true here)."""
        now = 1751500800.0
        anchor = now + 2 * 86400
        t = now - 3600
        self._row(t - 600, 9.0, anchor)
        self._row(t, 0.0, anchor)
        self._row(t + 600, 9.0, anchor)
        segs = self._segments(now)
        self.assertEqual([s for s in segs if s["end_kind"] == "granted"],
                         [])

    def test_drop_to_zero_after_expiry_not_granted(self):
        """Anchor passed between the two polls while the stored blob still
        carries the OLD anchor: the zero-wipe is natural expiry, not a
        grant (the zero rule requires the window active at the later
        poll)."""
        now = 1751500800.0
        expiry = now - 3600
        self._row(expiry - 600, 9.0, expiry)   # window ends at `expiry`
        self._row(expiry + 600, 0.0, expiry)   # stale anchor, meter rolled
        segs = self._segments(now, anchor=expiry + WINDOW_S)
        self.assertEqual([s for s in segs if s["end_kind"] == "granted"],
                         [])

    def test_ancient_event_bounded_output(self):
        """Review H1: one event with a garbage ts_epoch (epoch 0) must not
        blow up the look-back loops — segments stay <= MAX_SEGMENTS and
        months <= MAX_MONTHS, without building 100k intermediates."""
        from app.spend_history import (MAX_MONTHS, MAX_SEGMENTS,
                                       monthly_costs)
        now = 1751500800.0
        anchor = now + 2 * 86400
        self._row(now - 600, 10.0, anchor)
        _ins_event(self.conn, "old", "r-old", 0.0, inp=1_000_000)
        _ins_event(self.conn, "new", "r-new", now - 3600, inp=1_000_000)
        t0 = time.time()
        segs = self._segments(now)
        months = monthly_costs(self.conn, "personal", now=now)
        elapsed = time.time() - t0
        self.assertLessEqual(len(segs), MAX_SEGMENTS)
        self.assertLessEqual(len(months), MAX_MONTHS)
        # Bounded work, not "built 100k then truncated": generous wall
        # bound that still catches the unclamped behavior (~1s+).
        self.assertLess(elapsed, 5.0)

    def test_all_emitted_epochs_minute_floored(self):
        now = 1751500837.0                # NOT minute-aligned
        anchor = now + 2 * 86400 + 41     # NOT minute-aligned
        grant_t = now - 86400 + 13
        self._row(grant_t - 601, 55.0, anchor)
        self._row(grant_t, 3.0, anchor)
        _ins_event(self.conn, "e1", "r1", now - 20 * 86400, inp=1_000_000)
        for s in self._segments(now, anchor=anchor):
            self.assertEqual(s["start_epoch"] % 60, 0, s)
            if s["end_epoch"] is not None:
                self.assertEqual(s["end_epoch"] % 60, 0, s)
            if s.get("projected_end_epoch") is not None:
                self.assertEqual(s["projected_end_epoch"] % 60, 0, s)


# ---------------------------------------------------------------------------
# F5 — monthly_costs (true UTC months)
# ---------------------------------------------------------------------------

class MonthlyCostsTest(_SegBase):

    def test_empty_db_empty_list(self):
        from app.spend_history import monthly_costs
        self.assertEqual(monthly_costs(self.conn, "personal"), [])

    def test_utc_month_boundary_not_local_tz(self):
        """23:30Z on Jun 30 and 00:30Z on Jul 1 are BOTH June 30 in
        America/Chicago — a day-summary rollup would put both in June.
        True UTC bucketing splits them."""
        from app.spend_history import monthly_costs
        jun = datetime(2026, 6, 30, 23, 30, tzinfo=timezone.utc).timestamp()
        jul = datetime(2026, 7, 1, 0, 30, tzinfo=timezone.utc).timestamp()
        _ins_event(self.conn, "e1", "r1", jun, inp=1_000_000)
        _ins_event(self.conn, "e2", "r2", jul, inp=1_000_000)
        months = monthly_costs(self.conn, "personal", now=jul + 86400)
        by = {m["month"]: m for m in months}
        self.assertAlmostEqual(by["2026-06"]["total"], 5.0, places=2)
        self.assertAlmostEqual(by["2026-07"]["total"], 5.0, places=2)

    def test_by_model_split_and_partial_flag(self):
        from app.spend_history import monthly_costs
        t = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc).timestamp()
        _ins_event(self.conn, "e1", "r1", t, inp=1_000_000)
        _ins_event(self.conn, "e2", "r2", t + 60, inp=1_000_000,
                   model="claude-sonnet-5")
        months = monthly_costs(self.conn, "personal", now=t + 3600)
        self.assertEqual(len(months), 1)
        m = months[0]
        self.assertEqual(m["month"], "2026-07")
        self.assertTrue(m["partial"])
        self.assertIn("Opus 4.8", m["by_model"])
        self.assertIn("Sonnet 5", m["by_model"])
        self.assertAlmostEqual(m["total"],
                               sum(m["by_model"].values()), places=1)

    def test_months_are_contiguous_including_quiet_months(self):
        """A month with zero spend between active months still appears —
        a gap in the axis would misread as missing data."""
        from app.spend_history import monthly_costs
        mar = datetime(2026, 3, 15, tzinfo=timezone.utc).timestamp()
        may = datetime(2026, 5, 15, tzinfo=timezone.utc).timestamp()
        _ins_event(self.conn, "e1", "r1", mar, inp=1_000_000)
        _ins_event(self.conn, "e2", "r2", may, inp=1_000_000)
        months = monthly_costs(self.conn, "personal", now=may + 86400)
        self.assertEqual([m["month"] for m in months],
                         ["2026-03", "2026-04", "2026-05"])
        self.assertEqual(months[1]["total"], 0.0)
        self.assertEqual(months[1]["by_model"], {})


# ---------------------------------------------------------------------------
# F3 — compute_window_cost_by_model delegation
# ---------------------------------------------------------------------------

class ByModelDelegationTest(_SegBase):

    def test_total_equals_sum_of_by_model(self):
        from app.cost_windows import (compute_window_cost,
                                      compute_window_cost_by_model)
        t = 1751500800.0
        _ins_event(self.conn, "e1", "r1", t, inp=1_000_000)
        _ins_event(self.conn, "e2", "r2", t + 60, inp=2_000_000,
                   model="claude-sonnet-5")
        by = compute_window_cost_by_model(self.conn, t - 10, t + 120,
                                          scope="personal")
        total = compute_window_cost(self.conn, t - 10, t + 120,
                                    scope="personal")
        self.assertAlmostEqual(total, sum(by.values()), places=9)
        self.assertEqual(set(by), {"Opus 4.8", "Sonnet 5"})
        self.assertGreater(by["Opus 4.8"], 0)

    def test_empty_window_empty_dict(self):
        from app.cost_windows import compute_window_cost_by_model
        self.assertEqual(
            compute_window_cost_by_model(self.conn, 0, 1, scope="personal"),
            {})


# ---------------------------------------------------------------------------
# F6 — GET /api/spend-history gating
# ---------------------------------------------------------------------------

class SpendHistoryEndpointTest(_SegBase):

    def _seed_personal_world(self, now):
        anchor = now + 2 * 86400
        resets_iso = datetime.fromtimestamp(
            anchor, tz=timezone.utc).isoformat()
        stored = {"data": {"seven_day": {"utilization": 20,
                                         "resets_at": resets_iso},
                           "five_hour": {"utilization": 5,
                                         "resets_at": resets_iso}},
                  "updated_at": resets_iso}
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('oauth_usage',?)",
            (json.dumps(stored),))
        for i in range(4):
            self.conn.execute(
                "INSERT INTO limit_readings(fetched_epoch, source, bucket, "
                "utilization, resets_at, resets_at_epoch) "
                "VALUES(?, 'server', 'seven_day', ?, NULL, ?)",
                (now - 1800 + i * 600, 18.0 + i, anchor))
        self.conn.commit()
        _ins_event(self.conn, "e1", "r1", now - 3600, inp=1_000_000)

    def test_personal_scope_months_and_windows(self):
        now = time.time()
        self._seed_personal_world(now)
        with self.client() as c:
            r = c.get("/api/spend-history?scope=personal")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIn("months", body)
        self.assertIn("windows", body)
        self.assertEqual(body["windows"][-1]["end_kind"], "ongoing")
        self.assertGreaterEqual(len(body["months"]), 1)

    def test_enterprise_scope_never_carries_windows(self):
        """Compliance: personal Max window history must not exist for the
        enterprise view — not as an empty list, not at all."""
        now = time.time()
        self._seed_personal_world(now)
        with self.client() as c:
            r = c.get("/api/spend-history?scope=enterprise")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertNotIn("windows", body)
        # Personal-only events: the enterprise months axis is empty too.
        self.assertEqual(body["months"], [])

    def test_default_scope_is_enterprise_no_windows(self):
        now = time.time()
        self._seed_personal_world(now)
        with self.client() as c:
            r = c.get("/api/spend-history")
        self.assertEqual(r.status_code, 200)
        self.assertNotIn("windows", r.json())

    def test_invalid_scope_400(self):
        with self.client() as c:
            r = c.get("/api/spend-history?scope=bogus")
        self.assertEqual(r.status_code, 400)

    def test_months_present_without_oauth_data(self):
        """No oauth_usage row at all: months still served, windows omitted."""
        now = time.time()
        _ins_event(self.conn, "e1", "r1", now - 3600, inp=1_000_000)
        with self.client() as c:
            r = c.get("/api/spend-history?scope=personal")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertNotIn("windows", body)
        self.assertEqual(len(body["months"]), 1)

    def test_requires_dashboard_auth_when_password_set(self):
        saved_pw = self._config.DASHBOARD_PASSWORD
        saved_user = self._config.DASHBOARD_USER
        self._config.DASHBOARD_PASSWORD = "pw"
        self._config.DASHBOARD_USER = "u"
        try:
            with self.client() as c:
                r = c.get("/api/spend-history?scope=personal")
            self.assertEqual(r.status_code, 401)
        finally:
            self._config.DASHBOARD_PASSWORD = saved_pw
            self._config.DASHBOARD_USER = saved_user


if __name__ == "__main__":
    unittest.main()
