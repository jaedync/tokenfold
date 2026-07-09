"""Enterprise-only monthly $ budget: storage, pacing math, and the
GET/POST /api/enterprise-budget + /api/rate-limits wiring."""

import math
import unittest
from datetime import datetime, timezone

from app.monthly_budget import get_budget, monthly_budget_block, set_budget
from app.tests._support import TempDBTestCase


def _ins_event(conn, uuid, ts_epoch, inp=1_000_000, day=None):
    """Enterprise Opus 4.8 event: 1M input = $5 at static rates."""
    if day is None:
        day = datetime.fromtimestamp(ts_epoch, tz=timezone.utc).strftime("%Y-%m-%d")
    conn.execute(
        "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,request_id,"
        "source_machine,project_dir,model,is_sidechain,agent_id,input_tokens,"
        "output_tokens,cache_creation_tokens,cache_read_tokens,account_email,plan,"
        "org_name,is_human_prompt) "
        "VALUES(?,'assistant',?,?,?,'s1',?,'m1','proj','claude-opus-4-8',0,NULL,"
        "?,0,0,0,'jaedyn@acme.io','enterprise','Acme',0)",
        (uuid, day + "T12:00:00Z", ts_epoch, day, "r-" + uuid, inp))
    conn.commit()


def _ins_personal_event(conn, uuid, ts_epoch, inp=1_000_000, day=None):
    if day is None:
        day = datetime.fromtimestamp(ts_epoch, tz=timezone.utc).strftime("%Y-%m-%d")
    conn.execute(
        "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,request_id,"
        "source_machine,project_dir,model,is_sidechain,agent_id,input_tokens,"
        "output_tokens,cache_creation_tokens,cache_read_tokens,account_email,plan,"
        "org_name,is_human_prompt) "
        "VALUES(?,'assistant',?,?,?,'s1',?,'m1','proj','claude-opus-4-8',0,NULL,"
        "?,0,0,0,'me@personal.io','max',NULL,0)",
        (uuid, day + "T12:00:00Z", ts_epoch, day, "r-" + uuid, inp))
    conn.commit()


def _month_start_epoch(now: datetime) -> float:
    return now.replace(day=1, hour=0, minute=0, second=0,
                       microsecond=0, tzinfo=timezone.utc).timestamp()


# ---------------------------------------------------------------------------
# get_budget / set_budget roundtrip + validation
# ---------------------------------------------------------------------------

class BudgetStorageTest(TempDBTestCase):

    def test_no_budget_set_returns_none(self):
        self.assertIsNone(get_budget(self.conn))

    def test_set_then_get_roundtrip(self):
        set_budget(self.conn, 1000.0)
        self.assertEqual(get_budget(self.conn), 1000.0)

    def test_set_none_clears(self):
        set_budget(self.conn, 1000.0)
        set_budget(self.conn, None)
        self.assertIsNone(get_budget(self.conn))

    def test_garbage_meta_value_returns_none(self):
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?,?)",
            ("enterprise_monthly_budget_usd", "not-a-number"))
        self.conn.commit()
        self.assertIsNone(get_budget(self.conn))

    def test_validation_rejects_zero(self):
        with self.assertRaises(ValueError):
            set_budget(self.conn, 0)

    def test_validation_rejects_negative(self):
        with self.assertRaises(ValueError):
            set_budget(self.conn, -5.0)

    def test_validation_rejects_nan(self):
        with self.assertRaises(ValueError):
            set_budget(self.conn, float("nan"))

    def test_validation_rejects_inf(self):
        with self.assertRaises(ValueError):
            set_budget(self.conn, float("inf"))

    def test_validation_rejects_above_max(self):
        with self.assertRaises(ValueError):
            set_budget(self.conn, 2_000_000.0)

    def test_validation_accepts_max_boundary(self):
        set_budget(self.conn, 1_000_000.0)
        self.assertEqual(get_budget(self.conn), 1_000_000.0)

    def test_invalid_value_does_not_clobber_existing(self):
        set_budget(self.conn, 500.0)
        with self.assertRaises(ValueError):
            set_budget(self.conn, -1.0)
        self.assertEqual(get_budget(self.conn), 500.0)


# ---------------------------------------------------------------------------
# monthly_budget_block: pace math
# ---------------------------------------------------------------------------

class PaceMathTest(TempDBTestCase):
    """Mid-month fixture: 2026-07-09T12:00Z, 30-day July, elapsed ~ 8.5/31."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        self.now = datetime(2026, 7, 9, 12, 0, 0, tzinfo=timezone.utc)
        self.now_epoch = self.now.timestamp()
        self.month_start = _month_start_epoch(self.now)

    def test_under_pace(self):
        set_budget(self.conn, 1000.0)
        # elapsed_fraction ~ 8.5/31 ~= 0.274; expected ~= 274. Spend far under.
        _ins_event(self.conn, "e1", self.month_start + 3600, inp=1_000_000)  # $5
        block = monthly_budget_block(self.conn, self.now_epoch)
        self.assertEqual(block["pace"], "under")
        self.assertLess(block["projected_eom_usd"] / block["budget_usd"], 0.90)

    def test_on_pace(self):
        set_budget(self.conn, 1000.0)
        elapsed_fraction = (self.now_epoch - self.month_start) / (31 * 86400.0)
        # Target ratio exactly 1.0: mtd = elapsed_fraction * budget.
        target_mtd = elapsed_fraction * 1000.0
        # 1M input tokens = $5 (static Opus pricing) -> scale inp accordingly.
        inp = int(target_mtd / 5.0 * 1_000_000)
        _ins_event(self.conn, "e1", self.month_start + 3600, inp=inp)
        block = monthly_budget_block(self.conn, self.now_epoch)
        self.assertEqual(block["pace"], "on")

    def test_over_pace(self):
        set_budget(self.conn, 10.0)
        # Way over: spend $500 against a $10 budget.
        _ins_event(self.conn, "e1", self.month_start + 3600, inp=100_000_000)
        block = monthly_budget_block(self.conn, self.now_epoch)
        self.assertEqual(block["pace"], "over")
        self.assertGreater(block["projected_eom_usd"] / block["budget_usd"], 1.10)

    def _pace_for_ratio(self, elapsed_fraction, budget, ratio):
        """Drive monthly_budget_block with an mtd_cost engineered (via a
        compute_window_cost stub) so projected_eom_usd/budget == ratio,
        bypassing $/token quantization noise so the boundary is pinned
        precisely against the deadband constants.

        Multiplication order matters for exact float round-trip through
        the function's mtd_cost / elapsed_fraction: (budget * ratio) must
        be computed before multiplying by elapsed_fraction, so dividing by
        elapsed_fraction again cancels back to exactly budget * ratio."""
        mtd_cost = elapsed_fraction * (budget * ratio)
        import app.monthly_budget as mb
        orig = mb.compute_window_cost
        mb.compute_window_cost = lambda *a, **k: mtd_cost
        try:
            return monthly_budget_block(self.conn, self.now_epoch)
        finally:
            mb.compute_window_cost = orig

    def test_on_pace_boundary_lower_0_90(self):
        """ratio == 1 - PACE_DEADBAND (0.90) must be 'on' (< 0.90 is
        'under', not <=). PACE_DEADBAND is the comparison's own constant, so
        this pins the operator (strict <) rather than a hand-copied literal."""
        set_budget(self.conn, 1000.0)
        elapsed_fraction = (self.now_epoch - self.month_start) / (31 * 86400.0)
        from app.monthly_budget import EARLY_MONTH_FRACTION, PACE_DEADBAND
        self.assertGreaterEqual(elapsed_fraction, EARLY_MONTH_FRACTION)
        block = self._pace_for_ratio(elapsed_fraction, 1000.0,
                                     1.0 - PACE_DEADBAND)
        self.assertEqual(block["pace"], "on")

    def test_under_pace_just_below_lower_boundary(self):
        """A ratio a full percentage point under the 0.90 deadband edge must
        be 'under' — sanity check for the boundary test above."""
        set_budget(self.conn, 1000.0)
        elapsed_fraction = (self.now_epoch - self.month_start) / (31 * 86400.0)
        from app.monthly_budget import PACE_DEADBAND
        block = self._pace_for_ratio(elapsed_fraction, 1000.0,
                                     1.0 - PACE_DEADBAND - 0.01)
        self.assertEqual(block["pace"], "under")

    def test_on_pace_boundary_upper_1_10(self):
        """ratio == 1 + PACE_DEADBAND (1.10) must be 'on' (> 1.10 is
        'over', not >=)."""
        set_budget(self.conn, 1000.0)
        elapsed_fraction = (self.now_epoch - self.month_start) / (31 * 86400.0)
        from app.monthly_budget import PACE_DEADBAND
        block = self._pace_for_ratio(elapsed_fraction, 1000.0,
                                     1.0 + PACE_DEADBAND)
        self.assertEqual(block["pace"], "on")

    def test_over_pace_just_above_upper_boundary(self):
        """A ratio a full percentage point over the 1.10 deadband edge must
        be 'over' — sanity check for the boundary test above."""
        set_budget(self.conn, 1000.0)
        elapsed_fraction = (self.now_epoch - self.month_start) / (31 * 86400.0)
        from app.monthly_budget import PACE_DEADBAND
        block = self._pace_for_ratio(elapsed_fraction, 1000.0,
                                     1.0 + PACE_DEADBAND + 0.01)
        self.assertEqual(block["pace"], "over")

    def test_zero_spend_mid_month_is_under_with_zero_projection(self):
        set_budget(self.conn, 1000.0)
        block = monthly_budget_block(self.conn, self.now_epoch)
        self.assertEqual(block["pace"], "under")
        self.assertEqual(block["projected_eom_usd"], 0.0)
        self.assertEqual(block["mtd_cost"], 0.0)


# ---------------------------------------------------------------------------
# Early-month suppression
# ---------------------------------------------------------------------------

class EarlyMonthTest(TempDBTestCase):

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_pace_and_projection_null_before_threshold(self):
        set_budget(self.conn, 1000.0)
        now = datetime(2026, 7, 1, 6, 0, 0, tzinfo=timezone.utc)  # 6h into July
        now_epoch = now.timestamp()
        # elapsed_fraction = 6/(31*24) ~= 0.00806 < 0.05
        block = monthly_budget_block(self.conn, now_epoch)
        self.assertLess(block["elapsed_fraction"], 0.05)
        self.assertIsNone(block["pace"])
        self.assertIsNone(block["projected_eom_usd"])
        self.assertIn("expected_usd", block)
        self.assertIsNotNone(block["expected_usd"])

    def test_pace_present_at_and_after_threshold(self):
        set_budget(self.conn, 1000.0)
        # 31-day July: 0.05 * 31 * 86400 = 133,920s = 37.2h into the month.
        now = datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp() + 133_920
        block = monthly_budget_block(self.conn, now)
        self.assertGreaterEqual(block["elapsed_fraction"], 0.05)
        self.assertIsNotNone(block["pace"])
        self.assertIsNotNone(block["projected_eom_usd"])


# ---------------------------------------------------------------------------
# No budget set
# ---------------------------------------------------------------------------

class NoBudgetTest(TempDBTestCase):

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_block_is_none_when_no_budget(self):
        self.assertIsNone(monthly_budget_block(self.conn))

    def test_rate_limits_enterprise_has_no_monthly_budget_key(self):
        c = self.client()
        body = c.get("/api/rate-limits?scope=enterprise").json()
        self.assertNotIn("monthly_budget", body["weekly_budget"])


# ---------------------------------------------------------------------------
# Scope gating via /api/rate-limits
# ---------------------------------------------------------------------------

class ScopeGatingTest(TempDBTestCase):

    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        set_budget(self.conn, 1000.0)

    def test_enterprise_has_monthly_budget_no_oauth(self):
        c = self.client()
        body = c.get("/api/rate-limits?scope=enterprise").json()
        wb = body["weekly_budget"]
        self.assertIn("monthly_budget", wb)
        self.assertNotIn("oauth", wb)

    def test_personal_has_no_monthly_budget_even_when_set(self):
        c = self.client()
        body = c.get("/api/rate-limits?scope=personal").json()
        wb = body["weekly_budget"]
        self.assertNotIn("monthly_budget", wb)


# ---------------------------------------------------------------------------
# Month boundary math
# ---------------------------------------------------------------------------

class MonthBoundaryTest(TempDBTestCase):

    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        set_budget(self.conn, 1000.0)

    def test_july_2026_month_end(self):
        now = datetime(2026, 7, 9, 12, 0, 0, tzinfo=timezone.utc).timestamp()
        block = monthly_budget_block(self.conn, now)
        self.assertEqual(block["month"], "2026-07")
        expected_end = datetime(2026, 8, 1, 0, 0, 0,
                                tzinfo=timezone.utc).timestamp()
        self.assertEqual(block["month_end_epoch"], int(expected_end))

    def test_december_year_rollover(self):
        now = datetime(2026, 12, 15, 8, 0, 0, tzinfo=timezone.utc).timestamp()
        block = monthly_budget_block(self.conn, now)
        self.assertEqual(block["month"], "2026-12")
        expected_end = datetime(2027, 1, 1, 0, 0, 0,
                                tzinfo=timezone.utc).timestamp()
        self.assertEqual(block["month_end_epoch"], int(expected_end))


# ---------------------------------------------------------------------------
# MTD cost reuses the enterprise cost pipeline, excludes personal events
# ---------------------------------------------------------------------------

class MtdCostScopeTest(TempDBTestCase):

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_mtd_excludes_personal_events(self):
        set_budget(self.conn, 1000.0)
        now = datetime(2026, 7, 9, 12, 0, 0, tzinfo=timezone.utc)
        now_epoch = now.timestamp()
        month_start = _month_start_epoch(now)
        _ins_event(self.conn, "e1", month_start + 3600, inp=1_000_000)  # $5 enterprise
        _ins_personal_event(self.conn, "p1", month_start + 3600, inp=1_000_000)  # $5 personal
        block = monthly_budget_block(self.conn, now_epoch)
        self.assertAlmostEqual(block["mtd_cost"], 5.0, places=2)


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

class EnterpriseBudgetEndpointTest(TempDBTestCase):

    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        self._saved_pw = self._config.DASHBOARD_PASSWORD
        self._saved_user = self._config.DASHBOARD_USER
        self._config.DASHBOARD_PASSWORD = "pw"
        self._config.DASHBOARD_USER = "jaedyn"
        self.addCleanup(self._restore_auth)

    def _restore_auth(self):
        self._config.DASHBOARD_PASSWORD = self._saved_pw
        self._config.DASHBOARD_USER = self._saved_user

    def test_get_returns_null_when_unset(self):
        c = self.client()
        r = c.get("/api/enterprise-budget", auth=("jaedyn", "pw"))
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.json()["budget_usd"])

    def test_post_requires_basic_auth(self):
        c = self.client()
        r = c.post("/api/enterprise-budget", json={"budget_usd": 500.0})
        self.assertEqual(r.status_code, 401)

    def test_get_requires_basic_auth(self):
        c = self.client()
        r = c.get("/api/enterprise-budget")
        self.assertEqual(r.status_code, 401)

    def test_post_sets_and_get_reflects(self):
        c = self.client()
        r = c.post("/api/enterprise-budget", json={"budget_usd": 750.0},
                   auth=("jaedyn", "pw"))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["budget_usd"], 750.0)
        r2 = c.get("/api/enterprise-budget", auth=("jaedyn", "pw"))
        self.assertEqual(r2.json()["budget_usd"], 750.0)

    def test_post_null_clears(self):
        c = self.client()
        c.post("/api/enterprise-budget", json={"budget_usd": 750.0},
              auth=("jaedyn", "pw"))
        r = c.post("/api/enterprise-budget", json={"budget_usd": None},
                   auth=("jaedyn", "pw"))
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.json()["budget_usd"])

    def test_post_rejects_invalid_values(self):
        c = self.client()
        for bad in ("0", "-1", "2000000"):
            r = c.post("/api/enterprise-budget",
                       content='{"budget_usd": %s}' % bad,
                       headers={"Content-Type": "application/json"},
                       auth=("jaedyn", "pw"))
            self.assertEqual(r.status_code, 400, f"budget {bad!r} must be rejected")
        for bad in ("NaN", "Infinity"):
            r = c.post("/api/enterprise-budget",
                       content='{"budget_usd": %s}' % bad,
                       headers={"Content-Type": "application/json"},
                       auth=("jaedyn", "pw"))
            self.assertIn(r.status_code, (400, 422),
                         f"budget {bad!r} must be rejected")

    def test_post_rejects_non_numeric_string(self):
        c = self.client()
        r = c.post("/api/enterprise-budget", json={"budget_usd": "abc"},
                   auth=("jaedyn", "pw"))
        self.assertEqual(r.status_code, 422)  # pydantic type validation

    def test_fail_closed_without_dashboard_password(self):
        """Same policy as billing-readings: require_dashboard_auth is open
        when no password is set, but writes must still be rejected."""
        self._config.DASHBOARD_PASSWORD = ""
        c = self.client()
        r = c.post("/api/enterprise-budget", json={"budget_usd": 500.0})
        self.assertEqual(r.status_code, 403)

    def test_get_still_open_without_dashboard_password(self):
        self._config.DASHBOARD_PASSWORD = ""
        c = self.client()
        r = c.get("/api/enterprise-budget")
        self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()
