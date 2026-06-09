"""Tests for T1b: enterprise-lock gating of personal OAuth surfaces.

Fix 1: usage_fetcher.should_run() returns False when LOCKED_SCOPE='enterprise'.
Fix 2: /api/ha suppresses oauth fields when enterprise-locked; when unlocked the
       window spend is PERSONAL-scoped and implied_limit_usd is coherent.
"""

import json
import time
import unittest
from unittest.mock import patch

import app.config
from app.tests._support import TempDBTestCase


# ---------------------------------------------------------------------------
# Fix 1 — should_run() helper
# ---------------------------------------------------------------------------

class ShouldRunTest(unittest.TestCase):
    """Pure-function tests: no DB, no lifespan, no network."""

    def test_locked_enterprise_returns_false(self):
        with patch.object(app.config, 'LOCKED_SCOPE', 'enterprise'):
            from app import usage_fetcher
            self.assertFalse(usage_fetcher.should_run())

    def test_locked_personal_returns_true(self):
        with patch.object(app.config, 'LOCKED_SCOPE', 'personal'):
            from app import usage_fetcher
            self.assertTrue(usage_fetcher.should_run())

    def test_no_lock_returns_true(self):
        with patch.object(app.config, 'LOCKED_SCOPE', None):
            from app import usage_fetcher
            self.assertTrue(usage_fetcher.should_run())


# ---------------------------------------------------------------------------
# Helpers: event seeding with explicit plan/scope
# ---------------------------------------------------------------------------

def _insert_event(conn, uuid, req, plan, org, account_email,
                  model="claude-opus-4-8", ts=None, inp=1000, out=0):
    """Insert an assistant event with explicit scope fields."""
    if ts is None:
        ts = time.time()
    conn.execute(
        "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,request_id,"
        "source_machine,project_dir,model,is_sidechain,agent_id,"
        "input_tokens,output_tokens,cache_creation_tokens,cache_read_tokens,"
        "account_email,plan,org_name,is_human_prompt,user_type) VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (uuid, "assistant", "2026-06-09T12:00:00Z", ts, "2026-06-09",
         "sess", req, "machine", "proj",
         model, 0, None, inp, out, 0, 0,
         account_email, plan, org, 0, None),
    )
    conn.commit()


def _insert_enterprise(conn, uuid, req, ts=None, inp=1000, out=0):
    _insert_event(conn, uuid, req,
                  plan="enterprise", org="Acme", account_email="test@acme.io",
                  ts=ts, inp=inp, out=out)


def _insert_personal(conn, uuid, req, ts=None, inp=1000, out=0):
    _insert_event(conn, uuid, req,
                  plan="max", org=None, account_email="me@personal.io",
                  ts=ts, inp=inp, out=out)


def _seed_oauth_usage(conn, five_hour_resets_at, seven_day_resets_at,
                      fh_utilization=50.0, sd_utilization=50.0):
    """Seed a synthetic oauth_usage meta row."""
    payload = {
        "data": {
            "five_hour": {
                "resets_at": five_hour_resets_at,
                "utilization": fh_utilization,
            },
            "seven_day": {
                "resets_at": seven_day_resets_at,
                "utilization": sd_utilization,
            },
        },
        "updated_at": "2026-06-09T12:00:00+00:00",
    }
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        ("oauth_usage", json.dumps(payload)),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Fix 2a — enterprise-locked: oauth fields fully suppressed
# ---------------------------------------------------------------------------

class HALockedEnterpriseSuppressionTest(TempDBTestCase):
    """When LOCKED_SCOPE='enterprise', /api/ha must not emit any oauth fields."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_oauth_fields_absent_when_enterprise_locked(self):
        # Seed an oauth_usage row so it *would* appear without the fix.
        now = time.time()
        resets_at = _epoch_to_iso(now + 3600)  # resets in 1 hour

        _seed_oauth_usage(self.conn, resets_at, resets_at, fh_utilization=60.0, sd_utilization=40.0)

        # Also seed a personal event in the window (must not surface).
        _insert_personal(self.conn, "pA", "rA", ts=now - 100, inp=100_000)

        with patch.object(app.config, 'LOCKED_SCOPE', 'enterprise'):
            c = self.client()
            resp = c.get("/api/ha")
            self.assertEqual(resp.status_code, 200)
            body = resp.json()

        # The three oauth-derived fields must be null.
        self.assertIsNone(body["five_hour"],
                          f"five_hour must be None when enterprise-locked, got {body['five_hour']}")
        self.assertIsNone(body["weekly"],
                          f"weekly must be None when enterprise-locked, got {body['weekly']}")
        self.assertIsNone(body["updated_at_epoch"],
                          f"updated_at_epoch must be None when enterprise-locked, got {body['updated_at_epoch']}")

        # The raw JSON text must not leak any personal OAuth field names.
        raw = resp.text
        self.assertNotIn("pct_used", raw,
                         "pct_used must not appear in enterprise-locked response")
        self.assertNotIn("implied_limit_usd", raw,
                         "implied_limit_usd must not appear in enterprise-locked response")
        self.assertNotIn("resets_at", raw,
                         "resets_at must not appear in enterprise-locked response")

        # Enterprise cost totals must still be present.
        self.assertIn("cost_today_usd", body)
        self.assertIn("cost_total_usd", body)

    def test_cost_totals_still_present_when_locked(self):
        """Enterprise cost fields survive the lock — they're enterprise-scoped."""
        with patch.object(app.config, 'LOCKED_SCOPE', 'enterprise'):
            c = self.client()
            body = c.get("/api/ha").json()
        self.assertIsNotNone(body.get("cost_today_usd"))
        self.assertIsNotNone(body.get("cost_total_usd"))


# ---------------------------------------------------------------------------
# Fix 2b — unlocked: window spend is personal-scoped; implied_limit coherent
# ---------------------------------------------------------------------------

class HAUnlockedPersonalScopeTest(TempDBTestCase):
    """When not enterprise-locked, window spend must be PERSONAL-scoped only.

    Seeds:
      - P: personal event, $P cost in the weekly window
      - E: enterprise event, $E cost in the same window

    Expects weekly.spend_usd == P (not P+E), and implied_limit == P / 0.5
    when sd_utilization=50%.
    """

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_weekly_spend_is_personal_scoped(self):
        now = time.time()
        # resets_at is far enough in the future that events seeded near `now`
        # fall inside [resets_epoch - 7days, resets_epoch].
        resets_at_epoch = now + 3600  # 1 hour from now
        resets_at_iso = _epoch_to_iso(resets_at_epoch)

        # Seed oauth_usage with 50% seven_day utilization.
        _seed_oauth_usage(self.conn,
                          five_hour_resets_at=resets_at_iso,
                          seven_day_resets_at=resets_at_iso,
                          sd_utilization=50.0)

        # Personal event: claude-opus-4-8, 1M input = $5 personal cost.
        # Enterprise event: claude-haiku-4-5, 1M input = $1 enterprise cost.
        # Using different models/costs so blended ($6) != personal ($5).
        # Both seeded at `now` — well inside the 7-day window ending at resets_at.
        _insert_event(self.conn, "pA", "rA",
                      plan="max", org=None, account_email="me@personal.io",
                      model="claude-opus-4-8", ts=now - 60, inp=1_000_000)
        _insert_event(self.conn, "eA", "rB",
                      plan="enterprise", org="Acme", account_email="test@acme.io",
                      model="claude-haiku-4-5-20251001", ts=now - 60, inp=1_000_000)

        with patch.object(app.config, 'LOCKED_SCOPE', None):
            c = self.client()
            body = c.get("/api/ha").json()

        weekly = body.get("weekly")
        self.assertIsNotNone(weekly, "weekly block must be present when not locked")

        personal_cost = 5.0  # Opus 4.8 @ $5/M input (personal)
        # enterprise cost is $1/M (Haiku); blended would be $6 — must be excluded
        self.assertAlmostEqual(
            weekly["spend_usd"], personal_cost, places=2,
            msg=f"weekly.spend_usd must be personal cost ${personal_cost} "
                f"(not blended $6.0). Got: {weekly['spend_usd']}"
        )

        # implied_limit = personal_spend / (pct/100) = 5.0 / 0.5 = 10.0
        expected_implied = round(personal_cost / 0.5, 2)
        self.assertAlmostEqual(
            weekly["implied_limit_usd"], expected_implied, places=2,
            msg=f"implied_limit_usd must be {expected_implied} (personal spend / utilization). "
                f"Got: {weekly['implied_limit_usd']}"
        )

    def test_five_hour_spend_is_personal_scoped(self):
        now = time.time()
        resets_at_epoch = now + 3600
        resets_at_iso = _epoch_to_iso(resets_at_epoch)

        _seed_oauth_usage(self.conn,
                          five_hour_resets_at=resets_at_iso,
                          seven_day_resets_at=resets_at_iso,
                          fh_utilization=50.0)

        # Personal Opus 4.8: $5/M. Enterprise Haiku: $1/M. Blended would be $6.
        # five_hour window is 5h = 18000s; events at now-60 are inside it.
        _insert_event(self.conn, "pB", "rC",
                      plan="max", org=None, account_email="me@personal.io",
                      model="claude-opus-4-8", ts=now - 60, inp=1_000_000)
        _insert_event(self.conn, "eB", "rD",
                      plan="enterprise", org="Acme", account_email="test@acme.io",
                      model="claude-haiku-4-5-20251001", ts=now - 60, inp=1_000_000)

        with patch.object(app.config, 'LOCKED_SCOPE', None):
            c = self.client()
            body = c.get("/api/ha").json()

        five_hour = body.get("five_hour")
        self.assertIsNotNone(five_hour)
        self.assertAlmostEqual(
            five_hour["spend_usd"], 5.0, places=2,
            msg=f"five_hour.spend_usd must be personal-only $5.0 (not blended $6.0), "
                f"got {five_hour['spend_usd']}"
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _epoch_to_iso(epoch: float) -> str:
    """Convert epoch float to UTC ISO-8601 string (for seeding oauth_usage)."""
    from datetime import datetime, timezone
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


if __name__ == "__main__":
    unittest.main()
