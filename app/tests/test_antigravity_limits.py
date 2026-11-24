"""Antigravity provider windows are separate from Claude's fitted quota rates."""
import time

from app.tests._support import TempDBTestCase

GEMINI = "antigravity-gemini"
THIRD_PARTY = "antigravity-claude-gpt"


class AntigravityLimitsTest(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.now = time.time()
        self.http = self.client()

    def snapshot(self, provider=GEMINI, observed=None, **dollars):
        return {
            "provider": provider,
            "observed_at_epoch": self.now - 5 if observed is None else observed,
            "windows": [
                {"key": "five_hour", "pct": 0, "resets_at_epoch": self.now + 14_400,
                 **({"implied_window_usd": dollars["short"]} if "short" in dollars else {})},
                {"key": "seven_day", "pct": 40, "resets_at_epoch": self.now + 60_000,
                 **({"implied_window_usd": dollars["long"]} if "long" in dollars else {})},
            ],
        }

    def post(self, *snapshots, scope="personal"):
        return self.http.post("/api/provider-usage", json={
            "machine": "test-host", "account_class": scope, "limits": list(snapshots),
        }, headers={"X-API-Key": self.api_key})

    def runway(self, scope="personal"):
        r = self.http.get(f"/api/runway?scope={scope}")
        self.assertEqual(r.status_code, 200, r.text)
        return {w["id"]: w for w in r.json()["windows"]}

    def budget(self):
        r = self.http.get("/api/rate-limits?scope=personal")
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()["weekly_budget"]

    def test_both_families_show_four_percent_windows_without_calibration(self):
        r = self.post(self.snapshot(), self.snapshot(THIRD_PARTY))
        self.assertEqual(r.status_code, 200, r.text)
        windows = self.runway()
        self.assertEqual(set(windows), {f"{provider}:{key}"
                        for provider in (GEMINI, THIRD_PARTY) for key in ("five_hour", "seven_day")})
        for provider, label in ((GEMINI, "Antigravity Gemini"),
                                (THIRD_PARTY, "Antigravity Claude/GPT")):
            short, long = (windows[f"{provider}:{key}"] for key in ("five_hour", "seven_day"))
            self.assertEqual(short["label"], f"{label} · 5-hour")
            self.assertEqual(long["label"], f"{label} · 7-day")
            self.assertEqual((short["pct"], long["pct"]), (0, 40))
            self.assertEqual((short["window_s"], long["window_s"]), (18_000, 604_800))
            for window in (short, long):
                self.assertNotIn("implied_window_usd", window)
                self.assertIsNone(window["usd_per_point"])
                self.assertIsNone(window["usd_spent"])
                self.assertIsNone(window["usd_left"])
        self.assertEqual(self.runway("enterprise"), {})
        self.assertNotIn("month_cost", self.budget()["providers"][THIRD_PARTY])

    def test_supplied_gemini_capacity_has_no_fit_or_spend_gate(self):
        # No events, no fitted family, less than $100 of capacity and 0% in
        # the short window. Measured-spend and low-utilization gates do not apply.
        r = self.post(self.snapshot(short=30, long=99.25))
        self.assertEqual(r.status_code, 200, r.text)
        windows = self.runway()
        short = windows[f"{GEMINI}:five_hour"]
        long = windows[f"{GEMINI}:seven_day"]
        self.assertEqual(short["implied_window_usd"], 30)
        self.assertEqual(short["usd_left"], 30)
        self.assertEqual(long["implied_window_usd"], 99.25)
        self.assertAlmostEqual(long["usd_left"], 99.25 * 0.6)
        self.assertIsNone(long["usd_spent"])
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0], 0)
        classic = self.budget()["providers"][GEMINI]["windows"]
        self.assertEqual({w["key"]: w["implied_window_usd"] for w in classic},
                         {"five_hour": 30, "seven_day": 99.25})

    def test_no_calibration_for_third_party_and_no_work_report(self):
        self.assertEqual(self.post(self.snapshot(THIRD_PARTY, short=30)).status_code, 422)
        self.assertEqual(self.post(self.snapshot(), scope="work").status_code, 422)
        duplicate = self.snapshot()
        duplicate["windows"][1] = duplicate["windows"][0]
        self.assertEqual(self.post(duplicate).status_code, 422)
        self.assertEqual(self.runway(), {})

    def test_observation_order_keeps_percent_and_allows_calibration_refresh(self):
        first = self.snapshot()
        updated = self.snapshot(short=30, long=105)
        self.assertEqual(self.post(first).status_code, 200)
        self.assertEqual(self.post(updated).status_code, 200)
        changes = self.conn.total_changes
        self.assertEqual(self.post(updated).status_code, 200)
        self.assertEqual(self.conn.total_changes, changes)
        old = self.snapshot(observed=self.now - 10, long=1)
        old["windows"][1]["pct"] = 90
        self.assertEqual(self.post(old).status_code, 200)
        future = self.snapshot(observed=self.now + 3600, long=1)
        self.assertEqual(self.post(future).status_code, 200)
        self.assertEqual(self.conn.total_changes, changes)
        long = self.runway()[f"{GEMINI}:seven_day"]
        self.assertEqual(long["pct"], 40)
        self.assertEqual(long["implied_window_usd"], 105)

    def test_old_or_expired_windows_have_no_dollars(self):
        expired = self.snapshot(short=30, long=105)
        expired["windows"][0]["resets_at_epoch"] = self.now - 120
        self.assertEqual(self.post(expired).status_code, 200)
        short = self.runway()[f"{GEMINI}:five_hour"]
        self.assertTrue(short["stale"])
        self.assertNotIn("implied_window_usd", short)

    def test_stale_observation_can_show_pct_but_has_no_dollars(self):
        self.assertEqual(self.post(self.snapshot(observed=self.now - 7200, long=105)).status_code, 200)
        long = self.runway()[f"{GEMINI}:seven_day"]
        self.assertTrue(long["stale"])
        self.assertEqual(long["pct"], 40)
        self.assertNotIn("implied_window_usd", long)

    def test_both_weekly_windows_can_request_runway_trajectory_bounds(self):
        self.assertEqual(self.post(self.snapshot(), self.snapshot(THIRD_PARTY)).status_code, 200)
        for provider in (GEMINI, THIRD_PARTY):
            window = f"{provider}:seven_day"
            r = self.http.get("/api/runway/trajectory", params={"scope": "personal", "window": window})
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.json()["window"], window)
            self.assertEqual(r.json()["end"] - r.json()["start"], 604_800)
