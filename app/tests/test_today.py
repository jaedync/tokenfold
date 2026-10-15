"""GET /api/today: spend so far today against a typical day (BUILD.md)."""
import time
import unittest
from datetime import datetime, timedelta
from unittest import mock
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from app import agent_state
from app.tests._now_support import WORK, insert_event
from app.tests._support import TempDBTestCase


def _tz():
    from app.config import TZ_NAME
    return ZoneInfo(TZ_NAME)


def _local(day: datetime, hour: int, minute: int = 0) -> float:
    return day.replace(hour=hour, minute=minute, second=0, microsecond=0).timestamp()


class TodayTests(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        agent_state.reset()
        import app.today_typical as tt
        tt.clear_cache()
        p = mock.patch.object(tt._PROFILE, "wait_s", 5.0)
        p.start()
        self.addCleanup(p.stop)
        self.today = datetime.now(_tz()).replace(hour=0, minute=0, second=0,
                                                 microsecond=0)

    def _summarize(self):
        from app.summarizer import summarize_days
        summarize_days(None)

    def _build(self, now, scope="personal"):
        from app.today import build_today
        return build_today(scope, {"sessions": {}}, now)

    def test_today_totals_models_and_machines(self):
        now = time.time()
        t = max(self.today.timestamp() + 60, now - 3600)
        insert_event(self.conn, "a", session="s1", ts=t, inp=1_000_000,
                     machine="Laptop.local")
        insert_event(self.conn, "b", session="s2", ts=t, inp=0, out=0,
                     model="claude-opus-5-5", client="pi-agent",
                     provider="anthropic", reported_total=2.0, machine="workstation")
        insert_event(self.conn, "u", session="s1", ts=t, type_="user", human=1,
                     machine="Laptop.local")
        insert_event(self.conn, "w", session="s3", ts=t, inp=1_000_000,
                     account=WORK)
        self._summarize()
        body = self._build(now)
        self.assertEqual(body["day"], self.today.strftime("%Y-%m-%d"))
        self.assertAlmostEqual(body["spend_usd"], 7.0, places=4)
        self.assertEqual(body["prompts"], 1)
        models = {m["model"]: m for m in body["by_model"]}
        self.assertEqual(set(models), {"Opus 4.8", "Opus 5.5"})
        self.assertEqual(models["Opus 4.8"]["provider"], "anthropic")
        self.assertEqual(models["Opus 4.8"]["family"], "opus")
        self.assertEqual(models["Opus 4.8"]["tokens"], 1_000_000)
        self.assertAlmostEqual(models["Opus 5.5"]["usd"], 2.0, places=6)
        self.assertEqual(body["by_model"][0]["model"], "Opus 4.8")   # usd desc
        machines = {m["machine"]: m for m in body["by_machine"]}
        self.assertEqual(set(machines), {"laptop", "workstation"})
        self.assertAlmostEqual(machines["laptop"]["usd"], 5.0, places=4)
        self.assertEqual(machines["laptop"]["working"], 0)

    def test_typical_is_the_median_of_past_days(self):
        d1, d2, d3 = (self.today - timedelta(days=n) for n in (1, 2, 3))
        # d1: $5 at 01:00 and $5 at 23:00. d2: $10 at 01:00. d3: $5 at 23:00
        insert_event(self.conn, "a", session="s", ts=_local(d1, 1), inp=1_000_000)
        insert_event(self.conn, "b", session="s", ts=_local(d1, 23), inp=1_000_000)
        insert_event(self.conn, "c", session="s", ts=_local(d2, 1), inp=2_000_000)
        insert_event(self.conn, "d", session="s", ts=_local(d3, 23), inp=1_000_000)
        self._summarize()
        noon = _local(self.today, 12)
        body = self._build(noon)
        # By noon: d1 $5, d2 $10, d3 $0 -> median $5. Full days: 10, 10, 5.
        self.assertAlmostEqual(body["typical_by_now_usd"], 5.0, places=4)
        self.assertAlmostEqual(body["typical_day_usd"], 10.0, places=4)
        self.assertEqual(body["spend_usd"], 0.0)

    def test_typical_interpolates_inside_a_slot(self):
        d1 = self.today - timedelta(days=1)
        insert_event(self.conn, "a", session="s", ts=_local(d1, 10, 0), inp=1_000_000)
        self._summarize()
        # At 10:07:30 the 10:00 slot is half over: half its spend counts.
        body = self._build(_local(self.today, 10, 7) + 30)
        self.assertAlmostEqual(body["typical_by_now_usd"], 2.5, places=4)

    def test_no_history_gives_null_typical(self):
        body = self._build(time.time())
        self.assertIsNone(body["typical_by_now_usd"])
        self.assertIsNone(body["typical_day_usd"])
        self.assertEqual(body["by_model"], [])

    def test_live_working_counts_per_machine(self):
        now = time.time()
        insert_event(self.conn, "a", session="live-1", ts=now - 60, inp=1000,
                     machine="laptop")
        self._summarize()
        from app.today import build_today
        agent_state.update("live-1", "Laptop.local", "p", "working", now=now - 5)
        body = build_today("personal", agent_state.snapshot(now=now), now)
        machines = {m["machine"]: m for m in body["by_machine"]}
        self.assertEqual(machines["laptop"]["working"], 1)

    def test_endpoint_shape(self):
        from app.main import app
        r = TestClient(app).get("/api/today?scope=personal")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(set(r.json()), {
            "day", "spend_usd", "typical_by_now_usd", "typical_day_usd",
            "prompts", "active_s", "by_model", "by_machine"})


if __name__ == "__main__":
    unittest.main()
