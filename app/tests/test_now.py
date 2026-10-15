"""GET /api/now: live sessions joined to events (BUILD.md contract)."""
import time
import unittest
from datetime import datetime, timezone
from unittest import mock

from fastapi.testclient import TestClient

from app import agent_state
from app.tests._now_support import WORK, insert_event, set_oauth
from app.tests._support import TempDBTestCase


def _iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


class NowEndpointTests(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        agent_state.reset()
        import app.binding_rate as br
        import app.now_machines as nm
        import app.now_sessions as ns
        ns.reset_cache()
        br._CACHE.clear()
        nm.clear_cache()
        # Tiny DB: let the background caches finish inside the request.
        for cache in (br._CACHE, nm._DIRECTORY):
            p = mock.patch.object(cache, "wait_s", 5.0)
            p.start()
            self.addCleanup(p.stop)
        self.now = time.time()
        from app.main import app
        self.client = TestClient(app)

    def _summarize(self):
        from app.summarizer import summarize_days
        summarize_days(None)

    def _get(self, scope="personal"):
        r = self.client.get(f"/api/now?scope={scope}")
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_claude_session_joined_to_events(self):
        insert_event(self.conn, "e1", session="sid-c", ts=self.now - 600,
                     inp=1_000_000, machine="Laptop.local")
        insert_event(self.conn, "e0", session="sid-c", ts=self.now - 7200,
                     inp=1_000_000, machine="Laptop.local")
        self.conn.execute("INSERT INTO session_titles VALUES('sid-c','Fix it','')")
        self.conn.commit()
        agent_state.update("sid-c", "Laptop.local", "proj", "working",
                           now=self.now - 30, model="claude-opus-4-8")
        body = self._get()
        self.assertEqual(body["scope"], "personal")
        [s] = body["sessions"]
        self.assertEqual(s["id"], "sid-c")
        self.assertEqual(s["machine"], "laptop")
        self.assertEqual(s["project"], "proj")
        self.assertEqual(s["title"], "Fix it")
        self.assertEqual(s["state"], "working")
        self.assertAlmostEqual(s["state_since"], self.now - 30, delta=1)
        self.assertIsNone(s["waiting_reason"])
        self.assertEqual(s["model"], "Opus 4.8")
        self.assertEqual(s["family"], "opus")
        self.assertEqual(s["client"], "claude-code")
        self.assertAlmostEqual(s["cost_usd"], 10.0, places=4)
        # $5 in the last 30 min, session older than 30 min -> $10/h.
        self.assertAlmostEqual(s["burn_usd_per_h"], 10.0, places=4)
        self.assertIsNone(s["share_of_binding_pct_per_h"])
        self.assertEqual(body["live"]["working"], 1)
        self.assertAlmostEqual(body["live"]["burn_usd_per_h"]["m30"], 10.0, places=4)
        self.assertAlmostEqual(body["live"]["burn_usd_per_h"]["h1"], 5.0, places=4)

    def test_pi_session_matches_namespaced_events(self):
        insert_event(self.conn, "p1", session="pi:abc123:native-1",
                     ts=self.now - 300, model="claude-opus-4-8",
                     client="pi-agent", provider="anthropic",
                     reported_total=2.5, machine="workstation")
        agent_state.update("native-1", "workstation", "dev", "working",
                           now=self.now - 10, model="anthropic/claude-opus-4-8")
        [s] = self._get()["sessions"]
        self.assertEqual(s["client"], "pi-agent")
        self.assertAlmostEqual(s["cost_usd"], 2.5, places=6)
        self.assertEqual(s["model"], "Opus 4.8")

    def test_pi_subagent_run_name_is_the_fallback_title(self):
        insert_event(self.conn, "p1", session="pi:abc:child-1",
                     ts=self.now - 300, client="pi-agent", provider="anthropic",
                     reported_total=1.0, agent_id="pi:abc:ui-review-b0")
        agent_state.update("child-1", "workstation", "dev", "working", now=self.now - 10)
        [s] = self._get()["sessions"]
        self.assertEqual(s["title"], "ui-review-b0")

    def _pi_child(self, parent, child, file_name="x.jsonl", state="working"):
        """A Pi subagent session: its cursor file sits under its parent's
        subagents/<parent sid>/ folder, which is the only parent link."""
        from app.ingest import _pi_namespace
        f = f"subagents/{parent}/{file_name}"
        ns = _pi_namespace("personal", "workstation", f, child)
        self.conn.execute(
            "INSERT INTO sync_cursors(machine, project_dir, session_file, updated_at) "
            "VALUES('pi:personal:workstation', '/w', ?, ?)",
            (f, datetime.fromtimestamp(self.now).astimezone().isoformat()))
        insert_event(self.conn, "ev-" + child, session=f"{ns}:{child}",
                     ts=self.now - 120, client="pi-agent", provider="anthropic",
                     reported_total=1.0, agent_id=f"{ns}:run-{child}", machine="workstation")
        agent_state.update(child, "workstation", "w", state, now=self.now - 5)

    def test_pi_subagent_sessions_fold_into_their_parent(self):
        insert_event(self.conn, "pp", session="pi:aaa:parent-1", ts=self.now - 300,
                     client="pi-agent", provider="anthropic", reported_total=2.0,
                     machine="workstation")
        agent_state.update("parent-1", "workstation", "w", "working", now=self.now - 5)
        self._pi_child("parent-1", "child-1")
        self._pi_child("parent-1", "child-2", "y.jsonl")
        # A child of a parent that is not live stays its own row.
        self._pi_child("gone-parent", "orphan", "z.jsonl")
        body = self._get()
        rows = {s["id"]: s for s in body["sessions"]}
        self.assertEqual(set(rows), {"parent-1", "orphan"})
        self.assertEqual(rows["parent-1"]["subagents"], 2)
        self.assertAlmostEqual(rows["parent-1"]["cost_usd"], 4.0, places=6)
        self.assertEqual(body["live"]["subagents"], 2)
        self.assertEqual(body["live"]["working"], 2)

    def test_waiting_pi_subagent_stays_visible(self):
        insert_event(self.conn, "pp", session="pi:aaa:parent-1", ts=self.now - 300,
                     client="pi-agent", provider="anthropic", reported_total=2.0,
                     machine="workstation")
        agent_state.update("parent-1", "workstation", "w", "working", now=self.now - 5)
        self._pi_child("parent-1", "child-1", state="waiting")
        body = self._get()
        self.assertEqual([s["id"] for s in body["sessions"]], ["child-1", "parent-1"])
        self.assertEqual(body["sessions"][1]["subagents"], 1)

    def test_scope_follows_the_session_events(self):
        insert_event(self.conn, "w1", session="sid-w", ts=self.now - 60,
                     inp=1000, account=WORK)
        agent_state.update("sid-w", "laptop", "p", "working", now=self.now - 5)
        self.assertEqual(self._get("personal")["sessions"], [])
        self.assertEqual([s["id"] for s in self._get("enterprise")["sessions"]],
                         ["sid-w"])

    def test_session_without_events_shows_on_known_machines_only(self):
        insert_event(self.conn, "old", session="other", ts=self.now - 86400,
                     inp=1000, machine="laptop")
        self._summarize()
        agent_state.update("new-1", "laptop", "p", "working", now=self.now - 5)
        agent_state.update("new-2", "stranger", "p", "working", now=self.now - 5)
        [s] = self._get()["sessions"]
        self.assertEqual(s["id"], "new-1")
        self.assertIsNone(s["cost_usd"])
        self.assertIsNone(s["burn_usd_per_h"])
        self.assertIsNone(s["client"])

    def test_sort_waiting_longest_first_then_working_then_idle(self):
        insert_event(self.conn, "x", session="seed", ts=self.now - 60, inp=1)
        self._summarize()
        agent_state.update("idle-1", "laptop", "p", "idle", now=self.now - 50)
        agent_state.update("work-1", "laptop", "p", "working", now=self.now - 40)
        agent_state.update("wait-new", "laptop", "p", "waiting",
                           now=self.now - 10, reason="question")
        agent_state.update("wait-old", "laptop", "p", "waiting",
                           now=self.now - 100, reason="permission")
        body = self._get()
        self.assertEqual([s["id"] for s in body["sessions"]],
                         ["wait-old", "wait-new", "work-1", "idle-1"])
        self.assertEqual(body["sessions"][0]["waiting_reason"], "permission")
        live = body["live"]
        self.assertEqual((live["waiting"], live["working"], live["idle"]), (2, 1, 1))

    def test_meridian_twin_sessions_are_hidden(self):
        insert_event(self.conn, "t1", session="twin", ts=self.now - 60, inp=1)
        self.conn.execute("INSERT INTO pi_mirror_sessions VALUES('twin')")
        self.conn.commit()
        agent_state.update("twin", "laptop", "p", "working", now=self.now - 5)
        self.assertEqual(self._get()["sessions"], [])

    def test_machines_and_live_counts(self):
        insert_event(self.conn, "m1", session="s1", ts=self.now - 120,
                     inp=1000, machine="Laptop.local")
        insert_event(self.conn, "m2", session="s2", ts=self.now - 5 * 86400,
                     inp=1000, machine="desk")
        self._summarize()
        agent_state.update("s1", "laptop", "p", "working", now=self.now - 5)
        agent_state.add_subagent("s1", "a1", now=self.now - 4)
        body = self._get()
        machines = {m["name"]: m for m in body["machines"]}
        self.assertEqual(set(machines), {"laptop", "desk"})
        self.assertEqual(machines["laptop"]["working"], 1)
        self.assertAlmostEqual(machines["laptop"]["last_event"], self.now - 120, delta=1)
        self.assertAlmostEqual(machines["desk"]["last_event"], self.now - 5 * 86400, delta=1)
        live = body["live"]
        self.assertEqual(live["machines_total"], 2)
        self.assertEqual(live["machines_online"], 1)
        self.assertEqual(live["subagents"], 1)
        self.assertEqual(body["sessions"][0]["subagents"], 1)
        self.assertAlmostEqual(live["newest_data_age_s"], 120, delta=2)

    def test_share_of_binding_window(self):
        set_oauth(self.conn, {"seven_day": {
            "utilization": 10.0, "resets_at": _iso(self.now + 3 * 86400)}},
            observed=self.now - 60)
        # $10 in the window -> $1/point; $5 in the last 30 min of a long
        # session -> $10/h -> 10 points/h.
        insert_event(self.conn, "a", session="s", ts=self.now - 600, inp=1_000_000)
        insert_event(self.conn, "b", session="s", ts=self.now - 7200, inp=1_000_000)
        agent_state.update("s", "laptop", "p", "working", now=self.now - 5)
        [s] = self._get()["sessions"]
        self.assertAlmostEqual(s["share_of_binding_pct_per_h"], 10.0, places=3)

    def test_failing_join_degrades_to_nulls(self):
        insert_event(self.conn, "e1", session="sid", ts=self.now - 60, inp=1000)
        self._summarize()
        agent_state.update("sid", "laptop", "p", "working", now=self.now - 5)
        with mock.patch("app.now_sessions.session_rollups",
                        side_effect=RuntimeError("boom")):
            body = self._get()
        [s] = body["sessions"]
        self.assertIsNone(s["cost_usd"])
        self.assertEqual(s["state"], "working")

    def test_empty_fleet(self):
        body = self._get()
        self.assertEqual(body["sessions"], [])
        self.assertEqual(body["live"]["working"], 0)
        self.assertIsNone(body["live"]["newest_data_age_s"])

    def test_bad_scope_is_400(self):
        self.assertEqual(self.client.get("/api/now?scope=nope").status_code, 400)


class NowPureTests(unittest.TestCase):
    def test_burn_rate_window(self):
        from app.now import burn_rate
        now = 10_000.0
        # Old session: the full 30 min window.
        self.assertAlmostEqual(burn_rate(5.0, now - 7200, now), 10.0)
        # 10-minute-old session: its own age is the window.
        self.assertAlmostEqual(burn_rate(5.0, now - 600, now), 30.0)
        # A 1-minute-old session uses a 5 min floor, not an absurd rate.
        self.assertAlmostEqual(burn_rate(1.0, now - 60, now), 12.0)

    def test_model_display_and_client_inference(self):
        from app.now import infer_client, model_display
        self.assertEqual(model_display("anthropic/claude-opus-5-5"), "Opus 5.5")
        self.assertEqual(model_display("claude-sonnet-5-5"), "Sonnet 5.5")
        self.assertIsNone(model_display(""))
        self.assertEqual(infer_client("anthropic/claude-opus-5-5"), "pi-agent")
        self.assertEqual(infer_client("claude-opus-5-5"), "claude-code")
        self.assertIsNone(infer_client(""))


if __name__ == "__main__":
    unittest.main()
