"""agent_state: state_since and waiting_reason for /api/now.

age_s is time since the last event; the Now panel needs time in the current
state ("waiting 12 min") and why a session waits (permission, question,
attention), which only the notify event kind knows.
"""
import unittest

from fastapi.testclient import TestClient

from app import agent_state


class StateSinceTests(unittest.TestCase):
    def setUp(self):
        agent_state.reset()

    def _sess(self, sid="s1", now=None):
        return agent_state.snapshot(now=now)["sessions"][sid]

    def test_state_since_is_first_entry_into_state(self):
        agent_state.update("s1", "mac", "p", "working", now=1000.0)
        agent_state.update("s1", "mac", "p", "working", now=1050.0)
        s = self._sess(now=1060.0)
        self.assertEqual(s["state_since"], 1000.0)
        self.assertEqual(s["age_s"], 10.0)

    def test_state_change_moves_state_since(self):
        agent_state.update("s1", "mac", "p", "working", now=1000.0)
        agent_state.update("s1", "mac", "p", "waiting", now=1100.0)
        self.assertEqual(self._sess(now=1101.0)["state_since"], 1100.0)

    def test_tool_activity_keeps_working_since(self):
        agent_state.update("s1", "mac", "p", "working", now=1000.0)
        agent_state.tool_activity("s1", "mac", "p", now=1200.0)
        self.assertEqual(self._sess(now=1201.0)["state_since"], 1000.0)

    def test_stale_out_of_order_event_does_not_move_state_since(self):
        agent_state.update("s1", "mac", "p", "idle", now=1000.0, event_ts=12.0)
        agent_state.update("s1", "mac", "p", "working", now=1001.0, event_ts=10.0)
        s = self._sess(now=1002.0)
        self.assertEqual(s["state"], "idle")
        self.assertEqual(s["state_since"], 1000.0)

    def test_session_start_and_spawned_parent_have_state_since(self):
        agent_state.session_start("a", "mac", "p", now=1000.0)
        agent_state.add_subagent("b", "ag1", machine="mac", now=1010.0)
        agent_state.reconcile_subagents("c", [{"agent_id": "x"}], now=1020.0)
        snap = agent_state.snapshot(now=1030.0)["sessions"]
        self.assertEqual(snap["a"]["state_since"], 1000.0)
        self.assertEqual(snap["b"]["state_since"], 1010.0)
        self.assertEqual(snap["c"]["state_since"], 1020.0)

    def test_waiting_reason_set_and_cleared(self):
        agent_state.update("s1", "mac", "p", "waiting", now=1000.0,
                           reason="permission")
        self.assertEqual(self._sess(now=1001.0)["waiting_reason"], "permission")
        # A second waiting kind in the same spell: the latest reason wins,
        # the spell start stays.
        agent_state.update("s1", "mac", "p", "waiting", now=1005.0,
                           reason="question")
        s = self._sess(now=1006.0)
        self.assertEqual(s["waiting_reason"], "question")
        self.assertEqual(s["state_since"], 1000.0)
        agent_state.update("s1", "mac", "p", "working", now=1010.0)
        self.assertIsNone(self._sess(now=1011.0)["waiting_reason"])

    def test_snapshot_reports_raw_model(self):
        agent_state.update("s1", "mac", "p", "working", now=1000.0,
                           model="anthropic/claude-opus-5-5")
        self.assertEqual(self._sess(now=1001.0)["model"],
                         "anthropic/claude-opus-5-5")


class NotifyWaitingReasonTests(unittest.TestCase):
    TOKEN = "test-notify-token"

    def setUp(self):
        import app.notify as notify
        agent_state.reset()
        self._saved = notify._notify_token
        notify._notify_token = self.TOKEN
        self.addCleanup(setattr, notify, "_notify_token", self._saved)

    def test_each_waiting_event_kind_becomes_the_reason(self):
        from app.main import app
        auth = {"Authorization": f"Bearer {self.TOKEN}"}
        c = TestClient(app)
        for i, kind in enumerate(("permission", "question", "attention")):
            sid = f"s{i}"
            r = c.post("/api/notify", headers=auth, json={
                "event": kind, "session_id": sid, "machine": "m",
                "project": "p"})
            self.assertEqual(r.status_code, 200)
            snap = agent_state.snapshot()["sessions"][sid]
            self.assertEqual(snap["state"], "waiting")
            self.assertEqual(snap["waiting_reason"], kind)


if __name__ == "__main__":
    unittest.main()
