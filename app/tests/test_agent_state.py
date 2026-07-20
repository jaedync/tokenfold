"""Agent-state store + /api/notify notification policy.

Covers the 2026-07-18 consolidation: working events are state-only (the
push-spam regression this feature fixes), waiting pushes dedup per spell,
presence damping, aggregation, TTL decay, and the authed read endpoint.
"""
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from app import agent_state
from app.main import app


TOKEN = "test-notify-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class AgentStateStoreTests(unittest.TestCase):
    def setUp(self):
        agent_state.reset()

    def test_update_and_snapshot(self):
        agent_state.update("s1", "mac", "proj", "working", now=1000.0)
        snap = agent_state.snapshot(now=1001.0)
        self.assertEqual(snap["sessions"]["s1"]["state"], "working")
        self.assertEqual(snap["sessions"]["s1"]["machine"], "mac")
        self.assertTrue(snap["any_working"])
        self.assertFalse(snap["any_waiting"])

    def test_ttl_decay_forgets_dead_sessions(self):
        agent_state.update("s1", "mac", "proj", "waiting", now=1000.0)
        snap = agent_state.snapshot(now=1000.0 + 599)
        self.assertIn("s1", snap["sessions"])
        snap = agent_state.snapshot(now=1000.0 + 601)
        self.assertEqual(snap["sessions"], {})
        self.assertFalse(snap["any_waiting"])

    def test_working_clears_waiting_notified(self):
        agent_state.update("s1", "mac", "proj", "waiting", now=1000.0)
        agent_state.mark_waiting_notified("s1")
        self.assertTrue(agent_state.get_session("s1")["waiting_notified"])
        agent_state.update("s1", "mac", "proj", "working", now=1001.0)
        self.assertFalse(agent_state.get_session("s1")["waiting_notified"])

    def test_out_of_order_working_is_discarded(self):
        # The sub-2s-turn race: stop (sent later) arrives BEFORE the
        # delayed working retry. The stale working must not resurrect an
        # "active" cube for the whole TTL.
        agent_state.update("s1", "mac", "proj", "idle", now=1002.0, event_ts=12.0)
        agent_state.update("s1", "mac", "proj", "working", now=1003.0, event_ts=10.0)
        self.assertEqual(agent_state.get_session("s1")["state"], "idle")

    def test_in_order_events_apply_normally(self):
        agent_state.update("s1", "mac", "proj", "working", now=1000.0, event_ts=10.0)
        agent_state.update("s1", "mac", "proj", "idle", now=1001.0, event_ts=12.0)
        self.assertEqual(agent_state.get_session("s1")["state"], "idle")

    def test_events_without_client_ts_keep_arrival_order(self):
        agent_state.update("s1", "mac", "proj", "idle", now=1002.0)
        agent_state.update("s1", "mac", "proj", "working", now=1003.0)
        self.assertEqual(agent_state.get_session("s1")["state"], "working")

    def test_seconds_since_working_none_until_first_prompt(self):
        self.assertIsNone(agent_state.seconds_since_working())
        agent_state.update("s1", "mac", "proj", "working", now=1000.0)
        self.assertAlmostEqual(agent_state.seconds_since_working(now=1030.0), 30.0)


class NotifyPolicyTests(unittest.TestCase):
    """Drive the policy through the real endpoint with HA relay mocked."""

    def setUp(self):
        agent_state.reset()
        import app.notify as notify_mod
        self._notify_mod = notify_mod
        self._saved_token = notify_mod._notify_token
        notify_mod._notify_token = TOKEN
        self.pushes = []

        async def fake_relay(payload, devices=None):
            self.pushes.append(payload)
            return []  # no errors

        self._patcher = mock.patch.object(notify_mod, "_relay_to_ha", fake_relay)
        self._patcher.start()
        self.client = TestClient(app)

    def tearDown(self):
        self._patcher.stop()
        self._notify_mod._notify_token = self._saved_token
        agent_state.reset()

    def post(self, body):
        return self.client.post("/api/notify", json=body, headers=AUTH)

    def test_working_is_state_only_never_a_push(self):
        r = self.post({"event": "working", "project": "p", "machine": "mac",
                       "session_id": "s1"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json().get("state"), "working")
        self.assertEqual(self.pushes, [])
        self.assertEqual(agent_state.get_session("s1")["state"], "working")

    def test_waiting_pushes_once_per_spell(self):
        # no prior working event -> presence damping cannot apply
        r1 = self.post({"event": "permission", "project": "p", "machine": "mac",
                        "session_id": "s1"})
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(len(self.pushes), 1)
        # same spell re-fires (client retry, duplicate hook): suppressed
        r2 = self.post({"event": "permission", "project": "p", "machine": "mac",
                        "session_id": "s1"})
        self.assertEqual(r2.json().get("suppressed"), "duplicate")
        self.assertEqual(len(self.pushes), 1)
        # user answers (working), then a NEW waiting spell: pushes again
        # (working also arms presence damping, so disable it for this leg)
        with mock.patch.object(self._notify_mod, "AGENT_PRESENCE_DAMPING_S", 0):
            self.post({"event": "working", "project": "p", "machine": "mac",
                       "session_id": "s1"})
            self.post({"event": "permission", "project": "p", "machine": "mac",
                       "session_id": "s1"})
        self.assertEqual(len(self.pushes), 2)

    def test_presence_damping_suppresses_fresh_prompt(self):
        self.post({"event": "working", "project": "p", "machine": "mac",
                   "session_id": "s1"})
        r = self.post({"event": "permission", "project": "p", "machine": "mac",
                       "session_id": "s1"})
        self.assertEqual(r.json().get("suppressed"), "presence")
        self.assertEqual(self.pushes, [])
        # state still recorded for the ambient display
        self.assertEqual(agent_state.get_session("s1")["state"], "waiting")

    def test_aggregation_combines_multiple_waiting_sessions(self):
        with mock.patch.object(self._notify_mod, "AGENT_PRESENCE_DAMPING_S", 0):
            self.post({"event": "permission", "project": "alpha", "machine": "mac",
                       "session_id": "s1"})
            self.post({"event": "question", "project": "beta", "machine": "redarch",
                       "session_id": "s2"})
        self.assertEqual(len(self.pushes), 2)
        combined = self.pushes[1]
        self.assertIn("2 sessions waiting", combined["title"])
        self.assertIn("alpha@mac", combined["message"])
        self.assertIn("beta@redarch", combined["message"])
        # both spells are closed by the aggregate push
        self.assertTrue(agent_state.get_session("s1")["waiting_notified"])
        self.assertTrue(agent_state.get_session("s2")["waiting_notified"])

    def test_stop_still_pushes_and_goes_idle(self):
        r = self.post({"event": "stop", "project": "p", "machine": "mac",
                       "session_id": "s1", "duration_s": 42, "tool_count": 3})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(self.pushes), 1)
        self.assertIn("Response complete", self.pushes[0]["title"])
        self.assertEqual(agent_state.get_session("s1")["state"], "idle")

    def test_codex_style_payload_without_session_id(self):
        r = self.post({"event": "stop", "project": "p", "machine": "mac"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(agent_state.get_session("mac:p")["state"], "idle")

    def test_raw_payload_passthrough_unchanged(self):
        r = self.post({"title": "custom", "message": "hi"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.pushes, [{"title": "custom", "message": "hi"}])


class AgentStateEndpointTests(unittest.TestCase):
    def setUp(self):
        agent_state.reset()
        import app.notify as notify_mod
        self._notify_mod = notify_mod
        self._saved_token = notify_mod._notify_token
        notify_mod._notify_token = TOKEN
        self.client = TestClient(app)

    def tearDown(self):
        self._notify_mod._notify_token = self._saved_token
        agent_state.reset()

    def test_requires_bearer_auth(self):
        r = self.client.get("/api/agent-state")
        self.assertEqual(r.status_code, 401)
        r = self.client.get("/api/agent-state",
                            headers={"Authorization": "Bearer wrong"})
        self.assertEqual(r.status_code, 401)

    def test_snapshot_shape(self):
        agent_state.update("s1", "mac", "proj", "waiting")
        r = self.client.get("/api/agent-state", headers=AUTH)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["any_waiting"])
        self.assertEqual(body["summary"]["waiting"], 1)
        self.assertEqual(body["sessions"]["s1"]["project"], "proj")


if __name__ == "__main__":
    unittest.main()
