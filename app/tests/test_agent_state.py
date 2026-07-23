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
        # Non-waiting states use the default TTL (waiting has its own,
        # longer one; see test_waiting_outlives_the_default_ttl).
        agent_state.update("s1", "mac", "proj", "working", now=1000.0)
        snap = agent_state.snapshot(now=1000.0 + 599)
        self.assertIn("s1", snap["sessions"])
        snap = agent_state.snapshot(now=1000.0 + 601)
        self.assertEqual(snap["sessions"], {})
        self.assertFalse(snap["any_working"])

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

    def test_waiting_outlives_the_default_ttl(self):
        # A blocked permission prompt after 10 minutes is still waiting;
        # it must NOT decay to an idle-looking cube.
        agent_state.update("s1", "mac", "proj", "waiting", now=1000.0)
        snap = agent_state.snapshot(now=1000.0 + 601)
        self.assertEqual(snap["sessions"]["s1"]["state"], "waiting")
        self.assertTrue(snap["any_waiting"])
        # ...but a hard-crashed terminal still can't strand it forever
        snap = agent_state.snapshot(now=1000.0 + 7201)
        self.assertEqual(snap["sessions"], {})

    def test_remove_forgets_session_immediately(self):
        agent_state.update("s1", "mac", "proj", "waiting", now=1000.0)
        self.assertTrue(agent_state.remove("s1"))
        self.assertEqual(agent_state.snapshot(now=1001.0)["sessions"], {})
        self.assertFalse(agent_state.remove("s1"))   # idempotent

    def test_fleet_rev_recorded_and_rolled_up(self):
        agent_state.update("s1", "mac", "p", "working", now=1000.0, fleet_rev="abc1234")
        agent_state.update("s2", "htz", "p", "working", now=1001.0, fleet_rev="def5678")
        snap = agent_state.snapshot(now=1002.0)
        self.assertEqual(snap["sessions"]["s1"]["fleet_rev"], "abc1234")
        self.assertEqual(snap["fleet_revs"], {"mac": "abc1234", "htz": "def5678"})
        # newest event wins per machine (a just-synced machine stops looking stale)
        agent_state.update("s3", "mac", "p", "working", now=1003.0, fleet_rev="fff9999")
        self.assertEqual(agent_state.snapshot(now=1004.0)["fleet_revs"]["mac"], "fff9999")

    def test_fleet_rev_absent_stays_none(self):
        agent_state.update("s1", "mac", "p", "working", now=1000.0)
        snap = agent_state.snapshot(now=1001.0)
        self.assertIsNone(snap["sessions"]["s1"]["fleet_rev"])
        self.assertEqual(snap["fleet_revs"], {})

    def test_seconds_since_working_none_until_first_prompt(self):
        self.assertIsNone(agent_state.seconds_since_working())
        agent_state.update("s1", "mac", "proj", "working", now=1000.0)
        self.assertAlmostEqual(agent_state.seconds_since_working(now=1030.0), 30.0)

    # --- Fan-out (subagent) tracking --------------------------------------
    def test_add_and_remove_subagent_changes_fanout(self):
        agent_state.update("s1", "mac", "proj", "working", now=1000.0)
        self.assertEqual(agent_state.add_subagent("s1", "a1", now=1000.0), 1)
        self.assertEqual(agent_state.add_subagent("s1", "a2", now=1000.0), 2)
        snap = agent_state.snapshot(now=1001.0)
        self.assertEqual(snap["sessions"]["s1"]["fanout"], 2)
        self.assertEqual(snap["sessions"]["s1"]["subagents"], ["a1", "a2"])
        self.assertEqual(snap["total_fanout"], 2)
        self.assertEqual(agent_state.remove_subagent("s1", "a1", now=1002.0), 1)
        self.assertEqual(agent_state.snapshot(now=1003.0)["sessions"]["s1"]["fanout"], 1)

    def test_subagent_start_before_parent_creates_working_session(self):
        # A spawn that races ahead of any parent event still shows a working
        # session (spawning a subagent means the parent is working).
        agent_state.add_subagent("p1", "a1", machine="mac", project="proj", now=1000.0)
        snap = agent_state.snapshot(now=1001.0)
        self.assertEqual(snap["sessions"]["p1"]["state"], "working")
        self.assertEqual(snap["sessions"]["p1"]["fanout"], 1)

    def test_stale_subagent_decays_out_of_fanout(self):
        # A missed SubagentStop cannot strand a mote: it ages out.
        agent_state.update("s1", "mac", "proj", "working", now=1000.0)
        agent_state.add_subagent("s1", "a1", now=1000.0)
        self.assertEqual(agent_state.snapshot(now=1000.0 + 179)["sessions"]["s1"]["fanout"], 1)
        self.assertEqual(agent_state.snapshot(now=1000.0 + 181)["sessions"]["s1"]["fanout"], 0)

    def test_working_heartbeat_keeps_subagents_alive(self):
        # Long background fan-out: the parent's working heartbeats refresh
        # its live motes so they outlive the subagent TTL.
        agent_state.update("s1", "mac", "proj", "working", now=1000.0)
        agent_state.add_subagent("s1", "a1", now=1000.0)
        agent_state.update("s1", "mac", "proj", "working", now=1000.0 + 120)  # heartbeat
        self.assertEqual(agent_state.snapshot(now=1000.0 + 250)["sessions"]["s1"]["fanout"], 1)

    def test_remove_subagent_unknown_parent_is_noop(self):
        self.assertEqual(agent_state.remove_subagent("ghost", "a1"), 0)

    # --- Genuine-working timestamp (feeds the mood ladder) ----------------
    def test_working_event_sets_working_ts(self):
        agent_state.update("s1", "mac", "proj", "working", now=1000.0)
        self.assertEqual(agent_state.get_session("s1")["working_ts"], 1000.0)

    def test_fanout_activity_does_not_refresh_working_ts(self):
        # A subagent stop is activity (bumps ts) but is NOT a working
        # heartbeat, so working_ts must stay put: this is what lets the
        # mood tell "actively grinding" from "handed off to children".
        agent_state.update("s1", "mac", "proj", "working", now=1000.0)
        agent_state.add_subagent("s1", "a1", now=1000.0)
        agent_state.remove_subagent("s1", "a1", now=1200.0)
        s = agent_state.get_session("s1")
        self.assertEqual(s["ts"], 1200.0)          # activity bumped ts
        self.assertEqual(s["working_ts"], 1000.0)  # but not the working clock

    def test_subagent_spawn_marks_parent_working_ts(self):
        # Spawning a subagent means the parent is working right now.
        agent_state.add_subagent("p1", "a1", machine="mac", now=1000.0)
        self.assertEqual(agent_state.get_session("p1")["working_ts"], 1000.0)

    def test_heartbeat_refresh_does_not_bump_working_ts(self):
        # A subagent PostToolUse heartbeat re-adds the same mote; it must not
        # refresh the parent's working clock (else a handed-off parent never
        # goes stale and waiting_subagent could never fire).
        agent_state.update("s1", "mac", "proj", "working", now=1000.0)
        agent_state.add_subagent("s1", "a1", now=1000.0)
        agent_state.add_subagent("s1", "a1", now=1200.0)   # heartbeat refresh
        self.assertEqual(agent_state.get_session("s1")["working_ts"], 1000.0)
        self.assertEqual(agent_state.snapshot(now=1201.0)["sessions"]["s1"]["fanout"], 1)

    # --- Aggregate mood + drawable agents list ----------------------------
    def test_mood_idle_when_empty(self):
        self.assertEqual(agent_state.snapshot(now=1000.0)["mood"], "idle")

    def test_mood_working_when_grinding(self):
        agent_state.update("s1", "mac", "proj", "working", now=1000.0)
        self.assertEqual(agent_state.snapshot(now=1001.0)["mood"], "working")

    def test_mood_needs_you_beats_working(self):
        agent_state.update("s1", "mac", "proj", "working", now=1000.0)
        agent_state.update("s2", "mac", "proj", "waiting", now=1000.0)
        self.assertEqual(agent_state.snapshot(now=1001.0)["mood"], "needs_you")

    def test_mood_waiting_subagent_when_handed_off(self):
        # A working session that fanned out but whose working heartbeat has
        # gone stale (blocked in a synchronous fan-out) reads as handed off.
        agent_state.update("s1", "mac", "proj", "working", now=1000.0)
        agent_state.add_subagent("s1", "a1", now=1000.0)
        # 100s later: past AGENT_STATE_WORKING_FRESH_S (90), still has a mote
        # (mote refreshed to keep it live), so the mood is waiting_subagent.
        s = agent_state.get_session("s1")
        s["subagents"] = {"a1": 1099.0}     # mote kept alive by heartbeat
        snap = agent_state.snapshot(now=1100.0)
        self.assertEqual(snap["mood"], "waiting_subagent")

    def test_mood_working_beats_waiting_subagent_when_fresh(self):
        # Same fan-out but the parent is still heartbeating: active grind wins.
        agent_state.update("s1", "mac", "proj", "working", now=1000.0)
        agent_state.add_subagent("s1", "a1", now=1000.0)
        self.assertEqual(agent_state.snapshot(now=1001.0)["mood"], "working")

    def test_agents_list_enumerates_sessions_and_subagents(self):
        agent_state.update("s1", "mac", "proj", "working", now=1000.0)
        agent_state.add_subagent("s1", "a2", now=1000.0)
        agent_state.add_subagent("s1", "a1", now=1000.0)
        agent_state.update("s0", "mac", "proj", "waiting", now=1000.0)
        agents = agent_state.snapshot(now=1000.5)["agents"]
        # Sessions sorted by id; each session's subagents sorted and adjacent.
        self.assertEqual(
            [(a["id"], a["kind"], a["state"]) for a in agents],
            [("s0", "session", "waiting"),
             ("s1", "session", "working"),
             ("s1:a1", "subagent", "working"),
             ("s1:a2", "subagent", "working")])
        self.assertAlmostEqual(agents[0]["age_s"], 0.5, places=1)

    def test_agents_list_omits_stale_subagents(self):
        agent_state.update("s1", "mac", "proj", "working", now=1000.0)
        agent_state.add_subagent("s1", "a1", now=1000.0)
        # 200s later, past the 180s subagent TTL: the mote is gone.
        agents = agent_state.snapshot(now=1200.0)["agents"]
        self.assertEqual([a["id"] for a in agents], ["s1"])


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

    def test_state_only_stop_goes_straight_to_idle_without_ha_push(self):
        # Automated sessions emit their transitions too (complete picture);
        # the server records idle (never ready) and pushes nothing to HA.
        self.post({"event": "working", "machine": "mac", "project": "bot",
                   "session_id": "bot-1", "client_ts": 1.0})
        r = self.post({"event": "stop", "machine": "mac", "project": "bot",
                       "session_id": "bot-1", "client_ts": 2.0, "state_only": True})
        self.assertEqual(r.json(), {"ok": True, "state": "idle"})
        self.assertEqual(agent_state.get_session("bot-1")["state"], "idle")
        self.assertEqual(self.pushes, [])

    def test_interactive_stop_is_idle_and_still_pushes_to_ha(self):
        # Turn end: the cube reads idle (no gold "ready" state), but the
        # usage/cost receipt push to HA is unchanged.
        r = self.post({"event": "stop", "machine": "mac", "project": "proj",
                       "session_id": "s1", "client_ts": 2.0})
        self.assertTrue(r.json()["ok"])
        self.assertEqual(agent_state.get_session("s1")["state"], "idle")
        self.assertEqual(len(self.pushes), 1)
        snap = agent_state.snapshot()
        self.assertEqual(snap["summary"]["idle"], 1)
        self.assertEqual(snap["mood"], "idle")

    def test_idle_event_demotes_ready_without_push(self):
        self.post({"event": "stop", "machine": "mac", "project": "proj",
                   "session_id": "s1", "client_ts": 2.0})
        self.pushes.clear()
        r = self.post({"event": "idle", "machine": "mac", "project": "proj",
                       "session_id": "s1", "client_ts": 3.0, "state_only": True})
        self.assertEqual(r.json(), {"ok": True, "state": "idle"})
        self.assertEqual(agent_state.get_session("s1")["state"], "idle")
        self.assertEqual(self.pushes, [])

    def test_heartbeat_working_is_state_only(self):
        r = self.post({"event": "working", "machine": "mac", "project": "proj",
                       "session_id": "s1", "client_ts": 5.0,
                       "state_only": True, "heartbeat": True})
        self.assertEqual(r.json(), {"ok": True, "state": "working"})
        self.assertEqual(agent_state.get_session("s1")["state"], "working")
        self.assertEqual(self.pushes, [])

    def test_subagent_events_track_fanout_without_ha_push(self):
        # session_id is the recovered PARENT id; two spawns then one stop.
        self.post({"event": "working", "machine": "mac", "project": "proj",
                   "session_id": "p1", "client_ts": 1.0})
        r = self.post({"event": "subagent_start", "machine": "mac", "project": "proj",
                       "session_id": "p1", "agent_id": "a1", "agent_type": "Explore",
                       "client_ts": 2.0, "state_only": True})
        self.assertEqual(r.json(), {"ok": True, "state": "subagent_start", "fanout": 1})
        self.post({"event": "subagent_start", "machine": "mac", "project": "proj",
                   "session_id": "p1", "agent_id": "a2", "client_ts": 3.0})
        r = self.post({"event": "subagent_stop", "machine": "mac", "project": "proj",
                       "session_id": "p1", "agent_id": "a1", "client_ts": 4.0})
        self.assertEqual(r.json()["fanout"], 1)
        snap = agent_state.snapshot()
        self.assertEqual(snap["sessions"]["p1"]["fanout"], 1)
        self.assertEqual(snap["total_fanout"], 1)
        self.assertEqual(self.pushes, [])   # fan-out is ambient, never a phone buzz

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
        # A legacy client without state_only now lands idle at turn end
        # (was ready); current codex-relay sends state_only and also idles.
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
