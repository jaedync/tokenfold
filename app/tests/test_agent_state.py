"""Agent-state store + /api/notify notification policy.

Covers the 2026-07-18 consolidation: working events are state-only (the
push-spam regression this feature fixes), waiting pushes dedup per spell,
presence damping, aggregation, TTL decay, and the authed read endpoint.
"""
import asyncio
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
        # goes stale and waiting_subagent could never fire). The heartbeat gap
        # is within the subagent TTL (relay heartbeats every ~60s): with
        # sunsetting, a fully-gone mote (past TTL + sunset) is purged, so a
        # refresh past that window would be a genuine new spawn, not a refresh.
        agent_state.update("s1", "mac", "proj", "working", now=1000.0)
        agent_state.add_subagent("s1", "a1", now=1000.0)
        agent_state.add_subagent("s1", "a1", now=1120.0)   # heartbeat refresh
        self.assertEqual(agent_state.get_session("s1")["working_ts"], 1000.0)
        self.assertEqual(agent_state.snapshot(now=1121.0)["sessions"]["s1"]["fanout"], 1)

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
        # Mote kept alive by heartbeat: new record shape ({spawn_ts, ts, model,
        # stop_ts}); ts is what _active_subagents reads for freshness.
        s["subagents"] = {"a1": {"spawn_ts": 1000.0, "ts": 1099.0,
                                 "model": "", "stop_ts": None}}
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


# --- Model + color-family surfacing -------------------------------------------
# Standalone pytest functions (the file's store tests are unittest methods, but
# these mirror the Task 2 brief 1:1 and use the module-level agent_state import).
def test_family_classification():
    from app.agent_state import _family
    assert _family("claude-opus-4-8") == "opus"
    assert _family("claude-sonnet-5") == "sonnet"
    assert _family("claude-fable-5") == "fable"
    assert _family("claude-haiku-4-5-20251001") == "haiku"
    assert _family("gpt-5.6-sol") == "codex"
    assert _family("") == "unknown"
    assert _family("something-else") == "unknown"


def test_session_model_surfaces_in_agents():
    agent_state.reset()
    agent_state.update("s1", "mac", "proj", "working", now=100.0, model="claude-opus-4-8")
    snap = agent_state.snapshot(now=100.0)
    dot = [a for a in snap["agents"] if a["id"] == "s1"][0]
    assert dot["model"] == "claude-opus-4-8"
    assert dot["family"] == "opus"


def test_mote_model_and_age_from_spawn():
    agent_state.reset()
    agent_state.add_subagent("s1", "a1", model="claude-fable-5", now=100.0)
    # A later working heartbeat must NOT reset the mote's spawn age.
    agent_state.update("s1", "mac", "proj", "working", now=104.0, model="claude-opus-4-8")
    snap = agent_state.snapshot(now=105.0)
    mote = [a for a in snap["agents"] if a["kind"] == "subagent"][0]
    assert mote["family"] == "fable"          # mote keeps its own model
    assert mote["age_s"] == 5.0               # since spawn, not since heartbeat


# --- Sunsetting lifecycle (Task 3) --------------------------------------------
def test_stopped_mote_stays_visible_min_window():
    agent_state.reset()
    agent_state.add_subagent("s1", "a1", model="claude-opus-4-8", now=100.0)
    agent_state.remove_subagent("s1", "a1", now=101.0)   # stop 1s after spawn
    snap = agent_state.snapshot(now=101.5)               # 1.5s after spawn
    mote = [a for a in snap["agents"] if a["kind"] == "subagent"]
    assert mote and mote[0]["state"] == "sunsetting"
    assert mote[0]["stop_age_s"] == 0.5
    # gone only after both min-visible (3.0) and sunset (0.8) windows
    later = agent_state.snapshot(now=104.5)
    assert not [a for a in later["agents"] if a["kind"] == "subagent"]


def test_sunsetting_mote_excluded_from_fanout_and_mood():
    agent_state.reset()
    agent_state.add_subagent("s1", "a1", now=100.0)
    agent_state.remove_subagent("s1", "a1", now=100.2)
    # The parent's turn ended right after the sub returned: nobody is grinding
    # now, only a sunsetting mote lingers. (add_subagent marks the parent
    # working, so we must land it idle to observe the "nobody grinding" case;
    # a sunsetting mote must drive neither fan-out nor mood.)
    agent_state.update("s1", "mac", "proj", "idle", now=100.3)
    snap = agent_state.snapshot(now=100.5)
    assert snap["total_fanout"] == 0        # sunsetting is not live fan-out
    # only a sunsetting mote: mood is idle, not waiting_subagent
    assert snap["mood"] == "idle"


def test_session_gone_sunsets_then_drops():
    agent_state.reset()
    agent_state.update("s1", "mac", "proj", "working", now=100.0, model="claude-opus-4-8")
    agent_state.sunset_session("s1", now=101.0)
    snap = agent_state.snapshot(now=101.4)
    dot = [a for a in snap["agents"] if a["id"] == "s1"]
    assert dot and dot[0]["state"] == "sunsetting" and dot[0]["stop_age_s"] == 0.4
    assert snap["mood"] == "idle"           # a leaving session drives no mood
    assert not agent_state.snapshot(now=102.5)["agents"]   # dropped after window


def test_respawn_while_sunsetting_resets_the_mote():
    # A genuine re-spawn of an agent_id that is still sunsetting must come back
    # live, not be treated as a heartbeat that keeps the stale stop_ts and the
    # original spawn_ts. Keep-alive heartbeats flow through update(), so an
    # add_subagent on a stopped id can only be a real SubagentStart.
    agent_state.reset()
    agent_state.add_subagent("s1", "a1", model="claude-opus-4-8", now=100.0)
    agent_state.remove_subagent("s1", "a1", now=100.2)     # now sunsetting
    # Re-spawn inside the min-visible window (still drawable, so still in subs).
    agent_state.add_subagent("s1", "a1", now=100.5)
    snap = agent_state.snapshot(now=100.6)
    mote = [a for a in snap["agents"] if a["kind"] == "subagent"]
    assert mote, "the re-spawned mote should be drawn"
    assert mote[0]["state"] == "working"           # live again, not fading
    assert "stop_age_s" not in mote[0]             # no stale sunset stamp
    assert snap["total_fanout"] == 1               # counted as live fan-out
    # age is measured from the RESPAWN (spawn_ts reset to 100.5), not 100.0.
    assert mote[0]["age_s"] == 0.1


# --- Full-resolution session fields (v2.2): tool activity, trouble, compaction -
# Additive session-dot fields the cube overlays. Standalone pytest functions in
# the file's newer style, with fixed-clock injection.
def _dot(snap, sid):
    return [a for a in snap["agents"] if a["id"] == sid][0]


def test_session_start_creates_idle_session_with_family():
    agent_state.reset()
    agent_state.session_start("s1", "mac", "proj", model="claude-opus-4-8", now=100.0)
    dot = _dot(agent_state.snapshot(now=100.0), "s1")
    assert dot["state"] == "idle"           # SessionStart is not work
    assert dot["family"] == "opus"          # family resolved from the model
    # A SessionStart never marks the session working.
    assert "working_ts" not in agent_state.get_session("s1")


def test_second_session_start_refreshes_model_only():
    agent_state.reset()
    agent_state.session_start("s1", "mac", "proj", now=100.0)   # model unknown at startup
    agent_state.session_start("s1", "mac", "proj", model="claude-fable-5", now=105.0)
    s = agent_state.get_session("s1")
    assert s["model"] == "claude-fable-5"    # newly-resolved model backfilled
    assert s["state"] == "idle"              # still idle, not bumped to working
    assert "working_ts" not in s             # never bumps the working clock


def test_tool_activity_accumulates_count_and_last_tool():
    agent_state.reset()
    agent_state.tool_activity("s1", "mac", "proj", count=1, last_tool="Bash",
                              model="claude-opus-4-8", now=100.0)
    s = agent_state.get_session("s1")
    assert s["tool_count"] == 1
    assert s["last_tool"] == "Bash"
    assert s["working_ts"] == 100.0          # tool progress is the working heartbeat
    assert s["state"] == "working"
    agent_state.tool_activity("s1", "mac", "proj", count=2, last_tool="Read", now=110.0)
    s = agent_state.get_session("s1")
    assert s["tool_count"] == 3              # cumulative
    assert s["last_tool"] == "Read"
    assert s["working_ts"] == 110.0          # bumped again


def test_tool_activity_clears_earlier_trouble():
    agent_state.reset()
    agent_state.tool_activity("s1", "mac", "proj", count=1, last_tool="Bash", now=100.0)
    agent_state.tool_trouble("s1", now=101.0)
    assert _dot(agent_state.snapshot(now=102.0), "s1")["trouble"] is True
    agent_state.tool_activity("s1", "mac", "proj", count=1, last_tool="Read", now=103.0)
    assert _dot(agent_state.snapshot(now=104.0), "s1")["trouble"] is False


def test_tool_trouble_sets_then_self_clears_after_ttl():
    agent_state.reset()
    agent_state.tool_activity("s1", "mac", "proj", now=100.0)
    agent_state.tool_trouble("s1", now=101.0)
    assert _dot(agent_state.snapshot(now=101.0), "s1")["trouble"] is True
    # still troubled just before TROUBLE_TTL_S (45)
    assert _dot(agent_state.snapshot(now=101.0 + 44.0), "s1")["trouble"] is True
    # overlay self-clears once the TTL elapses (no clearing event needed)
    assert _dot(agent_state.snapshot(now=101.0 + 46.0), "s1")["trouble"] is False


def test_trouble_cleared_by_prompt_event():
    agent_state.reset()
    agent_state.tool_activity("s1", "mac", "proj", now=100.0)
    agent_state.tool_trouble("s1", now=101.0)
    assert _dot(agent_state.snapshot(now=102.0), "s1")["trouble"] is True
    agent_state.update("s1", "mac", "proj", "working", now=103.0)   # a fresh prompt
    assert _dot(agent_state.snapshot(now=104.0), "s1")["trouble"] is False


def test_stop_failure_marks_trouble_without_changing_state():
    agent_state.reset()
    agent_state.update("s1", "mac", "proj", "idle", now=100.0)
    agent_state.stop_failure("s1", now=101.0)
    assert agent_state.get_session("s1")["state"] == "idle"   # state unchanged
    assert _dot(agent_state.snapshot(now=101.0), "s1")["trouble"] is True


def test_compact_start_and_end_toggle_compacting():
    agent_state.reset()
    agent_state.tool_activity("s1", "mac", "proj", now=100.0)
    agent_state.compact_start("s1", now=101.0)
    assert _dot(agent_state.snapshot(now=102.0), "s1")["compacting"] is True
    assert agent_state.get_session("s1")["state"] == "working"   # compaction is an overlay
    agent_state.compact_end("s1", now=103.0)
    assert _dot(agent_state.snapshot(now=104.0), "s1")["compacting"] is False


def test_compacting_self_clears_without_compact_end():
    agent_state.reset()
    agent_state.tool_activity("s1", "mac", "proj", now=100.0)
    agent_state.compact_start("s1", now=101.0)
    # still compacting inside COMPACT_TTL_S (300)
    assert _dot(agent_state.snapshot(now=101.0 + 299.0), "s1")["compacting"] is True
    # fallback: overlay clears COMPACT_TTL_S after compact_start with no end
    assert _dot(agent_state.snapshot(now=101.0 + 301.0), "s1")["compacting"] is False


def test_plain_session_reports_new_field_defaults():
    agent_state.reset()
    agent_state.update("s1", "mac", "proj", "working", now=100.0)
    dot = _dot(agent_state.snapshot(now=100.0), "s1")
    assert dot["tool_count"] == 0
    assert dot["last_tool"] is None
    assert dot["trouble"] is False
    assert dot["compacting"] is False


def test_motes_never_carry_the_new_session_fields():
    agent_state.reset()
    agent_state.add_subagent("s1", "a1", model="claude-fable-5", now=100.0)
    agent_state.tool_activity("s1", "mac", "proj", count=1, last_tool="Bash", now=100.0)
    agent_state.compact_start("s1", now=100.0)
    snap = agent_state.snapshot(now=100.0)
    mote = [a for a in snap["agents"] if a["kind"] == "subagent"][0]
    for k in ("tool_count", "last_tool", "trouble", "compacting"):
        assert k not in mote                 # motes carry none of them
    for k in ("tool_count", "last_tool", "trouble", "compacting"):
        assert k in _dot(snap, "s1")         # sessions carry all four


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
        # Interactive stops now defer their receipt behind a quiet-window flush.
        # Stub the real 25s asyncio task so these policy tests never orphan one;
        # _flush drives the flush by hand where a push is asserted.
        self._sched_patcher = mock.patch.object(
            notify_mod, "_schedule_flush", lambda *a, **k: None)
        self._sched_patcher.start()
        notify_mod._reset_pending()
        self.client = TestClient(app)

    def tearDown(self):
        self._sched_patcher.stop()
        self._patcher.stop()
        self._notify_mod._reset_pending()
        self._notify_mod._notify_token = self._saved_token
        agent_state.reset()

    def post(self, body):
        return self.client.post("/api/notify", json=body, headers=AUTH)

    def _flush(self, session_id):
        pending = self._notify_mod._pending_receipts.get(session_id)
        seq = pending["stop_seq"] if pending else -1
        asyncio.run(self._notify_mod._flush_receipt(session_id, seq))

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

    def test_interactive_stop_is_idle_and_defers_ha_receipt(self):
        # Turn end: the cube reads idle immediately, but the usage/cost receipt
        # is held for the quiet window (no premature "Response complete").
        r = self.post({"event": "stop", "machine": "mac", "project": "proj",
                       "session_id": "s1", "client_ts": 2.0})
        self.assertTrue(r.json()["ok"])
        self.assertEqual(agent_state.get_session("s1")["state"], "idle")
        self.assertEqual(self.pushes, [])                 # deferred, not immediate
        self.assertIn("s1", self._notify_mod._pending_receipts)
        snap = agent_state.snapshot()
        self.assertEqual(snap["summary"]["idle"], 1)
        self.assertEqual(snap["mood"], "idle")
        # The quiet-window flush is what finally delivers the receipt.
        self._flush("s1")
        self.assertEqual(len(self.pushes), 1)

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

    def test_session_start_event_is_state_only_idle(self):
        r = self.post({"event": "session_start", "project": "p", "machine": "mac",
                       "session_id": "s1", "model": "claude-opus-4-8", "client_ts": 1.0})
        self.assertEqual(r.json(), {"ok": True, "state": "session_start"})
        self.assertEqual(self.pushes, [])
        self.assertEqual(agent_state.get_session("s1")["state"], "idle")

    def test_tool_activity_event_accumulates_state_only(self):
        r = self.post({"event": "tool_activity", "project": "p", "machine": "mac",
                       "session_id": "s1", "count": 2, "last_tool": "Bash",
                       "tools": {"Bash": 2}, "model": "claude-opus-4-8",
                       "client_ts": 1.0})
        self.assertEqual(r.json(), {"ok": True, "state": "tool_activity"})
        self.assertEqual(self.pushes, [])
        s = agent_state.get_session("s1")
        self.assertEqual(s["tool_count"], 2)
        self.assertEqual(s["last_tool"], "Bash")
        self.assertEqual(s["state"], "working")

    def test_trouble_and_compaction_events_are_state_only(self):
        self.post({"event": "tool_activity", "project": "p", "machine": "mac",
                   "session_id": "s1", "count": 1, "last_tool": "Bash", "client_ts": 1.0})
        r = self.post({"event": "tool_trouble", "project": "p", "machine": "mac",
                       "session_id": "s1", "tool_name": "Bash", "client_ts": 2.0})
        self.assertEqual(r.json(), {"ok": True, "state": "tool_trouble"})
        r = self.post({"event": "stop_failure", "project": "p", "machine": "mac",
                       "session_id": "s1", "reason": "boom", "client_ts": 3.0})
        self.assertEqual(r.json(), {"ok": True, "state": "stop_failure"})
        r = self.post({"event": "compact_start", "project": "p", "machine": "mac",
                       "session_id": "s1", "client_ts": 4.0})
        self.assertEqual(r.json(), {"ok": True, "state": "compact_start"})
        r = self.post({"event": "compact_end", "project": "p", "machine": "mac",
                       "session_id": "s1", "client_ts": 5.0})
        self.assertEqual(r.json(), {"ok": True, "state": "compact_end"})
        self.assertEqual(self.pushes, [])   # overlays are ambient, never a push
        # trouble/compaction never change the underlying state
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

    def test_stop_defers_then_flushes_response_complete(self):
        r = self.post({"event": "stop", "project": "p", "machine": "mac",
                       "session_id": "s1", "duration_s": 42, "tool_count": 3})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.pushes, [])                 # held for the quiet window
        self.assertEqual(agent_state.get_session("s1")["state"], "idle")
        self._flush("s1")
        self.assertEqual(len(self.pushes), 1)
        self.assertIn("Response complete", self.pushes[0]["title"])

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


class ReconcileHandshakeTests(unittest.TestCase):
    """v2.5 live_subagents snapshot handshake. Claude Code's hook dispatcher
    lags minutes and silently sheds queued lifecycle hooks under fan-out load
    (measured 2026-07-26: a SubagentStart ran 103s after its spawn; a
    SubagentStop was discarded at turn end and left a zombie mote). Every
    relay post may therefore carry a `live_subagents` ground-truth snapshot;
    reconcile_subagents heals the mote table from whichever post survives."""

    def setUp(self):
        agent_state.reset()

    def test_reconcile_creates_missing_motes_with_client_spawn_ts(self):
        agent_state.update("p1", "mac", "proj", "working", now=1000.0)
        n = agent_state.reconcile_subagents("p1", [
            {"agent_id": "kid1", "model": "opus", "spawn_ts": 940.0},
            {"agent_id": "kid2", "model": "sonnet", "spawn_ts": 950.0},
        ], now=1000.0)
        self.assertEqual(n, 2)
        snap = agent_state.snapshot(now=1001.0)
        self.assertEqual(snap["sessions"]["p1"]["fanout"], 2)
        motes = {a["id"]: a for a in snap["agents"] if a["kind"] == "subagent"}
        self.assertEqual(motes["p1:kid1"]["model"], "opus")
        # Displayed age runs from the CLIENT's true spawn moment, not from
        # whenever the snapshot finally got through.
        self.assertAlmostEqual(motes["p1:kid1"]["age_s"], 61.0, places=1)

    def test_reconcile_refreshes_live_mote_past_ttl(self):
        agent_state.update("p1", "mac", "proj", "working", now=1000.0)
        agent_state.add_subagent("p1", "kidA", now=1000.0)
        # Without a refresh the mote would TTL out at 1000+180; the snapshot
        # heartbeat at 1150 must carry it through to 1250.
        agent_state.reconcile_subagents(
            "p1", [{"agent_id": "kidA"}], now=1150.0)
        snap = agent_state.snapshot(now=1250.0)
        self.assertEqual(snap["sessions"].get("p1", {}).get("fanout"), 1)

    def test_reconcile_backfills_model_only_when_empty(self):
        agent_state.update("p1", "mac", "proj", "working", now=1000.0)
        agent_state.add_subagent("p1", "kidA", now=1000.0)          # no model
        agent_state.add_subagent("p1", "kidB", now=1000.0, model="opus")
        agent_state.reconcile_subagents("p1", [
            {"agent_id": "kidA", "model": "sonnet"},
            {"agent_id": "kidB", "model": "haiku"},
        ], now=1001.0)
        snap = agent_state.snapshot(now=1002.0)
        motes = {a["id"]: a for a in snap["agents"] if a["kind"] == "subagent"}
        self.assertEqual(motes["p1:kidA"]["model"], "sonnet")   # backfilled
        self.assertEqual(motes["p1:kidB"]["model"], "opus")     # kept

    def test_reconcile_never_revives_stopped_mote(self):
        agent_state.update("p1", "mac", "proj", "working", now=1000.0)
        agent_state.add_subagent("p1", "kidA", now=1000.0)
        agent_state.remove_subagent("p1", "kidA", now=1010.0)
        # The agent's transcript mtime stays fresh for a while after it
        # finishes; the snapshot may still list it. Stopped is authoritative.
        n = agent_state.reconcile_subagents(
            "p1", [{"agent_id": "kidA"}], now=1010.4)
        self.assertEqual(n, 0)
        snap = agent_state.snapshot(now=1010.5)
        motes = [a for a in snap["agents"] if a["kind"] == "subagent"]
        self.assertEqual(len(motes), 1)
        self.assertEqual(motes[0]["state"], "sunsetting")

    def test_tombstone_blocks_recreate_after_mote_is_pruned(self):
        agent_state.update("p1", "mac", "proj", "working", now=1000.0)
        agent_state.add_subagent("p1", "kidA", now=1000.0)
        agent_state.remove_subagent("p1", "kidA", now=1010.0)
        # Past min-visible + sunset: the mote record itself is pruned...
        snap = agent_state.snapshot(now=1020.0)
        self.assertEqual([a for a in snap["agents"] if a["kind"] == "subagent"],
                         [])
        # ...but a snapshot that still lists it (mtime lag) must not
        # resurrect it as a fresh live mote.
        n = agent_state.reconcile_subagents(
            "p1", [{"agent_id": "kidA"}], now=1030.0)
        self.assertEqual(n, 0)
        snap = agent_state.snapshot(now=1031.0)
        self.assertEqual(snap["sessions"]["p1"]["fanout"], 0)

    def test_stale_start_after_stop_is_suppressed_by_client_ts(self):
        """Out-of-order hook drain: the stop ran, then the delayed start
        finally drains from Claude Code's queue. Its client_ts predates the
        stop, so it is stale news and must not create a zombie mote."""
        agent_state.update("p1", "mac", "proj", "working", now=1000.0)
        agent_state.remove_subagent("p1", "kidA", now=1010.0)
        n = agent_state.add_subagent("p1", "kidA", now=1011.0, event_ts=1005.0)
        self.assertEqual(n, 0)
        snap = agent_state.snapshot(now=1012.0)
        self.assertEqual(snap["sessions"]["p1"]["fanout"], 0)

    def test_genuine_respawn_after_stop_wins_by_client_ts(self):
        agent_state.update("p1", "mac", "proj", "working", now=1000.0)
        agent_state.add_subagent("p1", "kidA", now=1000.0, event_ts=1000.0)
        agent_state.remove_subagent("p1", "kidA", now=1010.0)
        n = agent_state.add_subagent("p1", "kidA", now=1020.0, event_ts=1019.0)
        self.assertEqual(n, 1)
        snap = agent_state.snapshot(now=1021.0)
        self.assertEqual(snap["sessions"]["p1"]["fanout"], 1)

    def test_keep_alive_gated_by_refresh_subagents(self):
        """A working event that carried its own snapshot must NOT blanket-
        refresh every mote: that blanket refresh is what kept zombie motes
        alive forever while the parent stayed busy. Legacy events (no
        snapshot) keep the old keep-alive."""
        agent_state.update("p1", "mac", "proj", "working", now=1000.0)
        agent_state.add_subagent("p1", "zombie", now=1000.0)
        agent_state.update("p1", "mac", "proj", "working", now=1150.0,
                           refresh_subagents=False)
        snap = agent_state.snapshot(now=1250.0)
        self.assertEqual(snap["sessions"]["p1"]["fanout"], 0)
        # Legacy path (default True) still refreshes: the mote survives.
        agent_state.reset()
        agent_state.update("p1", "mac", "proj", "working", now=1000.0)
        agent_state.add_subagent("p1", "kid", now=1000.0)
        agent_state.update("p1", "mac", "proj", "working", now=1150.0)
        snap = agent_state.snapshot(now=1250.0)
        self.assertEqual(snap["sessions"]["p1"]["fanout"], 1)

    def test_reconcile_creates_parent_when_unknown(self):
        n = agent_state.reconcile_subagents(
            "ghost", [{"agent_id": "kid1", "spawn_ts": 990.0}], now=1000.0)
        self.assertEqual(n, 1)
        snap = agent_state.snapshot(now=1001.0)
        self.assertEqual(snap["sessions"]["ghost"]["state"], "working")

    def test_reconcile_ignores_garbage(self):
        agent_state.update("p1", "mac", "proj", "working", now=1000.0)
        n = agent_state.reconcile_subagents("p1", [
            "not-a-dict", {}, {"agent_id": ""}, {"agent_id": None},
            {"agent_id": "ok1", "spawn_ts": "bogus"},
            {"agent_id": "ok2", "spawn_ts": 99999999999.0},   # future clock
        ], now=1000.0)
        self.assertEqual(n, 2)
        snap = agent_state.snapshot(now=1001.0)
        motes = {a["id"]: a for a in snap["agents"] if a["kind"] == "subagent"}
        # Bad spawn_ts falls back to now; future spawn_ts clamps to now.
        self.assertAlmostEqual(motes["p1:ok1"]["age_s"], 1.0, places=1)
        self.assertAlmostEqual(motes["p1:ok2"]["age_s"], 1.0, places=1)
        n2 = agent_state.reconcile_subagents("p1", "not-a-list", now=1002.0)
        self.assertEqual(n2, 2)


class ReconcileNotifyTests(NotifyPolicyTests):
    """The handshake through the real endpoint (harness from NotifyPolicyTests:
    real /api/notify, HA relay mocked, receipts stubbed)."""

    def test_notify_payload_with_snapshot_reconciles(self):
        r = self.post({
            "event": "tool_activity", "machine": "mac", "project": "proj",
            "session_id": "api1", "count": 1, "last_tool": "Bash",
            "client_ts": 1.0, "state_only": True,
            "live_subagents": [
                {"agent_id": "kidX", "model": "opus", "spawn_ts": 1.0}]})
        self.assertEqual(r.status_code, 200)
        snap = agent_state.snapshot()
        self.assertEqual(snap["sessions"]["api1"]["fanout"], 1)
        motes = [a for a in snap["agents"] if a["kind"] == "subagent"]
        self.assertEqual(motes[0]["model"], "opus")
        self.assertEqual(self.pushes, [])       # ambient, never a phone buzz

    def test_notify_tool_activity_agent_id_heartbeats_mote(self):
        r = self.post({
            "event": "tool_activity", "machine": "mac", "project": "proj",
            "session_id": "api2", "count": 1, "last_tool": "Bash",
            "client_ts": 1.0, "state_only": True, "agent_id": "kidY"})
        self.assertEqual(r.status_code, 200)
        snap = agent_state.snapshot()
        self.assertEqual(snap["sessions"]["api2"]["fanout"], 1)
        self.assertEqual(self.pushes, [])

    def test_notify_empty_snapshot_does_not_blanket_refresh(self):
        """A tool_activity carrying live_subagents [] must not keep-alive
        stale motes: [] is the authoritative all-done statement, and the
        gated keep-alive is what lets a dropped-stop zombie finally age out."""
        self.post({"event": "subagent_start", "machine": "mac",
                   "project": "proj", "session_id": "api3",
                   "agent_id": "zomb", "client_ts": 1.0, "state_only": True})
        r = self.post({
            "event": "tool_activity", "machine": "mac", "project": "proj",
            "session_id": "api3", "count": 1, "last_tool": "Bash",
            "client_ts": 2.0, "state_only": True, "live_subagents": []})
        self.assertEqual(r.status_code, 200)
        s = agent_state.get_session("api3")
        spawn = s["subagents"]["zomb"]["spawn_ts"]
        # The mote's last-seen was NOT bumped past its spawn heartbeat: with
        # no further snapshots listing it, the TTL fade owns it from here.
        self.assertEqual(s["subagents"]["zomb"]["ts"], spawn)
