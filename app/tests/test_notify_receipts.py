"""Quiet-window stop receipts (S2): no premature "Response complete" pushes.

The Stop hook fires at the end of every Claude Code main-loop turn, including
turns that only hand off to a subagent or a background task (the harness
re-invokes and Stops again later). Pushing the HA receipt on every interactive
stop therefore spams the phone. The fix defers the receipt: an interactive stop
stores a pending receipt and schedules a flush RECEIPT_QUIET_S later, which
pushes only if the session has stayed quiet (no working signal, no newer stop)
and has no live subagent motes.

These tests drive the flush coroutine directly instead of sleeping the quiet
window, and stub out the real asyncio scheduling so no 25s task is ever
created.
"""
import asyncio
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from app import agent_state, notify
from app.main import app

TOKEN = "test-notify-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class QuietWindowReceiptTests(unittest.TestCase):
    """Drive the deferred-receipt policy through the real endpoint with the HA
    relay mocked and the quiet-window timer replaced by a hand-driven flush."""

    def setUp(self):
        agent_state.reset()
        notify._reset_pending()
        self._saved_token = notify._notify_token
        notify._notify_token = TOKEN
        self.pushes = []

        async def fake_relay(payload, devices=None):
            self.pushes.append(payload)
            return []

        self._relay_patcher = mock.patch.object(notify, "_relay_to_ha", fake_relay)
        self._relay_patcher.start()
        # Never spin a real 25s asyncio task in tests: the flush is driven by
        # hand via _flush() below.
        self._sched_patcher = mock.patch.object(
            notify, "_schedule_flush", lambda *a, **k: None)
        self._sched_patcher.start()
        self.client = TestClient(app)

    def tearDown(self):
        self._sched_patcher.stop()
        self._relay_patcher.stop()
        notify._notify_token = self._saved_token
        notify._reset_pending()
        agent_state.reset()

    def post(self, body):
        return self.client.post("/api/notify", json=body, headers=AUTH)

    def _flush(self, session_id):
        """Fire the pending receipt's flush with its current sequence."""
        pending = notify._pending_receipts.get(session_id)
        seq = pending["stop_seq"] if pending else -1
        asyncio.run(notify._flush_receipt(session_id, seq))

    # (a) an interactive stop pushes nothing immediately -------------------
    def test_interactive_stop_pushes_nothing_immediately(self):
        r = self.post({"event": "stop", "machine": "mac", "project": "proj",
                       "session_id": "s1", "client_ts": 2.0, "duration_s": 5})
        self.assertEqual(r.json()["state"], "idle")
        self.assertEqual(self.pushes, [])                 # no premature push
        self.assertIn("s1", notify._pending_receipts)     # receipt is pending
        # The idle state transition still happens immediately.
        self.assertEqual(agent_state.get_session("s1")["state"], "idle")

    # (b) quiet flush pushes exactly one receipt with the stop's payload ----
    def test_flush_after_quiet_window_pushes_one_receipt(self):
        self.post({"event": "stop", "machine": "mac", "project": "proj",
                   "session_id": "s1", "client_ts": 2.0, "duration_s": 42,
                   "tool_count": 3})
        self._flush("s1")
        self.assertEqual(len(self.pushes), 1)
        self.assertIn("Response complete", self.pushes[0]["title"])
        self.assertIn("42s, 3 tools", self.pushes[0]["message"])
        # A fired receipt is consumed: it can never push a second time.
        self.assertNotIn("s1", notify._pending_receipts)
        self._flush("s1")
        self.assertEqual(len(self.pushes), 1)

    # (c) a working signal between stop and flush cancels the receipt -------
    def test_tool_activity_between_stop_and_flush_cancels(self):
        self.post({"event": "stop", "machine": "mac", "project": "proj",
                   "session_id": "s1", "client_ts": 2.0, "duration_s": 5})
        self.post({"event": "tool_activity", "machine": "mac", "project": "proj",
                   "session_id": "s1", "count": 1, "last_tool": "Bash",
                   "client_ts": 3.0})
        self.assertNotIn("s1", notify._pending_receipts)  # cancelled at ingest
        self._flush("s1")
        self.assertEqual(self.pushes, [])

    def test_user_prompt_between_stop_and_flush_cancels(self):
        self.post({"event": "stop", "machine": "mac", "project": "proj",
                   "session_id": "s1", "client_ts": 2.0, "duration_s": 5})
        self.post({"event": "working", "machine": "mac", "project": "proj",
                   "session_id": "s1", "client_ts": 3.0})
        self.assertNotIn("s1", notify._pending_receipts)
        self._flush("s1")
        self.assertEqual(self.pushes, [])

    def test_subagent_start_between_stop_and_flush_cancels(self):
        self.post({"event": "stop", "machine": "mac", "project": "proj",
                   "session_id": "p1", "client_ts": 2.0, "duration_s": 5})
        self.post({"event": "subagent_start", "machine": "mac", "project": "proj",
                   "session_id": "p1", "agent_id": "a1", "client_ts": 3.0})
        self.assertNotIn("p1", notify._pending_receipts)
        self._flush("p1")
        self.assertEqual(self.pushes, [])

    # (d) a live subagent mote gates the flush; a later clean stop pushes ---
    def test_live_mote_gates_flush_then_clean_stop_pushes(self):
        # Parent with a live subagent, then it stops: this stop is a handoff,
        # not a real turn end, so the flush must drop even when quiet.
        self.post({"event": "working", "machine": "mac", "project": "proj",
                   "session_id": "p1", "client_ts": 1.0})
        self.post({"event": "subagent_start", "machine": "mac", "project": "proj",
                   "session_id": "p1", "agent_id": "a1", "client_ts": 2.0})
        self.post({"event": "stop", "machine": "mac", "project": "proj",
                   "session_id": "p1", "client_ts": 3.0, "duration_s": 5})
        self.assertTrue(agent_state.has_live_subagents("p1"))
        self._flush("p1")
        self.assertEqual(self.pushes, [])                 # gated by live mote
        # The child finishes (mote leaves live), then a NEW stop arrives: its
        # flush is quiet with no live mote, so it pushes.
        self.post({"event": "subagent_stop", "machine": "mac", "project": "proj",
                   "session_id": "p1", "agent_id": "a1", "client_ts": 4.0})
        self.assertFalse(agent_state.has_live_subagents("p1"))
        self.post({"event": "stop", "machine": "mac", "project": "proj",
                   "session_id": "p1", "client_ts": 5.0, "duration_s": 9})
        self._flush("p1")
        self.assertEqual(len(self.pushes), 1)
        self.assertIn("Response complete", self.pushes[0]["title"])

    # (e) two stops in a row: only the latest payload can push, exactly once -
    def test_two_stops_only_latest_pushes_once(self):
        self.post({"event": "stop", "machine": "mac", "project": "proj",
                   "session_id": "s1", "client_ts": 2.0, "duration_s": 5,
                   "tool_count": 1})
        seq1 = notify._pending_receipts["s1"]["stop_seq"]
        self.post({"event": "stop", "machine": "mac", "project": "proj",
                   "session_id": "s1", "client_ts": 3.0, "duration_s": 5,
                   "tool_count": 7})
        # The first receipt's timer lost: firing it does nothing.
        asyncio.run(notify._flush_receipt("s1", seq1))
        self.assertEqual(self.pushes, [])
        # The latest receipt fires exactly once with the latest payload.
        self._flush("s1")
        self.assertEqual(len(self.pushes), 1)
        self.assertIn("7 tools", self.pushes[0]["message"])

    # (f) a state_only stop is unaffected: no receipt is ever created -------
    def test_state_only_stop_creates_no_pending_receipt(self):
        r = self.post({"event": "stop", "machine": "mac", "project": "bot",
                       "session_id": "bot-1", "client_ts": 2.0,
                       "state_only": True})
        self.assertEqual(r.json(), {"ok": True, "state": "idle"})
        self.assertEqual(self.pushes, [])
        self.assertNotIn("bot-1", notify._pending_receipts)

    # scheduling glue: a real loop yields a cancellable task ----------------
    def test_schedule_flush_returns_cancellable_task_under_loop(self):
        self._sched_patcher.stop()          # exercise the real scheduler here
        try:
            async def make_and_cancel():
                task = notify._schedule_flush("s1", 1)
                self.assertIsNotNone(task)
                task.cancel()
                return task
            task = asyncio.run(make_and_cancel())
            self.assertTrue(task.cancelled())
        finally:
            self._sched_patcher.start()     # keep tearDown symmetric


if __name__ == "__main__":
    unittest.main()
