"""Per-tool-call SSE event stream (GET /api/agent-events) and its ingest hookup.

The push fan-out (agent_events.publish) and its ingest expansion in notify.py's
tool_activity branch are unit-tested here without driving a live streaming
connection: the SSE frame serializer is exercised directly, publish/queue
behavior is checked on real asyncio.Queue subscribers, and the ingest expansion
is observed by capturing publish() calls through the real /api/notify endpoint.

Auth (401) is checked on the non-streaming path (auth runs before the body), so
no test iterates an unbounded StreamingResponse (which would deadlock the pinned
TestClient exactly as test_stream.py documents).
"""
import asyncio
import json
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from app import agent_events, notify
from app.main import app

TOKEN = "test-notify-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class PublishTests(unittest.TestCase):
    """Direct publish/queue behavior: no endpoint, no DB."""

    def setUp(self):
        agent_events.reset()

    def tearDown(self):
        agent_events.reset()

    def test_publish_with_no_subscribers_is_noop(self):
        # Must not raise with an empty subscriber set.
        agent_events.publish("s1", "Read")

    def test_subscriber_receives_events_in_order(self):
        q = asyncio.Queue(maxsize=64)
        agent_events._subscribers.add(q)
        agent_events.publish("s1", "Read")
        agent_events.publish("s1", "Bash")
        first = q.get_nowait()
        second = q.get_nowait()
        self.assertEqual(
            first, {"type": "tool_call", "session_id": "s1", "tool": "Read"})
        self.assertEqual(
            second, {"type": "tool_call", "session_id": "s1", "tool": "Bash"})

    def test_queue_full_drops_for_that_subscriber_only(self):
        full = asyncio.Queue(maxsize=1)
        full.put_nowait({"type": "sentinel"})          # now at capacity
        ok = asyncio.Queue(maxsize=64)
        agent_events._subscribers.add(full)
        agent_events._subscribers.add(ok)
        agent_events.publish("s1", "Read")
        # The full queue silently dropped the event (still only the sentinel).
        self.assertEqual(full.qsize(), 1)
        self.assertEqual(full.get_nowait(), {"type": "sentinel"})
        # The other subscriber still received it.
        self.assertEqual(
            ok.get_nowait(),
            {"type": "tool_call", "session_id": "s1", "tool": "Read"})


class FrameSerializerTests(unittest.TestCase):
    """The SSE frame serializer directly: hello and tool_call frames."""

    def test_hello_frame_ends_blank_line_and_valid_json(self):
        frame = agent_events._frame({"type": "hello"})
        self.assertTrue(frame.startswith("data: "))
        self.assertTrue(frame.endswith("\n\n"))
        payload = json.loads(frame[len("data: "):].strip())
        self.assertEqual(payload, {"type": "hello"})

    def test_tool_call_frame_ends_blank_line_and_valid_json(self):
        event = {"type": "tool_call", "session_id": "s1", "tool": "Read"}
        frame = agent_events._frame(event)
        self.assertTrue(frame.startswith("data: "))
        self.assertTrue(frame.endswith("\n\n"))
        self.assertEqual(json.loads(frame[len("data: "):].strip()), event)


class RouteAuthTests(unittest.TestCase):
    def setUp(self):
        self._saved_token = notify._notify_token
        notify._notify_token = TOKEN
        self.client = TestClient(app)

    def tearDown(self):
        notify._notify_token = self._saved_token

    def test_no_bearer_returns_401(self):
        r = self.client.get("/api/agent-events")
        self.assertEqual(r.status_code, 401)

    def test_wrong_bearer_returns_401(self):
        r = self.client.get("/api/agent-events",
                            headers={"Authorization": "Bearer wrong"})
        self.assertEqual(r.status_code, 401)


class IngestExpansionTests(unittest.TestCase):
    """The tool_activity branch of /api/notify fans a batch out to per-tool
    events, is state-only (no HA push), and returns the same body as before."""

    def setUp(self):
        agent_events.reset()
        self._saved_token = notify._notify_token
        notify._notify_token = TOKEN
        self.published = []

        def fake_publish(session_id, tool):
            self.published.append((session_id, tool))

        self._publish_patch = mock.patch.object(
            agent_events, "publish", fake_publish)
        self._publish_patch.start()
        self._relay = mock.AsyncMock(return_value=[])
        self._relay_patch = mock.patch.object(notify, "_relay_to_ha", self._relay)
        self._relay_patch.start()
        self.client = TestClient(app)

    def tearDown(self):
        self._relay_patch.stop()
        self._publish_patch.stop()
        notify._notify_token = self._saved_token
        agent_events.reset()

    def post(self, body):
        return self.client.post("/api/notify", json=body, headers=AUTH)

    def test_tally_expands_in_insertion_order(self):
        r = self.post({"event": "tool_activity", "machine": "mac",
                       "project": "proj", "session_id": "s1",
                       "count": 3, "tools": {"Read": 2, "Bash": 1}})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(
            self.published,
            [("s1", "Read"), ("s1", "Read"), ("s1", "Bash")])

    def test_missing_tools_publishes_count_of_last_tool(self):
        self.post({"event": "tool_activity", "machine": "mac",
                   "project": "proj", "session_id": "s1",
                   "count": 4, "last_tool": "Grep"})
        self.assertEqual(self.published, [("s1", "Grep")] * 4)

    def test_count_missing_publishes_one(self):
        self.post({"event": "tool_activity", "machine": "mac",
                   "project": "proj", "session_id": "s1",
                   "last_tool": "Edit"})
        self.assertEqual(self.published, [("s1", "Edit")])

    def test_thirty_tool_tally_caps_at_eight(self):
        tools = {"T%02d" % i: 1 for i in range(30)}
        self.post({"event": "tool_activity", "machine": "mac",
                   "project": "proj", "session_id": "s1",
                   "count": 30, "tools": tools})
        self.assertEqual(len(self.published), 8)
        # The first 8 in tally order.
        self.assertEqual([t for _, t in self.published],
                         ["T%02d" % i for i in range(8)])

    def test_response_body_unchanged_and_no_ha_call(self):
        r = self.post({"event": "tool_activity", "machine": "mac",
                       "project": "proj", "session_id": "s1",
                       "count": 1, "tools": {"Read": 1}})
        self.assertEqual(r.json(), {"ok": True, "state": "tool_activity"})
        self.assertEqual(self._relay.await_count, 0)


if __name__ == "__main__":
    unittest.main()
