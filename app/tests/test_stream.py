"""SSE version-stream endpoint tests (GET /api/stats/stream).

TestClient (this pinned httpx/starlette) deadlocks on a never-ending
StreamingResponse: `client.stream(...)` buffers the whole body before returning,
so an infinite SSE generator hangs the suite. We therefore drive the async
generator `_version_events` directly on an event loop with a STUB request whose
`is_disconnected()` we control, and impose a HARD CAP on frames pulled so a
regression that stops emitting can never hang. POLL_S / KEEPALIVE_S are patched
down to milliseconds. Auth (401) and the response headers are checked on the
non-streaming path (auth runs before the body; headers are set on the
StreamingResponse object), which needs no body iteration.
"""

import asyncio
import json
import unittest
from unittest.mock import patch

import app.aggregator as agg
import app.config
import app.stream as stream
from app.tests._support import TempDBTestCase


class _StubRequest:
    """Minimal Request stand-in for the generator: only is_disconnected() is
    used. `disconnect_after` frames of asyncio.sleep, it reports disconnected so
    a test can prove the clean-exit path without hanging."""

    def __init__(self, disconnect_after=None):
        self._disconnect_after = disconnect_after
        self._checks = 0

    async def is_disconnected(self):
        self._checks += 1
        if self._disconnect_after is not None and self._checks >= self._disconnect_after:
            return True
        return False


async def _collect(request, max_frames, timeout=5.0):
    """Pull up to max_frames from the generator with a wall-clock timeout so a
    stuck generator fails fast instead of hanging the suite. Returns the frames
    and always aclose()s the generator (exercising the cancellation path)."""
    gen = stream._version_events(request)
    frames = []

    async def _pull():
        async for frame in gen:
            frames.append(frame)
            if len(frames) >= max_frames:
                return

    try:
        await asyncio.wait_for(_pull(), timeout=timeout)
    except asyncio.TimeoutError:
        pass
    finally:
        await gen.aclose()
    return frames


def _data_versions(frames):
    """Extract the version ints from `data:` frames, ignoring keepalives."""
    out = []
    for f in frames:
        if f.startswith("data:"):
            out.append(json.loads(f[len("data:"):].strip())["version"])
    return out


class StreamAuthTest(TempDBTestCase):
    def test_unauthenticated_returns_401(self):
        with patch.object(app.config, "DASHBOARD_PASSWORD", "secret"):
            c = self.client()
            r = c.get("/api/stats/stream")
        self.assertEqual(r.status_code, 401)


class StreamFirstEventTest(TempDBTestCase):
    def test_first_event_carries_current_version(self):
        async def run():
            with patch.object(stream, "POLL_S", 0.001), \
                 patch.object(stream, "KEEPALIVE_S", 100):
                return await _collect(_StubRequest(), max_frames=1)

        frames = asyncio.run(run())
        self.assertEqual(_data_versions(frames), [agg.get_cache_version()])


class StreamChangeDetectionTest(TempDBTestCase):
    def test_version_bump_emits_new_event(self):
        async def run():
            with patch.object(stream, "POLL_S", 0.001), \
                 patch.object(stream, "KEEPALIVE_S", 100):
                gen = stream._version_events(_StubRequest())
                first = await gen.__anext__()          # immediate frame
                before = json.loads(first[len("data:"):].strip())["version"]
                agg.trigger_eager_rebuild()            # bump the version
                # next data frame must reflect the new (higher) version; bounded
                nxt = await asyncio.wait_for(gen.__anext__(), timeout=5.0)
                after = json.loads(nxt[len("data:"):].strip())["version"]
                await gen.aclose()
                return before, after

        before, after = asyncio.run(run())
        self.assertGreater(after, before)


class StreamKeepaliveTest(TempDBTestCase):
    def test_keepalive_comment_when_idle(self):
        async def run():
            with patch.object(stream, "POLL_S", 0.001), \
                 patch.object(stream, "KEEPALIVE_S", 0.005):
                # first frame is the data event; a keepalive must follow while idle
                return await _collect(_StubRequest(), max_frames=2)

        frames = asyncio.run(run())
        self.assertTrue(any(f.startswith(": keepalive") for f in frames),
                        "no keepalive comment while idle: %r" % frames)


class StreamDisconnectTest(TempDBTestCase):
    def test_generator_exits_cleanly_on_disconnect(self):
        """is_disconnected() True → generator stops (no unbounded frames, no
        traceback). We cap at many frames but expect only the initial one."""
        async def run():
            with patch.object(stream, "POLL_S", 0.001), \
                 patch.object(stream, "KEEPALIVE_S", 100):
                # disconnect reported on the first in-loop check
                return await _collect(_StubRequest(disconnect_after=1),
                                      max_frames=50)

        frames = asyncio.run(run())
        # Only the pre-loop initial data frame should have been emitted.
        self.assertEqual(len(frames), 1)
        self.assertTrue(frames[0].startswith("data:"))


class StreamHeadersTest(TempDBTestCase):
    def test_response_headers(self):
        async def run():
            from starlette.requests import Request
            scope = {"type": "http", "method": "GET", "headers": []}

            async def receive():
                return {"type": "http.request"}

            resp = await stream.stats_stream(Request(scope, receive))
            return resp

        resp = asyncio.run(run())
        self.assertEqual(resp.media_type, "text/event-stream")
        self.assertEqual(resp.headers.get("cache-control"), "no-cache")
        self.assertEqual(resp.headers.get("x-accel-buffering"), "no")


if __name__ == "__main__":
    unittest.main()
