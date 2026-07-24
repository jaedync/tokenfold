"""Per-tool-call event push over Server-Sent Events (GET /api/agent-events).

Every Claude Code tool call the fleet relays as a tool_activity ingest fans out
here as one SSE frame, so an ambient consumer (the LED cube) can ripple once per
individual call in realtime instead of once per coalesced poll batch. This is a
push-only side channel: the 250ms /api/agent-state poll stays the state source
AND the ripple fallback when this link is down.

Design: a set of per-subscriber asyncio.Queue(maxsize 64). publish() is a
synchronous put_nowait fan-out; a full queue (a slow consumer) drops the event
for that subscriber only, so one lagging reader can never block ingest. Every
caller runs on the single uvicorn event loop, so no locking is needed. The route
yields a hello frame, then serves queued frames, emitting a keepalive comment
after KEEPALIVE_S idle; a disconnect unregisters the queue in a finally.

Auth is the same fleet ingest token as /api/notify and /api/agent-state; it is
imported inside the handler (like agent_state.py) to avoid an import cycle.
"""
import asyncio
import json

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

router = APIRouter()

# Per-subscriber queues. Bounded so a slow consumer's backlog is dropped rather
# than grown without limit; the fan-out drops on QueueFull instead of blocking.
_subscribers: "set[asyncio.Queue]" = set()

# Queue depth per subscriber before publish() starts dropping for it.
QUEUE_MAXSIZE = 64
# Emit an SSE keepalive comment after this long with nothing sent, so idle
# proxies with short read timeouts do not drop the connection.
KEEPALIVE_S = 20.0


def publish(session_id: str, tool: str) -> None:
    """Fan one tool-call event out to every subscriber. Synchronous: all callers
    run on the uvicorn event loop, so no locking. A subscriber whose queue is
    full (slow consumer) drops this event and keeps its place; ingest never
    blocks on a lagging reader."""
    event = {"type": "tool_call", "session_id": session_id, "tool": tool}
    for q in _subscribers:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass


def _frame(payload: dict) -> str:
    """Serialize one SSE data frame carrying the payload as JSON, terminated by
    the blank line that ends an SSE event."""
    return "data: %s\n\n" % json.dumps(payload)


async def _event_stream(request: Request, queue: "asyncio.Queue"):
    """Async generator: a hello frame, then one data frame per queued tool_call,
    with a keepalive comment after KEEPALIVE_S idle. Any exit (client disconnect
    surfaces as CancelledError or is_disconnected()) unregisters the queue."""
    try:
        yield _frame({"type": "hello"})
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), KEEPALIVE_S)
            except asyncio.TimeoutError:
                if await request.is_disconnected():
                    return
                yield ": keepalive\n\n"
                continue
            yield _frame(event)
    except asyncio.CancelledError:
        # Client went away mid-await; propagate cancellation without logging.
        raise
    finally:
        _subscribers.discard(queue)


def reset() -> None:
    """Test support: forget every subscriber."""
    _subscribers.clear()


@router.get("/api/agent-events")
async def agent_events(request: Request,
                       authorization: str | None = Header(default=None)):
    # Same bearer token the fleet ingest and /api/agent-state use; never
    # unauthenticated. Import inside the handler to avoid a notify <-> events
    # import cycle.
    from .notify import _check_auth

    if not _check_auth(authorization):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    queue: "asyncio.Queue" = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    _subscribers.add(queue)
    return StreamingResponse(
        _event_stream(request, queue),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
