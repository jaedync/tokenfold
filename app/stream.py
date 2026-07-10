"""Server-Sent Events stream of the aggregator cache version.

The dashboard opens an EventSource against GET /api/stats/stream and refetches
/api/stats whenever the emitted version changes, turning the existing polled
/api/stats/version fallback into a push. Each SSE frame is a tiny JSON payload
`data: {"version": <int>}` — never the stats blob itself, so the browser stays
in control of when (and at what scope) it actually refetches.

WHY 300ms polling of the version counter instead of cross-thread asyncio
signaling from the rebuild worker: the cache is rebuilt on a plain daemon
`threading.Thread` (see aggregator.trigger_eager_rebuild), which has no handle
on this request's event loop. Bridging a worker-thread notification into each
connected client's loop would mean loop-bound Events / call_soon_threadsafe
plumbing — a source of leaked loop references and shutdown deadlocks, and a
per-connection registration the worker would have to fan out to. Reading
get_cache_version() is a lock-guarded int compare: effectively free. Polling it
every POLL_S (300ms) in-process cannot deadlock, holds no cross-thread state,
and drops naturally to nothing when the client disconnects and the generator is
cancelled. A 300ms worst-case latency is imperceptible for a usage dashboard.

Constants POLL_S / KEEPALIVE_S live at module top and are patched to small
values in tests so change-detection and keepalive fire in milliseconds.
"""

import asyncio
import json

from fastapi import APIRouter, Depends, Request
from starlette.responses import StreamingResponse

from .aggregator import get_cache_version
from .auth import require_dashboard_auth

router = APIRouter()

# How often the generator re-reads the (cheap, lock-guarded) cache version.
POLL_S = 0.3
# Emit an SSE comment after this long with nothing sent, so idle proxies with
# short read timeouts don't drop the connection.
KEEPALIVE_S = 20


def _data_frame(version: int) -> str:
    """Serialize one SSE data frame carrying the version as JSON."""
    return "data: %s\n\n" % json.dumps({"version": version})


async def _version_events(request: Request):
    """Async generator yielding SSE frames until the client disconnects.

    Emits the current version immediately (so a reconnecting client detects a
    missed bump), then on each POLL_S tick emits a new data frame when the
    version changed, or a keepalive comment once KEEPALIVE_S elapses idle.

    Normal disconnects surface as asyncio.CancelledError (generator cancelled)
    or as request.is_disconnected() going True; both exit quietly with no
    traceback — CancelledError is re-raised, never logged, so it can't spew.
    """
    last_version = get_cache_version()
    yield _data_frame(last_version)
    idle = 0.0
    try:
        while True:
            await asyncio.sleep(POLL_S)
            if await request.is_disconnected():
                return
            version = get_cache_version()
            if version != last_version:
                last_version = version
                idle = 0.0
                yield _data_frame(version)
                continue
            idle += POLL_S
            if idle >= KEEPALIVE_S:
                idle = 0.0
                yield ": keepalive\n\n"
    except asyncio.CancelledError:
        # Client went away mid-await; propagate cancellation without logging.
        raise


@router.get("/api/stats/stream", dependencies=[Depends(require_dashboard_auth)])
async def stats_stream(request: Request):
    return StreamingResponse(
        _version_events(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
