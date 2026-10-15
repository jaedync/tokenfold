"""FastAPI app entry point."""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .db import (ROLLUP_DAYS_KEY, close_conn, get_conn, queued_rollup_days,
                 write_txn)
from .light import init_light, start_watchdog, stop_watchdog
from .notify import init_notify_token
from .pricing import load_pricing
from . import usage_fetcher

logger = logging.getLogger("tokenfold.main")
# Progress note interval for the one-time all-days rollup rebuild (~75 s on prod).
BACKFILL_PROGRESS_EVERY = 50


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: init DB and load pricing
    get_conn()
    try:
        load_pricing()
    except Exception:
        pass
    if not config.DASHBOARD_PASSWORD:
        print(
            "[auth] WARNING: DASHBOARD_PASSWORD unset — dashboard is OPEN (no login). "
            "Set DASHBOARD_USER/DASHBOARD_PASSWORD to lock it down.",
            flush=True,
        )
    init_notify_token()
    if usage_fetcher.should_run():
        usage_fetcher.start()
    else:
        print("[usage_fetcher] usage_fetcher disabled (instance locked to enterprise scope)", flush=True)
    await init_light()
    start_watchdog()

    # Backfill daily_summary if empty, then warm the cache
    from .aggregator import trigger_eager_rebuild, start_sweeps, stop_sweeps
    _backfill_if_needed()
    trigger_eager_rebuild()

    # Start periodic sweep timers
    start_sweeps()

    yield

    # Shutdown
    stop_sweeps()
    stop_watchdog()
    usage_fetcher.stop()
    close_conn()


def _backfill_if_needed():
    """Rebuild empty or invalidated rollups before serving/warming the cache.

    Each day commits atomically. A failing day is logged and skipped so a bad
    day cannot crash-loop startup; the pending marker then stays durable, so
    the next startup retries ALL days even if some rows already exist. Days
    queued by queue_rollup_days alone are re-rolled the same way.
    """
    conn = get_conn()
    pending = conn.execute(
        "SELECT 1 FROM meta WHERE key='daily_summary_rebuild_pending'"
    ).fetchone()
    queued = queued_rollup_days(conn)
    every_day = pending or not conn.execute("SELECT 1 FROM daily_summary LIMIT 1").fetchone()
    if not every_day and not queued:
        return
    from .summarizer import summarize_days
    if every_day:
        with write_txn(conn):
            conn.execute(
                "INSERT OR REPLACE INTO meta(key,value) "
                "VALUES('daily_summary_rebuild_pending','1')")
        days = [r['day'] for r in conn.execute('SELECT DISTINCT day FROM events ORDER BY day')]
    else:
        days = sorted(queued)
    started, failed = time.monotonic(), []
    logger.warning("rebuilding daily rollups for %d days before serving", len(days))
    for i, day in enumerate(days, 1):
        # Match normal daily ingest: never group requests from unrelated days
        # together in one all-history accumulator; keep startup memory bounded.
        try:
            summarize_days([day])
        except Exception:
            logger.exception("daily rollup rebuild failed for %s; retried next startup", day)
            failed.append(day)
        if i % BACKFILL_PROGRESS_EVERY == 0:
            logger.warning("rebuilt %d/%d days (%.0f s)", i, len(days), time.monotonic() - started)
    if failed:
        logger.error("daily rollup rebuild incomplete: %d of %d days failed", len(failed), len(days))
        return
    with write_txn(conn):
        # A full rebuild also covered the queued days.
        conn.execute("DELETE FROM meta WHERE key IN "
                     "('daily_summary_rebuild_pending', ?)", (ROLLUP_DAYS_KEY,))
    logger.warning("daily rollups rebuilt in %.0f s", time.monotonic() - started)


# Disable the auto-generated API docs on this confidential dashboard: /docs,
# /redoc, and /openapi.json otherwise expose route paths and the X-API-Key
# header name to unauthenticated callers (info-disclosure, no data leak).
app = FastAPI(
    title="Tokenfold",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


# Defense-in-depth; the XSS fix itself is the </script>-encoding + esc() in dashboard.py/template.
@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    if request.url.path == "/" or (request.url.path.startswith("/api/") and
                                    "text/event-stream" not in response.headers.get("content-type", "")):
        # Authenticated snapshots must not be replayed by browser/proxy caches
        # after a scope change, wake-up, or reconnect.
        response.headers["Cache-Control"] = "private, no-store"
    return response


from .ingest import router as ingest_router
from .api import router as api_router
from .dashboard import router as dashboard_router
from .notify import router as notify_router
from .light import router as light_router
from .ha import router as ha_router
from .desktop_sessions import router as desktop_sessions_router
from .limit_readings import router as limit_readings_router
from .spend_history import router as spend_history_router
from .served_models import router as served_models_router
from .install import router as install_router
from .monthly_budget import router as monthly_budget_router
from .stream import router as stream_router
from .agent_state import router as agent_state_router
from .agent_events import router as agent_events_router

app.include_router(ingest_router)
app.include_router(api_router)
app.include_router(dashboard_router)
app.include_router(notify_router)
app.include_router(light_router)
app.include_router(ha_router)
app.include_router(desktop_sessions_router)
app.include_router(limit_readings_router)
app.include_router(spend_history_router)
app.include_router(served_models_router)
app.include_router(install_router)
app.include_router(monthly_budget_router)
app.include_router(stream_router)
app.include_router(agent_state_router)
app.include_router(agent_events_router)

_static = Path(__file__).resolve().parent.parent / "static"
if _static.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static)), name="static")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(_static / "favicon.ico", media_type="image/x-icon")


@app.get("/health")
async def health():
    return {"status": "ok"}
