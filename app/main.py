"""FastAPI app entry point."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .db import close_conn, get_conn
from .light import init_light, start_watchdog, stop_watchdog
from .notify import init_notify_token
from .pricing import load_pricing
from . import usage_fetcher


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
    """Populate daily_summary table if it's empty but events exist."""
    conn = get_conn()
    summary_count = conn.execute("SELECT COUNT(*) FROM daily_summary").fetchone()[0]
    if summary_count > 0:
        return  # Already populated
    event_count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    if event_count == 0:
        return  # No data at all
    from .summarizer import summarize_days
    summarize_days(None)  # Backfill all days


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
    return response


from .ingest import router as ingest_router
from .api import router as api_router
from .dashboard import router as dashboard_router
from .notify import router as notify_router
from .light import router as light_router
from .ha import router as ha_router
from .desktop_sessions import router as desktop_sessions_router
from .billing_readings import router as billing_readings_router
from .limit_readings import router as limit_readings_router
from .spend_history import router as spend_history_router
from .install import router as install_router

app.include_router(ingest_router)
app.include_router(api_router)
app.include_router(dashboard_router)
app.include_router(notify_router)
app.include_router(light_router)
app.include_router(ha_router)
app.include_router(desktop_sessions_router)
app.include_router(billing_readings_router)
app.include_router(limit_readings_router)
app.include_router(spend_history_router)
app.include_router(install_router)

_static = Path(__file__).resolve().parent.parent / "static"
if _static.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static)), name="static")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(_static / "favicon.ico", media_type="image/x-icon")


@app.get("/health")
async def health():
    return {"status": "ok"}
