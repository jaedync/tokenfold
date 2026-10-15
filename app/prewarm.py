"""Startup prewarm for the slow reads of /next. No routes.

After a restart the first visit used to wait for builds that only a request
started: today's typical-spend profile (Runway said "not ready"), the
/api/spend-history scan (History said "Loading" for about 6 s) and the
/api/stats build of the scope that is not the default (about 11 s for the
work scope). It also builds the default 30-day ledger, which leaves the
closed days and the Pi lineage cached (a cold ledger read is about 2 s).
start() runs those builds once, in order, on one background thread, so
startup never waits for them.

Every build reads on its own read-only connection (db.read_conn), never on
the shared ingest connection, so the prewarm cannot hold ingest back and
needs no write lock. join() stops the thread between steps and waits for it;
the app shutdown calls it before it closes the database.

The limit-rate fit has its own prewarm (limit_rate.prewarm): it writes its
result to meta through the shared connection.
"""
import logging
import sys
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)

SCOPES = ("personal", "enterprise")
TYPICAL_WAIT_S = 120.0

_lock = threading.Lock()
_stop = threading.Event()
_thread: Optional[threading.Thread] = None


def _cfg():
    return sys.modules["app.config"]


def _scopes() -> list[str]:
    locked = _cfg().LOCKED_SCOPE
    return [locked] if locked else list(SCOPES)


def _warm_typical(scope: str) -> None:
    from . import today_typical
    today_typical.warm(scope, time.time(), wait_s=TYPICAL_WAIT_S)


def _warm_ledger(scope: str) -> None:
    from . import ledger
    ledger.warm(scope)


def _warm_history(scope: str) -> None:
    from . import spend_history
    spend_history.warm(scope)


def _warm_stats(scope: str) -> None:
    """Build the /api/stats payload of a scope that is not the default.

    The default scope comes first: the eager rebuild already builds it, and
    build_dashboard_data waits for that build instead of starting a second
    one. A built scope stays warm while it is in the aggregator's warm-scope
    set (WARM_SCOPE_TTL_S), as if a visitor had asked for it.
    """
    from . import aggregator
    aggregator.build_dashboard_data(_cfg().DEFAULT_SCOPE)
    if scope != _cfg().DEFAULT_SCOPE:
        aggregator.build_dashboard_data(scope)


def _steps() -> list[tuple[str, object, str]]:
    scopes = _scopes()
    steps = [("typical", _warm_typical, s) for s in scopes]
    steps += [("ledger", _warm_ledger, s) for s in scopes]
    steps += [("spend history", _warm_history, s) for s in scopes]
    # A scope-locked instance serves one scope: the eager rebuild has it.
    if len(scopes) > 1:
        steps += [("stats", _warm_stats, s) for s in scopes if s != _cfg().DEFAULT_SCOPE]
    return steps


def _run(steps) -> None:
    for name, fn, scope in steps:
        if _stop.is_set():
            return
        started = time.monotonic()
        try:
            fn(scope)
        except Exception:
            log.exception("prewarm of %s (%s) failed; the first request builds it", name, scope)
            continue
        log.info("prewarmed %s (%s) in %.1f s", name, scope, time.monotonic() - started)


def start() -> Optional[threading.Thread]:
    """Start the prewarm thread unless one runs. Never blocks, never raises."""
    global _thread
    try:
        with _lock:
            if _thread is not None and _thread.is_alive():
                return _thread
            _stop.clear()
            _thread = threading.Thread(target=_run, args=(_steps(),),
                                       name="next-prewarm", daemon=True)
            _thread.start()
            return _thread
    except Exception:
        log.exception("prewarm did not start; the first requests build their data")
        return None


def join(timeout: float) -> None:
    """Stop after the running step and wait up to timeout for the thread."""
    _stop.set()
    with _lock:
        thread = _thread
    if thread is not None:
        thread.join(timeout)
