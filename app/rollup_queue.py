"""Coalesced daily_summary re-roll after ingest.

Each ingest batch (about one each 4 s on ms01) re-rolled today in the request
thread, and one re-roll of a busy day takes about 1.4 s, so the re-roll alone
used about a third of a core (2026-11-13). request() now records the days and
a worker re-rolls their union at most once per ROLLUP_MIN_INTERVAL_S; the first
request after a quiet interval runs at once.

The days are durable before the request returns: they go to the startup
re-roll list (db.ROLLUP_DAYS_KEY) and leave it only after a re-roll that began
after their last request. A stop before the worker runs leaves them for the
next startup (main._backfill_if_needed); the hourly sweep also re-rolls the
last 7 days.

The worker re-rolls each day alone (a failing day cannot block today; it gets
MAX_ATTEMPTS tries), reads on its own connection and writes through
db.background_write, like the other background work.

Lock order: db.WRITE_LOCK, then _lock. request() never holds _lock while it
writes, so the two orders cannot meet.
"""
import json
import logging
import os
import threading
import time

from . import db

logger = logging.getLogger(__name__)

_ROLLUP_MIN_INTERVAL_DEFAULT_S = 30.0


def _interval_from_env() -> float:
    raw = os.environ.get("TOKENFOLD_ROLLUP_MIN_INTERVAL_S")
    if raw is None or raw.strip() == "":
        return _ROLLUP_MIN_INTERVAL_DEFAULT_S
    try:
        value = float(raw)
    except ValueError:
        value = float("nan")
    if not value >= 0 or value == float("inf"):
        logger.warning("TOKENFOLD_ROLLUP_MIN_INTERVAL_S=%r is not a number >= 0; "
                       "using %s s", raw, _ROLLUP_MIN_INTERVAL_DEFAULT_S)
        return _ROLLUP_MIN_INTERVAL_DEFAULT_S
    return value


# 0 re-rolls in the calling thread, as before (the test support sets 0).
ROLLUP_MIN_INTERVAL_S = _interval_from_env()

# A day that fails this many re-rolls in a row leaves the worker; it stays in
# the durable list for the startup re-roll (and the hourly sweep re-rolls the
# last 7 days), so one bad day cannot block the others.
MAX_ATTEMPTS = 3

_lock = threading.Lock()
_pending: set[str] = set()
_failures: dict[str, int] = {}
_thread: threading.Thread | None = None
_last_started: float | None = None
_wake = threading.Event()


def _roll_day(day: str) -> None:
    from .summarizer import summarize_days
    summarize_days([day], background=True)


def _roll(days: list[str]) -> set[str]:
    """Re-roll each day alone, so one failing day cannot hold back the others.
    Returns the days that were re-rolled. Stops between days at shutdown."""
    done: set[str] = set()
    for day in days:
        if db.shutting_down():
            break
        try:
            _roll_day(day)
        except Exception:
            logger.exception("coalesced rollup failed for %s", day)
            continue
        done.add(day)
    if done:
        from .aggregator import trigger_eager_rebuild
        trigger_eager_rebuild()
    return done


def request(days) -> None:
    """Re-roll these days soon. Raises only when the durable record fails
    (interval > 0) or, with interval 0, when the re-roll fails."""
    days = {d for d in days if d}
    if not days:
        return
    if ROLLUP_MIN_INTERVAL_S <= 0:
        from .aggregator import trigger_eager_rebuild
        from .summarizer import summarize_days
        summarize_days(sorted(days))
        trigger_eager_rebuild()
        return
    # Pending first, then durable: _unqueue reads _pending inside its write
    # transaction, so it either sees these days or runs before this write.
    with _lock:
        _pending.update(days)
    try:
        with db.write_txn() as conn:
            db.queue_rollup_days(conn, days)
    finally:
        _ensure_worker()  # also when the durable write failed


def _ensure_worker() -> None:
    global _thread
    with _lock:
        if _thread is not None or not _pending:
            return
        thread = threading.Thread(target=_run, daemon=True, name="rollup")
        thread.start()  # under _lock: the worker cannot exit before it is recorded
        _thread = thread


def _wait_interval() -> None:
    with _lock:
        last = _last_started
    if last is not None:
        delay = last + ROLLUP_MIN_INTERVAL_S - time.monotonic()
        if delay > 0:
            _wake.wait(delay)
    _wake.clear()


def _run() -> None:
    global _thread, _last_started
    me = threading.current_thread()
    try:
        while True:
            _wait_interval()
            with _lock:
                if db.shutting_down() or not _pending:
                    if _thread is me:
                        _thread = None
                    return
                days = sorted(_pending)
                _pending.clear()
                _last_started = time.monotonic()
            try:
                done = _roll(days)
            except Exception:
                logger.exception("coalesced rollup failed (days=%s)", days)
                done = set()
            if db.shutting_down():
                return  # the days not done stay listed for the startup re-roll
            _retry_or_give_up(set(days) - done, done)
            if done:
                _unqueue(sorted(done))
    finally:
        with _lock:
            if _thread is me:
                _thread = None


def _retry_or_give_up(failed: set[str], done: set[str]) -> None:
    with _lock:
        for day in done:
            _failures.pop(day, None)
        for day in failed:
            n = _failures.get(day, 0) + 1
            if n >= MAX_ATTEMPTS:
                _failures.pop(day, None)
                logger.error("rollup of %s failed %d times; left for the startup re-roll",
                             day, n)
                continue
            _failures[day] = n
            _pending.add(day)


def _unqueue(days: list[str]) -> None:
    """Drop rolled days from the durable list, except days requested again."""
    try:
        with db.background_write() as conn:
            if conn is None:
                return  # shutdown: the startup re-roll covers them
            with _lock:
                done = set(days) - _pending
            queued = db.queued_rollup_days(conn)
            left = queued - done
            if left == queued:
                return
            if left:
                conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                             (db.ROLLUP_DAYS_KEY, json.dumps(sorted(left))))
            else:
                conn.execute("DELETE FROM meta WHERE key=?", (db.ROLLUP_DAYS_KEY,))
    except Exception:
        # Harmless: the next startup re-rolls the days that stay listed.
        logger.exception("rollup queue cleanup failed (days=%s)", days)


def join(timeout: float) -> None:
    """End a wait and wait up to timeout for the worker (shutdown, tests)."""
    with _lock:
        thread = _thread
    _wake.set()
    if thread is not None:
        thread.join(timeout)
