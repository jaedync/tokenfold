"""Stale-while-revalidate cache for slow inputs of fast live endpoints.

/api/now (polled every 5 s) and /api/today (every 60 s) have hard latency
targets, but a few of their inputs need a multi-day event scan (the binding
window's $/point, the 28-day typical-day profile). Those inputs change
slowly, so a request never waits for them longer than wait_s: a cold miss
starts one builder thread and returns None (the endpoint reports null) when
the build is slower than that, and a stale value is served while a refresh
runs. Failures are logged and never cached. Builders must open their own
read connection (db.read_conn): they run off the request thread.

stale_s bounds how old a served value may get: a value older than
ttl + stale_s counts as missing, so the request waits (up to wait_s) for the
build. None (the default) serves a stale value at any age.
"""
import logging
import threading
import time
import weakref

from . import db

log = logging.getLogger(__name__)

# Every cache, so the app shutdown can wait for their builder threads.
_instances: "weakref.WeakSet[BackgroundCache]" = weakref.WeakSet()


def join_all(timeout: float) -> None:
    """Wait up to timeout in total for every running build (app shutdown)."""
    deadline = time.monotonic() + timeout
    for cache in list(_instances):
        with cache._lock:
            threads = list(cache._inflight.values())
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))


class BackgroundCache:
    def __init__(self, ttl: float, wait_s: float, max_entries: int = 16,
                 stale_s: float | None = None):
        self.ttl = ttl
        self.wait_s = wait_s
        self.max_entries = max_entries
        self.stale_s = stale_s
        self._lock = threading.Lock()
        self._entries: dict = {}          # key -> (monotonic_ts, value)
        self._inflight: dict = {}         # key -> Thread
        _instances.add(self)

    def get(self, key, build, wait_s: float | None = None):
        """wait_s overrides the instance wait (0 = start the build, never wait)."""
        with self._lock:
            entry = self._entries.get(key)
            age = time.monotonic() - entry[0] if entry else None
            if entry and age < self.ttl:
                return entry[1]
            if entry and self.stale_s is not None and age >= self.ttl + self.stale_s:
                entry = None
            thread = self._inflight.get(key)
            if thread is None and db.shutting_down():
                # Checked under the lock, as join_all reads _inflight: no
                # build starts that the app shutdown could miss.
                return entry[1] if entry is not None else None
            if thread is None:
                thread = threading.Thread(target=self._run, args=(key, build),
                                          daemon=True, name=f"bg-cache:{key}")
                self._inflight[key] = thread
                thread.start()
        if entry is not None:
            return entry[1]                 # stale but usable
        thread.join(self.wait_s if wait_s is None else wait_s)
        with self._lock:
            entry = self._entries.get(key)
        return entry[1] if entry else None

    def _run(self, key, build):
        try:
            value = build()
        except Exception:
            log.exception("background cache build failed for %r", key)
            with self._lock:
                self._inflight.pop(key, None)
            return
        with self._lock:
            self._entries[key] = (time.monotonic(), value)
            self._inflight.pop(key, None)
            while len(self._entries) > self.max_entries:
                oldest = min(self._entries, key=lambda k: self._entries[k][0])
                del self._entries[oldest]

    def clear(self):
        """Test support: forget every value (in-flight builds still land)."""
        with self._lock:
            self._entries.clear()
