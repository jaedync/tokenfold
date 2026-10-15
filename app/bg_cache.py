"""Stale-while-revalidate cache for slow inputs of fast live endpoints.

/api/now (polled every 5 s) and /api/today (every 60 s) have hard latency
targets, but a few of their inputs need a multi-day event scan (the binding
window's $/point, the 28-day typical-day profile). Those inputs change
slowly, so a request never waits for them longer than wait_s: a cold miss
starts one builder thread and returns None (the endpoint reports null) when
the build is slower than that, and a stale value is served while a refresh
runs. Failures are logged and never cached. Builders must open their own
read connection (db.read_conn): they run off the request thread.
"""
import logging
import threading
import time

log = logging.getLogger(__name__)


class BackgroundCache:
    def __init__(self, ttl: float, wait_s: float, max_entries: int = 16):
        self.ttl = ttl
        self.wait_s = wait_s
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._entries: dict = {}          # key -> (monotonic_ts, value)
        self._inflight: dict = {}         # key -> Thread

    def get(self, key, build, wait_s: float | None = None):
        """wait_s overrides the instance wait (0 = start the build, never wait)."""
        with self._lock:
            entry = self._entries.get(key)
            if entry and time.monotonic() - entry[0] < self.ttl:
                return entry[1]
            thread = self._inflight.get(key)
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
