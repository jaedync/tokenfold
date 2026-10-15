"""Bounded single-flight cache for secondary, expensive read-only analytics.

ttl is the age after which a value is built again. Ingest bursts do not
trigger another all-history scan per tab: reads coalesce per scoped key.
Failures are never cached. No raw events or credentials enter this cache.

stale_s (stale-while-revalidate): a value older than ttl but younger than
ttl + stale_s is served at once, and one background thread builds the next
one. A dashboard poll then never waits for a rebuild once the key is warm,
so a body can be up to ttl + stale_s old (and as old as one build).

version (optional): a function that returns the data version (the
aggregator cache version, which every ingest changes). A value built before
the version changed is stale at any age: with stale_s it is served at once
while one refresh runs, without stale_s it is built again inline. lookup()
says when it served a stale value, so a route can tell the browser to read
again soon (the X-Tokenfold-Stale header).

Builds must open their own read connection (db.read_conn): a refresh runs
off the request thread. No refresh starts after the app shutdown began.
"""
import logging
import threading
import time
import weakref
from collections import OrderedDict

from . import db

log = logging.getLogger(__name__)

_MISS = object()

# Response header of a route that served a stale value (static/next/js/api.js).
STALE_HEADER = "X-Tokenfold-Stale"

# Every cache, so the app shutdown can wait for their refresh threads.
_instances: "weakref.WeakSet[ReadCache]" = weakref.WeakSet()


def join_all(timeout: float) -> None:
    """Wait up to timeout in total for every running refresh (app shutdown)."""
    deadline = time.monotonic() + timeout
    for cache in list(_instances):
        with cache._lock:
            threads = list(cache._refreshing.values())
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))


class ReadCache:
    def __init__(self, ttl=30, max_entries=16, stale_s=0, version=None):
        self.ttl = ttl
        self.max_entries = max_entries
        self.stale_s = stale_s
        self._version = version
        self._lock = threading.Lock()          # guards the dicts only
        self._entries = OrderedDict()          # key -> (monotonic time, value)
        self._versions: dict = {}              # key -> data version at the build start
        # key -> [build lock, threads waiting on it]; dropped with the last user.
        self._key_locks: dict = {}
        self._refreshing: dict = {}            # key -> refresh Thread
        _instances.add(self)

    def _current(self):
        # Read outside self._lock: the version function takes its own lock.
        return self._version() if self._version else None

    def _fresh(self, key, max_age=None, version=None):
        entry = self._entries.get(key)
        limit = self.ttl if max_age is None else max_age
        if entry and time.monotonic() - entry[0] < limit \
                and (max_age is not None or self._versions.get(key) == version):
            self._entries.move_to_end(key)
            return entry[1]
        return _MISS

    def get(self, key, build):
        return self.lookup(key, build)[0]

    def lookup(self, key, build):
        """(value, stale): stale is True when the value was served while a
        refresh builds a newer one (past ttl, or built before the version
        changed)."""
        # Callers are worker-thread HTTP handlers, never the event loop.
        # One build per key at a time (a second caller waits for it and
        # reads its value); a slow build of one key (a cold 30-day ledger)
        # never blocks a hit or a build of another key.
        version = self._current()
        with self._lock:
            value = self._fresh(key, version=version)
            if value is not _MISS:
                return value, False
            if self.stale_s:
                value = self._fresh(key, self.ttl + self.stale_s)
                if value is not _MISS:
                    self._start_refresh(key, build)
                    return value, True
        return self._build(key, build), False

    def _start_refresh(self, key, build):
        # Caller holds self._lock.
        if key in self._refreshing or db.shutting_down():
            return
        thread = threading.Thread(target=self._refresh, args=(key, build),
                                  daemon=True, name=f"read-cache:{key!r}"[:60])
        self._refreshing[key] = thread
        thread.start()

    def _refresh(self, key, build):
        try:
            self._build(key, build, force=True)
        except Exception:
            # The stale value stays until ttl + stale_s; the next read retries.
            log.exception("read cache refresh failed for %r", key)
        finally:
            with self._lock:
                self._refreshing.pop(key, None)

    def _build(self, key, build, force=False):
        with self._lock:
            slot = self._key_locks.setdefault(key, [threading.Lock(), 0])
            slot[1] += 1
        try:
            with slot[0]:
                version = self._current()
                if not force:
                    with self._lock:
                        value = self._fresh(key, version=version)
                    if value is not _MISS:
                        return value
                value = build()
                with self._lock:
                    self._entries[key] = (time.monotonic(), value)
                    self._versions[key] = version
                    self._entries.move_to_end(key)
                    while len(self._entries) > self.max_entries:
                        old, _ = self._entries.popitem(last=False)
                        self._versions.pop(old, None)
                return value
        finally:
            with self._lock:
                slot[1] -= 1
                if slot[1] == 0:
                    del self._key_locks[key]
