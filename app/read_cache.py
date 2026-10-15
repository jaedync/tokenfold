"""Bounded single-flight cache for secondary, expensive read-only analytics.

These feeds tolerate at most 30 seconds of server cache age. Ingest bursts do
not trigger another all-history scan per tab: reads coalesce per scoped key.
Failures are never cached. No raw events or credentials enter this cache.
"""
import threading
import time
from collections import OrderedDict

_MISS = object()


class ReadCache:
    def __init__(self, ttl=30, max_entries=16):
        self.ttl = ttl
        self.max_entries = max_entries
        self._lock = threading.Lock()          # guards the two dicts only
        self._entries = OrderedDict()
        # key -> [build lock, threads waiting on it]; dropped with the last user.
        self._key_locks: dict = {}

    def _fresh(self, key):
        entry = self._entries.get(key)
        if entry and time.monotonic() - entry[0] < self.ttl:
            self._entries.move_to_end(key)
            return entry[1]
        return _MISS

    def get(self, key, build):
        # Callers are worker-thread HTTP handlers, never the event loop.
        # One build per key at a time (a second caller waits for it and
        # reads its value); a slow build of one key (a cold 30-day ledger)
        # never blocks a hit or a build of another key.
        with self._lock:
            value = self._fresh(key)
            if value is not _MISS:
                return value
            slot = self._key_locks.setdefault(key, [threading.Lock(), 0])
            slot[1] += 1
        try:
            with slot[0]:
                with self._lock:
                    value = self._fresh(key)
                if value is not _MISS:
                    return value
                value = build()
                with self._lock:
                    self._entries[key] = (time.monotonic(), value)
                    self._entries.move_to_end(key)
                    while len(self._entries) > self.max_entries:
                        self._entries.popitem(last=False)
                return value
        finally:
            with self._lock:
                slot[1] -= 1
                if slot[1] == 0:
                    del self._key_locks[key]
