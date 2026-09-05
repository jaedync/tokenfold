"""Bounded single-flight cache for secondary, expensive read-only analytics.

These feeds tolerate at most 30 seconds of server cache age. Ingest bursts do
not trigger another all-history scan per tab: reads coalesce per scoped key.
Failures are never cached. No raw events or credentials enter this cache.
"""
import threading
import time
from collections import OrderedDict


class ReadCache:
    def __init__(self, ttl=30, max_entries=16):
        self.ttl = ttl
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._entries = OrderedDict()

    def get(self, key, build):
        # Callers are worker-thread HTTP handlers, never the event loop.
        # Serializing misses bounds total scan concurrency as well as duplicate
        # work. Hits for a different key wait only for this bounded read.
        with self._lock:
            entry = self._entries.get(key)
            if entry and time.monotonic() - entry[0] < self.ttl:
                self._entries.move_to_end(key)
                return entry[1]
            value = build()
            self._entries[key] = (time.monotonic(), value)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
            return value
