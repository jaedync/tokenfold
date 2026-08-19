"""client/backfill-transcripts.py harvest: server-tool counts ride along with
the cache-tier split and titles when repairing history from local transcripts."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "backfill_transcripts",
    Path(__file__).resolve().parents[2] / "client" / "backfill-transcripts.py")
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)


def _arec(uuid, usage):
    return {"type": "assistant", "uuid": uuid,
            "message": {"usage": usage}}


class HarvestServerToolsTest(unittest.TestCase):

    def _write(self, records):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        self.addCleanup(os.unlink, path)
        return path

    def test_harvest_collects_server_tool_counts(self):
        path = self._write([_arec("u1", {
            "server_tool_use": {"web_search_requests": 2,
                                "web_fetch_requests": 1}})])
        cache_tiers, server_tools, titles, _sig = bt.harvest_file(path)
        self.assertEqual(server_tools, {"u1": [2, 1]})
        self.assertEqual(cache_tiers, {})
        self.assertEqual(titles, {})

    def test_zero_counts_not_collected(self):
        """All-zero counts are the common case — sending them would bloat
        every batch for no repair value (the server treats 0 as unset)."""
        path = self._write([_arec("u1", {
            "server_tool_use": {"web_search_requests": 0,
                                "web_fetch_requests": 0}})])
        _, server_tools, _, _ = bt.harvest_file(path)
        self.assertEqual(server_tools, {})

    def test_cache_tiers_and_titles_still_harvested(self):
        path = self._write([
            _arec("u1", {
                "cache_creation": {"ephemeral_5m_input_tokens": 5,
                                   "ephemeral_1h_input_tokens": 7},
                "server_tool_use": {"web_search_requests": 3,
                                    "web_fetch_requests": 0}}),
            {"type": "ai-title", "sessionId": "s1", "aiTitle": "My Session"},
        ])
        cache_tiers, server_tools, titles, _sig = bt.harvest_file(path)
        self.assertEqual(cache_tiers, {"u1": [5, 7]})
        self.assertEqual(server_tools, {"u1": [3, 0]})
        self.assertEqual(titles, {"s1": "My Session"})

    def test_malformed_counts_skipped(self):
        path = self._write([_arec("u1", {"server_tool_use": "garbage"}),
                            _arec("u2", {"server_tool_use": {
                                "web_search_requests": "2",
                                "web_fetch_requests": 1}})])
        _, server_tools, _, _ = bt.harvest_file(path)
        self.assertEqual(server_tools, {})


class CacheVersionTest(unittest.TestCase):
    """The skip-cache must be versioned: caches written before a harvest
    field existed mark files 'done' that were never scanned for that field,
    silently skipping ~all history on incremental re-runs."""

    def setUp(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        self.cache_path = Path(path)
        self._saved = bt.CACHE_PATH
        bt.CACHE_PATH = self.cache_path
        self.addCleanup(self._restore)

    def _restore(self):
        bt.CACHE_PATH = self._saved
        try:
            os.unlink(self.cache_path)
        except OSError:
            pass

    def test_legacy_unversioned_cache_forces_rescan(self):
        self.cache_path.write_text(json.dumps({"/some/file.jsonl": [123, 456]}))
        self.assertEqual(bt._load_cache(full=False), {})

    def test_stale_version_forces_rescan(self):
        self.cache_path.write_text(json.dumps(
            {"v": bt.CACHE_VERSION - 1, "files": {"/f.jsonl": [1, 2]}}))
        self.assertEqual(bt._load_cache(full=False), {})

    def test_current_version_roundtrip(self):
        sigs = {"/f.jsonl": [1, 2]}
        bt._save_cache(sigs)
        self.assertEqual(bt._load_cache(full=False), sigs)


if __name__ == "__main__":
    unittest.main()
