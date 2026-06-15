"""Push-path performance contracts.

The pusher fires on every Stop/SessionEnd/PostToolUse hook, so its steady-state
cost must be tiny. Three regressions this locks down:

1. No multi-second sleep in the push path (a 0-90s jitter sleep, a relic of the
   cron paradigm, used to run on every fire — even enterprise machines with no
   OAuth token slept ~45s avg before the fetch no-op'd).
2. Unchanged transcript files are skipped via a stat (mtime+size) cache instead
   of being read end-to-end every run. With 3000+ files / 1GB+ that was ~900ms
   of wasted I/O per fire.
3. The OAuth usage fetch is interval-gated so it doesn't hit Anthropic on every
   single hook fire.
"""

import importlib.util
import json
import pathlib
import tempfile
import time
import unittest


def _load():
    spec = importlib.util.spec_from_file_location(
        "csp_perf", pathlib.Path(__file__).resolve().parent / "claude-stats-push.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MOD = _load()


class _PushBase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.dir.name)
        self.claude = root / "projects"
        (self.claude / "projA").mkdir(parents=True)
        self.addCleanup(self.dir.cleanup)

        self.cursor = root / "cursor.json"
        self._save = {
            "CLAUDE_DIR": MOD.CLAUDE_DIR, "CURSOR_FILE": MOD.CURSOR_FILE,
            "SERVER_URL": MOD.SERVER_URL, "API_KEY": MOD.API_KEY,
            "push_batch": MOD.push_batch, "read_account": MOD.read_account,
            "desktop_dir": MOD.desktop_dir, "_fetch_and_push_usage": MOD._fetch_and_push_usage,
        }
        MOD.CLAUDE_DIR = self.claude
        MOD.CURSOR_FILE = self.cursor
        MOD.SERVER_URL = "http://x"
        MOD.API_KEY = "x"
        MOD.read_account = lambda *_a, **_k: {}
        MOD.desktop_dir = lambda: None
        MOD._fetch_and_push_usage = lambda: None

        self.sent = []
        MOD.push_batch = lambda pd, sf, cl, ev, acct: (
            self.sent.append(len(ev)) or {"accepted": len(ev), "duplicates": 0})

        # track which jsonl files get opened — shadow builtins.open in the module
        self.reads = []
        real_open = open

        def counting_open(file, *a, **k):
            if str(file).endswith(".jsonl"):
                self.reads.append(str(file))
            return real_open(file, *a, **k)
        MOD.open = counting_open
        self.addCleanup(self._restore)

    def _restore(self):
        for k, v in self._save.items():
            setattr(MOD, k, v)
        if "open" in MOD.__dict__:
            del MOD.__dict__["open"]

    def _write_session(self, name, n_events):
        p = self.claude / "projA" / name
        lines = [json.dumps({"uuid": f"{name}-{i}", "type": "assistant",
                             "timestamp": "2026-06-15T00:00:00Z",
                             "message": {"model": "claude-opus-4-8",
                                         "usage": {"input_tokens": 1}}}) + "\n"
                 for i in range(n_events)]
        p.write_text("".join(lines))
        return p


class SkipCacheTest(_PushBase):
    def test_unchanged_files_not_reread(self):
        self._write_session("s1.jsonl", 10)
        self._write_session("s2.jsonl", 5)

        MOD.main()                       # cold: both read
        self.assertEqual(sorted(set(self.reads)),
                         sorted({str(self.claude / "projA" / n)
                                 for n in ("s1.jsonl", "s2.jsonl")}))
        first_events = sum(self.sent)
        self.assertEqual(first_events, 15)

        self.reads.clear(); self.sent.clear()
        MOD.main()                       # warm: nothing changed -> no reads
        self.assertEqual(self.reads, [], "unchanged files must not be re-read")
        self.assertEqual(sum(self.sent), 0)

    def test_appended_file_is_reread_and_new_lines_sent(self):
        p = self._write_session("s1.jsonl", 10)
        MOD.main()
        self.reads.clear(); self.sent.clear()

        with open(p, "a") as f:
            f.write(json.dumps({"uuid": "new-1", "type": "assistant",
                                "timestamp": "2026-06-15T01:00:00Z",
                                "message": {"model": "claude-opus-4-8",
                                            "usage": {"input_tokens": 1}}}) + "\n")
        MOD.main()
        self.assertIn(str(p), self.reads, "appended file must be re-read")
        self.assertEqual(sum(self.sent), 1, "only the new line should be sent")

    def test_legacy_int_cursor_still_works(self):
        """Old cursor files store bare int line counts. Upgrade must not
        reset (re-send) — a fully-consumed file stays consumed."""
        p = self._write_session("s1.jsonl", 10)
        self.cursor.write_text(json.dumps({str(p): 10}))  # old format, at end
        MOD.main()
        self.assertEqual(sum(self.sent), 0, "legacy at-end cursor must not re-send")


class NoJitterSleepTest(_PushBase):
    def test_push_path_has_no_long_sleep(self):
        self._write_session("s1.jsonl", 3)
        slept = []
        real_sleep = time.sleep
        MOD.time.sleep = lambda s: slept.append(s)
        try:
            MOD.main()
        finally:
            MOD.time.sleep = real_sleep
        self.assertTrue(all(s <= 1 for s in slept),
                        f"push path slept too long: {slept}")


class UsageFetchGateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stamp = pathlib.Path(self.tmp.name) / "usage-stamp"
        self._saved = getattr(MOD, "USAGE_FETCH_STAMP", None)
        MOD.USAGE_FETCH_STAMP = self.stamp
        self.addCleanup(lambda: setattr(MOD, "USAGE_FETCH_STAMP", self._saved))

    def test_recent_fetch_is_skipped(self):
        self.stamp.write_text(json.dumps({"last": time.time()}))
        self.assertTrue(MOD._usage_fetch_too_soon(),
                        "a fetch seconds ago must gate the next one out")

    def test_stale_fetch_allowed(self):
        self.stamp.write_text(json.dumps({"last": time.time() - 99999}))
        self.assertFalse(MOD._usage_fetch_too_soon())

    def test_missing_stamp_allows_fetch(self):
        self.assertFalse(MOD._usage_fetch_too_soon())


if __name__ == "__main__":
    unittest.main()
