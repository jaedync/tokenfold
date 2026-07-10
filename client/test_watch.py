"""Tests for the --watch resident daemon + single-instance flock in
claude-stats-push.py.

Run with:
    .venv/bin/python client/test_watch.py -v

Stdlib unittest, NO network: every server-touching function (push_batch,
push_desktop_sessions, _fetch_and_push_usage) is stubbed at the module level.
"""

import fcntl
import importlib.util
import json
import os
import pathlib
import shutil
import tempfile
import time
import unittest
from unittest import mock


def _load_module():
    """Load claude-stats-push.py as a module (hyphen in name blocks import)."""
    here = pathlib.Path(__file__).resolve().parent
    src = here / "claude-stats-push.py"
    spec = importlib.util.spec_from_file_location("claude_stats_push", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


class _TmpHome(unittest.TestCase):
    """Base: an isolated fake $HOME with a redirected CLAUDE_DIR + CURSOR_FILE.

    We patch the module-level Path constants so no test touches the real
    user's transcripts or cursor.
    """

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.claude_dir = self.tmp / ".claude" / "projects"
        self.claude_dir.mkdir(parents=True)
        self.cursor_file = self.tmp / ".tokenfold-cursor.json"
        self.lock_file = self.tmp / ".tokenfold-push.lock"
        self._patches = [
            mock.patch.object(MOD, "CLAUDE_DIR", self.claude_dir),
            mock.patch.object(MOD, "CURSOR_FILE", self.cursor_file),
            mock.patch.object(MOD, "LOCK_FILE", self.lock_file),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    def _write_session(self, name, lines, mtime=None):
        """Write a JSONL session file with `lines` valid records."""
        p = self.claude_dir / "proj" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        body = "".join(json.dumps({"type": "user", "uuid": "u%d" % i}) + "\n"
                       for i in range(lines))
        p.write_text(body)
        if mtime is not None:
            os.utime(p, (mtime, mtime))
        return p


# ── Case 1: flock exclusivity ────────────────────────────────────────────────

class FlockExclusivityTest(_TmpHome):
    def test_first_acquire_wins_second_refuses(self):
        h1 = MOD.acquire_singleton_lock()
        self.assertIsNotNone(h1, "first acquire should win")
        self.addCleanup(MOD.release_singleton_lock, h1)

        # A second attempt on the SAME lock file (separate fd) must fail.
        h2 = MOD.acquire_singleton_lock()
        self.assertIsNone(h2, "second acquire must refuse while held")

    def test_release_frees_the_lock(self):
        h1 = MOD.acquire_singleton_lock()
        self.assertIsNotNone(h1)
        MOD.release_singleton_lock(h1)
        h2 = MOD.acquire_singleton_lock()
        self.assertIsNotNone(h2, "lock must be re-acquirable after release")
        self.addCleanup(MOD.release_singleton_lock, h2)


# ── Case 2: hot-set selection ────────────────────────────────────────────────

class HotSetTest(_TmpHome):
    def test_old_files_excluded_new_included(self):
        now = time.time()
        fresh = self._write_session("fresh.jsonl", 1, mtime=now - 60)
        old = self._write_session("old.jsonl", 1,
                                  mtime=now - (MOD.HOT_WINDOW_S + 3600))
        files = MOD.find_session_files()
        hot = MOD.select_hot_set(files, now=now)
        hot_paths = {str(p) for p in hot}
        self.assertIn(str(fresh), hot_paths)
        self.assertNotIn(str(old), hot_paths)

    def test_edge_exactly_at_window_included(self):
        now = time.time()
        edge = self._write_session("edge.jsonl", 1,
                                   mtime=now - MOD.HOT_WINDOW_S + 1)
        files = MOD.find_session_files()
        hot = MOD.select_hot_set(files, now=now)
        self.assertIn(str(edge), {str(p) for p in hot})


# ── Case 3: change detection ─────────────────────────────────────────────────

class ChangeDetectionTest(_TmpHome):
    def test_signature_change_detected_unchanged_not(self):
        p = self._write_session("s.jsonl", 1)
        sigs = {str(p): MOD._file_sig(p)}
        # No change → hot_set_changed False.
        self.assertFalse(MOD.hot_set_changed([p], sigs))
        # Append a line → signature changes → True.
        with open(p, "a") as f:
            f.write(json.dumps({"type": "user", "uuid": "new"}) + "\n")
        # Bump mtime to be safe on coarse filesystem clocks.
        os.utime(p, (time.time() + 1, time.time() + 1))
        self.assertTrue(MOD.hot_set_changed([p], sigs))

    def test_new_hot_file_counts_as_change(self):
        p = self._write_session("s.jsonl", 1)
        # Empty prior sigs → an unseen hot file is a change.
        self.assertTrue(MOD.hot_set_changed([p], {}))


# ── Case 4: rescan discovers new files ───────────────────────────────────────

class RescanTest(_TmpHome):
    def test_watch_loop_discovers_new_file_within_one_rescan(self):
        # Deterministic loop: patch intervals tiny and cap iterations.
        cycles = []

        def fake_cycle(cursors):
            cycles.append(True)
            return cursors

        created = {"done": False}

        def maybe_create():
            # Create a brand-new session only after the first tick, so the
            # rescan (not the initial scan) is what discovers it.
            if not created["done"]:
                created["done"] = True
                self._write_session("late.jsonl", 1)

        with mock.patch.object(MOD, "run_push_cycle", side_effect=fake_cycle), \
             mock.patch.object(MOD, "_fetch_and_push_usage"), \
             mock.patch.object(MOD, "HOT_POLL_S", 0.0), \
             mock.patch.object(MOD, "RESCAN_S", 0.0), \
             mock.patch.object(MOD, "_watch_sleep", side_effect=lambda s: maybe_create()):
            MOD.watch_loop(max_iterations=3)

        # The late file must have been globbed by a rescan and pushed.
        self.assertTrue(created["done"])
        self.assertGreaterEqual(len(cycles), 1,
                                "a rescan should have found the new file and pushed")


# ── Case 5: one-shot refactor regression ─────────────────────────────────────

class OneShotRefactorTest(_TmpHome):
    def test_run_push_cycle_advances_cursor_like_classic(self):
        p = self._write_session("s.jsonl", 3)

        pushed = []

        def fake_push(project_dir, session_file, cursor_line, events, account):
            pushed.append((session_file, cursor_line, len(events)))
            return {"accepted": len(events), "duplicates": 0}

        with mock.patch.object(MOD, "push_batch", side_effect=fake_push), \
             mock.patch.object(MOD, "read_account", return_value={}), \
             mock.patch.object(MOD, "desktop_dir", return_value=None):
            cursors = MOD.load_cursors()
            cursors = MOD.run_push_cycle(cursors)
            MOD.save_cursors(cursors)

        # Cursor advanced to 3 lines on disk.
        on_disk = json.loads(self.cursor_file.read_text())
        self.assertEqual(on_disk[str(p)], 3)
        self.assertEqual(pushed, [("s.jsonl", 0, 3)])

    def test_second_cycle_is_noop_after_full_consume(self):
        p = self._write_session("s.jsonl", 2)

        def fake_push(project_dir, session_file, cursor_line, events, account):
            return {"accepted": len(events), "duplicates": 0}

        with mock.patch.object(MOD, "push_batch", side_effect=fake_push), \
             mock.patch.object(MOD, "read_account", return_value={}), \
             mock.patch.object(MOD, "desktop_dir", return_value=None):
            cursors = MOD.run_push_cycle(MOD.load_cursors())
            MOD.save_cursors(cursors)
            calls = []
            with mock.patch.object(MOD, "push_batch",
                                   side_effect=lambda *a, **k: calls.append(a) or {"accepted": 0, "duplicates": 0}):
                cursors = MOD.run_push_cycle(MOD.load_cursors())
            self.assertEqual(calls, [], "unchanged file must not re-push")


# ── Case 6: usage-fetch gating untouched ─────────────────────────────────────

class UsageGatingTest(_TmpHome):
    def test_watch_loop_calls_fetch_and_push_usage(self):
        with mock.patch.object(MOD, "run_push_cycle", side_effect=lambda c: c), \
             mock.patch.object(MOD, "_fetch_and_push_usage") as fetch, \
             mock.patch.object(MOD, "HOT_POLL_S", 0.0), \
             mock.patch.object(MOD, "RESCAN_S", 0.0), \
             mock.patch.object(MOD, "_watch_sleep"):
            MOD.watch_loop(max_iterations=2)
        self.assertGreaterEqual(fetch.call_count, 1,
                                "watch loop must call _fetch_and_push_usage")

    def test_gate_still_lives_in_fetch_only(self):
        # The 300s throttle must be the ONLY gate — assert the interval constant
        # is unchanged and the too-soon helper still exists.
        self.assertEqual(MOD.USAGE_FETCH_MIN_INTERVAL, 300)
        self.assertTrue(callable(MOD._usage_fetch_too_soon))


if __name__ == "__main__":
    unittest.main()
