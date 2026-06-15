"""Cursor persistence must survive concurrent pushers.

Stop/SessionEnd hooks are undebounced, so a push can run while a PostToolUse
push is still in flight. The pushers race on the cursor file. The server
dedupes by uuid so correctness is safe, but a non-atomic write (truncate +
rewrite) can be read mid-write as truncated JSON -> load_cursors() falls back
to {} -> every cursor resets to 0 -> the entire transcript history is
re-sent. An atomic write (tmp + os.replace) makes a concurrent reader always
see either the complete old or complete new file, never a torn one.
"""

import importlib.util
import json
import pathlib
import tempfile
import threading
import unittest


def _load_module():
    here = pathlib.Path(__file__).resolve().parent
    src = here / "claude-stats-push.py"
    spec = importlib.util.spec_from_file_location("claude_stats_push", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


class CursorAtomicWriteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp.close()
        self.path = pathlib.Path(self.tmp.name)
        self._saved = MOD.CURSOR_FILE
        MOD.CURSOR_FILE = self.path
        self.addCleanup(self._restore)

    def _restore(self):
        MOD.CURSOR_FILE = self._saved
        for p in (self.path, self.path.with_suffix(".json.tmp"),
                  self.path.with_suffix(".tmp")):
            try:
                p.unlink()
            except OSError:
                pass

    def test_concurrent_reads_never_see_torn_file(self):
        # a large dict so the write spans many bytes -> torn reads are likely
        # under a non-atomic writer
        big = {f"/some/very/long/project/path/session-{i}.jsonl": i
               for i in range(4000)}
        MOD.save_cursors(big)

        stop = threading.Event()
        torn = []

        def writer():
            n = 0
            while not stop.is_set():
                d = dict(big)
                d["counter"] = n
                MOD.save_cursors(d)
                n += 1

        def reader():
            while not stop.is_set():
                # load_cursors() swallows torn reads into {} — detect that
                got = MOD.load_cursors()
                if not got:
                    torn.append(1)

        threads = [threading.Thread(target=writer) for _ in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        threading.Event().wait(0.8)
        stop.set()
        for t in threads:
            t.join()

        self.assertEqual(
            len(torn), 0,
            f"{len(torn)} reads saw a torn/empty cursor file — a concurrent "
            "pusher would reset to 0 and re-send all history")

    def test_no_tmp_file_left_behind(self):
        MOD.save_cursors({"a": 1})
        leftovers = list(self.path.parent.glob(self.path.name + "*.tmp")) + \
            list(self.path.parent.glob(self.path.stem + ".tmp"))
        self.assertEqual(leftovers, [], f"stale tmp files: {leftovers}")


if __name__ == "__main__":
    unittest.main()
