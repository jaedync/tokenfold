"""Client-side capture of thinking-block signature headers.

Every thinking block carries a base64 `signature` blob (mean 2.4 KB, 7% of all
bytes this client uploads) whose only plaintext is a ~200 byte header naming
the model that actually served the block. The client splits the blob, ships the
header, and drops the rest. What is locked down here:

  1. split_signature agrees with the real captured blobs in
     app/tests/fixtures/thinking_signatures.json (7 shapes, four header format
     versions), and degrades to (0, None, 0) on anything it cannot read
  2. the two client copies of split_signature stay byte-identical to each other
     (the server holds a third copy and asserts the same thing)
  3. strip_content emits the derived fields, drops the blob, and never raises
  4. an unparseable blob still ships a diagnosable sample
  5. the backfill harvest picks the header out of a transcript line
  6. CACHE_VERSION moved, so machines that ran an older backfill rescan
"""

import importlib.util
import inspect
import json
import pathlib
import re
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = HERE.parent / "app" / "tests" / "fixtures" / "thinking_signatures.json"


def _load(name, filename):
    """Load a hyphenated client script as a module."""
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PUSH = _load("csp_sig", "claude-stats-push.py")
BF = _load("bf_sig", "backfill-transcripts.py")
SIGS = json.loads(FIXTURES.read_text())

# A real sig_header is up to 260 base64 chars (kettle_v2), so "the blob is
# gone" cannot be checked at 200. 400 sits above every header and below the
# shortest blob in the fixture set (552 chars).
_MAX_B64_RUN = 400
_B64_RUN = re.compile(r"[A-Za-z0-9+/]{%d,}" % _MAX_B64_RUN)


class SplitSignatureTest(unittest.TestCase):
    def test_matches_every_captured_fixture(self):
        for name, fix in SIGS.items():
            with self.subTest(fixture=name):
                want = (fix["expect"]["sig_version"],
                        fix["expect"]["sig_header_b64"],
                        fix["expect"]["sig_cipher_len"])
                self.assertEqual(PUSH.split_signature(fix["signature"]), want)

    def test_garbage_never_raises(self):
        # Anything unreadable must look identical to "no signature here":
        # a format change may not cost us the event it rode in on.
        for junk in ("", "not a signature", "!!!!", "AAAA", "x" * 500,
                     "CBAY", None, 12345, b"bytes", [], {}):
            with self.subTest(input=repr(junk)[:40]):
                self.assertEqual(PUSH.split_signature(junk), (0, None, 0))

    def test_truncated_blob_degrades(self):
        blob = SIGS["fable_v2"]["signature"]
        self.assertEqual(PUSH.split_signature(blob[:40]), (0, None, 0))

    def test_unpadded_base64_still_parses(self):
        # Some transcript writers drop the padding; the header is the same.
        blob = SIGS["opus5_v2"]["signature"]
        unpadded = blob.rstrip("=")
        self.assertNotEqual(unpadded, blob)  # keep the case real
        self.assertEqual(PUSH.split_signature(unpadded),
                         PUSH.split_signature(blob))

    def test_client_copies_are_byte_identical(self):
        # Three copies exist (both client scripts and the server decoder) and
        # only identical sources are safe: a drift would change what the
        # dashboard reports without changing a single test.
        self.assertEqual(inspect.getsource(PUSH.split_signature),
                         inspect.getsource(BF.split_signature))


class StripContentTest(unittest.TestCase):
    def _thinking_record(self, signature, thinking="reasoning text " * 40):
        return {
            "uuid": "evt-1",
            "type": "assistant",
            "message": {
                "model": "claude-fable-5",
                "content": [{"type": "thinking", "thinking": thinking,
                             "signature": signature}],
            },
        }

    def _stripped_block(self, rec):
        return rec["message"]["content"][0]

    def test_emits_header_fields_and_drops_blob(self):
        for name, fix in SIGS.items():
            with self.subTest(fixture=name):
                blob = fix["signature"]
                rec = self._thinking_record(blob)
                out = PUSH.strip_content(rec)
                blk = self._stripped_block(out)
                self.assertEqual(blk["signature"], "[%d chars]" % len(blob))
                self.assertEqual(blk["sig_version"], fix["expect"]["sig_version"])
                self.assertEqual(blk["sig_header"], fix["expect"]["sig_header_b64"])
                self.assertEqual(blk["sig_cipher_len"],
                                 fix["expect"]["sig_cipher_len"])
                self.assertNotIn("sig_error", blk)

                wire = json.dumps(out)
                self.assertNotIn(blob[:300], wire)
                self.assertEqual(_B64_RUN.findall(wire), [])

    def test_payload_shrinks(self):
        blob = SIGS["fable_v2"]["signature"]  # 3104 chars
        rec = self._thinking_record(blob)
        before = len(json.dumps(rec))
        after = len(json.dumps(PUSH.strip_content(rec)))
        self.assertLess(after, before * 0.25)

    def test_does_not_mutate_the_input_record(self):
        blob = SIGS["kettle_v2"]["signature"]
        rec = self._thinking_record(blob)
        PUSH.strip_content(rec)
        self.assertEqual(rec["message"]["content"][0]["signature"], blob)

    def test_unparseable_blob_ships_a_sample(self):
        blob = "n0tAsignature" * 40  # 520 chars, decodes to nothing useful
        out = PUSH.strip_content(self._thinking_record(blob))
        blk = self._stripped_block(out)
        self.assertEqual(blk["signature"], "[%d chars]" % len(blob))
        self.assertTrue(blk["sig_error"])
        self.assertEqual(blk["sig_sample"], blob[:256])
        self.assertNotIn("sig_header", blk)
        self.assertNotIn("sig_version", blk)

    def test_thinking_block_without_signature_is_untouched(self):
        rec = self._thinking_record(None)
        del rec["message"]["content"][0]["signature"]
        blk = self._stripped_block(PUSH.strip_content(rec))
        self.assertNotIn("sig_header", blk)
        self.assertNotIn("sig_error", blk)
        self.assertTrue(blk["thinking"].endswith("chars]"))

    def test_non_string_signature_is_ignored(self):
        # Never raise, never drop the event, whatever the transcript holds.
        for bogus in (None, 12345, {"a": 1}, []):
            with self.subTest(value=repr(bogus)):
                rec = self._thinking_record(bogus)
                blk = self._stripped_block(PUSH.strip_content(rec))
                self.assertNotIn("sig_header", blk)
                self.assertNotIn("sig_error", blk)


class BackfillHarvestTest(unittest.TestCase):
    def _write_transcript(self, lines):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                          delete=False)
        self.addCleanup(lambda: pathlib.Path(tmp.name).unlink(missing_ok=True))
        for line in lines:
            tmp.write(json.dumps(line) + "\n")
        tmp.close()
        return tmp.name

    @staticmethod
    def _assistant(uuid, blocks):
        return {"type": "assistant", "uuid": uuid,
                "message": {"content": blocks, "usage": {}}}

    @staticmethod
    def _thinking(signature):
        return {"type": "thinking", "thinking": "x" * 30,
                "signature": signature}

    def test_harvests_headers_from_a_transcript(self):
        fable, kettle = SIGS["fable_v2"], SIGS["kettle_v2"]
        path = self._write_transcript([
            {"type": "user", "uuid": "u0", "message": {"content": "hi"}},
            self._assistant("a1", [self._thinking(fable["signature"])]),
            # text before thinking: the thinking block still has to be found
            self._assistant("a2", [{"type": "text", "text": "hello"},
                                   self._thinking(kettle["signature"])]),
            self._assistant("a3", [{"type": "text", "text": "no thinking"}]),
            self._assistant("a4", [self._thinking("garbage-not-a-signature")]),
        ])
        _ct, _st, _titles, sig_headers = BF.harvest_file(path)
        self.assertEqual(sorted(sig_headers), ["a1", "a2"])
        self.assertEqual(sig_headers["a1"],
                         [fable["expect"]["sig_version"],
                          fable["expect"]["sig_header_b64"],
                          fable["expect"]["sig_cipher_len"]])
        self.assertEqual(sig_headers["a2"][1],
                         kettle["expect"]["sig_header_b64"])

    def test_first_thinking_block_wins(self):
        first, second = SIGS["opus5_v2"], SIGS["sonnet5_v0"]
        path = self._write_transcript([
            self._assistant("a1", [self._thinking(first["signature"]),
                                   self._thinking(second["signature"])]),
        ])
        _ct, _st, _titles, sig_headers = BF.harvest_file(path)
        self.assertEqual(sig_headers["a1"][1],
                         first["expect"]["sig_header_b64"])

    def test_records_without_a_uuid_are_skipped(self):
        path = self._write_transcript([
            {"type": "assistant", "message": {
                "content": [self._thinking(SIGS["fable_v2"]["signature"])]}},
        ])
        _ct, _st, _titles, sig_headers = BF.harvest_file(path)
        self.assertEqual(sig_headers, {})

    def test_unreadable_file_yields_empty_maps(self):
        result = BF.harvest_file("/nonexistent/transcript.jsonl")
        self.assertEqual(result, ({}, {}, {}, {}))


class BackfillPostTest(unittest.TestCase):
    """The wire shape is what the server parses, so pin it here."""

    def setUp(self):
        self.posted = []
        saved = {name: getattr(BF, name)
                 for name in ("post", "read_config", "harvest", "_save_cache",
                              "BATCH")}
        self.addCleanup(
            lambda: [setattr(BF, k, v) for k, v in saved.items()])

        BF.read_config = lambda: ("http://server", "key")
        BF._save_cache = lambda sigs: None
        BF.post = lambda url, key, payload: (
            self.posted.append(payload) or
            {"updated_events": 0, "updated_titles": 0,
             "updated_server_tools": 0,
             "updated_sig_headers": len(payload.get("sig_headers", {})),
             "touched_days": []})

    def _run_with(self, sig_headers):
        BF.harvest = lambda **kw: ({}, {}, {}, sig_headers, {})
        argv = sys.argv
        sys.argv = ["backfill-transcripts.py"]
        try:
            BF.main()
        finally:
            sys.argv = argv

    def test_sig_headers_ride_their_own_batches(self):
        fix = SIGS["fable_v2"]["expect"]
        self._run_with({"evt-1": [fix["sig_version"], fix["sig_header_b64"],
                                  fix["sig_cipher_len"]]})
        sig_posts = [p for p in self.posted if "sig_headers" in p]
        self.assertEqual(len(sig_posts), 1)
        self.assertFalse(sig_posts[0]["reroll"])  # one re-roll at the end
        self.assertEqual(sig_posts[0]["sig_headers"]["evt-1"],
                         [fix["sig_version"], fix["sig_header_b64"],
                          fix["sig_cipher_len"]])

    def test_batches_at_the_server_cap(self):
        self.assertEqual(BF.BATCH, 20_000)  # server-side cap per request
        BF.BATCH = 2
        self._run_with({"e%d" % i: [2, "aGVhZGVy", 10] for i in range(5)})
        sizes = [len(p["sig_headers"]) for p in self.posted if "sig_headers" in p]
        self.assertEqual(sizes, [2, 2, 1])

    def test_nothing_to_send_posts_nothing(self):
        self._run_with({})
        self.assertEqual([p for p in self.posted if "sig_headers" in p], [])


class CacheVersionTest(unittest.TestCase):
    """A cache written before sig_headers existed marks files 'done' that were
    never scanned for them; only a version bump forces the rescan."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = pathlib.Path(self.tmp.name) / "cache.json"
        original = BF.CACHE_PATH
        BF.CACHE_PATH = self.path
        self.addCleanup(lambda: setattr(BF, "CACHE_PATH", original))

    def test_version_moved_past_the_server_tool_release(self):
        self.assertGreater(BF.CACHE_VERSION, 2)

    def test_stale_version_forces_a_full_rescan(self):
        self.path.write_text(json.dumps(
            {"v": BF.CACHE_VERSION - 1, "files": {"/t.jsonl": [1, 2]}}))
        self.assertEqual(BF._load_cache(full=False), {})

    def test_current_version_skips_unchanged_files(self):
        files = {"/t.jsonl": [1, 2]}
        BF._save_cache(files)
        self.assertEqual(BF._load_cache(full=False), files)
        self.assertEqual(json.loads(self.path.read_text())["v"],
                         BF.CACHE_VERSION)


if __name__ == "__main__":
    unittest.main()
