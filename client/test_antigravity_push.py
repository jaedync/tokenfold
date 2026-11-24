"""Privacy and replay tests for the stdlib Antigravity reporter."""
import importlib.util
import json
import os
import random
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("agy_push", Path(__file__).with_name("antigravity-push.py"))
push = importlib.util.module_from_spec(spec)
spec.loader.exec_module(push)
CID = "fbdad862-c5cd-4255-9849-ef2bcfec60b9"
OTHER = "b948ea48-ec36-4ac1-80f2-b2156c58ae2b"
# Temp folders go under tempfile.gettempdir(), not next to this file: an
# interrupted run must not leave them in the repository (review Low).


def step(index=1, **overrides):
    return dict({
        "step_index": index, "source": "MODEL", "type": "PLANNER_RESPONSE",
        "status": "DONE", "created_at": "2026-11-24T05:58:00Z",
        "model": "gemini-4-neon-xhigh", "input_tokens": 100,
        "cache_read_tokens": 200, "output_tokens": 20,
        "thinking": "synthetic secret", "content": "synthetic reply",
        "tool_calls": [{"arguments": "synthetic command"}],
    }, **overrides)


def line(index=1, **overrides):
    return json.dumps(step(index, **overrides)).encode() + b"\n"


class ParseTranscriptTest(unittest.TestCase):
    def test_projection_has_only_usage_and_identity(self):
        self.assertEqual(push.parse_transcript_step(line(), CID.upper()), {
            "conversation_id": CID, "step_index": 1, "timestamp": "2026-11-24T05:58:00Z",
            "model": "gemini-4-neon-xhigh", "input_tokens": 100,
            "cache_read_tokens": 200, "output_tokens": 20,
        })

    def test_bad_or_incomplete_steps_never_leave_the_client(self):
        for raw in (b"\xff", b"{", b"[]", b"123", line(source="USER"),
                    line(status="RUNNING"), line(model=" hi there"), line(input_tokens=True),
                    line(cache_read_tokens=-10), line(output_tokens=10**12 + 1),
                    line(created_at="2026-11-24T05:58:00"), line(created_at="bad time"),
                    line(step_index=True)):
            with self.subTest(raw=raw):
                self.assertIsNone(push.parse_transcript_step(raw, CID))
        self.assertIsNone(push.parse_transcript_step(line(), "not-a-conversation"))


class QuotaTest(unittest.TestCase):
    document = {"schema": 1, "read_at": "2026-11-24T05:00:00Z", "windows": [
        {"bucket_id": "gemini-5h", "group": "ignored label", "used_pct": 0,
         "reset_at": "2026-11-24T10:00:00Z", "implied_window_usd": 30},
        {"bucket_id": "gemini-models-weekly", "remaining_fraction": 0.75,
         "reset_at": "2026-12-01T05:00:00Z", "implied_window_usd": 105},
        {"bucket_id": "3p-5h", "used_pct": 10, "reset_at": "2026-11-24T10:00:00Z",
         "implied_window_usd": 99, "unexpected_text": "never post this"},
        {"bucket_id": "claude-and-gpt-models-weekly", "used_pct": 25,
         "reset_at": "2026-12-01T05:00:00Z"},
        {"bucket_id": "unknown", "used_pct": 10, "reset_at": "2026-12-01T05:00:00Z"},
    ]}

    def test_quota_uses_only_structured_window_fields(self):
        snapshots = {s["provider"]: s for s in push.parse_quota_snapshot(self.document)}
        self.assertEqual(snapshots.keys(), {"antigravity-gemini", "antigravity-claude-gpt"})
        gemini = snapshots["antigravity-gemini"]
        self.assertEqual(gemini["observed_at_epoch"], 1795496400)
        self.assertEqual(gemini["windows"], [
            {"key": "five_hour", "pct": 0, "resets_at_epoch": 1795514400, "implied_window_usd": 30},
            {"key": "seven_day", "pct": 25, "resets_at_epoch": 1796101200, "implied_window_usd": 105},
        ])
        other = snapshots["antigravity-claude-gpt"]["windows"]
        self.assertEqual([w["pct"] for w in other], [10, 25])
        self.assertTrue(all(w.keys() == {"key", "pct", "resets_at_epoch"} for w in other))
        self.assertEqual(push.parse_quota_snapshot(dict(self.document, read_at="not a timestamp")), [])
        self.assertEqual(push.parse_quota_snapshot(dict(self.document, schema=2)), [])

    def test_invalid_and_duplicate_windows_do_not_replace_other_group(self):
        short, long, first_3p, *_ = self.document["windows"]
        windows = [short, dict(short), long, first_3p, {"bucket_id": [], "used_pct": 30}]
        snapshots = push.parse_quota_snapshot(dict(self.document, windows=windows))
        self.assertEqual([s["provider"] for s in snapshots], ["antigravity-claude-gpt"])
        broken = dict(short, used_pct=True)
        windows = [broken, long, first_3p]
        snapshots = push.parse_quota_snapshot(dict(self.document, windows=windows))
        self.assertEqual([s["provider"] for s in snapshots], ["antigravity-claude-gpt"])

    def test_quota_posts_even_when_spend_fails(self):
        calls = []

        def send(endpoint, body):
            calls.append((endpoint, body))
            if endpoint == "/api/ingest/antigravity":
                raise push.PushError("simulated failure")
            return {"status": "ok", "providers": [s["provider"] for s in body["limits"]]}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / CID / push.TRANSCRIPT
            path.parent.mkdir(parents=True)
            path.write_bytes(line())
            latest = root / "latest.json"
            latest.write_text(json.dumps(self.document), encoding="utf-8")
            errors = push.push_once(root, latest, root / "cursor.json", send, "test-host")
        self.assertEqual(len(errors), 1)
        self.assertEqual([endpoint for endpoint, _ in calls],
                         ["/api/ingest/antigravity", "/api/provider-usage"])
        self.assertEqual(calls[-1][1]["account_class"], "personal")
        self.assertEqual(calls[-1][1]["machine"], "test-host")


class HttpTest(unittest.TestCase):
    def test_keyed_ingest_does_not_follow_redirects(self):
        hits = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                hits.append((self.command, self.path))
                self.send_response(302)
                self.send_header("Location", "/another-path")
                self.end_headers()

            def do_GET(self):
                hits.append((self.command, self.path))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"accepted":1,"duplicates":0}')

            def log_message(self, *_):
                pass  # Do not print HTTP requests in this test.

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                endpoint = "/api/ingest/antigravity"
                with patch.dict(os.environ, {"no_proxy": "127.0.0.1"}):
                    with self.assertRaisesRegex(push.PushError, "HTTP 302"):
                        push._post(f"http://127.0.0.1:{server.server_port}", "test-key", endpoint, {})
                self.assertEqual(hits, [("POST", endpoint)])
            finally:
                server.shutdown()
                thread.join(5)


class CursorTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.state_path = self.root / "cursor.json"
        self.state = {}
        self.deliveries = []

    def path(self, cid):
        path = self.root / cid / ".system_generated/logs/transcript_full.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def sync(self, post=None):
        push.sync_transcripts(self.root, self.state, post or self.accept)
        push.save_cursors(self.state_path, self.state)
        self.state = push.load_cursors(self.state_path)

    def accept(self, batch):
        self.deliveries.extend((e["conversation_id"], e["step_index"]) for e in batch)

    def test_random_chunkings_rotations_and_reloads_deliver_only_complete_steps_once(self):
        rng = random.Random(20261124)
        p1, p2 = self.path(CID), self.path(OTHER)
        for trial in range(32):
            with self.subTest(trial=trial):
                self.state.clear()
                self.deliveries.clear()
                expected = set()
                for path, cid in ((p1, CID), (p2, OTHER)):
                    path.write_bytes(b"")
                    ids = rng.sample(range(1, 900), rng.randrange(5, 15))
                    for i in ids:
                        data = line(i)
                        split = rng.randrange(1, len(data))
                        with path.open("ab") as stream:
                            stream.write(data[:split])
                        self.sync()
                        self.assertCountEqual(self.deliveries, expected)
                        with path.open("ab") as stream:
                            stream.write(data[split:])
                        self.sync()
                        expected.add((cid, i))
                        self.assertCountEqual(self.deliveries, expected)
                    # Replaced inode, duplicate old steps, and a late step with a lower ID.
                    new = path.with_suffix(".next")
                    new.write_bytes(b"".join(line(i) for i in ids) + line(0))
                    os.replace(new, path)
                    self.sync()
                    expected.add((cid, 0))
                    self.assertCountEqual(self.deliveries, expected)
                    # Same-inode rewrite can change early lines and keep the same size.
                    data = path.read_bytes().replace(b'"step_index": 0,', b'"step_index": 9,')
                    path.write_bytes(data)
                    self.sync()
                    expected.add((cid, 9))
                    self.assertCountEqual(self.deliveries, expected)
                self.sync()
                self.assertCountEqual(self.deliveries, expected)
                p1.unlink()
                p2.unlink()

    def test_growing_transcript_snapshot_does_not_skip_an_updated_step(self):
        path = self.path(CID)
        path.write_bytes(line(1, status="RUNNING"))
        self.sync()
        self.assertEqual(self.deliveries, [])
        path.write_bytes(line(1, status="DONE"))
        self.sync()
        self.assertEqual(self.deliveries, [(CID, 1)])

    def test_bad_ingest_ack_preserves_replay(self):
        path = self.path(CID)
        path.write_bytes(line(1))
        with self.assertRaises(push.PushError):
            push.sync_spend(self.root, self.state_path,
                            lambda *_: {"accepted": 0, "duplicates": 0}, "test-host")
        self.state = push.load_cursors(self.state_path)
        self.sync()
        self.assertEqual(self.deliveries, [(CID, 1)])

    def test_retries_failed_batch_without_resending_acknowledged_steps(self):
        path = self.path(CID)
        path.write_bytes(line(1) + line(2) + line(2) + line(3))
        seen_batches = []

        def post(batch):
            seen_batches.append([e["step_index"] for e in batch])
            if len(seen_batches) == 2:
                raise push.PushError("simulated delivery failure")
            self.accept(batch)

        with patch.object(push, "BATCH_SIZE", 2):
            with self.assertRaises(push.PushError):
                self.sync(post)
            self.assertEqual(seen_batches, [[1, 2], [3]])
            self.assertEqual(self.deliveries, [(CID, 1), (CID, 2)])
            self.sync()
        self.assertEqual(self.deliveries, [(CID, 1), (CID, 2), (CID, 3)])
