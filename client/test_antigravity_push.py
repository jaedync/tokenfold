"""Privacy and replay tests for the stdlib Antigravity reporter."""
import importlib.util
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("agy_push", Path(__file__).with_name("antigravity-push.py"))
push = importlib.util.module_from_spec(spec)
spec.loader.exec_module(push)
CID = "fbdad862-c5cd-4255-9849-ef2bcfec60b9"
OTHER = "b948ea48-ec36-4ac1-80f2-b2156c58ae2b"
CLIENT_DIR = Path(__file__).resolve().parent


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


class CursorTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=CLIENT_DIR)
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
