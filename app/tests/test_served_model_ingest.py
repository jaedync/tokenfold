"""POST /api/ingest: capture the served model from a thinking block.

Two client generations must both work. A current client splits the signature
itself and ships ~200 bytes of header (sig_header/sig_version/sig_cipher_len);
an older one still ships the whole blob under `signature` and the server
splits it. One decoder either way, same five columns.
"""

import json
import unittest
from pathlib import Path

from app.tests._support import TempDBTestCase

FIXTURES = json.loads(
    (Path(__file__).resolve().parent / "fixtures"
     / "thinking_signatures.json").read_text())

SIG_COLS = "served_model, sig_version, sig_header, sig_cipher_len, sig_fields"


def _rec(uuid, blocks, model="claude-fable-5"):
    return {
        "uuid": uuid,
        "type": "assistant",
        "timestamp": "2026-06-09T12:00:00Z",
        "sessionId": "s1",
        "requestId": "r-" + uuid,
        "message": {
            "model": model,
            "id": "m-" + uuid,
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "content": blocks,
        },
    }


def _thinking(**extra):
    blk = {"type": "thinking", "thinking": "some reasoning"}
    blk.update(extra)
    return blk


class ServedModelIngestTest(TempDBTestCase):

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def _post(self, events):
        c = self.client()
        return c.post("/api/ingest", json={
            "machine": "m", "project_dir": "p", "session_file": "s.jsonl",
            "cursor": {"last_line_num": 0}, "events": events,
        }, headers={"X-API-Key": self.api_key})

    def _sig(self, uuid):
        return self.conn.execute(
            f"SELECT {SIG_COLS} FROM events WHERE uuid=?", (uuid,)).fetchone()

    # ── client-decoded fields (the live path) ─────────────────────────────

    def test_client_decoded_header_stored_and_model_derived(self):
        exp = FIXTURES["kettle_v2"]["expect"]
        r = self._post([_rec("u1", [_thinking(
            signature="[2345 chars]",
            sig_header=exp["sig_header_b64"],
            sig_version=exp["sig_version"],
            sig_cipher_len=exp["sig_cipher_len"],
        )])])
        self.assertEqual(r.status_code, 200, r.text)
        row = self._sig("u1")
        self.assertEqual(row["served_model"], exp["served_model"])
        self.assertEqual(row["sig_version"], exp["sig_version"])
        self.assertEqual(row["sig_header"], exp["sig_header_b64"])
        self.assertEqual(row["sig_cipher_len"], exp["sig_cipher_len"])
        self.assertEqual(row["sig_fields"], exp["sig_fields"])

    def test_served_model_can_differ_from_requested_model(self):
        """The whole point: `model` is what was asked for, `served_model` is
        what answered."""
        exp = FIXTURES["carafe_v2"]["expect"]
        self._post([_rec("u1", [_thinking(
            sig_header=exp["sig_header_b64"], sig_version=exp["sig_version"],
            sig_cipher_len=exp["sig_cipher_len"])], model="claude-opus-4-8")])
        row = self.conn.execute(
            "SELECT model, served_model FROM events WHERE uuid='u1'").fetchone()
        self.assertEqual(row["model"], "claude-opus-4-8")
        self.assertEqual(row["served_model"], "claude-carafe-416c93ba-v1-prod")

    def test_v4_header_stores_shape_without_a_model(self):
        """A header that no longer names a model is still worth keeping: the
        field list is how we see the format change."""
        exp = FIXTURES["fable_v4"]["expect"]
        self._post([_rec("u1", [_thinking(
            sig_header=exp["sig_header_b64"], sig_version=4,
            sig_cipher_len=exp["sig_cipher_len"])])])
        row = self._sig("u1")
        self.assertIsNone(row["served_model"])
        self.assertEqual(row["sig_version"], 4)
        self.assertEqual(row["sig_fields"], "1,3,7,8")

    def test_first_thinking_block_wins(self):
        a = FIXTURES["fable_v2"]["expect"]
        b = FIXTURES["kettle_v2"]["expect"]
        self._post([_rec("u1", [
            _thinking(sig_header=a["sig_header_b64"], sig_version=2,
                      sig_cipher_len=a["sig_cipher_len"]),
            _thinking(sig_header=b["sig_header_b64"], sig_version=2,
                      sig_cipher_len=b["sig_cipher_len"]),
        ])])
        self.assertEqual(self._sig("u1")["served_model"], a["served_model"])

    # ── raw-signature fallback (older clients) ────────────────────────────

    def test_raw_signature_is_split_server_side(self):
        fx = FIXTURES["fable_v2"]
        exp = fx["expect"]
        r = self._post([_rec("u1", [_thinking(signature=fx["signature"])])])
        self.assertEqual(r.status_code, 200, r.text)
        row = self._sig("u1")
        self.assertEqual(row["served_model"], exp["served_model"])
        self.assertEqual(row["sig_version"], exp["sig_version"])
        self.assertEqual(row["sig_header"], exp["sig_header_b64"])
        self.assertEqual(row["sig_cipher_len"], exp["sig_cipher_len"])
        self.assertEqual(row["sig_fields"], exp["sig_fields"])

    def test_placeholder_signature_stores_nothing(self):
        """A current client replaces the blob with '[N chars]'; that is not a
        signature and must not become one."""
        self._post([_rec("u1", [_thinking(signature="[2345 chars]")])])
        self.assertIsNone(self._sig("u1")["sig_header"])

    def test_thinking_block_without_signature_leaves_nulls(self):
        self._post([_rec("u1", [_thinking()])])
        row = self._sig("u1")
        self.assertIsNone(row["sig_header"])
        self.assertIsNone(row["served_model"])
        self.assertIsNone(row["sig_version"])

    def test_non_thinking_event_leaves_nulls(self):
        self._post([_rec("u1", [{"type": "text", "text": "hello"}])])
        self.assertIsNone(self._sig("u1")["sig_header"])

    # ── untrusted input ───────────────────────────────────────────────────

    def test_hostile_values_never_500_and_never_store_junk(self):
        """Transcript JSON is untrusted: a dict bound to sqlite raises and
        would 500 the whole batch, a str in an INTEGER column poisons every
        SUM over it. Neither may happen."""
        exp = FIXTURES["fable_v2"]["expect"]
        events = [
            _rec("bad1", [_thinking(sig_header={"nope": 1})]),
            _rec("bad2", [_thinking(sig_header=["a"], sig_version=[1])]),
            _rec("bad3", [_thinking(sig_header="!!! not base64 !!!")]),
            _rec("bad4", [_thinking(sig_header="A" * 5000)]),
            _rec("bad5", [_thinking(signature={"blob": True})]),
            _rec("ok", [_thinking(sig_header=exp["sig_header_b64"],
                                  sig_version="two",
                                  sig_cipher_len="lots")]),
        ]
        r = self._post(events)
        self.assertEqual(r.status_code, 200, r.text)
        for uuid in ("bad1", "bad2", "bad3", "bad4", "bad5"):
            with self.subTest(uuid=uuid):
                self.assertIsNone(self._sig(uuid)["sig_header"])
        # The good header still lands; the junk ints coerce to 0, not to text.
        row = self._sig("ok")
        self.assertEqual(row["served_model"], exp["served_model"])
        self.assertEqual(row["sig_version"], 0)
        self.assertEqual(row["sig_cipher_len"], 0)

    def test_oversized_header_rejected(self):
        """4096 chars is generous headroom for a ~200 byte header; a payload
        past it is not a header we should be storing."""
        self._post([_rec("u1", [_thinking(sig_header="A" * 4097)])])
        self.assertIsNone(self._sig("u1")["sig_header"])

    def test_absurd_raw_blob_is_not_parsed(self):
        self._post([_rec("u1", [_thinking(signature="A" * 70_000)])])
        self.assertIsNone(self._sig("u1")["sig_header"])


if __name__ == "__main__":
    unittest.main()
