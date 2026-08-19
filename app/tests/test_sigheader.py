"""app/sigheader.py: split a thinking-block signature, decode its header.

Every positive case is a REAL signature captured off live transcripts
(fixtures/thinking_signatures.json), paired with the decode we expect. The
format is undocumented and moves, so the fixtures ARE the specification: when
Anthropic ships a fifth header shape, a new fixture is the whole change.
"""

import base64
import json
import unittest
from pathlib import Path

from app.sigheader import decode_header, split_signature

FIXTURE_PATH = (Path(__file__).resolve().parent
                / "fixtures" / "thinking_signatures.json")
FIXTURES = json.loads(FIXTURE_PATH.read_text())

# Nothing here may raise, whatever it is handed: an unreadable signature must
# cost us the signature, never the event carrying it.
GARBAGE = [
    None, 12, 3.5, True, [], {}, b"bytes",
    "", "   ", "not base64!!!", "@@@@", "AAAA", "=", "z",
    "CAI=",                       # valid protobuf, no envelope
    "[2345 chars]",               # the placeholder a current client sends
    "A" * 10_000,                 # long but meaningless
]


def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _protobuf(varints, envelope=None):
    """Hand-build a base64 signature blob for shapes no fixture covers.

    varints: {field: value} at the top level. envelope: {field: bytes} wrapped
    as the top-level f2, or None for no envelope at all.
    """
    body = b"".join(_varint(f << 3) + _varint(v) for f, v in varints.items())
    if envelope is not None:
        inner = b"".join(_varint((f << 3) | 2) + _varint(len(p)) + p
                         for f, p in envelope.items())
        body += _varint((2 << 3) | 2) + _varint(len(inner)) + inner
    return base64.b64encode(body).decode("ascii")


class SplitSignatureTest(unittest.TestCase):

    def test_every_fixture_splits_to_expected_triple(self):
        for name, fx in FIXTURES.items():
            with self.subTest(fixture=name):
                exp = fx["expect"]
                self.assertEqual(
                    split_signature(fx["signature"]),
                    (exp["sig_version"], exp["sig_header_b64"],
                     exp["sig_cipher_len"]),
                )

    def test_versions_covered(self):
        """Fixtures span every top-level format version seen so far."""
        seen = {fx["expect"]["sig_version"] for fx in FIXTURES.values()}
        self.assertEqual(seen, {0, 2, 4})

    def test_garbage_returns_empty_triple_and_never_raises(self):
        for bad in GARBAGE:
            with self.subTest(value=bad):
                version, header, cipher_len = split_signature(bad)
                self.assertIsNone(header)
                self.assertIsInstance(version, int)
                self.assertEqual(cipher_len, 0)

    def test_truncated_blob_yields_no_header(self):
        """Half a real signature is not half a header: it is no header."""
        blob = FIXTURES["fable_v2"]["signature"]
        version, header, _ = split_signature(blob[:40])
        self.assertIsNone(header)

    def test_unpadded_base64_still_parses(self):
        """Transcripts store the blob without '=' padding often enough to
        matter; a stricter decoder loses those blocks silently."""
        for name in ("fable_v2", "opus5_v2", "carafe_v2"):
            with self.subTest(fixture=name):
                exp = FIXTURES[name]["expect"]
                stripped = FIXTURES[name]["signature"].rstrip("=")
                self.assertEqual(
                    split_signature(stripped),
                    (exp["sig_version"], exp["sig_header_b64"],
                     exp["sig_cipher_len"]),
                )

    def test_envelope_without_header_reports_no_cipher_len(self):
        """No header means nothing to store, so the triple carries no partial
        state either: (version, None, 0), never a stray length."""
        blob = _protobuf({1: 2}, envelope={5: b"\x00" * 4})
        self.assertEqual(split_signature(blob), (2, None, 0))


class DecodeHeaderTest(unittest.TestCase):

    def _decode(self, name):
        return decode_header(FIXTURES[name]["expect"]["sig_header_b64"])

    def test_every_fixture_header_decodes_as_expected(self):
        for name, fx in FIXTURES.items():
            with self.subTest(fixture=name):
                exp = fx["expect"]
                got = self._decode(name)
                self.assertEqual(got["served_model"], exp["served_model"])
                self.assertEqual(got["fields"], exp["sig_fields"])
                self.assertEqual(got["kind"], exp["kind"])
                self.assertEqual(got["tag"], exp["tag"])

    def test_v4_header_has_no_model_but_still_fingerprints(self):
        """The whole point of storing `fields`: a header that stopped naming
        the model still tells us WHICH shape it is."""
        got = self._decode("fable_v4")
        self.assertIsNone(got["served_model"])
        self.assertEqual(got["fields"], "1,3,7,8")

    def test_served_model_differs_from_requested_model(self):
        """The signal this feature exists for: asked for one model, served
        by another."""
        for name in ("kettle_v2", "carafe_v2"):
            with self.subTest(fixture=name):
                fx = FIXTURES[name]
                self.assertNotEqual(fx["expect"]["served_model"],
                                    fx["requested_model"])

    def test_kettle_carries_kind_and_tag(self):
        got = self._decode("kettle_v2")
        self.assertEqual(got["kind"], "narration")
        self.assertEqual(got["tag"], "MYCRO_MODEL_MANATEE")

    def test_fields_are_sorted_numerically_not_lexically(self):
        """'10' sorts before '2' as text; the field list must not."""
        self.assertEqual(self._decode("kettle_v2")["fields"],
                         "1,3,5,6,7,8,10,11,14,17")

    def test_garbage_returns_blank_and_never_raises(self):
        for bad in GARBAGE:
            with self.subTest(value=bad):
                got = decode_header(bad)
                self.assertIsNone(got["served_model"])
                self.assertIsNone(got["kind"])
                self.assertIsNone(got["tag"])
                self.assertIsInstance(got["fields"], str)

    def test_blank_input_has_empty_fields(self):
        """Empty `fields` is how ingest recognizes "not a header at all"."""
        for bad in (None, "", "not base64!!!", 12):
            with self.subTest(value=bad):
                self.assertEqual(decode_header(bad)["fields"], "")

    def test_full_blob_is_not_a_header(self):
        """Passing the whole signature where a header belongs must not
        accidentally decode: the envelope has no f6."""
        got = decode_header(FIXTURES["fable_v2"]["signature"])
        self.assertIsNone(got["served_model"])


if __name__ == "__main__":
    unittest.main()
