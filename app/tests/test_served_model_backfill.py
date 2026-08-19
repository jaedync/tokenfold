"""POST /api/backfill: fill in signature headers for events already stored.

Every event ingested before this feature has NULL signature columns while the
transcripts on disk still carry the blobs. The backfill repairs those rows
under the same fill-only-unset contract as the cache-tier and server-tool
maps: a row that already has a header is never rewritten.
"""

import json
import unittest
from pathlib import Path

from app.tests._support import TempDBTestCase

FIXTURES = json.loads(
    (Path(__file__).resolve().parent / "fixtures"
     / "thinking_signatures.json").read_text())

FABLE = FIXTURES["fable_v2"]["expect"]
KETTLE = FIXTURES["kettle_v2"]["expect"]


def _triple(exp):
    """The wire shape: [sig_version, sig_header_b64, sig_cipher_len]."""
    return [exp["sig_version"], exp["sig_header_b64"], exp["sig_cipher_len"]]


class SigHeaderBackfillTest(TempDBTestCase):

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def _ins(self, uuid, day="2026-06-09", **cols):
        base = {
            "served_model": None, "sig_version": None, "sig_header": None,
            "sig_cipher_len": None, "sig_fields": None,
        }
        base.update(cols)
        self.conn.execute(
            "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,"
            "request_id,source_machine,project_dir,model,is_sidechain,agent_id,"
            "input_tokens,output_tokens,cache_creation_tokens,cache_read_tokens,"
            "has_thinking,is_human_prompt,served_model,sig_version,sig_header,"
            "sig_cipher_len,sig_fields) "
            "VALUES(?,'assistant',?,1781000000.0,?,'s1',?,'m1','proj',"
            "'claude-fable-5',0,NULL,1,1,0,0,1,0,?,?,?,?,?)",
            (uuid, day + "T12:00:00Z", day, "r-" + uuid,
             base["served_model"], base["sig_version"], base["sig_header"],
             base["sig_cipher_len"], base["sig_fields"]),
        )
        self.conn.commit()

    def _post(self, sig_headers):
        c = self.client()
        return c.post("/api/backfill", json={"sig_headers": sig_headers},
                      headers={"X-API-Key": self.api_key})

    def _sig(self, uuid):
        return self.conn.execute(
            "SELECT served_model, sig_version, sig_header, sig_cipher_len, "
            "sig_fields FROM events WHERE uuid=?", (uuid,)).fetchone()

    def test_unset_row_is_filled_and_counted(self):
        self._ins("u1")
        r = self._post({"u1": _triple(KETTLE)})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["updated_sig_headers"], 1)
        row = self._sig("u1")
        self.assertEqual(row["served_model"], KETTLE["served_model"])
        self.assertEqual(row["sig_version"], KETTLE["sig_version"])
        self.assertEqual(row["sig_header"], KETTLE["sig_header_b64"])
        self.assertEqual(row["sig_cipher_len"], KETTLE["sig_cipher_len"])
        self.assertEqual(row["sig_fields"], KETTLE["sig_fields"])

    def test_second_backfill_does_not_overwrite(self):
        """Fill-only-unset: re-running a backfill (or running an older one
        after a newer) must never rewrite a header that is already there."""
        self._ins("u1")
        first = self._post({"u1": _triple(KETTLE)})
        self.assertEqual(first.json()["updated_sig_headers"], 1)

        second = self._post({"u1": _triple(FABLE)})
        self.assertEqual(second.json()["updated_sig_headers"], 0)
        row = self._sig("u1")
        self.assertEqual(row["served_model"], KETTLE["served_model"])
        self.assertEqual(row["sig_header"], KETTLE["sig_header_b64"])

    def test_unknown_uuid_ignored(self):
        r = self._post({"ghost": _triple(FABLE)})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["updated_sig_headers"], 0)

    def test_malformed_entries_skipped_without_error(self):
        """Bad triples are skipped one by one; the good ones in the same
        batch still land."""
        self._ins("u2")
        self._ins("u3")
        self._ins("u4")
        self._ins("ok")
        r = self._post({
            "u2": [2, FABLE["sig_header_b64"]],          # wrong arity
            "u3": [2, "!!! not base64 !!!", 10],
            "u4": [2, {"header": "nope"}, 10],
            "ok": _triple(FABLE),
        })
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["updated_sig_headers"], 1)
        for uuid in ("u2", "u3", "u4"):
            with self.subTest(uuid=uuid):
                self.assertIsNone(self._sig(uuid)["sig_header"])
        self.assertEqual(self._sig("ok")["served_model"], FABLE["served_model"])

    def test_non_list_value_rejected_at_the_boundary(self):
        """Schema-level validation, same as the other backfill maps: a value
        that is not a list fails fast with 422 rather than being ignored."""
        r = self._post({"u1": "not a list"})
        self.assertEqual(r.status_code, 422, r.text)

    def test_out_of_range_numbers_coerced_not_stored(self):
        self._ins("u1")
        self._post({"u1": [-5, FABLE["sig_header_b64"], 10 ** 15]})
        row = self._sig("u1")
        self.assertEqual(row["sig_version"], 0)
        self.assertEqual(row["sig_cipher_len"], 0)
        self.assertEqual(row["served_model"], FABLE["served_model"])

    def test_sig_only_backfill_does_not_reroll_days(self):
        """No rollup reads served_model (the chip and the API both query
        events), so a signature-only backfill must not pay for a re-roll."""
        self._ins("u1")
        r = self._post({"u1": _triple(FABLE)})
        self.assertEqual(r.json()["touched_days"], [])

    def test_other_backfill_maps_still_work_alongside(self):
        self._ins("u1")
        c = self.client()
        r = c.post("/api/backfill", json={
            "sig_headers": {"u1": _triple(FABLE)},
            "titles": {"sessA": "a title"},
        }, headers={"X-API-Key": self.api_key})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["updated_sig_headers"], 1)
        self.assertEqual(body["updated_titles"], 1)

    def test_requires_api_key(self):
        c = self.client()
        r = c.post("/api/backfill", json={"sig_headers": {}})
        self.assertEqual(r.status_code, 401)


if __name__ == "__main__":
    unittest.main()
