"""Audit follow-ups from the 2026-07-12 incident review.

1. A summary-rebuild failure AFTER events are durably committed must not 500
   the ingest: the client would retry, dedupe to accepted=0, and the touched
   days would stay stale until the next sweep. Events are the truth; the
   rollup self-heals — log and return the counts.
2. Token counts come from untrusted transcript JSON. sqlite3 binds a dict/list
   with InterfaceError (failing the whole batch) and silently STORES a str in
   the INTEGER column (poisoning SUM() aggregation) — coerce at extract time
   like service_tier/speed already do.
"""
from app.tests._support import TempDBTestCase


def _rec(uuid, **usage_extra):
    u = {"input_tokens": 5, "output_tokens": 5}
    u.update(usage_extra)
    return {
        "uuid": uuid,
        "type": "assistant",
        "timestamp": "2026-06-09T12:00:00Z",
        "sessionId": "s1",
        "requestId": "r-" + uuid,
        "message": {"model": "claude-opus-4-8", "id": "m-" + uuid, "usage": u},
    }


def _body(*events):
    return {
        "machine": "m",
        "project_dir": "p",
        "session_file": "s.jsonl",
        "cursor": {"last_line_num": 0},
        "events": list(events),
    }


class SummaryFailureDoesNotFailIngestTest(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_ingest_returns_200_when_rebuild_raises(self):
        import app.summarizer as s
        real = s.summarize_days

        def boom(days=None):
            raise RuntimeError("rebuild boom")

        s.summarize_days = boom
        self.addCleanup(setattr, s, "summarize_days", real)

        c = self.client()
        with self.assertLogs("app.ingest", level="ERROR"):
            r = c.post("/api/ingest", json=_body(_rec("u1")),
                       headers={"X-API-Key": self.api_key})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["accepted"], 1)
        # The events must be durable regardless of the rebuild failure.
        n = self.conn.execute(
            "SELECT COUNT(*) FROM events WHERE uuid='u1'").fetchone()[0]
        self.assertEqual(n, 1)


class TokenFieldCoercionTest(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def _ingest(self, *events):
        c = self.client()
        return c.post("/api/ingest", json=_body(*events),
                      headers={"X-API-Key": self.api_key})

    def test_dict_token_count_does_not_fail_batch(self):
        r = self._ingest(_rec("bad", input_tokens={"nested": 1}),
                         _rec("good"))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["accepted"], 2)
        row = self.conn.execute(
            "SELECT input_tokens FROM events WHERE uuid='bad'").fetchone()
        self.assertEqual(row["input_tokens"], 0)

    def test_string_token_count_coerced_to_zero(self):
        r = self._ingest(_rec("s1e", output_tokens="loads"))
        self.assertEqual(r.status_code, 200, r.text)
        row = self.conn.execute(
            "SELECT output_tokens FROM events WHERE uuid='s1e'").fetchone()
        self.assertEqual(row["output_tokens"], 0)

    def test_bool_and_negative_rejected_valid_int_kept(self):
        r = self._ingest(_rec("b1", input_tokens=True, output_tokens=-7,
                              cache_read_input_tokens=123))
        self.assertEqual(r.status_code, 200, r.text)
        row = self.conn.execute(
            "SELECT input_tokens, output_tokens, cache_read_tokens "
            "FROM events WHERE uuid='b1'").fetchone()
        self.assertEqual(row["input_tokens"], 0)
        self.assertEqual(row["output_tokens"], 0)
        self.assertEqual(row["cache_read_tokens"], 123)
