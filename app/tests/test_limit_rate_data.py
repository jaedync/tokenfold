"""SQL loaders for /api/limit-rate (app/limit_rate_data.py)."""

import unittest

from app.tests._support import TempDBTestCase

T = 1_791_000_000.0  # 2026-10-03 UTC


def insert_event(conn, uuid, ts, *, model="claude-opus-4-8", request_id=None,
                 message_id=None, source="claude-code", provider=None,
                 inp=0, out=0, cw=0, cr=0, reported_total=None,
                 email="me@example.test", plan="max", speed=None):
    conn.execute(
        "INSERT INTO events(uuid, type, timestamp, ts_epoch, day, source_machine, "
        "session_id, request_id, message_id, model, source_client, provider, "
        "input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens, "
        "reported_cost_total, account_email, plan, speed) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (uuid, "assistant", "2026-10-03T00:00:00Z", ts, "2026-10-03", "m1",
         "s-" + uuid, request_id or "req-" + uuid, message_id, model, source,
         provider, inp, out, cw, cr, reported_total, email, plan, speed))


def insert_reading(conn, bucket, t, pct, resets_at_epoch, source="meridian-oauth"):
    conn.execute(
        "INSERT INTO limit_readings(fetched_epoch, source, bucket, utilization, "
        "resets_at, resets_at_epoch) VALUES(?,?,?,?,?,?)",
        (t, source, bucket, pct, None, resets_at_epoch))


class FamilyTest(unittest.TestCase):

    def test_family_of(self):
        from app.limit_rate_data import family_of
        self.assertEqual(family_of("claude-opus-5-5"), "opus")
        self.assertEqual(family_of("claude-fable-5-1"), "fable")
        self.assertEqual(family_of("claude-sonnet-5-5"), "sonnet")
        self.assertEqual(family_of("claude-haiku-4-5-20251001"), "haiku")
        self.assertEqual(family_of("mystery"), "other")
        self.assertEqual(family_of(None), "other")


class LoadRequestsTest(TempDBTestCase):

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_cost_split_by_type_at_list_price(self):
        from app.limit_rate_data import load_requests
        # Opus 4.8 static price: 5 / 25 / 6.25 / 0.50 per MTok.
        insert_event(self.conn, "a", T + 10, inp=1_000_000, out=1_000_000,
                     cw=1_000_000, cr=1_000_000)
        self.conn.commit()
        rows = load_requests(self.conn, T, T + 3600)
        self.assertEqual(len(rows), 1)
        t, fam, parts = rows[0]
        self.assertEqual(fam, "opus")
        self.assertAlmostEqual(t, T + 10)
        for got, want in zip(parts, (5.0, 25.0, 6.25, 0.5)):
            self.assertAlmostEqual(got, want)

    def test_streaming_repeats_and_pi_twins_count_once(self):
        from app.limit_rate_data import load_requests
        # Two streaming chunks of one request: MAX, not SUM.
        insert_event(self.conn, "c1", T + 5, request_id="req1", out=100_000)
        insert_event(self.conn, "c2", T + 6, request_id="req1", out=200_000)
        # A Pi row with a positive reported cost and its SDK twin (same msg_):
        # the canonical predicate keeps only the Pi row.
        insert_event(self.conn, "pi", T + 20, source="pi-agent", provider="anthropic",
                     request_id="pi:x:msg_1", message_id="msg_1", out=400_000,
                     reported_total=10.0, email=None, plan=None)
        insert_event(self.conn, "sdk", T + 21, request_id="req_sdk",
                     message_id="msg_1", out=400_000)
        self.conn.commit()
        rows = load_requests(self.conn, T, T + 3600)
        total_out = sum(r[2][1] for r in rows)
        self.assertAlmostEqual(total_out, (200_000 + 400_000) * 25 / 1e6)

    def test_pi_reported_components_win(self):
        from app.limit_rate_data import load_requests
        insert_event(self.conn, "pi", T + 20, source="pi-agent", provider="anthropic",
                     request_id="pi:x:msg_7", message_id="msg_7", out=1_000_000,
                     cw=1_000_000, reported_total=40.0, email=None, plan=None)
        self.conn.execute("UPDATE events SET reported_cost_output=20, "
                          "reported_cost_cache_write=10 WHERE uuid='pi'")
        self.conn.commit()
        (t, fam, parts), = load_requests(self.conn, T, T + 3600)
        # Components give the split; the reported total wins (20+10 -> 40).
        self.assertAlmostEqual(parts[1], 40 * 20 / 30)
        self.assertAlmostEqual(parts[2], 40 * 10 / 30)
        self.assertAlmostEqual(sum(parts), 40.0)

    def test_pi_copies_on_two_machines_count_once(self):
        from app.limit_rate_data import load_requests
        # A Pi session file copied to a new machine is re-reported with a new
        # request_id but the same Anthropic message ID.
        for uuid, rid in (("p1", "pi:aaa:msg_9"), ("p2", "pi:bbb:msg_9")):
            insert_event(self.conn, uuid, T + 30, source="pi-agent", provider="anthropic",
                         request_id=rid, message_id="msg_9", out=1_000_000,
                         reported_total=25.0, email=None, plan=None)
        self.conn.commit()
        rows = load_requests(self.conn, T, T + 3600)
        self.assertAlmostEqual(sum(r[2][1] for r in rows), 25.0)

    def test_scope_and_provider_filters(self):
        from app.limit_rate_data import load_requests
        insert_event(self.conn, "work", T + 5, out=1_000_000,
                     email="me@corp.test", plan="enterprise")
        insert_event(self.conn, "codex", T + 6, source="pi-agent", provider="openai-codex",
                     model="gpt-5", out=1_000_000, reported_total=3.0)
        insert_event(self.conn, "old", T - 100, out=1_000_000)
        insert_event(self.conn, "ok", T + 7, out=1_000_000)
        self.conn.commit()
        rows = load_requests(self.conn, T, T + 3600)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0][2][1], 25.0)


class LoadReadingsTest(TempDBTestCase):

    def test_active_source_only_sorted_and_deduped(self):
        from app.limit_rate_data import load_readings
        insert_reading(self.conn, "seven_day", T + 200, 5.0, T + 9e5)
        insert_reading(self.conn, "seven_day", T + 100, 4.0, T + 9e5)
        insert_reading(self.conn, "seven_day", T + 100.2, 4.5, T + 9e5)
        insert_reading(self.conn, "seven_day", T + 150, 99.0, T + 9e5, source="client")
        insert_reading(self.conn, "five_hour", T + 120, 1.0, T + 9e3)
        self.conn.commit()
        rows = load_readings(self.conn, "seven_day", T, "meridian-oauth")
        self.assertEqual(rows, [(T + 100.2, 4.5, T + 9e5), (T + 200, 5.0, T + 9e5)])

    def test_granted_resets(self):
        from app.limit_rate_data import load_granted
        for i, pct in enumerate([60.0, 61.0, 62.0, 0.0, 1.0, 2.0]):
            insert_reading(self.conn, "seven_day", T + i * 600, pct, T + 5 * 86400)
        self.conn.commit()
        self.assertEqual(load_granted(self.conn, T - 1, "meridian-oauth"), [T + 1800])

    def test_natural_weekly_rollover_is_not_a_grant(self):
        from app.limit_rate_data import load_granted
        end = T + 1000
        for i, (pct, reset) in enumerate([(80.0, end), (81.0, end), (0.0, end + 7 * 86400),
                                          (1.0, end + 7 * 86400)]):
            insert_reading(self.conn, "seven_day", T + i * 600, pct, reset)
        self.conn.commit()
        self.assertEqual(load_granted(self.conn, T - 1, "meridian-oauth"), [])


if __name__ == "__main__":
    unittest.main()
