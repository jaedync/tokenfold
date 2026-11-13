"""Coalesced daily_summary re-roll after ingest (app.rollup_queue).

Production, 2026-11-13: each ingest batch (about one each 4 s) re-rolled today
in the request thread, about 1.4 s each, so the re-roll used about a third of
a core. These tests pin the coalescing, the durable day list and the shutdown.
"""

import random
import threading
import time
import unittest
from unittest.mock import patch

import app.db as db
import app.rollup_queue as rq
from app.tests._support import TempDBTestCase


def _wait_idle(timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with rq._lock:
            if rq._thread is None:
                return True
        time.sleep(0.005)
    return False


class _QueueCase(TempDBTestCase):
    INTERVAL = 0.3

    def setUp(self):
        super().setUp()
        self.rolls: list[tuple[float, list[str]]] = []
        self.roll_lock = threading.Lock()
        self.fail_next = 0
        self.block = None  # threading.Event the fake roll waits on

        def fake_roll(days):
            start = time.monotonic()
            with self.roll_lock:
                self.rolls.append((start, list(days)))
                fail = self.fail_next > 0
                if fail:
                    self.fail_next -= 1
            if self.block is not None:
                self.block.wait(5.0)
            if fail:
                raise RuntimeError("roll failed (test)")

        p1 = patch.object(rq, "_roll", fake_roll)
        p2 = patch.object(rq, "ROLLUP_MIN_INTERVAL_S", self.INTERVAL)
        p1.start(); p2.start()

        def stop():
            if self.block is not None:
                self.block.set()
            rq.join(5.0)
            p2.stop(); p1.stop()
        self.addCleanup(stop)

    def queued(self):
        return db.queued_rollup_days(db.get_conn())


class RollupQueueTest(_QueueCase):
    def test_zero_interval_rolls_in_the_calling_thread(self):
        rq.ROLLUP_MIN_INTERVAL_S = 0.0
        rq.request({"2026-11-12"})
        self.assertEqual([d for _, d in self.rolls], [["2026-11-12"]])
        self.assertIsNone(rq._thread)
        self.assertEqual(self.queued(), set())

    def test_first_request_after_idle_rolls_at_once(self):
        t0 = time.monotonic()
        rq.request({"2026-11-12"})
        self.assertTrue(_wait_idle())
        self.assertEqual(len(self.rolls), 1)
        self.assertLess(self.rolls[0][0] - t0, self.INTERVAL / 2)
        self.assertEqual(self.queued(), set())

    def test_steady_requests_coalesce_to_one_roll_per_interval(self):
        end = time.monotonic() + 1.2
        n = 0
        while time.monotonic() < end:
            rq.request({"2026-11-12", f"2026-11-{n % 3 + 1:02d}"})
            n += 1
            time.sleep(0.02)
        self.assertTrue(_wait_idle())
        starts = [s for s, _ in self.rolls]
        gaps = [b - a for a, b in zip(starts, starts[1:])]
        # 1.2 s / 0.3 s: rolls at about 0, 0.3, 0.6, 0.9 and 1.2 s, plus the trailing one.
        self.assertLessEqual(len(self.rolls), 6, starts)
        self.assertGreater(n, 30)
        self.assertTrue(all(g >= self.INTERVAL * 0.95 for g in gaps), gaps)
        rolled = set().union(*(set(d) for _, d in self.rolls))
        self.assertEqual(rolled, {"2026-11-12", "2026-11-01", "2026-11-02", "2026-11-03"})
        self.assertEqual(self.queued(), set())

    def wait_rolls(self, n, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.roll_lock:
                if len(self.rolls) >= n:
                    return True
            time.sleep(0.005)
        return False

    def test_days_wait_in_the_durable_list_until_rolled(self):
        rq.request({"2026-11-10"})
        self.assertTrue(self.wait_rolls(1))  # the worker now waits its interval
        self.block = threading.Event()
        rq.request({"2026-11-11"})  # inside the interval: waits
        self.assertIn("2026-11-11", self.queued())
        self.assertEqual(len(self.rolls), 1)
        self.block.set()
        self.assertTrue(_wait_idle())
        self.assertEqual(self.rolls[-1][1], ["2026-11-11"])
        self.assertEqual(self.queued(), set())

    def test_day_requested_again_during_its_roll_stays_listed_and_rolls_again(self):
        self.block = threading.Event()
        rq.request({"2026-11-12"})
        deadline = time.monotonic() + 2
        while not self.rolls and time.monotonic() < deadline:
            time.sleep(0.005)
        rq.request({"2026-11-12"})  # new events after the roll began
        self.block.set()
        time.sleep(self.INTERVAL / 3)  # first roll done, second not yet started
        self.assertEqual(len(self.rolls), 1)
        self.assertIn("2026-11-12", self.queued(), "dropped before its second roll")
        self.assertTrue(_wait_idle())
        self.assertEqual([d for _, d in self.rolls], [["2026-11-12"], ["2026-11-12"]])
        self.assertEqual(self.queued(), set())

    def test_failed_roll_keeps_the_days_and_retries(self):
        self.fail_next = 1
        with self.assertLogs("app.rollup_queue", "ERROR"):
            rq.request({"2026-11-12"})
            self.assertTrue(_wait_idle(timeout=3.0))
        self.assertEqual([d for _, d in self.rolls], [["2026-11-12"], ["2026-11-12"]])
        self.assertGreaterEqual(self.rolls[1][0] - self.rolls[0][0], self.INTERVAL * 0.95)
        self.assertEqual(self.queued(), set())

    def test_shutdown_ends_the_wait_and_keeps_the_days_for_startup(self):
        rq.request({"2026-11-10"})
        self.assertTrue(_wait_idle())
        rq.ROLLUP_MIN_INTERVAL_S = 30.0
        rq.request({"2026-11-11"})
        time.sleep(0.05)
        db.begin_shutdown()
        try:
            t0 = time.monotonic()
            rq.join(5.0)
            self.assertLess(time.monotonic() - t0, 2.0)
            self.assertIsNone(rq._thread)
            self.assertEqual(len(self.rolls), 1, "no roll after the shutdown started")
        finally:
            db._shutdown.clear()
        self.assertIn("2026-11-11", self.queued())

    def test_startup_rolls_the_days_left_in_the_list(self):
        import app.main as main
        with db.write_txn() as conn:
            db.queue_rollup_days(conn, {"2026-11-11"})
            # No full-rebuild marker (a new database has one) and a non-empty
            # daily_summary, so startup re-rolls only the listed days.
            conn.execute("DELETE FROM meta WHERE key='daily_summary_rebuild_pending'")
            conn.execute("INSERT INTO daily_summary(day, account_email, updated_at) "
                         "VALUES('2026-11-01', 'a', '2026-11-01T00:00:00')")
        calls = []
        with patch("app.summarizer.summarize_days", lambda days: calls.append(list(days))):
            main._backfill_if_needed()
        self.assertEqual(calls, [["2026-11-11"]])
        self.assertEqual(self.queued(), set())


class RollupQueuePropertyTest(_QueueCase):
    """Random request orders from 4 threads, with slow and failing rolls:
    every day is rolled by a roll that began after its last request, and
    the durable list is empty at the end."""

    INTERVAL = 0.03

    def test_every_request_is_covered_by_a_later_roll(self):
        for seed in range(6):
            with self.subTest(seed=seed):
                self._one_run(random.Random(seed))

    def _one_run(self, rnd):
        self.rolls.clear()
        last_request: dict[str, float] = {}
        req_lock = threading.Lock()
        days = [f"2026-11-{i:02d}" for i in range(1, 6)]

        def client(seed):
            r = random.Random(seed)
            for _ in range(25):
                chosen = set(r.sample(days, r.randint(1, 3)))
                with req_lock:
                    now = time.monotonic()
                    for d in chosen:
                        last_request[d] = now
                rq.request(chosen)
                time.sleep(r.random() * 0.012)

        self.fail_next = rnd.randint(0, 2)
        with self.assertNoLogs("app.rollup_queue", "CRITICAL"):
            threads = [threading.Thread(target=client, args=(rnd.random(),)) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertTrue(_wait_idle(timeout=10.0))
        for d, t_req in last_request.items():
            later = [s for s, ds in self.rolls if d in ds and s >= t_req]
            self.assertTrue(later, f"{d} requested at {t_req} never rolled after it")
        self.assertEqual(self.queued(), set())


class RollupEnvTest(unittest.TestCase):
    def _parse(self, raw):
        with patch.dict("os.environ", {}, clear=False) as env:
            env.pop("TOKENFOLD_ROLLUP_MIN_INTERVAL_S", None)
            if raw is not None:
                env["TOKENFOLD_ROLLUP_MIN_INTERVAL_S"] = raw
            return rq._interval_from_env()

    def test_values(self):
        self.assertEqual(self._parse(None), 30.0)
        self.assertEqual(self._parse("0"), 0.0)
        self.assertEqual(self._parse("7.5"), 7.5)
        for raw in ("x", "-3", "nan", "inf"):
            with self.subTest(raw=raw), self.assertLogs("app.rollup_queue", "WARNING"):
                self.assertEqual(self._parse(raw), 30.0)


class IngestUsesTheQueueTest(TempDBTestCase):
    """Both batch routes hand their days to the queue and answer without a
    re-roll in the request thread when the interval is set."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        self.requested = []
        p1 = patch.object(rq, "request", lambda days: self.requested.append(set(days)))
        p1.start()
        self.addCleanup(p1.stop)
        self.rolled = []
        p2 = patch("app.summarizer.summarize_days", lambda days=None: self.rolled.append(days))
        p2.start()
        self.addCleanup(p2.stop)

    def _today(self):
        from datetime import datetime
        from app.config import TZ_NAME
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(TZ_NAME)).strftime("%Y-%m-%d")

    def test_claude_ingest_requests_touched_days_and_today(self):
        body = {"machine": "m", "project_dir": "p", "session_file": "s.jsonl",
                "cursor": {"last_line_num": 0}, "events": [{
                    "uuid": "u1", "type": "assistant", "timestamp": "2026-06-09T12:00:00Z",
                    "sessionId": "s1", "requestId": "r-u1",
                    "message": {"model": "claude-opus-4-8", "id": "m-u1",
                                "usage": {"input_tokens": 5, "output_tokens": 5}}}]}
        r = self.client().post("/api/ingest", json=body, headers={"X-API-Key": self.api_key})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(len(self.requested), 1)
        self.assertIn(self._today(), self.requested[0])
        self.assertEqual(self.rolled, [], "no re-roll in the request thread")

    def test_pi_ingest_requests_touched_days_and_today(self):
        body = {"machine": "m", "account_class": "personal", "project_dir": "/p",
                "session_file": "pi.jsonl", "cursor": {"last_line_num": 0}, "events": [{
                    "event_id": "e1", "timestamp": "2026-06-09T12:00:00Z",
                    "session_id": "s", "kind": "assistant", "provider": "openai",
                    "api": "responses", "model": "gpt-4o", "request_id": "req",
                    "usage": {"input": 10, "output": 20, "cost_total": 1.0}}]}
        r = self.client().post("/api/ingest/pi", json=body, headers={"X-API-Key": self.api_key})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(len(self.requested), 1)
        self.assertTrue({self._today(), "2026-06-09"} <= self.requested[0], self.requested)
        self.assertEqual(self.rolled, [], "no re-roll in the request thread")


if __name__ == "__main__":
    unittest.main()
