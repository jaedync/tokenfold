"""Regression tests for the 2026-07-12 daily_summary wipe incident.

Root cause: summarize_days ran DELETE(days) → minutes of accumulate → INSERT
→ commit on the shared cross-thread connection with no lock. A concurrent
summarize (ingest re-rolls touched days on every batch) inserted rows inside
that window; the sweep's INSERT hit UNIQUE(day, account_email), aborted
mid-rebuild, and the already-visible DELETE was persisted by other threads'
commits — months of rollup rows vanished while raw events stayed intact.
The sweep wrappers swallowed the exception, so nothing was logged.

These tests pin the fix:
  * accumulate happens BEFORE the destructive write, and the DELETE+INSERT
    runs as one serialized atomic phase (interleaved writers can't abort it,
    failures can't leave the table half-rebuilt or emptied);
  * db.write_txn() serializes writers and owns commit/rollback so one
    thread's rollback can never destroy another thread's half-done write;
  * sweep timers and the drain worker LOG failures instead of hiding them;
  * WAL is bounded (journal_size_limit) and checkpointable on demand.
"""
import os
import threading
import time

from app.tests._support import TempDBTestCase


def _ins(conn, uuid, req, acct, day="2026-06-09", ts=1781000000.0,
         inp=1_000_000, machine="m", session=None):
    conn.execute(
        "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,request_id,"
        "source_machine,project_dir,model,is_sidechain,input_tokens,output_tokens,"
        "cache_creation_tokens,cache_read_tokens,account_email) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (uuid, "assistant", "2026-06-09T12:00:00Z", ts, day,
         session or "s-" + uuid, req, machine, "proj", "claude-opus-4-8",
         0, inp, 0, 0, 0, acct))
    conn.commit()


class InterleavedWriteTest(TempDBTestCase):
    """The incident itself: a concurrent writer lands a summary row while a
    rebuild is accumulating. The rebuild must complete and win — not abort
    on UNIQUE and leave the table gutted."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_concurrent_insert_mid_accumulate_does_not_abort_rebuild(self):
        _ins(self.conn, "u1", "r1", "a@x", day="2026-06-09")
        _ins(self.conn, "u2", "r2", "b@y", day="2026-06-10")

        import app.summarizer as s
        real = s._accumulate
        injected = {"done": False}

        def patched(conn_, days, placeholders, account):
            if not injected["done"]:
                injected["done"] = True
                # Simulate a concurrent summarize_days committing a row for a
                # target day while this rebuild is still accumulating.
                self.conn.execute(
                    "INSERT INTO daily_summary(day, account_email, updated_at) "
                    "VALUES('2026-06-09', 'a@x', 'injected')")
                self.conn.commit()
            return real(conn_, days, placeholders, account)

        s._accumulate = patched
        self.addCleanup(setattr, s, "_accumulate", real)

        from app.summarizer import summarize_days
        summarize_days(["2026-06-09", "2026-06-10"])  # must not raise

        rows = {(r["day"], r["account_email"]): r for r in self.conn.execute(
            "SELECT day, account_email, cost, updated_at FROM daily_summary")}
        self.assertEqual(set(rows), {("2026-06-09", "a@x"),
                                     ("2026-06-10", "b@y")})
        # The rebuild's row must have won over the injected stub.
        self.assertNotEqual(rows[("2026-06-09", "a@x")]["updated_at"],
                            "injected")
        self.assertGreater(rows[("2026-06-09", "a@x")]["cost"], 0)

    def test_failed_rebuild_preserves_existing_rows(self):
        _ins(self.conn, "u1", "r1", "a@x", day="2026-06-09")
        from app.summarizer import summarize_days
        summarize_days(["2026-06-09"])
        self.assertIsNotNone(self.conn.execute(
            "SELECT 1 FROM daily_summary WHERE day='2026-06-09'").fetchone())

        import app.summarizer as s
        real = s._accumulate

        def boom(conn_, days, placeholders, account):
            raise RuntimeError("accumulate blew up")

        s._accumulate = boom
        self.addCleanup(setattr, s, "_accumulate", real)

        with self.assertRaises(RuntimeError):
            summarize_days(["2026-06-09"])

        # A failed rebuild must leave yesterday's rollup intact — the incident
        # deleted first and never got to re-insert.
        row = self.conn.execute(
            "SELECT cost FROM daily_summary WHERE day='2026-06-09' "
            "AND account_email='a@x'").fetchone()
        self.assertIsNotNone(row)
        self.assertGreater(row["cost"], 0)


class WriteTxnTest(TempDBTestCase):
    """db.write_txn(): one writer at a time, commit/rollback owned by the
    holder so cross-thread commit/rollback bleed is impossible."""

    def test_commits_on_success(self):
        from app.db import write_txn
        with write_txn() as conn:
            conn.execute("INSERT INTO meta(key, value) VALUES('k', 'v')")
        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(self.conn.execute(
            "SELECT value FROM meta WHERE key='k'").fetchone()["value"], "v")

    def test_rolls_back_on_exception(self):
        from app.db import write_txn
        with self.assertRaises(RuntimeError):
            with write_txn() as conn:
                conn.execute("INSERT INTO meta(key, value) VALUES('k2', 'v')")
                raise RuntimeError("boom")
        self.assertFalse(self.conn.in_transaction)
        self.assertIsNone(self.conn.execute(
            "SELECT value FROM meta WHERE key='k2'").fetchone())

    def test_serializes_concurrent_writers(self):
        from app.db import write_txn
        order = []
        entered = threading.Event()

        def slow_writer():
            with write_txn() as conn:
                conn.execute("INSERT INTO meta(key, value) VALUES('a', '1')")
                entered.set()
                time.sleep(0.2)
                order.append("A-exit")

        def fast_writer():
            entered.wait(2.0)
            with write_txn() as conn:
                order.append("B-enter")
                conn.execute("INSERT INTO meta(key, value) VALUES('b', '1')")

        ta = threading.Thread(target=slow_writer)
        tb = threading.Thread(target=fast_writer)
        ta.start()
        tb.start()
        ta.join(5.0)
        tb.join(5.0)
        self.assertEqual(order, ["A-exit", "B-enter"])


class SweepLoggingTest(TempDBTestCase):
    """Sweep timers must not swallow failures silently — the incident's full
    sweep died invisibly behind `except Exception: pass`."""

    def test_full_sweep_logs_failure_and_does_not_raise(self):
        import app.aggregator as agg
        import app.summarizer as s
        real_sum, real_sched = s.summarize_days, agg._schedule_full_sweep

        def boom(days=None):
            raise RuntimeError("sweep boom")

        s.summarize_days = boom
        agg._schedule_full_sweep = lambda: None
        self.addCleanup(setattr, s, "summarize_days", real_sum)
        self.addCleanup(setattr, agg, "_schedule_full_sweep", real_sched)

        with self.assertLogs("app.aggregator", level="ERROR") as cm:
            agg._run_full_sweep()  # must not raise
        self.assertTrue(any("sweep" in m.lower() for m in cm.output))

    def test_periodic_sweep_logs_failure_and_does_not_raise(self):
        import app.aggregator as agg
        import app.summarizer as s
        real_sum, real_sched = s.summarize_days, agg._schedule_periodic_sweep
        saved_last_full = agg._last_full_sweep

        def boom(days=None):
            raise RuntimeError("sweep boom")

        s.summarize_days = boom
        agg._schedule_periodic_sweep = lambda: None
        agg._last_full_sweep = 0.0  # ensure the sweep body runs
        self.addCleanup(setattr, s, "summarize_days", real_sum)
        self.addCleanup(setattr, agg, "_schedule_periodic_sweep", real_sched)
        self.addCleanup(setattr, agg, "_last_full_sweep", saved_last_full)

        with self.assertLogs("app.aggregator", level="ERROR") as cm:
            agg._run_periodic_sweep()  # must not raise
        self.assertTrue(any("sweep" in m.lower() for m in cm.output))


class DrainWorkerLoggingTest(TempDBTestCase):
    """The drain worker deliberately drops a failed iteration (next
    invalidation retries) — but it must say so in the log."""

    def test_build_error_is_logged(self):
        import app.aggregator as agg
        real = agg._build_dashboard_data_inner

        def boom(scope):
            raise RuntimeError("build boom")

        agg._build_dashboard_data_inner = boom
        self.addCleanup(setattr, agg, "_build_dashboard_data_inner", real)

        with self.assertLogs("app.aggregator", level="ERROR") as cm:
            agg.trigger_eager_rebuild()
            deadline = time.time() + 5.0
            while time.time() < deadline:
                with agg._cache_lock:
                    if not agg._rebuilding:
                        break
                time.sleep(0.01)
        self.assertTrue(any("rebuild" in m.lower() for m in cm.output))


class WalMaintenanceTest(TempDBTestCase):
    """WAL must be bounded: journal_size_limit at connect, and an on-demand
    truncating checkpoint for the hourly sweep."""

    def test_journal_size_limit_configured(self):
        val = self.conn.execute("PRAGMA journal_size_limit").fetchone()[0]
        self.assertGreater(val, 0)
        self.assertLessEqual(val, 256 * 1024 * 1024)

    def test_checkpoint_wal_truncates(self):
        from app.db import checkpoint_wal
        for i in range(200):
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?)",
                (f"pad-{i}", "x" * 512))
        self.conn.commit()
        wal_path = self.db_path + "-wal"
        self.assertTrue(os.path.exists(wal_path))
        self.assertGreater(os.path.getsize(wal_path), 0)

        checkpoint_wal()

        self.assertEqual(os.path.getsize(wal_path), 0)
