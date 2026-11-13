"""Last active time per machine without a full scan of events.

Production, 2026-11-13: the dashboard build's GROUP BY source_machine with
MAX(ts_epoch) scanned every event and ran the activity subqueries per row:
2.8 s of each personal build on the production copy. Per machine, the newest
row of the (source_machine, ts_epoch) index that passes the filters gives the
same answer: 0.18 s on that copy.
"""

import random
import sqlite3

import app.aggregator as agg
import app.db as db
from app.config import scope_predicate
from app.event_dedupe import counts_as_activity
from app.tests._support import TempDBTestCase


def _old(conn, pred):
    return {r[0]: r[1] for r in conn.execute(
        f"SELECT source_machine, MAX(ts_epoch) FROM events "
        f"WHERE {pred} AND {counts_as_activity()} GROUP BY source_machine") if r[1]}


class MachineLastActiveTest(TempDBTestCase):
    def _insert(self, rows):
        with db.write_txn() as conn:
            for i, (machine, ts, client, sid, typ, email, plan, copy) in enumerate(rows):
                conn.execute(
                    "INSERT INTO events(uuid, type, timestamp, ts_epoch, day, source_machine, "
                    "source_client, session_id, account_email, plan, pi_copy, model, request_id) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (f"u{i}", typ, "2026-11-12T00:00:00Z", ts, "2026-11-12", machine,
                     client, sid, email, plan, copy, "claude-opus-5-5", f"r{i}"))

    def _random_rows(self, rnd, n=600):
        machines = ["mac", "Mac.local", "op01", "ms01", "htz", "lonely"]
        rows = []
        for _ in range(n):
            client = rnd.choice(["claude-code", "claude-code", "pi-agent", "claude-desktop"])
            rows.append((
                rnd.choice(machines), rnd.uniform(1.7e9, 1.8e9), client,
                rnd.choice([f"s{k}" for k in range(25)] + [None]),
                rnd.choice(["user", "assistant", "system"]),
                rnd.choice(["", "unknown", "me@example.com", "work@example.com"]),
                rnd.choice([None, "max", "enterprise"]),
                rnd.choice([0, 0, 0, 1]),
            ))
        # Claude Code sessions with no reply (not activity), the newest rows.
        for k in range(8):
            rows.append(("lonely" if k % 2 else "op01", 1.8e9 + k, "claude-code",
                         f"solo{k}", "user", "", None, 0))
        return rows

    def test_same_answer_as_the_grouped_scan(self):
        filtered_out = 0
        for seed in range(8):
            with self.subTest(seed=seed):
                with db.write_txn() as conn:
                    conn.execute("DELETE FROM events")
                    conn.execute("DELETE FROM pi_mirror_sessions")
                rnd = random.Random(seed)
                self._insert(self._random_rows(rnd))
                with db.write_txn() as conn:
                    for k in rnd.sample(range(25), 4):
                        conn.execute("INSERT OR IGNORE INTO pi_mirror_sessions(session_id) VALUES(?)",
                                     (f"s{k}",))
                with db.read_conn() as conn:
                    for scope in ("personal", "enterprise"):
                        pred = scope_predicate(scope)
                        self.assertEqual(dict(agg._machine_last_active(conn, pred)),
                                         _old(conn, pred), scope)
                        unfiltered = dict(conn.execute(
                            "SELECT source_machine, MAX(ts_epoch) FROM events "
                            "GROUP BY source_machine").fetchall())
                        filtered_out += sum(1 for m, ts in _old(conn, pred).items()
                                            if ts != unfiltered.get(m))
        # The filters really drop newer rows, so the check is not trivial.
        self.assertGreater(filtered_out, 16)

    def test_per_machine_lookup_uses_the_index(self):
        self._insert(self._random_rows(random.Random(1), n=50))
        with db.read_conn() as conn:
            plan = " ".join(r[3] for r in conn.execute(
                "EXPLAIN QUERY PLAN " + agg._LAST_ACTIVE_PER_MACHINE_SQL.format(
                    pred=scope_predicate("personal"), activity=counts_as_activity()),
                ("op01",)))
        self.assertIn("idx_events_source_ts", plan)
        self.assertNotIn("SCAN events", plan.replace("SCAN events USING INDEX", "").replace(
            "SCAN events USING COVERING INDEX", ""))

    def test_migration_replaces_the_single_column_index(self):
        db.close_conn()
        raw = sqlite3.connect(self.db_path)
        raw.execute("CREATE INDEX IF NOT EXISTS idx_events_source ON events(source_machine)")
        raw.commit()
        raw.close()
        conn = db.open_conn()
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        self.assertIn("idx_events_source_ts", names)
        self.assertNotIn("idx_events_source", names)


if __name__ == "__main__":
    import unittest
    unittest.main()
