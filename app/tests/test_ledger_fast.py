"""The fast billable scan (one-level grouping for Pi rows) gives the same
groups and the same ledger as the exact two-level scan (app/ledger_query.py)."""
from unittest.mock import patch

from app.tests.test_ledger import WORK, LedgerTestBase

KEY = ("session_id", "agent_id", "model", "provider", "source_client", "speed",
       "inference_geo")
VALUES = ("requests", "first_ts", "last_ts", "inp", "outp", "cc", "cr", "c5m", "c1h",
          "ws", "reported_input", "reported_output", "reported_cache_read",
          "reported_cache_write", "reported_total", "machine", "project_dir", "org_type")


def table(rows):
    out = {}
    for r in rows:
        key = tuple(r[k] for k in KEY) + ((r["era"],) if "era" in r.keys() else ())
        assert key not in out, key
        out[key] = {k: r[k] for k in VALUES}
    return out


class FastBillableTest(LedgerTestBase):
    def setUp(self):
        super().setUp()
        t = self.t0
        # Claude Code: streaming repeats one request three times; a subagent.
        self.cc("cc-1.jsonl", [
            self.cc_user("u1", "cc-1", t),
            self.cc_reply("a1", "cc-1", t + 10, "r1", inp=100),
            self.cc_reply("a1b", "cc-1", t + 11, "r1", inp=300),
            self.cc_reply("a1c", "cc-1", t + 12, "r1", inp=200),
            self.cc_reply("a2", "cc-1", t + 40, "r2", model="claude-sonnet-4-6"),
            self.cc_user("su1", "cc-1", t + 50, agent="agent-a", human=False),
            self.cc_reply("sa1", "cc-1", t + 60, "r3", agent="agent-a"),
            self.cc_reply("sa1b", "cc-1", t + 61, "r3", agent="agent-a", inp=5)])
        self.cc("cc-w.jsonl", [
            self.cc_user("wu", "cc-w", t + 5),
            self.cc_reply("wa", "cc-w", t + 6, "rw"),
            self.cc_reply("wa2", "cc-w", t + 7, "rw")], account=WORK)
        # Meridian twin of a Pi row.
        self.cc("twin.jsonl", [
            self.cc_user("tw-u", "twin", t),
            self.cc_reply("tw-a", "twin", t + 1, "r-twin", msg_id="msg_twin")])
        self.pi("bucket/2026-10-14T00-00-00-000Z_P1.jsonl", [
            self.pi_event("p-u", "P1", t, kind="user"),
            self.pi_event("p-a", "P1", t + 2, cost=2.0, request_id="msg_twin"),
            self.pi_event("p-b", "P1", t + 900, cost=1.0),
            self.pi_event("p-c", "P1", t + 950, cost=0.0, model="claude-sonnet-4-6"),
            self.pi_event("p-d", "P1", t + 990, cost=0.5, provider="openai", model="gpt-5.5")])
        # A Pi session that repeats one request id: the duplicate guard path.
        self.pi("bucket/2026-10-14T00-00-00-000Z_P2.jsonl", [
            self.pi_event("q-u", "P2", t + 30, kind="user"),
            self.pi_event("q-a", "P2", t + 31, cost=1.5, request_id="dup"),
            self.pi_event("q-b", "P2", t + 32, cost=2.5, request_id="dup"),
            self.pi_event("q-c", "P2", t + 33, cost=0.25)])
        self.pi("subagents/P1/2026-10-14T00-10-00-000Z_be-ledger_aaaa1111.jsonl", [
            self.pi_event("c-a", "C1", t + 60, cost=4.0, model="claude-sonnet-4-6",
                          agent="be-ledger"),
            self.pi_event("c-b", "C1", t + 120, cost=0.5, agent="be-ledger")])
        self.pi("bucket/2026-10-14T00-00-00-000Z_W1.jsonl", [
            self.pi_event("w-a", "W1", t + 70, cost=3.0)], account_class="work")

    def groups(self, fn, scope="personal", lo=None, hi=None, **kw):
        from app.config import scope_predicate
        from app.db import read_conn
        lo = self.t0 - 60 if lo is None else lo
        hi = self.t0 + 3600 if hi is None else hi
        with read_conn() as conn:
            return fn(conn, scope_predicate(scope), lo, hi, **kw)

    def assertSameGroups(self, fast, exact):
        fast, exact = table(fast), table(exact)
        self.assertEqual(set(fast), set(exact))
        for key, want in exact.items():
            got = fast[key]
            for k, v in want.items():
                if isinstance(v, float):
                    self.assertAlmostEqual(got[k], v, places=9, msg=(key, k))
                else:
                    self.assertEqual(got[k], v, (key, k))

    def test_fast_scan_matches_the_exact_scan_in_both_scopes(self):
        from app.ledger_query import billable_groups, billable_groups_exact
        for scope in ("personal", "enterprise"):
            with self.subTest(scope=scope):
                exact = self.groups(billable_groups_exact, scope)
                self.assertTrue(exact)
                self.assertSameGroups(self.groups(billable_groups, scope), exact)

    def test_repeated_pi_request_is_counted_once(self):
        from app.ledger_query import billable_groups
        rows = [r for r in self.groups(billable_groups) if r["session_id"].endswith(":P2")]
        by_cost = sorted((r["requests"], r["reported_total"]) for r in rows)
        # The exact scan keeps MAX per request: $2.50 for "dup", then $0.25.
        self.assertEqual(by_cost, [(2, 2.75)])

    def test_session_filter_and_partial_ranges_match(self):
        from app.ledger_query import billable_groups, billable_groups_exact
        exact = self.groups(billable_groups_exact)
        ids = sorted({r["session_id"] for r in exact})[:2]
        self.assertSameGroups(self.groups(billable_groups, session_ids=ids),
                              self.groups(billable_groups_exact, session_ids=ids))
        lo, hi = self.t0 + 11, self.t0 + 100
        self.assertSameGroups(self.groups(billable_groups, lo=lo, hi=hi),
                              self.groups(billable_groups_exact, lo=lo, hi=hi))

    def test_a_pricing_era_inside_the_range_uses_the_exact_scan(self):
        from app.ledger_query import billable_groups, billable_groups_exact
        with patch("app.ledger_query.era_boundaries", return_value=[self.t0 + 11.5]):
            self.assertSameGroups(self.groups(billable_groups),
                                  self.groups(billable_groups_exact))

    def test_ledger_payload_is_unchanged(self):
        import app.ledger as ledger
        import app.ledger_days as ledger_days
        import app.ledger_query as lq
        fast = self.ledger()
        ledger._cache._entries.clear()
        ledger_days._days._entries.clear()
        with patch.object(ledger_days, "billable_groups", lq.billable_groups_exact):
            exact = self.ledger()
        self.assertEqual(len(fast["sessions"]), len(exact["sessions"]))
        self.assertAlmostEqual(fast["totals"]["cost_usd"], exact["totals"]["cost_usd"], places=9)
        for a, b in zip(fast["sessions"], exact["sessions"]):
            self.assertEqual(a["id"], b["id"])
            self.assertAlmostEqual(a["cost_usd"], b["cost_usd"], places=9)
            self.assertEqual(a["tokens"], b["tokens"])
            self.assertEqual({k: v for k, v in a.items() if k not in ("cost_usd", "orch_cost_usd", "agent_cost_usd", "orch_models", "runs")},
                             {k: v for k, v in b.items() if k not in ("cost_usd", "orch_cost_usd", "agent_cost_usd", "orch_models", "runs")})
