"""Tests for scope-partitioned aggregator: enterprise + personal are a TRUE partition.

Keystone: ent_total + per_total == blended_total, and the NULL-plan row C lands in
personal (not lost) — proving the COALESCE guard against SQL 3-valued-logic NULL trap.
"""

import json
import time
import unittest

from app.tests._support import TempDBTestCase


def ins(conn, uuid, req, acct, plan, org, machine, project, session,
        model="claude-opus-4-8", day="2026-06-09", ts=1781000000.0, inp=0, out=0):
    conn.execute(
        "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,request_id,"
        "source_machine,project_dir,model,is_sidechain,agent_id,input_tokens,"
        "output_tokens,cache_creation_tokens,cache_read_tokens,account_email,plan,"
        "org_name,is_human_prompt,user_type) VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (uuid, "assistant", "2026-06-09T12:00:00Z", ts, day, session, req, machine,
         project, model, 0, None, inp, out, 0, 0, acct, plan, org, 0, None))
    conn.commit()


def ins_null_plan(conn, uuid, req, machine, project, session,
                  model="claude-haiku-4-5", day="2026-06-09", ts=1781000200.0, inp=0):
    """Insert a row with NULL plan, NULL account_email, NULL org — the NULL-trap test."""
    conn.execute(
        "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,request_id,"
        "source_machine,project_dir,model,is_sidechain,agent_id,input_tokens,"
        "output_tokens,cache_creation_tokens,cache_read_tokens,account_email,plan,"
        "org_name,is_human_prompt,user_type) VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (uuid, "assistant", "2026-06-09T12:00:00Z", ts, day, session, req, machine,
         project, model, 0, None, inp, 0, 0, 0, None, None, None, 0, None))
    conn.commit()


class PartitionCompletenessTest(TempDBTestCase):
    """The keystone test: ent + per = blended, NULL row lands in personal (not lost)."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        # Reset scope cache so scopes don't bleed between tests
        import app.aggregator as agg
        agg._cached_data.clear()

    def tearDown(self):
        import app.aggregator as agg
        agg._cached_data.clear()
        super().tearDown()

    def test_partition_completeness_with_null_row(self):
        """Seed A (enterprise $5), B (max plan $3), C (NULL plan $1).
        Totals: ent=5, per=4, blended=9. C must land in personal (not lost).
        """
        from app.summarizer import summarize_days
        import app.aggregator as agg

        # A: enterprise, Opus 4.8, 1M input = $5
        ins(self.conn, "eA", "rA", "a@acme.io", "enterprise", "Acme",
            "mA", "proj-a", "sA", model="claude-opus-4-8", inp=1_000_000, ts=1781000000.0)
        # B: max plan (personal), Sonnet 4.6, 1M input = $3
        ins(self.conn, "eB", "rB", "b@personal.io", "max", None,
            "mB", "proj-b", "sB", model="claude-sonnet-4-6", inp=1_000_000, ts=1781000100.0)
        # C: NULL plan, NULL account, NULL org (the SQL 3-valued-logic trap)
        ins_null_plan(self.conn, "eC", "rC", "mC", "proj-c", "sC",
                      model="claude-haiku-4-5", inp=1_000_000, ts=1781000200.0)

        summarize_days(None)
        agg._cached_data.clear()

        ent = agg.build_dashboard_data("enterprise")
        per = agg.build_dashboard_data("personal")

        # Blended total from DB
        row = self.conn.execute("SELECT SUM(cost) as t FROM daily_summary").fetchone()
        blended = round(row["t"] or 0.0, 2)

        # Partition correctness
        self.assertAlmostEqual(ent["total_cost"], 5.0, places=2,
                               msg=f"enterprise must be $5, got {ent['total_cost']}")
        self.assertAlmostEqual(per["total_cost"], 4.0, places=2,
                               msg=f"personal must be $4 (B $3 + C $1), got {per['total_cost']}")
        self.assertAlmostEqual(blended, 9.0, places=2,
                               msg=f"blended must be $9, got {blended}")
        self.assertAlmostEqual(ent["total_cost"] + per["total_cost"], blended, places=2,
                               msg="ent + per must equal blended (partition completeness)")

        # NULL row C must land in personal (not lost)
        # C contributes $1 to personal; without C personal would be $3
        # We've already checked per==4.0 which proves C landed in personal

    def test_scope_key_in_payload(self):
        """Both scopes must include 'scope' key in their payload, no org_name key."""
        from app.summarizer import summarize_days
        import app.aggregator as agg

        ins(self.conn, "eA", "rA", "a@acme.io", "enterprise", "Acme",
            "mA", "proj-a", "sA", inp=1_000_000)
        summarize_days(None)
        agg._cached_data.clear()

        ent = agg.build_dashboard_data("enterprise")
        per = agg.build_dashboard_data("personal")

        self.assertEqual(ent["scope"], "enterprise")
        self.assertEqual(per["scope"], "personal")
        self.assertNotIn("org_name", ent, "org_name must not be in payload")
        self.assertNotIn("org_name", per, "org_name must not be in payload")

    def test_no_cross_bleedover_enterprise_to_personal(self):
        """Enterprise payload must not contain personal account/machine strings."""
        from app.summarizer import summarize_days
        import app.aggregator as agg

        ins(self.conn, "eA", "rA", "a@acme.io", "enterprise", "Acme",
            "acme-hpc1", "proj-a", "sA", inp=1_000_000, ts=1781000000.0)
        ins(self.conn, "eB", "rB", "b@personal.io", "max", None,
            "personal-mbp", "secret-proj", "sB", inp=1_000_000, ts=1781000100.0)
        ins_null_plan(self.conn, "eC", "rC", "null-machine", "null-proj", "sC",
                      inp=1_000_000, ts=1781000200.0)

        summarize_days(None)
        agg._cached_data.clear()

        ent_blob = json.dumps(agg.build_dashboard_data("enterprise"))
        per_blob = json.dumps(agg.build_dashboard_data("personal"))

        # Enterprise payload must not contain personal/null strings
        self.assertNotIn("personal-mbp", ent_blob)
        self.assertNotIn("secret-proj", ent_blob)
        self.assertNotIn("null-machine", ent_blob)
        self.assertNotIn("null-proj", ent_blob)

        # Personal payload must not contain enterprise strings
        self.assertNotIn("acme-hpc1", per_blob)
        self.assertNotIn("a@acme.io", per_blob)

    def test_cache_isolation_between_scopes(self):
        """build_dashboard_data('enterprise') and ('personal') must return different totals."""
        from app.summarizer import summarize_days
        import app.aggregator as agg

        # A: enterprise, Opus 4.8 = $5/M
        ins(self.conn, "eA", "rA", "a@acme.io", "enterprise", "Acme",
            "mA", "proj-a", "sA", model="claude-opus-4-8", inp=1_000_000, ts=1781000000.0)
        # B: personal, Haiku 4.5 = $1/M (different cost so scopes are distinguishable)
        ins(self.conn, "eB", "rB", "b@personal.io", "max", None,
            "mB", "proj-b", "sB", model="claude-haiku-4-5-20251001", inp=1_000_000, ts=1781000100.0)

        summarize_days(None)
        agg._cached_data.clear()

        ent = agg.build_dashboard_data("enterprise")
        per = agg.build_dashboard_data("personal")

        # Calling enterprise first must not pollute the personal cache slot
        self.assertNotAlmostEqual(ent["total_cost"], per["total_cost"], places=2,
                                  msg="enterprise and personal must return different totals")

        # Second call to enterprise must return same value (cache hit)
        ent2 = agg.build_dashboard_data("enterprise")
        self.assertEqual(ent["total_cost"], ent2["total_cost"])

    def test_empty_payload_has_scope_key(self):
        """_empty_dashboard must include 'scope' key, not org_name."""
        from app.aggregator import _empty_dashboard
        emp = _empty_dashboard("2026-06-01", "personal")
        self.assertEqual(emp["scope"], "personal")
        self.assertNotIn("org_name", emp)


if __name__ == "__main__":
    unittest.main()
