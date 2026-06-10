"""Tests for config-driven, fail-closed enterprise classification.

Covers:
  - ENTERPRISE_ORG_UUIDS env var: a row with a matching org_uuid classifies enterprise
  - ENTERPRISE_EMAIL_DOMAINS env var: a row matching an email domain classifies enterprise
  - Env empty → those rows classify personal (fail-closed)
  - SQL injection safety: single-quote in env value must not break query execution
  - End-to-end: ingest org_type/org_uuid via POST, summarize, assert classification

Reload-safety approach:
  importlib.reload(app.config) runs inside mock.patch.dict(os.environ, ...) so the
  module-level predicate builder sees the patched env during import. scope_predicate()
  reads module-level ENTERPRISE_PRED/PERSONAL_PRED from app.config's __dict__, which
  reload replaces in-place on the same module object — so any existing 'from .config
  import scope_predicate' references in other modules also see the new values.
  Teardown reloads again with a clean env to restore the original state.
"""

import importlib
import json
import os
import sqlite3
import unittest
from unittest import mock

from app.tests._support import TempDBTestCase


# ── helpers ──────────────────────────────────────────────────────────────────

def _insert_with_org(conn, uuid, req, acct, plan, org, org_type=None, org_uuid=None,
                     machine="m", project="proj", session="s",
                     model="claude-haiku-4-5", day="2026-06-09",
                     ts=1781000000.0, inp=1_000_000):
    conn.execute(
        "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,request_id,"
        "source_machine,project_dir,model,is_sidechain,agent_id,input_tokens,"
        "output_tokens,cache_creation_tokens,cache_read_tokens,account_email,plan,"
        "org_name,org_type,org_uuid,is_human_prompt,user_type) VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (uuid, "assistant", "2026-06-09T12:00:00Z", ts, day, session, req, machine,
         project, model, 0, None, inp, 0, 0, 0, acct, plan, org, org_type, org_uuid,
         0, None),
    )
    conn.commit()


def _reload_config_with_env(env_patch: dict):
    """Reload app.config with the given env vars patched in. Returns the module."""
    import app.config as cfg
    with mock.patch.dict(os.environ, env_patch, clear=False):
        importlib.reload(cfg)
    return cfg


def _restore_config():
    """Reload app.config with no extra env vars to restore default state."""
    import app.config as cfg
    importlib.reload(cfg)
    return cfg


# ── tests ─────────────────────────────────────────────────────────────────────

class OrgUuidEnvTest(TempDBTestCase):
    """ENTERPRISE_ORG_UUIDS env var controls org_uuid-based classification."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def tearDown(self):
        # Restore config to default so subsequent tests aren't affected.
        _restore_config()
        super().tearDown()

    def test_org_uuid_classifies_enterprise_when_env_set(self):
        """A row whose org_uuid matches ENTERPRISE_ORG_UUIDS classifies enterprise."""
        test_uuid = "test-org-uuid-0001"
        _insert_with_org(
            self.conn, "e1", "r1", "user@corp.io", None, "Corp",
            org_type=None, org_uuid=test_uuid, machine="m1")
        # Also a personal row without the uuid
        _insert_with_org(
            self.conn, "e2", "r2", "per@home.io", None, None,
            org_type=None, org_uuid=None, machine="m2", ts=1781000100.0)

        # Reload config with the uuid in the env var
        _reload_config_with_env({"ENTERPRISE_ORG_UUIDS": test_uuid})

        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data.clear()

        ent = agg.build_dashboard_data("enterprise")
        per = agg.build_dashboard_data("personal")

        self.assertAlmostEqual(ent["total_cost"], 1.0, places=2,
                               msg=f"org_uuid-matched row must be in enterprise, got {ent['total_cost']}")
        self.assertAlmostEqual(per["total_cost"], 1.0, places=2,
                               msg=f"non-matched row must be in personal, got {per['total_cost']}")
        # Verify partition
        row = self.conn.execute("SELECT SUM(cost) FROM daily_summary").fetchone()
        blended = round(row[0] or 0.0, 2)
        self.assertAlmostEqual(ent["total_cost"] + per["total_cost"], blended, places=2)

    def test_org_uuid_personal_when_env_empty(self):
        """Same org_uuid row classifies personal when ENTERPRISE_ORG_UUIDS is not set."""
        _insert_with_org(
            self.conn, "e1", "r1", "user@corp.io", None, "Corp",
            org_type=None, org_uuid="some-uuid", machine="m1")

        # Reload with empty / unset uuid list
        _reload_config_with_env({"ENTERPRISE_ORG_UUIDS": ""})

        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data.clear()

        ent = agg.build_dashboard_data("enterprise")
        per = agg.build_dashboard_data("personal")

        self.assertAlmostEqual(ent["total_cost"], 0.0, places=2,
                               msg="enterprise must be $0 when no uuid matches")
        self.assertAlmostEqual(per["total_cost"], 1.0, places=2,
                               msg="row must fall through to personal (fail-closed)")


class EmailDomainEnvTest(TempDBTestCase):
    """ENTERPRISE_EMAIL_DOMAINS env var controls email-domain-based classification."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def tearDown(self):
        _restore_config()
        super().tearDown()

    def test_email_domain_classifies_enterprise(self):
        """A row whose email matches ENTERPRISE_EMAIL_DOMAINS classifies enterprise."""
        _insert_with_org(
            self.conn, "e1", "r1", "jane@acme.io", None, None,
            org_type=None, org_uuid=None, machine="m1")
        _insert_with_org(
            self.conn, "e2", "r2", "bob@personal.com", None, None,
            org_type=None, org_uuid=None, machine="m2", ts=1781000100.0)

        _reload_config_with_env({"ENTERPRISE_EMAIL_DOMAINS": "acme.io"})

        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data.clear()

        ent = agg.build_dashboard_data("enterprise")
        per = agg.build_dashboard_data("personal")

        self.assertAlmostEqual(ent["total_cost"], 1.0, places=2,
                               msg=f"acme.io email must classify enterprise, got {ent['total_cost']}")
        self.assertAlmostEqual(per["total_cost"], 1.0, places=2,
                               msg=f"personal email must stay personal, got {per['total_cost']}")

    def test_email_domain_with_at_prefix_stripped(self):
        """@acme.io (with @ prefix) must work the same as acme.io."""
        _insert_with_org(
            self.conn, "e1", "r1", "jane@acme.io", None, None,
            org_type=None, org_uuid=None, machine="m1")

        _reload_config_with_env({"ENTERPRISE_EMAIL_DOMAINS": "@acme.io"})

        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data.clear()

        ent = agg.build_dashboard_data("enterprise")
        self.assertAlmostEqual(ent["total_cost"], 1.0, places=2,
                               msg="@ prefix stripped; acme.io email must classify enterprise")

    def test_like_wildcard_domain_is_ignored(self):
        """A domain containing SQL LIKE wildcards ('%.io') must NOT match arbitrary
        emails: '%' is a live LIKE wildcard, so without sanitization '%.io' would
        classify EVERY *.io personal account as enterprise. The config builder must
        skip such entries (fail-closed -> row stays personal)."""
        _insert_with_org(
            self.conn, "e1", "r1", "someone@personal.io", None, None,
            org_type=None, org_uuid=None, machine="m1")

        _reload_config_with_env({"ENTERPRISE_EMAIL_DOMAINS": "%.io"})

        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data.clear()

        ent = agg.build_dashboard_data("enterprise")
        per = agg.build_dashboard_data("personal")
        self.assertAlmostEqual(
            ent["total_cost"], 0.0, places=2,
            msg="'%.io' wildcard domain must NOT classify someone@personal.io as enterprise")
        self.assertAlmostEqual(
            per["total_cost"], 1.0, places=2,
            msg="row must remain personal when the only signal is a wildcard domain")


class SqlInjectionSafetyTest(TempDBTestCase):
    """Single quotes in env values must not break predicate SQL execution."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def tearDown(self):
        _restore_config()
        super().tearDown()

    def test_single_quote_in_org_type_env(self):
        """ENTERPRISE_ORG_TYPES with a single-quote value must not raise at query time."""
        _insert_with_org(
            self.conn, "e1", "r1", "user@x.io", None, None,
            org_type="o'brien", org_uuid=None, machine="m1")

        _reload_config_with_env({"ENTERPRISE_ORG_TYPES": "o'brien"})

        from app.summarizer import summarize_days
        import app.aggregator as agg
        # Must not raise sqlite3.OperationalError or any exception
        try:
            summarize_days(None)
            agg._cached_data.clear()
            ent = agg.build_dashboard_data("enterprise")
            per = agg.build_dashboard_data("personal")
        except Exception as exc:
            self.fail(f"single-quote in env must not cause SQL error: {exc}")

        # Regardless of result, ent + per = blended must hold
        row = self.conn.execute("SELECT SUM(cost) FROM daily_summary").fetchone()
        blended = round(row[0] or 0.0, 2)
        combined = round(ent["total_cost"] + per["total_cost"], 2)
        self.assertAlmostEqual(combined, blended, places=2,
                               msg="partition must still hold even with quoted env value")

    def test_single_quote_in_org_uuid_env(self):
        """ENTERPRISE_ORG_UUIDS with a single-quote value must not raise."""
        _insert_with_org(
            self.conn, "e1", "r1", "user@x.io", None, None,
            org_type=None, org_uuid="uuid'danger", machine="m1")

        _reload_config_with_env({"ENTERPRISE_ORG_UUIDS": "uuid'danger"})

        from app.summarizer import summarize_days
        import app.aggregator as agg
        try:
            summarize_days(None)
            agg._cached_data.clear()
            agg.build_dashboard_data("enterprise")
            agg.build_dashboard_data("personal")
        except Exception as exc:
            self.fail(f"single-quote in ENTERPRISE_ORG_UUIDS must not cause SQL error: {exc}")


class IngestOrgTypeUuidEndToEndTest(TempDBTestCase):
    """End-to-end: POST /api/ingest with org_type/org_uuid; summarize; verify
    daily_summary carries the fields and classification works."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_ingest_and_summarize_carry_org_type_org_uuid(self):
        """org_type and org_uuid from the ingest payload are stored in events and
        stamped into daily_summary; a row with org_type='claude_enterprise' classifies
        enterprise through build_dashboard_data."""
        event = {
            "uuid": "test-evt-001",
            "type": "assistant",
            "timestamp": "2026-06-09T12:00:00Z",
            "sessionId": "sess-001",
            "requestId": "req-001",
            "message": {
                "model": "claude-haiku-4-5",
                "id": "msg-001",
                "usage": {"input_tokens": 1_000_000, "output_tokens": 0},
            },
        }
        body = {
            "machine": "test-machine",
            "project_dir": "test-proj",
            "session_file": "test.jsonl",
            "cursor": {"last_line_num": 0},
            "events": [event],
            "account_email": "test@enterprise.io",
            "org_name": "TestCorp",
            "plan": None,
            "rate_limit_tier": None,
            "org_type": "claude_enterprise",
            "org_uuid": "test-org-uuid-9999",
        }
        c = self.client()
        r = c.post("/api/ingest", json=body, headers={"X-API-Key": self.api_key})
        self.assertEqual(r.status_code, 200, r.text)

        # Verify events row carries org_type and org_uuid
        ev_row = self.conn.execute(
            "SELECT org_type, org_uuid FROM events WHERE uuid='test-evt-001'"
        ).fetchone()
        self.assertIsNotNone(ev_row, "event row must exist after ingest")
        self.assertEqual(ev_row["org_type"], "claude_enterprise")
        self.assertEqual(ev_row["org_uuid"], "test-org-uuid-9999")

        # Summarize and verify daily_summary carries them
        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data.clear()

        ds_row = self.conn.execute(
            "SELECT org_type, org_uuid FROM daily_summary "
            "WHERE account_email='test@enterprise.io'"
        ).fetchone()
        self.assertIsNotNone(ds_row, "daily_summary row must exist after summarize")
        self.assertEqual(ds_row["org_type"], "claude_enterprise",
                         "daily_summary must carry org_type")
        self.assertEqual(ds_row["org_uuid"], "test-org-uuid-9999",
                         "daily_summary must carry org_uuid")

        # Classification: org_type='claude_enterprise' → enterprise scope (default config)
        ent = agg.build_dashboard_data("enterprise")
        self.assertAlmostEqual(ent["total_cost"], 1.0, places=2,
                               msg="org_type=claude_enterprise must classify as enterprise via default config")


if __name__ == "__main__":
    unittest.main()
