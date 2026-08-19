"""GET /api/served-models: grouped served-model capture, personal scope only.

Gated exactly like /api/limit-history (dashboard auth, neutral 404 for
enterprise scope and for an enterprise-locked instance) so an instance that
must never surface personal data does not surface this either.
"""

import json
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from app.config import TZ_NAME
from app.tests._support import TempDBTestCase

FIXTURES = json.loads(
    (Path(__file__).resolve().parent / "fixtures"
     / "thinking_signatures.json").read_text())

TZ = ZoneInfo(TZ_NAME)
FABLE = FIXTURES["fable_v2"]["expect"]
KETTLE = FIXTURES["kettle_v2"]["expect"]
V4 = FIXTURES["fable_v4"]["expect"]


def _day(offset=0):
    return (datetime.now(TZ) + timedelta(days=offset)).strftime("%Y-%m-%d")


class ServedModelsApiTest(TempDBTestCase):

    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        self._n = 0

    def _ins(self, day, model, exp=None, count=1, personal=True):
        """Insert `count` thinking events for one (day, model, header).

        exp is a fixture `expect` block; exp=None inserts UNSIGNED events
        (every signature column NULL), which is what pre-feature history and
        every non-thinking turn look like.
        """
        header = exp["sig_header_b64"] if exp else None
        served = exp["served_model"] if exp else None
        version = exp["sig_version"] if exp else None
        fields = exp["sig_fields"] if exp else None
        clen = exp["sig_cipher_len"] if exp else None
        for _ in range(count):
            self._n += 1
            uuid = f"e{self._n}"
            self.conn.execute(
                "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,"
                "request_id,source_machine,project_dir,model,is_sidechain,"
                "agent_id,input_tokens,output_tokens,cache_creation_tokens,"
                "cache_read_tokens,has_thinking,is_human_prompt,account_email,"
                "plan,served_model,sig_version,sig_header,sig_cipher_len,"
                "sig_fields) VALUES(?,'assistant',?,1781000000.0,?,'s1',?,'m1',"
                "'proj',?,0,NULL,1,1,0,0,1,0,?,?,?,?,?,?,?)",
                (uuid, day + "T12:00:00Z", day, "r-" + uuid, model,
                 "me@gmail.com" if personal else "me@acme.io",
                 "max" if personal else "enterprise",
                 served, version, header, clen, fields),
            )
        self.conn.commit()

    def _get(self, qs="scope=personal"):
        return self.client().get(f"/api/served-models?{qs}")

    # ── grouping ──────────────────────────────────────────────────────────

    def test_groups_by_day_model_served_model(self):
        day = _day()
        self._ins(day, "claude-fable-5", FABLE, count=3)
        self._ins(day, "claude-fable-5", KETTLE, count=2)
        r = self._get()
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["days"], 30)
        by_served = {row["served_model"]: row for row in body["rows"]}
        self.assertEqual(set(by_served), {FABLE["served_model"],
                                          KETTLE["served_model"]})
        self.assertEqual(by_served[FABLE["served_model"]]["blocks"], 3)
        self.assertEqual(by_served[KETTLE["served_model"]]["blocks"], 2)
        row = by_served[KETTLE["served_model"]]
        self.assertEqual(row["day"], day)
        self.assertEqual(row["model"], "claude-fable-5")
        self.assertEqual(row["sig_version"], KETTLE["sig_version"])
        self.assertEqual(row["sig_fields"], KETTLE["sig_fields"])
        self.assertEqual(row["cipher_bytes"], 2 * KETTLE["sig_cipher_len"])

    def test_null_served_model_rows_are_included(self):
        """The v4 share is the point: hiding unnamed blocks would make the
        named ones look like the whole population."""
        day = _day()
        self._ins(day, "claude-fable-5", FABLE, count=1)
        self._ins(day, "claude-fable-5", V4, count=4)
        rows = self._get().json()["rows"]
        nulls = [r for r in rows if r["served_model"] is None]
        self.assertEqual(len(nulls), 1)
        self.assertEqual(nulls[0]["blocks"], 4)
        self.assertEqual(nulls[0]["sig_version"], 4)
        self.assertEqual(nulls[0]["sig_fields"], "1,3,7,8")

    def test_days_window_excludes_older_rows(self):
        self._ins(_day(), "claude-fable-5", KETTLE, count=1)
        self._ins(_day(-40), "claude-fable-5", FABLE, count=1)
        rows = self._get("days=30&scope=personal").json()["rows"]
        self.assertEqual([r["served_model"] for r in rows],
                         [KETTLE["served_model"]])
        wide = self._get("days=90&scope=personal").json()["rows"]
        self.assertEqual(len(wide), 2)

    def test_days_is_clamped_never_an_error(self):
        for qs, expect in (("days=0", 1), ("days=-3", 1), ("days=9999", 400)):
            with self.subTest(qs=qs):
                r = self._get(qs + "&scope=personal")
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual(r.json()["days"], expect)

    def test_unsigned_events_are_not_listed(self):
        self._ins(_day(), "claude-fable-5", None, count=3)
        self.assertEqual(self._get().json()["rows"], [])

    def test_enterprise_rows_never_leak_into_personal(self):
        day = _day()
        self._ins(day, "claude-fable-5", KETTLE, count=2, personal=False)
        self._ins(day, "claude-fable-5", FABLE, count=1, personal=True)
        rows = self._get().json()["rows"]
        self.assertEqual([r["served_model"] for r in rows],
                         [FABLE["served_model"]])

    # ── gating ────────────────────────────────────────────────────────────

    def test_enterprise_scope_is_404(self):
        self.assertEqual(self._get("scope=enterprise").status_code, 404)

    def test_invalid_scope_is_400(self):
        self.assertEqual(self._get("scope=bogus").status_code, 400)

    def test_enterprise_locked_instance_is_404(self):
        import app.config as cfg
        saved = cfg.LOCKED_SCOPE
        cfg.LOCKED_SCOPE = "enterprise"
        self.addCleanup(setattr, cfg, "LOCKED_SCOPE", saved)
        self.assertEqual(self._get("scope=personal").status_code, 404)
        self.assertEqual(self._get("").status_code, 404)

    def test_requires_dashboard_auth(self):
        import app.config as cfg
        saved = (cfg.DASHBOARD_USER, cfg.DASHBOARD_PASSWORD)
        cfg.DASHBOARD_USER, cfg.DASHBOARD_PASSWORD = "admin", "s3cret"

        def _restore():
            cfg.DASHBOARD_USER, cfg.DASHBOARD_PASSWORD = saved
        self.addCleanup(_restore)

        c = self.client()
        self.assertEqual(c.get("/api/served-models?scope=personal").status_code,
                         401)
        ok = c.get("/api/served-models?scope=personal",
                   auth=("admin", "s3cret"))
        self.assertEqual(ok.status_code, 200, ok.text)


if __name__ == "__main__":
    unittest.main()
