import json
import time
from datetime import datetime, timezone
from pathlib import Path

from app.tests._support import TempDBTestCase

FIXTURE_PATH = (Path(__file__).resolve().parent
                / "fixtures" / "oauth_usage_live_2026-07-01.json")


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


class EnterpriseOnlyGateTest(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_zero_consumer_bleedover(self):
        # Enterprise: 1M input Opus 4.8 = $5
        ins(self.conn, "e1", "re1", "jaedyn@acme.io", "enterprise", "Acme",
            "acme-hpc1", "acme-portal", "sE", inp=1_000_000)
        # Consumer: 2M input = $10 — must NEVER appear anywhere
        ins(self.conn, "c1", "rc1", "me@gmail.com", "max", None,
            "personal-mbp", "secret-side-project", "sC", inp=2_000_000,
            ts=1781000100.0)
        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data.clear()
        d = agg.build_dashboard_data()
        blob = json.dumps(d)
        # ZERO consumer bleedover: none of these strings anywhere in the payload
        self.assertNotIn("me@gmail.com", blob)
        self.assertNotIn("personal-mbp", blob)
        self.assertNotIn("secret-side-project", blob)
        self.assertNotIn('"max"', blob)  # consumer plan label shouldn't surface
        # Enterprise IS present, and totals are enterprise-only ($5, not blended $15)
        self.assertIn("acme-hpc1", blob)
        self.assertAlmostEqual(sum(m["cost"] for m in d["model_breakdown"]), 5.0, places=2)
        self.assertAlmostEqual(sum(x["cost"] for x in d["daily"]), 5.0, places=2)

    def test_no_enterprise_data_is_empty_not_crash(self):
        ins(self.conn, "c1", "rc1", "me@gmail.com", "max", None,
            "personal-mbp", "proj", "sC", inp=1_000_000)
        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data.clear()
        d = agg.build_dashboard_data()  # only consumer data -> enterprise view empty, no crash
        self.assertAlmostEqual(sum(x["cost"] for x in d["daily"]), 0.0, places=2)
        self.assertNotIn("me@gmail.com", json.dumps(d))

    def _seed_oauth_usage(self, resets_epoch):
        """Insert an oauth_usage meta row so /api/rate-limits and /api/ha
        populate their windowed spend (both need seven_day.resets_at)."""
        resets_iso = datetime.fromtimestamp(
            resets_epoch, tz=timezone.utc).isoformat()
        stored = {
            "data": {
                "seven_day": {"utilization": 50, "resets_at": resets_iso},
                "five_hour": {"utilization": 50, "resets_at": resets_iso},
            },
            "updated_at": resets_iso,
        }
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES('oauth_usage', ?)",
            (json.dumps(stored),))
        self.conn.commit()

    def _seed_live_fixture(self):
        """Seed meta.oauth_usage with the EXACT live prod payload — its
        limits[] carries a weekly_scoped 'Fable' entry, so the suppression
        tests below exercise real bucket data (mirrors test_oauth_gating)."""
        usage = json.loads(FIXTURE_PATH.read_text())
        stored = {"data": usage, "updated_at": "2026-07-01T12:00:00+00:00"}
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('oauth_usage', ?)",
            (json.dumps(stored),))
        self.conn.commit()

    def test_routes_exclude_consumer_spend(self):
        # Events in the CURRENT week window (rolling-7d window; no oauth row needed).
        now = time.time()
        # enterprise: 1M input Opus 4.8 = $5
        ins(self.conn, "e1", "re1", "jaedyn@acme.io", "enterprise", "Acme",
            "acme-hpc1", "acme-portal", "sE", inp=1_000_000, ts=now - 3600)
        # personal/consumer: 2M input Opus 4.8 = $10
        ins(self.conn, "c1", "rc1", "me@gmail.com", "max", None,
            "personal-mbp", "secret", "sC", inp=2_000_000, ts=now - 1800)
        # oauth row still needed for /api/ha (ha.py reads it directly)
        self._seed_oauth_usage(now + 3600)
        self.conn.commit()
        import app.aggregator as agg
        agg._cached_data.clear()
        c = self.client()  # NOTE: NO `with` — lifespan triggers usage_fetcher network/segfault
        rl = c.get("/api/rate-limits").json()["weekly_budget"]
        # week_cost must be enterprise-only $5, not blended $15
        self.assertAlmostEqual(rl["week_cost"], 5.0, places=2)
        ha = c.get("/api/ha", headers={"X-API-Key": self.api_key}).json()
        # /api/ha windows are PERSONAL-scoped (they track Max personal budget utilization).
        # Enterprise spend ($5) is excluded; only personal/consumer ($10) appears.
        self.assertAlmostEqual(ha["weekly"]["spend_usd"], 10.0, places=2)
        self.assertAlmostEqual(ha["five_hour"]["spend_usd"], 10.0, places=2)

    def test_rate_limits_works_without_oauth_row(self):
        """Decoupling: /api/rate-limits must return enterprise week_cost
        even when no oauth_usage meta row exists at all."""
        now = time.time()
        ins(self.conn, "e1", "re1", "jaedyn@acme.io", "enterprise", "Acme",
            "acme-hpc1", "acme-portal", "sE", inp=1_000_000, ts=now - 3600)
        ins(self.conn, "c1", "rc1", "me@gmail.com", "max", None,
            "personal-mbp", "secret", "sC", inp=2_000_000, ts=now - 1800)
        # No _seed_oauth_usage here — that's the whole point
        self.conn.commit()
        import app.aggregator as agg
        agg._cached_data.clear()
        c = self.client()
        rl = c.get("/api/rate-limits").json()["weekly_budget"]
        self.assertIsNotNone(rl, "weekly_budget should not be None when no oauth row")
        self.assertAlmostEqual(rl["week_cost"], 5.0, places=2)

    def test_rate_limits_no_personal_gauge_fields(self):
        """Enterprise /api/rate-limits response must contain NONE of the personal-Max
        gauge keys — neither in weekly_budget directly nor in any nested oauth sub-object.

        Contract: enterprise scope (or no scope, which defaults to enterprise) NEVER
        returns the 'oauth' key or any of its children.  The template JS contains these
        property names as source-code strings, so this assertion intentionally targets
        the API JSON response, NOT the raw HTML source.
        """
        now = time.time()
        ins(self.conn, "e1", "re1", "jaedyn@acme.io", "enterprise", "Acme",
            "acme-hpc1", "acme-portal", "sE", inp=1_000_000, ts=now - 3600)
        # Seed the LIVE fixture — its limits[] yields a scoped:fable bucket,
        # so the suppression assertions below are non-vacuous. It must still
        # be suppressed for enterprise scope.
        self._seed_live_fixture()
        c = self.client()
        # Default (no scope param) resolves to enterprise.
        resp = c.get("/api/rate-limits")
        body = resp.json()
        wb = body.get("weekly_budget", {})
        # Top-level oauth key must be absent.
        self.assertNotIn("oauth", wb,
                         "'oauth' key must not appear in enterprise rate-limits response")
        # Belt-and-suspenders: none of the gauge field names in the raw JSON text.
        raw = resp.text
        for forbidden in ("weekly_pct", "five_hour_pct", "weekly_resets_at",
                          "five_hour_resets_at", "opus_pct", "sonnet_pct",
                          "extra_usage"):
            self.assertNotIn(f'"{forbidden}"', raw,
                             f"personal gauge field '{forbidden}' must not appear in "
                             f"enterprise /api/rate-limits response JSON")
        # Normalized bucket data is personal Max data too — none of the bucket
        # list, its scoped keys, or the model display name may appear.
        for forbidden in ("scoped:", "Fable", "buckets"):
            self.assertNotIn(forbidden, raw,
                             f"'{forbidden}' must not appear in enterprise "
                             f"/api/rate-limits response body")

    def test_rate_limits_personal_has_oauth_gauges(self):
        """Personal /api/rate-limits with a seeded oauth_usage row must return the
        'oauth' sub-object with all required gauge fields."""
        now = time.time()
        ins(self.conn, "c1", "rc1", "me@gmail.com", "max", None,
            "personal-mbp", "proj", "sC", inp=1_000_000, ts=now - 3600)
        self._seed_oauth_usage(now + 3600)
        self.conn.commit()
        import app.aggregator as agg
        agg._cached_data.clear()
        c = self.client()
        body = c.get("/api/rate-limits?scope=personal").json()
        wb = body.get("weekly_budget", {})
        self.assertIn("oauth", wb,
                      "personal scope with oauth_usage row must return 'oauth' key")
        oauth = wb["oauth"]
        for required in ("weekly_pct", "five_hour_pct", "weekly_resets_at",
                         "five_hour_resets_at", "updated_at"):
            self.assertIn(required, oauth,
                          f"oauth block must contain '{required}'")
        # Values from seeded row: utilization=50 for both windows
        self.assertEqual(oauth["weekly_pct"], 50)
        self.assertEqual(oauth["five_hour_pct"], 50)

    def test_rate_limits_personal_no_oauth_row_no_oauth_key(self):
        """Personal /api/rate-limits with NO oauth_usage meta row must NOT return the
        'oauth' key — no error, no empty object, simply absent."""
        now = time.time()
        ins(self.conn, "c1", "rc1", "me@gmail.com", "max", None,
            "personal-mbp", "proj", "sC", inp=1_000_000, ts=now - 3600)
        # No oauth_usage row seeded
        self.conn.commit()
        c = self.client()
        body = c.get("/api/rate-limits?scope=personal").json()
        wb = body.get("weekly_budget", {})
        self.assertIsNotNone(wb, "weekly_budget must not be None")
        self.assertNotIn("oauth", wb,
                         "'oauth' key must be absent when no oauth_usage row exists")

    def test_rate_limits_enterprise_locked_no_oauth_key(self):
        """Enterprise-locked instance must never return the 'oauth' key even if an
        oauth_usage row is present — belt-and-suspenders lock enforcement."""
        now = time.time()
        ins(self.conn, "c1", "rc1", "me@gmail.com", "max", None,
            "personal-mbp", "proj", "sC", inp=1_000_000, ts=now - 3600)
        # LIVE fixture (limits[] with scoped:fable) — lock must suppress it all.
        self._seed_live_fixture()
        import app.config as cfg
        from unittest.mock import patch
        with patch.object(cfg, "LOCKED_SCOPE", "enterprise"):
            c = self.client()
            # Locked enterprise — any personal request returns 403
            resp = c.get("/api/rate-limits")
            body = resp.json()
        wb = body.get("weekly_budget", {})
        self.assertNotIn("oauth", wb,
                         "'oauth' key must not appear when LOCKED_SCOPE='enterprise'")
        # No normalized bucket data may leak into the raw body either.
        raw = resp.text
        for forbidden in ("scoped:", "Fable", "buckets"):
            self.assertNotIn(forbidden, raw,
                             f"'{forbidden}' must not appear in enterprise-locked "
                             f"/api/rate-limits response body")

    def test_oauth_panel_show_path_sets_real_display_value(self):
        """Visibility regression pin (P1): #oauthGaugesPanel is hidden by the
        stylesheet (display:none), so the JS show-path MUST set a real inline
        value ('block'). Setting style.display = '' merely removes the inline
        style and the stylesheet's display:none wins — gauges build their DOM
        but render 0x0/invisible. Crude source-level assertion, but it pins the
        exact bug class so it can't silently return."""
        from pathlib import Path
        tpl = (Path(__file__).resolve().parent.parent.parent
               / "templates" / "dashboard.html").read_text()
        # The CSS hide must exist (panel hidden until oauth payload arrives)...
        self.assertIn("#oauthGaugesPanel", tpl)
        # ...and the show-path must use a non-empty display value.
        self.assertIn(
            "oauthPanel.style.display = 'block'", tpl,
            "oauth gauges show-path must set display to 'block' (a real value); "
            "style.display = '' lets the stylesheet's display:none win and the "
            "gauges never become visible")
        self.assertNotIn(
            "oauthPanel.style.display = ''", tpl,
            "show-path must never clear the inline display style — the "
            "stylesheet default is display:none")

    def test_dashboard_data_json_never_contains_gauge_fields(self):
        """The embedded data_json (aggregator payload served in /) must never contain
        oauth gauge fields in ANY scope — gauges come from /api/rate-limits, not the
        aggregator payload."""
        now = time.time()
        ins(self.conn, "e1", "re1", "jaedyn@acme.io", "enterprise", "Acme",
            "acme-hpc1", "acme-portal", "sE", inp=1_000_000, ts=now - 3600)
        ins(self.conn, "c1", "rc1", "me@gmail.com", "max", None,
            "personal-mbp", "proj", "sC", inp=1_000_000, ts=now - 1800)
        self._seed_oauth_usage(now + 3600)
        self.conn.commit()
        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data.clear()
        c = self.client()
        # Check both scopes
        for scope in ("enterprise", "personal"):
            html = c.get(f"/?scope={scope}").text
            import re as _re
            # data_json ships in a type=application/json tag (guarded JSON.parse
            # boot), not an inline `const D = {...}` assignment.
            m = _re.search(
                r'<script type="application/json" id="tf-data">({.*?})</script>',
                html, _re.DOTALL)
            self.assertIsNotNone(
                m, f"data_json embed (#tf-data JSON tag) not found in page (scope={scope})")
            data_blob = m.group(1)
            for field in ("weekly_pct", "opus_pct", "five_hour_pct", "extra_usage"):
                self.assertNotIn(f'"{field}"', data_blob,
                                 f"'{field}' must not be in embedded data_json "
                                 f"(scope={scope})")

    def test_aggregator_exposes_scope_not_org_name(self):
        """build_dashboard_data must expose 'scope', NOT org_name or plan_scope."""
        ins(self.conn, "e1", "re1", "jaedyn@acme.io", "enterprise", "Acme",
            "acme-hpc1", "acme-portal", "sE", inp=1_000_000)
        ins(self.conn, "c1", "rc1", "me@gmail.com", "max", None,
            "personal-mbp", "secret-side-project", "sC", inp=2_000_000,
            ts=1781000100.0)
        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data.clear()
        d = agg.build_dashboard_data("enterprise")
        self.assertEqual(d["scope"], "enterprise")
        self.assertNotIn("org_name", d, "org_name must not be served (no org value on site)")
        self.assertNotIn("plan_scope", d, "plan_scope superseded by scope")

    def test_aggregator_personal_scope_no_org_name(self):
        """Personal scope payload must not include org_name."""
        ins(self.conn, "c1", "rc1", "me@gmail.com", "max", None,
            "personal-mbp", "proj", "sC", inp=1_000_000)
        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data.clear()
        d = agg.build_dashboard_data("personal")
        self.assertEqual(d["scope"], "personal")
        self.assertNotIn("org_name", d, "org_name must not be in payload")
        self.assertNotIn("plan_scope", d, "plan_scope superseded by scope")

    def test_dashboard_html_enterprise_badge_no_consumer_strings(self):
        """Dashboard HTML must show ENTERPRISE scope label; must NOT show Acme org name;
        no consumer strings; the embedded data_json payload must not contain oauth gauge
        field values (gauges are rendered client-side from /api/rate-limits, not embedded).

        NOTE: The template's JS source now legitimately contains the string literals
        'weekly_pct', 'opus_pct', etc. as property names in the oauth gauge renderer.
        The old assertions checking for their absence in the full HTML source have been
        moved to the API-response level: see test_rate_limits_no_personal_gauge_fields
        (enterprise /api/rate-limits must never return these keys) and
        test_rate_limits_personal_has_oauth_gauges (personal scope must return them when
        an oauth_usage meta row is present).
        """
        now = time.time()
        ins(self.conn, "e1", "re1", "jaedyn@acme.io", "enterprise", "Acme",
            "acme-hpc1", "acme-portal", "sE", inp=1_000_000, ts=now - 3600)
        ins(self.conn, "c1", "rc1", "me@gmail.com", "max", None,
            "personal-mbp", "secret-side-project", "sC", inp=2_000_000,
            ts=now - 1800)
        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data.clear()
        c = self.client()
        html = c.get("/").text
        # Scope label badge: band markup must be present and show "ENTERPRISE" scope.
        # The org VALUE (Acme) must NOT appear — we serve scope label, not org name.
        self.assertIn("header-enterprise-band", html)
        self.assertIn("ENTERPRISE", html)
        self.assertNotIn("Acme", html, "org name must not be rendered in page")
        self.assertNotIn("me@gmail.com", html)
        self.assertNotIn("personal-mbp", html)
        self.assertNotIn("secret-side-project", html)
        # The embedded data_json (#tf-data JSON tag) is the aggregator payload — must NOT
        # contain oauth gauge values.  Extract just the embedded JSON blob and check it.
        # (The surrounding template JS source contains the property name strings as code,
        # so we cannot check full HTML; we check the data payload specifically.)
        import re as _re
        m = _re.search(
            r'<script type="application/json" id="tf-data">({.*?})</script>',
            html, _re.DOTALL)
        self.assertIsNotNone(m, "data_json embed (#tf-data JSON tag) not found in page")
        data_blob = m.group(1)
        self.assertNotIn('"weekly_pct"', data_blob,
                         "weekly_pct must not appear in embedded data payload")
        self.assertNotIn('"opus_pct"', data_blob,
                         "opus_pct must not appear in embedded data payload")


class EnterprisePredicateUnattributedTest(TempDBTestCase):
    """Fix 3: unattributed rows (account_email=NULL) must be excluded even if
    plan='enterprise' and org_name is set."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_null_account_excluded_from_dashboard_and_rate_limits(self):
        """Seed event with account_email=NULL, plan='enterprise', org_name='Spoof'.
        After summarize, it must appear in NEITHER build_dashboard_data() totals
        NOR /api/rate-limits week_cost.
        """
        # Unattributed spoof row
        now = time.time()
        self.conn.execute(
            "INSERT INTO events("
            "uuid,type,timestamp,ts_epoch,day,session_id,request_id,"
            "source_machine,project_dir,model,is_sidechain,agent_id,"
            "input_tokens,output_tokens,cache_creation_tokens,cache_read_tokens,"
            "account_email,plan,org_name"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("spoof1", "assistant", "2026-06-09T12:00:00Z", now - 3600, "2026-06-09",
             "sSP", "rSP", "spoof-machine", "proj", "claude-opus-4-8", 0, None,
             1_000_000, 0, 0, 0, None, "enterprise", "Spoof"),
        )
        self.conn.commit()

        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data.clear()

        # dashboard totals must be 0 (spoof row excluded)
        d = agg.build_dashboard_data()
        self.assertAlmostEqual(
            d["total_cost"], 0.0, places=2,
            msg=f"spoof row must not contribute to total_cost, got {d['total_cost']}")

        # /api/rate-limits week_cost must also be 0
        c = self.client()
        rl = c.get("/api/rate-limits").json()["weekly_budget"]
        self.assertAlmostEqual(
            rl["week_cost"], 0.0, places=2,
            msg=f"spoof row must not contribute to week_cost, got {rl['week_cost']}")


class MultiEnterpriseAccountSameDayTest(TempDBTestCase):
    """Fix 5: two enterprise accounts on the same day must accumulate, not last-write-wins."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_two_enterprise_accounts_same_day(self):
        """Account A ($5 Opus 4.8) + Account B ($3 Sonnet 4.6) same day = $8 total.
        Also verifies _merge_summary_rows deep-merge via today panel.
        """
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from app.config import TZ_NAME

        today_str = datetime.now(ZoneInfo(TZ_NAME)).strftime("%Y-%m-%d")
        now = time.time()

        # Account A: Opus 4.8, 1M input = $5
        ins(self.conn, "eA", "rA", "a@x.io", "enterprise", "OrgA",
            "mA", "acme-portal", "sA",
            model="claude-opus-4-8", day=today_str, ts=now - 7200, inp=1_000_000)
        # Account B: Sonnet 4.6, 1M input = $3
        ins(self.conn, "eB", "rB", "b@y.io", "enterprise", "OrgB",
            "mB", "acme-portal", "sB",
            model="claude-sonnet-4-6", day=today_str, ts=now - 3600, inp=1_000_000)
        # Consumer event (must be excluded)
        ins(self.conn, "cC", "rC", "me@gmail.com", "max", None,
            "personal-mbp", "secret", "sC",
            model="claude-opus-4-8", day=today_str, ts=now - 1800, inp=1_000_000)

        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data.clear()

        # 1. Two enterprise rows in daily_summary for today
        rows = self.conn.execute(
            "SELECT account_email FROM daily_summary "
            "WHERE day=? AND plan='enterprise'", (today_str,)
        ).fetchall()
        accounts_in_summary = {r["account_email"] for r in rows}
        self.assertEqual(
            accounts_in_summary, {"a@x.io", "b@y.io"},
            f"expected 2 enterprise rows, got {accounts_in_summary}")

        d = agg.build_dashboard_data()

        # 2. total_cost accumulates both ($5 + $3 = $8)
        self.assertAlmostEqual(
            d["total_cost"], 8.0, places=2,
            msg=f"total_cost must be 8.0, got {d['total_cost']}")

        # 3. daily entry for today has cost 8.0
        today_daily = next((x for x in d["daily"] if x["date"] == today_str), None)
        self.assertIsNotNone(today_daily, "today must appear in daily list")
        self.assertAlmostEqual(
            today_daily["cost"], 8.0, places=2,
            msg=f"today daily cost must be 8.0, got {today_daily['cost']}")

        # 4. model_breakdown has both Opus 4.8 ($5) and Sonnet 4.6 ($3)
        mb_models = {m["model"]: m["cost"] for m in d["model_breakdown"]}
        self.assertIn("Opus 4.8", mb_models, "Opus 4.8 must appear in model_breakdown")
        self.assertIn("Sonnet 4.6", mb_models, "Sonnet 4.6 must appear in model_breakdown")
        self.assertAlmostEqual(mb_models["Opus 4.8"], 5.0, places=2)
        self.assertAlmostEqual(mb_models["Sonnet 4.6"], 3.0, places=2)

        # 5. machine_summary includes both machines (canonical names are lowercase)
        ms_machines = {m["machine"] for m in d["machine_summary"]}
        self.assertIn("ma", ms_machines, "mA must appear in machine_summary")
        self.assertIn("mb", ms_machines, "mB must appear in machine_summary")

        # 6. today panel: cost 8.0 and both models present (_merge_summary_rows path)
        today_panel = d["today"]
        self.assertAlmostEqual(
            today_panel["cost"], 8.0, places=2,
            msg=f"today panel cost must be 8.0, got {today_panel['cost']}")
        today_mb_models = {m["model"] for m in today_panel["model_breakdown"]}
        self.assertIn("Opus 4.8", today_mb_models,
                      "today panel must include Opus 4.8")
        self.assertIn("Sonnet 4.6", today_mb_models,
                      "today panel must include Sonnet 4.6")
