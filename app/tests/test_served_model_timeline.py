"""GET /api/served-models/timeline: reroutes as runs, not as a share.

A reroute is sticky: a session runs on the model it asked for for a stretch,
flips to another for a stretch, and flips back. The grouped /api/served-models
rows and the chip percentage both flatten that into one number for the whole
range, which is exactly the shape that hides WHEN it happened. These tests pin
the timeline contract the dashboard draws: binned cells per model, compressed
runs per session, a first-seen ledger of every (model, served, header) combo,
and the latest fleet-wide state per model.
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
CARAFE = FIXTURES["carafe_v2"]["expect"]
OPUS = FIXTURES["opus48_v0"]["expect"]
SONNET = FIXTURES["sonnet5_v0"]["expect"]
V4 = FIXTURES["fable_v4"]["expect"]

KETTLE_SLUG = "kettle-e2c95a10-v2"
CARAFE_SLUG = "carafe-416c93ba-v1"


def _at(days_ago=0, hour=12, minute=0):
    """Local wall-clock datetime `days_ago` days back."""
    return (datetime.now(TZ) - timedelta(days=days_ago)).replace(
        hour=hour, minute=minute, second=0, microsecond=0)


def _midnight(days_ago=0):
    """Epoch of the local midnight starting the day `days_ago` days back."""
    day = (datetime.now(TZ) - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    return datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=TZ).timestamp()


class TimelineTestCase(TempDBTestCase):
    """Seeding shared by every timeline test."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        self._n = 0

    def _ins(self, model, exp, when, count=1, session="s1", machine="m1",
             personal=True, step_s=60):
        """Insert `count` signed thinking events `step_s` apart from `when`.

        exp is a fixture `expect` block; exp=None inserts UNSIGNED events (every
        signature column NULL), which is what every non-thinking turn looks
        like. `when` is a local datetime: it fixes both `day` and `ts_epoch`, so
        the local day and the bin a block lands in never disagree.
        """
        header = exp["sig_header_b64"] if exp else None
        served = exp["served_model"] if exp else None
        version = exp["sig_version"] if exp else None
        fields = exp["sig_fields"] if exp else None
        clen = exp["sig_cipher_len"] if exp else None
        for i in range(count):
            self._n += 1
            uuid = f"t{self._n:04d}"
            ts = when.timestamp() + i * step_s
            day = datetime.fromtimestamp(ts, TZ).strftime("%Y-%m-%d")
            self.conn.execute(
                "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,"
                "request_id,source_machine,project_dir,model,is_sidechain,"
                "agent_id,input_tokens,output_tokens,cache_creation_tokens,"
                "cache_read_tokens,has_thinking,is_human_prompt,account_email,"
                "plan,served_model,sig_version,sig_header,sig_cipher_len,"
                "sig_fields) VALUES(?,'assistant',?,?,?,?,?,?,'proj',?,0,NULL,"
                "1,1,0,0,1,0,?,?,?,?,?,?,?)",
                (uuid, day + "T12:00:00Z", ts, day, session, "r-" + uuid,
                 machine, model,
                 "me@gmail.com" if personal else "me@acme.io",
                 "max" if personal else "enterprise",
                 served, version, header, clen, fields),
            )
        self.conn.commit()

    def _get(self, qs="scope=personal"):
        return self.client().get(f"/api/served-models/timeline?{qs}")

    def _body(self, qs="scope=personal"):
        r = self._get(qs)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()


class TimelineWindowTest(TimelineTestCase):
    """days -> bin size, window start, clamping."""

    def test_bin_seconds_per_days_window(self):
        """Cell width follows the window so a strip never draws thousands of
        cells: half-hourly up close, daily across a year."""
        for days, expect in ((1, 1800), (2, 1800), (3, 3600), (14, 3600),
                             (15, 21600), (90, 21600), (91, 86400),
                             (400, 86400)):
            with self.subTest(days=days):
                body = self._body(f"days={days}&scope=personal")
                self.assertEqual(body["days"], days)
                self.assertEqual(body["bin_seconds"], expect)

    def test_since_epoch_is_local_midnight_of_the_first_day(self):
        body = self._body("days=7&scope=personal")
        self.assertEqual(body["since_epoch"], _midnight(6))

    def test_days_is_clamped_never_an_error(self):
        for qs, expect in (("days=0", 1), ("days=-3", 1), ("days=9999", 400)):
            with self.subTest(qs=qs):
                self.assertEqual(self._body(qs + "&scope=personal")["days"],
                                 expect)

    def test_older_rows_are_outside_the_window(self):
        self._ins("claude-fable-5", KETTLE, _at(0))
        self._ins("claude-fable-5", CARAFE, _at(40))
        served = {r["served_model"] for r in self._body()["ledger"]}
        self.assertEqual(served, {KETTLE["served_model"]})
        wide = {r["served_model"]
                for r in self._body("days=90&scope=personal")["ledger"]}
        self.assertEqual(wide, {KETTLE["served_model"],
                                CARAFE["served_model"]})

    def test_unsigned_events_are_invisible(self):
        self._ins("claude-fable-5", None, _at(0), count=3)
        body = self._body()
        self.assertEqual(body["models"], [])
        self.assertEqual(body["ledger"], [])
        self.assertEqual(body["bins"], [])
        self.assertEqual(body["sessions"], {})
        self.assertEqual(body["latest"], {})


class TimelineModelsTest(TimelineTestCase):
    """`models`: who has something to report, biggest first."""

    def test_only_models_with_a_non_self_block(self):
        self._ins("claude-fable-5", KETTLE, _at(0, 9))
        self._ins("claude-fable-5", FABLE, _at(0, 10))
        self._ins("claude-sonnet-5", SONNET, _at(0, 11), count=50)
        self.assertEqual(self._body()["models"], ["Fable 5"])

    def test_models_are_ordered_by_blocks_desc(self):
        """Blocks of the model, self included: the strip draws every block it
        has, so the longest strip sorts first."""
        self._ins("claude-fable-5", FABLE, _at(0, 9), count=8)
        self._ins("claude-fable-5", KETTLE, _at(0, 10))
        self._ins("claude-opus-4-8", CARAFE, _at(0, 11), count=3)
        self.assertEqual(self._body()["models"], ["Fable 5", "Opus 4.8"])
        self._ins("claude-opus-4-8", OPUS, _at(0, 12), count=20)
        self.assertEqual(self._body()["models"], ["Opus 4.8", "Fable 5"])

    def test_hidden_counts_as_non_self(self):
        """A v4 header names no model. That is a state change worth drawing,
        not an absence."""
        self._ins("claude-fable-5", V4, _at(0, 9), count=2)
        body = self._body()
        self.assertEqual(body["models"], ["Fable 5"])
        self.assertEqual({b[2] for b in body["bins"]}, {"hidden"})


class TimelineBinsTest(TimelineTestCase):
    """`bins`: [model, bin_start, state, blocks], floor-aligned, non-empty."""

    def test_cells_are_floor_aligned_and_counted(self):
        when = _at(0, 9)
        self._ins("claude-fable-5", KETTLE, when, count=2, step_s=600)
        self._ins("claude-fable-5", KETTLE, _at(0, 10), count=1)
        body = self._body("days=1&scope=personal")
        bin_s = body["bin_seconds"]
        self.assertEqual(bin_s, 1800)
        first = (int(when.timestamp()) // bin_s) * bin_s
        second = (int(_at(0, 10).timestamp()) // bin_s) * bin_s
        self.assertEqual(
            sorted(body["bins"], key=lambda b: b[1]),
            [["Fable 5", first, KETTLE_SLUG, 2],
             ["Fable 5", second, KETTLE_SLUG, 1]])

    def test_self_and_foreign_cells_live_side_by_side(self):
        self._ins("claude-fable-5", FABLE, _at(0, 9), count=2)
        self._ins("claude-fable-5", KETTLE, _at(0, 9, 5), count=1)
        body = self._body("days=1&scope=personal")
        cells = {(b[2], b[3]) for b in body["bins"]}
        self.assertEqual(cells, {("self", 2), (KETTLE_SLUG, 1)})

    def test_no_empty_cells_and_none_for_self_only_models(self):
        self._ins("claude-fable-5", KETTLE, _at(0, 9))
        self._ins("claude-sonnet-5", SONNET, _at(0, 9), count=4)
        bins = self._body()["bins"]
        self.assertTrue(bins)
        self.assertTrue(all(b[3] > 0 for b in bins), bins)
        self.assertEqual({b[0] for b in bins}, {"Fable 5"})


class TimelineSessionsTest(TimelineTestCase):
    """`sessions`: consecutive same (model, state) compressed into runs."""

    def test_a_flip_mid_session_is_two_runs(self):
        self._ins("claude-fable-5", FABLE, _at(0, 9), count=3, session="sA")
        self._ins("claude-fable-5", KETTLE, _at(0, 11), count=2, session="sA")
        runs = self._body()["sessions"]["sA"]["runs"]
        self.assertEqual(
            runs,
            [["Fable 5", "self", _at(0, 9).timestamp(),
              _at(0, 9).timestamp() + 120, 3],
             ["Fable 5", KETTLE_SLUG, _at(0, 11).timestamp(),
              _at(0, 11).timestamp() + 60, 2]])

    def test_a_flip_back_is_a_third_run(self):
        self._ins("claude-fable-5", FABLE, _at(0, 9), session="sA")
        self._ins("claude-fable-5", KETTLE, _at(0, 10), session="sA")
        self._ins("claude-fable-5", FABLE, _at(0, 11), session="sA")
        states = [r[1] for r in self._body()["sessions"]["sA"]["runs"]]
        self.assertEqual(states, ["self", KETTLE_SLUG, "self"])

    def test_two_models_in_one_session_never_merge(self):
        self._ins("claude-fable-5", KETTLE, _at(0, 9), session="sA")
        self._ins("claude-opus-4-8", OPUS, _at(0, 10), session="sA")
        runs = self._body()["sessions"]["sA"]["runs"]
        self.assertEqual([(r[0], r[1]) for r in runs],
                         [("Fable 5", KETTLE_SLUG), ("Opus 4.8", "self")])

    def test_self_only_session_is_absent(self):
        self._ins("claude-fable-5", KETTLE, _at(0, 9), session="sA")
        self._ins("claude-fable-5", FABLE, _at(0, 9), session="sB", count=5)
        sessions = self._body()["sessions"]
        self.assertEqual(list(sessions), ["sA"])

    def test_session_carries_its_machine(self):
        self._ins("claude-fable-5", KETTLE, _at(0, 9), session="sA",
                  machine="mini")
        self.assertEqual(self._body()["sessions"]["sA"]["machine"], "mini")

    def test_a_sessionless_row_still_counts_everywhere_else(self):
        """Odd data must cost the session bar, not the endpoint."""
        self._ins("claude-fable-5", KETTLE, _at(0, 9), session=None)
        body = self._body()
        self.assertEqual(body["sessions"], {})
        self.assertEqual(body["models"], ["Fable 5"])
        self.assertEqual(len(body["ledger"]), 1)


class TimelineLedgerTest(TimelineTestCase):
    """`ledger`: one row per (model, served, sig_version, sig_fields)."""

    def test_row_carries_first_last_and_reach(self):
        self._ins("claude-fable-5", KETTLE, _at(1, 9), session="sA",
                  machine="mini")
        self._ins("claude-fable-5", KETTLE, _at(0, 9), count=2, session="sB",
                  machine="air")
        rows = self._body()["ledger"]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["model"], "Fable 5")
        self.assertEqual(row["state"], KETTLE_SLUG)
        self.assertEqual(row["served_model"], KETTLE["served_model"])
        self.assertEqual(row["sig_version"], KETTLE["sig_version"])
        self.assertEqual(row["sig_fields"], KETTLE["sig_fields"])
        self.assertEqual(row["first_seen"], _at(1, 9).timestamp())
        self.assertEqual(row["last_seen"], _at(0, 9).timestamp() + 60)
        self.assertEqual(row["blocks"], 3)
        self.assertEqual(row["sessions"], 2)
        self.assertEqual(row["machines"], ["air", "mini"])
        self.assertEqual(row["first_session"], "sA")
        self.assertEqual(row["first_machine"], "mini")

    def test_self_combos_are_listed_too(self):
        """The ledger is the header-format record: a self row that changes
        sig_fields is the earliest warning that the capture is drifting."""
        self._ins("claude-sonnet-5", SONNET, _at(0, 9), count=2)
        rows = self._body()["ledger"]
        self.assertEqual([r["state"] for r in rows], ["self"])
        self.assertEqual(rows[0]["served_model"], SONNET["served_model"])

    def test_hidden_row_has_a_null_served_model(self):
        self._ins("claude-fable-5", V4, _at(0, 9))
        row = self._body()["ledger"][0]
        self.assertEqual(row["state"], "hidden")
        self.assertIsNone(row["served_model"])
        self.assertEqual(row["sig_version"], 4)
        self.assertEqual(row["sig_fields"], "1,3,7,8")

    def test_rows_are_sorted_by_first_seen(self):
        self._ins("claude-fable-5", KETTLE, _at(0, 11))
        self._ins("claude-fable-5", FABLE, _at(0, 9))
        self._ins("claude-fable-5", V4, _at(0, 10))
        rows = self._body()["ledger"]
        self.assertEqual([r["state"] for r in rows],
                         ["self", "hidden", KETTLE_SLUG])
        self.assertEqual([r["first_seen"] for r in rows],
                         sorted(r["first_seen"] for r in rows))

    def test_the_same_served_model_under_two_headers_is_two_rows(self):
        self._ins("claude-fable-5", KETTLE, _at(0, 9))
        self.conn.execute("UPDATE events SET sig_version=3, "
                          "sig_fields='1,3,5,6,7,8' WHERE uuid='t0001'")
        self.conn.commit()
        self._ins("claude-fable-5", KETTLE, _at(0, 10))
        rows = self._body()["ledger"]
        self.assertEqual([(r["sig_version"], r["sig_fields"]) for r in rows],
                         [(3, "1,3,5,6,7,8"),
                          (KETTLE["sig_version"], KETTLE["sig_fields"])])


class TimelineLatestTest(TimelineTestCase):
    """`latest`: the state the fleet is in right now, and since when."""

    def test_latest_is_the_final_run_across_sessions(self):
        self._ins("claude-fable-5", FABLE, _at(0, 9), count=4)
        self._ins("claude-fable-5", KETTLE, _at(0, 11), count=2, session="sB")
        self._ins("claude-fable-5", KETTLE, _at(0, 12), count=1, session="sC")
        self.assertEqual(self._body()["latest"],
                         {"Fable 5": {"state": KETTLE_SLUG,
                                      "since": _at(0, 11).timestamp(),
                                      "blocks": 3}})

    def test_a_flip_back_to_self_resets_the_run(self):
        self._ins("claude-fable-5", KETTLE, _at(0, 9), count=5)
        self._ins("claude-fable-5", FABLE, _at(0, 12), count=1)
        self.assertEqual(self._body()["latest"]["Fable 5"],
                         {"state": "self", "since": _at(0, 12).timestamp(),
                          "blocks": 1})

    def test_only_reported_models_get_a_latest(self):
        self._ins("claude-fable-5", KETTLE, _at(0, 9))
        self._ins("claude-sonnet-5", SONNET, _at(0, 9))
        self.assertEqual(list(self._body()["latest"]), ["Fable 5"])


class TimelineScopeTest(TimelineTestCase):
    """Gated exactly like /api/served-models."""

    def test_enterprise_rows_never_leak_into_personal(self):
        self._ins("claude-fable-5", CARAFE, _at(0, 9), personal=False)
        self._ins("claude-fable-5", KETTLE, _at(0, 10), personal=True)
        served = {r["served_model"] for r in self._body()["ledger"]}
        self.assertEqual(served, {KETTLE["served_model"]})

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
        url = "/api/served-models/timeline?scope=personal"
        self.assertEqual(c.get(url).status_code, 401)
        ok = c.get(url, auth=("admin", "s3cret"))
        self.assertEqual(ok.status_code, 200, ok.text)


if __name__ == "__main__":
    unittest.main()
