"""The served-model chip on the Model Breakdown rows.

The chip says WHAT is serving a model and WHEN, not what share of a range it
took: "kettle-e2c95a10-v2 since Aug 17" while the reroute is live, and
"kettle-e2c95a10-v2 Aug 17-18" once it stops. The old share text survives as
the hover title. Model breakdown rows come from daily_summary, which has no
served-model dimension, so the chip is computed by its own small query over
events and shipped in the dashboard JSON, keyed by mode ('all' | '14d' |
'today') and by DISPLAY model name.
"""

import json
import re
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from app.config import PERSONAL_PRED, TZ_NAME
from app.served_models import _span_label, served_model_chips, slug
from app.tests._support import TempDBTestCase

TEMPLATE = (Path(__file__).resolve().parents[2] / "templates" / "dashboard.html")
FIXTURES = json.loads(
    (Path(__file__).resolve().parent / "fixtures"
     / "thinking_signatures.json").read_text())

TZ = ZoneInfo(TZ_NAME)
FABLE = FIXTURES["fable_v2"]["expect"]
KETTLE = FIXTURES["kettle_v2"]["expect"]
V4 = FIXTURES["fable_v4"]["expect"]

KETTLE_SLUG = "kettle-e2c95a10-v2"


def _day(offset=0):
    return (datetime.now(TZ) + timedelta(days=offset)).strftime("%Y-%m-%d")


def _label(offset=0):
    """The chip's rendering of the local day `offset` days from today."""
    return (datetime.now(TZ) + timedelta(days=offset)).strftime("%b %-d")


class SlugTest(unittest.TestCase):

    def test_trims_the_fixed_affixes_only(self):
        self.assertEqual(slug("claude-kettle-e2c95a10-v2-prod"), KETTLE_SLUG)
        self.assertEqual(slug("claude-fable-5"), "fable-5")
        self.assertEqual(slug("something-else"), "something-else")


class SpanLabelTest(unittest.TestCase):
    """The first-last date span, written the way a person would say it."""

    def _ts(self, y, m, d):
        return datetime(y, m, d, 12, tzinfo=TZ).timestamp()

    def test_one_day_is_one_date(self):
        self.assertEqual(_span_label(self._ts(2026, 8, 17),
                                     self._ts(2026, 8, 17)), "Aug 17")

    def test_same_month_writes_the_month_once(self):
        self.assertEqual(_span_label(self._ts(2026, 8, 17),
                                     self._ts(2026, 8, 18)), "Aug 17-18")

    def test_across_months_both_dates_are_written(self):
        self.assertEqual(_span_label(self._ts(2026, 7, 31),
                                     self._ts(2026, 8, 2)), "Jul 31-Aug 2")

    def test_across_years_both_dates_are_written(self):
        self.assertEqual(_span_label(self._ts(2025, 12, 31),
                                     self._ts(2026, 1, 1)), "Dec 31-Jan 1")


class ServedModelChipsTest(TempDBTestCase):
    """State, dates and windowing, straight off the helper."""

    def setUp(self):
        super().setUp()
        self._n = 0

    def _ins(self, day, model, exp, count=1, hour=12):
        base = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=TZ, hour=hour)
        for i in range(count):
            self._n += 1
            uuid = f"c{self._n}"
            self.conn.execute(
                "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,"
                "request_id,source_machine,project_dir,model,is_sidechain,"
                "agent_id,input_tokens,output_tokens,cache_creation_tokens,"
                "cache_read_tokens,has_thinking,is_human_prompt,account_email,"
                "plan,served_model,sig_version,sig_header,sig_cipher_len,"
                "sig_fields) VALUES(?,'assistant',?,?,?,'s1',?,'m1',"
                "'proj',?,0,NULL,1,1,0,0,1,0,'me@gmail.com','max',?,?,?,?,?)",
                (uuid, day + "T12:00:00Z", base.timestamp() + i * 60, day,
                 "r-" + uuid, model,
                 exp["served_model"], exp["sig_version"],
                 exp["sig_header_b64"], exp["sig_cipher_len"],
                 exp["sig_fields"]),
            )
        self.conn.commit()

    def _chips(self):
        return served_model_chips(self.conn, PERSONAL_PRED,
                                  _day(-13), _day())

    def _fable(self, mode="all"):
        return self._chips()[mode]["Fable 5"]

    def test_no_chip_when_served_model_matches_requested(self):
        self._ins(_day(), "claude-fable-5", FABLE, count=10)
        self.assertEqual(self._chips()["all"], {})

    def test_no_chip_when_nothing_is_signed(self):
        self.assertEqual(self._chips(),
                         {"all": {}, "14d": {}, "today": {}})

    def test_text_names_the_state_and_when_it_started(self):
        """A live reroute reads as a state, not as a percentage."""
        self._ins(_day(), "claude-fable-5", KETTLE, count=58)
        self._ins(_day(), "claude-fable-5", FABLE, count=42)
        self.assertEqual(self._fable()["text"],
                         f"{KETTLE_SLUG} since {_label()}")

    def test_title_keeps_the_share_of_signed_blocks(self):
        """58% kettle: 58 of the 100 signed blocks were served by kettle, the
        other 42 by Fable itself. Hover text, so the number is still there."""
        self._ins(_day(), "claude-fable-5", KETTLE, count=58)
        self._ins(_day(), "claude-fable-5", FABLE, count=42)
        self.assertEqual(self._fable()["title"], f"58% {KETTLE_SLUG}")

    def test_since_uses_the_first_day_the_state_appeared(self):
        self._ins(_day(-3), "claude-fable-5", KETTLE, count=2)
        self._ins(_day(), "claude-fable-5", KETTLE, count=2)
        self.assertEqual(self._fable()["text"],
                         f"{KETTLE_SLUG} since {_label(-3)}")

    def test_a_state_that_went_quiet_reads_as_a_date_range(self):
        """More than 24 h since the last block of that state, and the model
        has kept working since: it is history, not a live state."""
        self._ins(_day(-10), "claude-fable-5", KETTLE, count=2)
        self._ins(_day(-9), "claude-fable-5", KETTLE, count=2)
        self._ins(_day(), "claude-fable-5", FABLE, count=5)
        text = self._fable()["text"]
        self.assertTrue(text.startswith(f"{KETTLE_SLUG} "), text)
        self.assertNotIn("since", text)
        self.assertIn(_label(-10), text)

    def test_a_single_quiet_day_is_one_date(self):
        self._ins(_day(-10), "claude-fable-5", KETTLE, count=2)
        self._ins(_day(), "claude-fable-5", FABLE, count=5)
        self.assertEqual(self._fable()["text"],
                         f"{KETTLE_SLUG} {_label(-10)}")

    def test_hidden_is_a_state_with_its_header_version(self):
        """A v4 header names no model. That is worth a chip of its own: the
        observatory going dark is the observation."""
        self._ins(_day(), "claude-fable-5", V4, count=2)
        self.assertEqual(self._fable()["text"], f"hidden (v4) since {_label()}")

    def test_hidden_blocks_dilute_the_title_share(self):
        """1 kettle of 4 signed blocks is 25%, the same denominator the
        statusline uses, and the 2 unnamed blocks are now named in the title
        rather than silently swelling the divisor."""
        self._ins(_day(), "claude-fable-5", KETTLE, count=1)
        self._ins(_day(), "claude-fable-5", FABLE, count=1)
        self._ins(_day(), "claude-fable-5", V4, count=2)
        self.assertEqual(self._fable()["title"],
                         f"50% hidden · 25% {KETTLE_SLUG}")

    def test_entries_are_ordered_by_blocks_desc(self):
        self._ins(_day(), "claude-fable-5", KETTLE, count=3)
        self._ins(_day(), "claude-fable-5", V4, count=1)
        self.assertEqual(
            self._fable()["text"],
            f"{KETTLE_SLUG} since {_label()} · hidden (v4) since {_label()}")

    def test_chip_is_keyed_by_display_model_name(self):
        """The dashboard rows are named 'Opus 4.8', not 'claude-opus-4-8'."""
        exp = FIXTURES["carafe_v2"]["expect"]
        self._ins(_day(), "claude-opus-4-8", exp, count=1)
        self.assertEqual(list(self._chips()["all"]), ["Opus 4.8"])

    def test_multiple_slugs_most_common_first(self):
        self._ins(_day(), "claude-opus-4-8", KETTLE, count=1)
        self._ins(_day(), "claude-opus-4-8",
                  FIXTURES["carafe_v2"]["expect"], count=3)
        chip = self._chips()["all"]["Opus 4.8"]
        self.assertEqual(chip["title"],
                         f"75% carafe-416c93ba-v1 · 25% {KETTLE_SLUG}")
        self.assertTrue(chip["text"].startswith("carafe-416c93ba-v1 since "),
                        chip["text"])

    def test_modes_window_independently(self):
        self._ins(_day(), "claude-fable-5", KETTLE, count=1)
        self._ins(_day(-5), "claude-fable-5", KETTLE, count=1)
        self._ins(_day(-40), "claude-fable-5", KETTLE, count=1)
        chips = self._chips()
        self.assertIn("Fable 5", chips["today"])
        self.assertIn("Fable 5", chips["14d"])
        self.assertIn("Fable 5", chips["all"])
        # The window decides the first day the state is known to have started.
        self.assertEqual(chips["today"]["Fable 5"]["text"],
                         f"{KETTLE_SLUG} since {_label()}")
        self.assertEqual(chips["14d"]["Fable 5"]["text"],
                         f"{KETTLE_SLUG} since {_label(-5)}")
        self.assertEqual(chips["all"]["Fable 5"]["text"],
                         f"{KETTLE_SLUG} since {_label(-40)}")
        # Older-than-14d rows exist only in 'all'.
        self._ins(_day(-40), "claude-opus-4-8",
                  FIXTURES["carafe_v2"]["expect"], count=1)
        chips = self._chips()
        self.assertIn("Opus 4.8", chips["all"])
        self.assertNotIn("Opus 4.8", chips["14d"])
        self.assertNotIn("Opus 4.8", chips["today"])

    def test_tiny_share_floors_at_one_percent(self):
        """A single odd block out of thousands is exactly what this is for;
        rendering it as '0%' would read as 'never happened'."""
        self._ins(_day(), "claude-fable-5", KETTLE, count=1)
        self._ins(_day(), "claude-fable-5", FABLE, count=999)
        self.assertEqual(self._fable()["title"], f"1% {KETTLE_SLUG}")

    def test_enterprise_rows_excluded_by_the_scope_predicate(self):
        self._ins(_day(), "claude-fable-5", KETTLE, count=1)
        self.conn.execute("UPDATE events SET plan='enterprise', "
                          "account_email='me@acme.io'")
        self.conn.commit()
        self.assertEqual(self._chips()["all"], {})


class ChipRenderTest(TempDBTestCase):
    """End to end: the chip text reaches the page, and only when earned."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def _ins(self, uuid, exp, day=None):
        day = day or _day()
        self.conn.execute(
            "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,"
            "request_id,source_machine,project_dir,model,is_sidechain,agent_id,"
            "input_tokens,output_tokens,cache_creation_tokens,cache_read_tokens,"
            "has_thinking,is_human_prompt,account_email,plan,org_name,"
            "served_model,sig_version,sig_header,sig_cipher_len,sig_fields) "
            "VALUES(?,'assistant',?,?,?,'s1',?,'m1','proj','claude-fable-5',0,"
            "NULL,1000,1000,0,0,1,0,'jaedyn@acme.io','enterprise','Acme',"
            "?,?,?,?,?)",
            (uuid, day + "T12:00:00Z",
             datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=TZ).timestamp()
             + 43200, day, "r-" + uuid,
             exp["served_model"], exp["sig_version"], exp["sig_header_b64"],
             exp["sig_cipher_len"], exp["sig_fields"]),
        )
        self.conn.commit()

    def _render(self):
        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data.clear()
        r = self.client().get("/")
        self.assertEqual(r.status_code, 200)
        return r.text

    def _payload(self, html):
        m = re.search(r'<script type="application/json" id="tf-data">(.*?)'
                      r'</script>', html, re.S)
        self.assertIsNotNone(m, "embedded data payload not found")
        return json.loads(m.group(1))

    def _served_block(self, html):
        return self._payload(html)["served_models"]

    def test_chip_text_reaches_the_page(self):
        self._ins("s1", KETTLE)
        self._ins("s2", FABLE)
        served = self._served_block(self._render())
        self.assertEqual(served["all"],
                         {"Fable 5": {"text": f"{KETTLE_SLUG} since {_label()}",
                                      "title": f"50% {KETTLE_SLUG}"}})

    def test_nothing_rendered_when_nothing_to_report(self):
        self._ins("s1", FABLE)
        served = self._served_block(self._render())
        self.assertEqual(served, {"all": {}, "14d": {}, "today": {}})

    def test_model_name_itself_is_never_altered(self):
        """The chip is a separate payload key; no breakdown row's `model`
        may carry the served slug inside it."""
        self._ins("s1", KETTLE)
        payload = self._payload(self._render())
        names = [m["model"] for m in payload["model_breakdown"]]
        self.assertIn("Fable 5", names)
        for name in names:
            self.assertNotIn(KETTLE_SLUG, name)


class ChipTemplateTest(unittest.TestCase):
    """Source-level: the chip stays a dim footnote outside the model name."""

    @classmethod
    def setUpClass(cls):
        cls.tpl = TEMPLATE.read_text()

    def _chip_lines(self):
        lines = [ln for ln in self.tpl.splitlines()
                 if "served-chip" in ln and "servedMap[m.model]" in ln]
        self.assertTrue(lines, "chip render not found in the template")
        return lines

    def test_chip_is_rendered_from_the_served_models_payload(self):
        self.assertIn("D.served_models", self.tpl)
        self.assertIn("servedMap[m.model]", self.tpl)

    def test_chip_text_is_escaped(self):
        """Same rule as every other string sink in this template. Asserted as
        the invariant rather than one spelling: the chip payload is an object
        now, so how the template digs out .text is its own business."""
        for line in self._chip_lines():
            self.assertIn("esc(", line)

    def test_chip_has_its_own_muted_class(self):
        self.assertIn(".served-chip {", self.tpl)
        chip_css = self.tpl.split(".served-chip {", 1)[1].split("}", 1)[0]
        self.assertIn("var(--gray-dim)", chip_css)

    def test_chip_is_part_of_the_model_table_render_key(self):
        """Chip text can change on a refresh that leaves the numbers alone;
        the keyed rebuild must notice."""
        self.assertIn("JSON.stringify(servedMap)", self.tpl)


if __name__ == "__main__":
    unittest.main()
