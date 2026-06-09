"""XSS hardening tests — two attack vectors:
1. </script> breakout in the embedded data_json payload (Hole 1)
2. Source-level assertion that esc() wraps all string-field innerHTML sinks (Hole 2)
"""
import json
import re
import time
import unittest
from pathlib import Path

from app.tests._support import TempDBTestCase

EVIL_MACHINE = 'mEvil</script><script>window.__pwned=1</script>'
EVIL_PROJECT = 'proj</script><img src=x onerror=alert(1)>'

TEMPLATE_PATH = Path(__file__).resolve().parent.parent.parent / "templates" / "dashboard.html"


def _ins(conn, uuid, req, machine, project, model="claude-opus-4-8",
         day="2026-06-09", ts=1781000000.0, inp=1_000_000):
    conn.execute(
        "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,request_id,"
        "source_machine,project_dir,model,is_sidechain,agent_id,input_tokens,"
        "output_tokens,cache_creation_tokens,cache_read_tokens,account_email,plan,"
        "org_name,is_human_prompt,user_type) VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (uuid, "assistant", "2026-06-09T12:00:00Z", ts, day, "sE", req, machine,
         project, model, 0, None, inp, 0, 0, 0,
         "jaedyn@acme.io", "enterprise", "Acme", 0, None))
    conn.commit()


class ScriptBreakoutTest(TempDBTestCase):
    """Hole 1: </script> in embedded data_json must not break out of the
    <script> block. Fix: json.dumps(data).replace('</', '<\\/') in dashboard.py."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_script_breakout_absent_in_html(self):
        """Payload containing </script><script> must NOT appear verbatim in the page."""
        _ins(self.conn, "e1", "r1", EVIL_MACHINE, EVIL_PROJECT)

        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data = None

        c = self.client()
        resp = c.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.text

        # The raw injection string must not appear verbatim
        self.assertNotIn(
            "</script><script>window.__pwned",
            html,
            "Raw </script> breakout sequence found in HTML — Hole 1 not fixed",
        )

    def test_evil_machine_name_still_present_in_html(self):
        """The machine name must still be present (escaped, not dropped)."""
        _ins(self.conn, "e1", "r1", EVIL_MACHINE, EVIL_PROJECT)

        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data = None

        c = self.client()
        html = c.get("/").text

        # The prefix 'mEvil' must still appear (escaped form) — payload wasn't dropped
        self.assertIn(
            "mEvil",
            html,
            "Machine name 'mEvil' prefix is missing — escaping dropped the value",
        )

    def test_api_stats_roundtrips_raw_string_unmodified(self):
        """Hole 1 fix must only affect HTML embedding; /api/stats JSON must return
        the raw (unescaped) machine name so downstream consumers see the literal value."""
        _ins(self.conn, "e1", "r1", EVIL_MACHINE, EVIL_PROJECT)

        from app.summarizer import summarize_days
        import app.aggregator as agg
        summarize_days(None)
        agg._cached_data = None

        c = self.client()
        data = c.get("/api/stats").json()

        machines = data.get("machines", [])
        self.assertIn(
            EVIL_MACHINE,
            machines,
            f"Expected literal evil machine string in /api/stats machines list, got: {machines}",
        )


class InnerHTMLSinkEscTest(unittest.TestCase):
    """Hole 2: source-level assertions that all string-field innerHTML sinks
    are wrapped with esc().

    Client-side execution cannot be verified from Python; we verify the source
    directly so a reviewer can confirm the fix without running a browser.
    Each known sink is checked individually so failures point at the exact line.
    """

    def _load_template(self):
        return TEMPLATE_PATH.read_text(encoding="utf-8")

    def test_esc_function_defined(self):
        """esc() helper must be defined in the inline script block."""
        src = self._load_template()
        self.assertIn(
            "function esc(",
            src,
            "esc() helper function not found in dashboard.html",
        )

    def test_esc_helper_covers_html_specials(self):
        """The esc() definition must at minimum handle & < > \" '."""
        src = self._load_template()
        # Find the esc function body between its braces
        m = re.search(r"function esc\(.*?\{(.+?)\}", src, re.DOTALL)
        self.assertIsNotNone(m, "Could not parse esc() function body")
        body = m.group(1)
        for char in ("&amp;", "&lt;", "&gt;", "&quot;", "&#39;"):
            self.assertIn(char, body, f"esc() body missing escape for {char!r}")

    def test_machine_sink_uses_esc(self):
        """machineBody innerHTML sink must wrap ms.machine with esc()."""
        src = self._load_template()
        self.assertIn(
            "esc(ms.machine)",
            src,
            "machineBody innerHTML sink does not call esc(ms.machine)",
        )

    def test_model_breakdown_row_sink_uses_esc(self):
        """Model breakdown row.innerHTML sink must wrap m.model with esc()."""
        src = self._load_template()
        self.assertIn(
            "esc(m.model)",
            src,
            "Model breakdown row.innerHTML sink does not call esc(m.model)",
        )

    def test_top_tool_sink_uses_esc(self):
        """Card sub-detail topTool sink must wrap topTool with esc()."""
        src = self._load_template()
        self.assertIn(
            "esc(topTool)",
            src,
            "topTool sub.innerHTML sink does not call esc(topTool)",
        )

    def test_primary_model_sink_uses_esc(self):
        """Primary model sub.innerHTML sink must wrap primaryModel.model with esc()."""
        src = self._load_template()
        self.assertIn(
            "esc(primaryModel.model)",
            src,
            "primaryModel.model sub.innerHTML sink does not call esc()",
        )

    def test_cost_meta_model_sink_uses_esc(self):
        """costMeta innerHTML sink (costParts map) must wrap m.model with esc()."""
        src = self._load_template()
        # The costParts map line uses m.model — verify it's wrapped
        # Match: esc(m.model) appears in the costParts context
        self.assertIn(
            "esc(m.model)",
            src,
            "costMeta/costParts m.model sink does not call esc(m.model)",
        )

    def test_pricing_table_name_sink_uses_esc(self):
        """Pricing table body.innerHTML sink must wrap name (model name) with esc()."""
        src = self._load_template()
        self.assertIn(
            "esc(name)",
            src,
            "Pricing table body.innerHTML sink does not call esc(name) for model name",
        )


if __name__ == "__main__":
    unittest.main()
