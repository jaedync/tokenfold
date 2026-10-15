"""Browser checks for the /next Ledger view in mock mode (next_dom_ledger.test.cjs).

Renders the real /next shell through the FastAPI app on an isolated DB, then
drives the Ledger in headless Chromium. Static files and fixtures come from
the repo through page.route; no live server, network or production data.
"""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from app.tests._support import TempDBTestCase
from app.tests.test_dashboard_dom import find_playwright

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "app" / "tests"


class NextLedgerDomTest(TempDBTestCase):
    def test_ledger_view_in_browser(self):
        playwright = find_playwright()
        if not playwright or not shutil.which("node"):
            self.skipTest("Playwright unavailable; set TOKENFOLD_PLAYWRIGHT to a playwright module path")
        response = self.client().get("/next?scope=personal")
        self.assertEqual(response.status_code, 200)
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp, "next.html")
            page.write_text(response.text)
            env = dict(os.environ, TOKENFOLD_PLAYWRIGHT=playwright, TOKENFOLD_HTML_NEXT=str(page))
            result = subprocess.run(["node", str(SCRIPTS / "next_dom_ledger.test.cjs")], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
