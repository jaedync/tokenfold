"""Browser checks for the History and Lab views of /next (next_dom_history_lab.test.cjs).

Same harness as test_next_dom: the real /next shell from the FastAPI app on an
isolated DB, static files and fixtures from the repo through page.route.
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


class NextHistoryLabDomTest(TempDBTestCase):
    def test_history_and_lab_in_browser(self):
        playwright = find_playwright()
        if not playwright or not shutil.which("node"):
            self.skipTest("Playwright unavailable; set TOKENFOLD_PLAYWRIGHT to a playwright module path")
        response = self.client().get("/next?scope=personal")
        self.assertEqual(response.status_code, 200)
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp, "next.html")
            page.write_text(response.text)
            env = dict(os.environ, TOKENFOLD_PLAYWRIGHT=playwright, TOKENFOLD_HTML_NEXT=str(page))
            result = subprocess.run(["node", str(SCRIPTS / "next_dom_history_lab.test.cjs")], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=240)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
