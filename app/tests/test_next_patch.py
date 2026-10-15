"""Unit checks for patch() in static/next/js/dom.js (next_patch.test.cjs).

Runs dom.js on a blank page in headless Chromium; no server, network or
production data.
"""
import os
import shutil
import subprocess
import unittest
from pathlib import Path

from app.tests.test_dashboard_dom import find_playwright

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "app" / "tests"


class NextPatchTest(unittest.TestCase):
    def test_patch_in_browser(self):
        playwright = find_playwright()
        if not playwright or not shutil.which("node"):
            self.skipTest("Playwright unavailable; set TOKENFOLD_PLAYWRIGHT to a playwright module path")
        env = dict(os.environ, TOKENFOLD_PLAYWRIGHT=playwright)
        result = subprocess.run(["node", str(SCRIPTS / "next_patch.test.cjs")], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
