"""Unit checks for static/next/js/feeds.js (next_feeds.test.cjs).

Runs feeds.js with a fake api on a blank page in headless Chromium; no server, network or
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


class NextFeedsTest(unittest.TestCase):
    def test_feeds_in_browser(self):
        playwright = find_playwright()
        if not playwright or not shutil.which("node"):
            self.skipTest("Playwright unavailable; set TOKENFOLD_PLAYWRIGHT to a playwright module path")
        env = dict(os.environ, TOKENFOLD_PLAYWRIGHT=playwright)
        result = subprocess.run(["node", str(SCRIPTS / "next_feeds.test.cjs")], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
