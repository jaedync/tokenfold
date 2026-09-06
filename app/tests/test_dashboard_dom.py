"""Executable browser regressions for quota refresh presentation.

Renders the real dashboard template through the FastAPI app on an isolated DB,
then drives it in headless Chromium (Playwright) with route-served quota
fixtures. No live server, network, or production data is involved.

Playwright is resolved from TOKENFOLD_PLAYWRIGHT or a local npx cache; the
test skips with an explicit message when neither is present.
"""
import base64
import glob
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.tests._support import TempDBTestCase

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "app" / "tests"


def find_playwright():
    candidates = [os.environ.get("TOKENFOLD_PLAYWRIGHT", "")]
    candidates += sorted(glob.glob(os.path.expanduser("~/.npm/_npx/*/node_modules/playwright")))
    for candidate in candidates:
        if candidate and Path(candidate, "package.json").is_file():
            return candidate
    return None


class DashboardDomTest(TempDBTestCase):
    def render(self, scope, password=""):
        headers = {}
        if password:
            token = base64.b64encode(f"admin:{password}".encode()).decode()
            headers["Authorization"] = "Basic " + token
        with patch.object(self._config, "DASHBOARD_PASSWORD", password):
            response = self.client().get(f"/?scope={scope}", headers=headers)
        self.assertEqual(response.status_code, 200)
        return response.text

    def test_quota_refresh_presentation_in_browser(self):
        self.run_browser("dashboard_dom.test.cjs")

    def test_independent_review_transitions_in_browser(self):
        self.run_browser("dashboard_review.test.cjs")

    def run_browser(self, script):
        playwright = find_playwright()
        if not playwright or not shutil.which("node"):
            self.skipTest("Playwright unavailable; set TOKENFOLD_PLAYWRIGHT to a playwright module path")
        with tempfile.TemporaryDirectory() as tmp:
            personal = Path(tmp, "personal.html")
            personal.write_text(self.render("personal"))
            # A writable enterprise instance (Basic auth) exposes the budget affordance.
            enterprise = Path(tmp, "enterprise.html")
            enterprise.write_text(self.render("enterprise", "secret"))
            env = dict(os.environ, TOKENFOLD_PLAYWRIGHT=playwright,
                       TOKENFOLD_HTML_PERSONAL=str(personal),
                       TOKENFOLD_HTML_ENTERPRISE=str(enterprise))
            result = subprocess.run(["node", str(SCRIPTS / script)], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=240)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
