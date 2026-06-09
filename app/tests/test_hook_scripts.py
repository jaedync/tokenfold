import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # app/tests/ -> repo root


class HookWrapperTest(unittest.TestCase):
    def test_wrapper_exists_and_valid_bash(self):
        s = ROOT / "client" / "tokenfold-usage-push.sh"
        self.assertTrue(s.exists(), "wrapper missing")
        r = subprocess.run(["bash", "-n", str(s)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_wrapper_detaches_and_reads_config(self):
        t = (ROOT / "client" / "tokenfold-usage-push.sh").read_text()
        self.assertIn("notify-relay-url", t)        # reuses existing config files
        self.assertTrue("setsid" in t or "nohup" in t)  # detaches


class InstallerRegistrationTest(unittest.TestCase):
    def test_installer_mentions_wrapper_and_sessionend(self):
        t = (ROOT / "client" / "install-hooks.sh").read_text()
        self.assertIn("tokenfold-usage-push.sh", t)
        self.assertIn('"SessionEnd"', t)


class InstallerIntegrationTest(unittest.TestCase):
    def setUp(self):
        if subprocess.run(["bash", "-c", "command -v jq"],
                          capture_output=True).returncode != 0:
            self.skipTest("jq not available")

    def test_fresh_install_registers_usage_push_on_stop_and_sessionend(self):
        with tempfile.TemporaryDirectory() as home:
            r = subprocess.run(
                ["bash", str(ROOT / "client" / "install-hooks.sh"),
                 "https://x.example", "tok"],
                capture_output=True, text=True, env={**os.environ, "HOME": home})
            self.assertEqual(r.returncode, 0, r.stderr)
            settings = json.loads(
                (Path(home) / ".claude" / "settings.json").read_text())
            hooks = settings["hooks"]
            self.assertIn("tokenfold-usage-push.sh", json.dumps(hooks.get("SessionEnd", [])))
            self.assertIn("tokenfold-usage-push.sh", json.dumps(hooks.get("Stop", [])))
            self.assertTrue(
                (Path(home) / ".claude" / "hooks" / "tokenfold-usage-push.sh").exists())


if __name__ == "__main__":
    unittest.main()
