import subprocess
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


if __name__ == "__main__":
    unittest.main()
