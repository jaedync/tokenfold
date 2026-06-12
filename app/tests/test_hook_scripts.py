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


class DebouncedPostToolUseTest(unittest.TestCase):
    """Long turns run for hours before Stop fires; the PostToolUse hook with a
    TOKENFOLD_MIN_INTERVAL debounce keeps server data <=1 min stale. Source-level
    contract: the wrapper implements the debounce and the installer registers
    the debounced PostToolUse group alongside Stop/SessionEnd."""

    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        base = Path(__file__).resolve().parents[2] / "client"
        cls.wrapper = (base / "tokenfold-usage-push.sh").read_text()
        cls.installer = (base / "install-tokenfold-hook.sh").read_text()

    def test_wrapper_debounces(self):
        self.assertIn("TOKENFOLD_MIN_INTERVAL", self.wrapper)
        self.assertIn(".tokenfold-last-push", self.wrapper)

    def test_installer_registers_posttooluse(self):
        self.assertIn('"PostToolUse"', self.installer)
        self.assertIn("TOKENFOLD_MIN_INTERVAL=60", self.installer)
        self.assertNotIn("TOKENFOLD_MIN_INTERVAL=300", self.installer)

    def test_stop_and_sessionend_have_no_debounce(self):
        """Stop must ALWAYS fire — only PostToolUse gets the cooldown env."""
        import re
        # the plain group command (used for Stop/SessionEnd) carries no env prefix
        plain = re.findall(r'"command": \'"?\$HOME/\.claude/usage-telemetry/hook\.sh"?\'',
                           self.installer)
        self.assertTrue(plain, "plain (undebounced) hook command missing")
        # exactly one debounced command, and it is the PostToolUse one
        debounced = re.findall(r"TOKENFOLD_MIN_INTERVAL=\d+", self.installer)
        self.assertEqual(len(set(debounced)), 1)


def _run_registration(settings: dict) -> dict:
    """Run the installer's embedded settings-registration python verbatim
    against a temp settings.json; return the resulting settings."""
    installer = (ROOT / "client" / "install-tokenfold-hook.sh").read_text()
    start = installer.index("<<'PY'") + len("<<'PY'\n")
    end = installer.index("\nPY\n", start)
    py = installer[start:end]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "settings.json")
        with open(path, "w") as f:
            json.dump(settings, f)
        r = subprocess.run(["python3", "-", path], input=py,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError(f"registration failed: {r.stderr}")
        with open(path) as f:
            return json.load(f)


class RegistrationNormalizationTest(unittest.TestCase):
    """The fleet upgrades by re-running this installer (auto-update). A stale
    debounce value in an existing PostToolUse group must be rewritten, not
    skipped — presence-only idempotency would freeze old installs at 300s."""

    def test_fresh_install_registers_all_three_events(self):
        out = _run_registration({})
        for event in ("Stop", "SessionEnd", "PostToolUse"):
            cmds = [h["command"] for g in out["hooks"][event] for h in g["hooks"]]
            self.assertTrue(any("usage-telemetry/hook.sh" in c for c in cmds), event)
        post = json.dumps(out["hooks"]["PostToolUse"])
        self.assertIn("TOKENFOLD_MIN_INTERVAL=60", post)
        stop = json.dumps(out["hooks"]["Stop"])
        self.assertNotIn("TOKENFOLD_MIN_INTERVAL", stop)

    def test_stale_debounce_value_is_rewritten(self):
        stale = {"hooks": {"PostToolUse": [{"hooks": [{
            "type": "command",
            "command": 'TOKENFOLD_MIN_INTERVAL=300 "$HOME/.claude/usage-telemetry/hook.sh"',
            "timeout": 10}]}]}}
        out = _run_registration(stale)
        post = json.dumps(out["hooks"]["PostToolUse"])
        self.assertIn("TOKENFOLD_MIN_INTERVAL=60", post)
        self.assertNotIn("TOKENFOLD_MIN_INTERVAL=300", post)
        # still exactly one tokenfold group — normalize, don't duplicate
        marked = [g for g in out["hooks"]["PostToolUse"]
                  if "usage-telemetry/hook.sh" in json.dumps(g)]
        self.assertEqual(len(marked), 1)

    def test_foreign_hooks_untouched(self):
        foreign = {"hooks": {"Stop": [{"hooks": [{
            "type": "command", "command": "/usr/local/bin/other-hook.sh"}]}]}}
        out = _run_registration(foreign)
        stop = json.dumps(out["hooks"]["Stop"])
        self.assertIn("other-hook.sh", stop)
        self.assertIn("usage-telemetry/hook.sh", stop)

    def test_current_state_is_a_noop(self):
        once = _run_registration({})
        twice = _run_registration(once)
        self.assertEqual(once, twice)
