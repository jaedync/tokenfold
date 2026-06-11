"""Hook auto-update: client/tokenfold-update.sh keeps fleet clients current.

Spawned detached after each debounced push, it compares the latest GitHub
commit sha touching client/ against a local stamp and, on change, downloads
the sha-pinned tarball and re-runs the DOWNLOADED idempotent installer.
Every failure is a silent skip — it can never break a push.
"""

import io
import json
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPDATER = ROOT / "client" / "tokenfold-update.sh"

SHA_A = "a1" * 20  # 40-char lowercase hex, like a real commit sha
SHA_B = "b2" * 20


class UpdaterSourceTest(unittest.TestCase):

    def test_exists_and_valid_sh(self):
        self.assertTrue(UPDATER.exists(), "client/tokenfold-update.sh missing")
        r = subprocess.run(["sh", "-n", str(UPDATER)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_source_contract(self):
        t = UPDATER.read_text()
        # cheap change check against the client/ subtree, sha-pinned download
        self.assertIn("api.github.com/repos/jaedync/tokenfold/commits", t)
        self.assertIn("path=client", t)
        self.assertIn("codeload.github.com/jaedync/tokenfold/tar.gz", t)
        # stamp + lock + log + kill switch
        self.assertIn(".client-sha", t)
        self.assertIn(".update-lock", t)
        self.assertIn("update.log", t)
        self.assertIn("TOKENFOLD_NO_UPDATE", t)
        # runs the DOWNLOADED installer, never the installed copy
        self.assertIn("install-tokenfold-hook.sh", t)

    def test_wrapper_spawns_updater_detached(self):
        t = (ROOT / "client" / "tokenfold-usage-push.sh").read_text()
        self.assertIn("tokenfold-update.sh", t)

    def test_installer_copies_updater(self):
        t = (ROOT / "client" / "install-tokenfold-hook.sh").read_text()
        self.assertIn("tokenfold-update.sh", t)


class UpdaterIntegrationTest(unittest.TestCase):
    """Run the real script against file:// fixtures in an isolated HOME."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.hooks = self.home / ".claude" / "usage-telemetry"
        self.hooks.mkdir(parents=True)

    def _fixture(self, sha=SHA_B, installer_rc=0):
        """Create an api.json + sha-named tarball serving a stub installer."""
        api = self.home / "api.json"
        api.write_text(json.dumps([{"sha": sha}]))

        stub = (
            "#!/bin/sh\n"
            f"echo ran >> \"$HOME/.claude/usage-telemetry/installer-ran\"\n"
            f"exit {installer_rc}\n"
        )
        tardir = self.home / "tarballs"
        tardir.mkdir(exist_ok=True)
        buf = io.BytesIO(stub.encode())
        with tarfile.open(tardir / sha, "w:gz") as tf:
            info = tarfile.TarInfo(f"tokenfold-{sha}/client/install-tokenfold-hook.sh")
            info.size = len(stub.encode())
            info.mode = 0o755
            tf.addfile(info, io.BytesIO(stub.encode()))
        return api, tardir

    def _run(self, api, tardir, extra_env=None):
        env = {**os.environ,
               "HOME": str(self.home),
               "TOKENFOLD_UPDATE_API": f"file://{api}",
               "TOKENFOLD_UPDATE_TARBALL_BASE": f"file://{tardir}"}
        env.pop("TOKENFOLD_NO_UPDATE", None)
        env.update(extra_env or {})
        return subprocess.run(["sh", str(UPDATER)], capture_output=True,
                              text=True, env=env, timeout=60)

    def _stamp(self):
        p = self.hooks / ".client-sha"
        return p.read_text() if p.exists() else None

    def test_new_sha_runs_installer_and_writes_stamp(self):
        api, tardir = self._fixture(sha=SHA_B)
        (self.hooks / ".client-sha").write_text(SHA_A)
        r = self._run(api, tardir)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((self.hooks / "installer-ran").exists(),
                        "downloaded installer was not executed")
        self.assertEqual(self._stamp(), SHA_B)

    def test_same_sha_is_a_noop(self):
        api, tardir = self._fixture(sha=SHA_B)
        (self.hooks / ".client-sha").write_text(SHA_B)
        r = self._run(api, tardir)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse((self.hooks / "installer-ran").exists())

    def test_installer_failure_does_not_write_stamp(self):
        api, tardir = self._fixture(sha=SHA_B, installer_rc=1)
        (self.hooks / ".client-sha").write_text(SHA_A)
        r = self._run(api, tardir)
        self.assertEqual(r.returncode, 0, "updater must never fail the caller")
        self.assertEqual(self._stamp(), SHA_A,
                         "stamp must only advance on installer success")

    def test_no_update_guard(self):
        api, tardir = self._fixture(sha=SHA_B)
        r = self._run(api, tardir, extra_env={"TOKENFOLD_NO_UPDATE": "1"})
        self.assertEqual(r.returncode, 0)
        self.assertFalse((self.hooks / "installer-ran").exists())
        self.assertIsNone(self._stamp())

    def test_held_lock_skips(self):
        api, tardir = self._fixture(sha=SHA_B)
        (self.hooks / ".update-lock").mkdir()  # fresh lock = update in flight
        r = self._run(api, tardir)
        self.assertEqual(r.returncode, 0)
        self.assertFalse((self.hooks / "installer-ran").exists())
        self.assertTrue((self.hooks / ".update-lock").exists(),
                        "a held lock must not be stolen")

    def test_garbage_sha_rejected(self):
        """A compromised/erroring API response must not reach the URL or shell."""
        api = self.home / "api.json"
        api.write_text(json.dumps([{"sha": "../../../etc; rm -rf /"}]))
        tardir = self.home / "tarballs"
        tardir.mkdir(exist_ok=True)
        r = self._run(api, tardir)
        self.assertEqual(r.returncode, 0)
        self.assertFalse((self.hooks / "installer-ran").exists())
        self.assertIsNone(self._stamp())


if __name__ == "__main__":
    unittest.main()
