"""Tests for client/bootstrap.sh — the curl-pipe-bash entry served at /install.sh.

Run with:
    .venv/bin/python -m unittest client.test_bootstrap -v

The bootstrap is exercised end to end against a local file:// tarball whose
install-tokenfold-hook.sh is a STUB that prints its argv, so assertions can
cover URL baking, --url override order, arg passthrough, tmpdir cleanup, and
exit-code propagation without touching the network or the real installer.
"""

import os
import pathlib
import shutil
import subprocess
import tarfile
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
BOOTSTRAP_SRC = HERE / "bootstrap.sh"

# Absolute path so the preflight test can restrict the script's PATH without
# breaking subprocess's own lookup of the interpreter.
BASH = shutil.which("bash") or "/bin/bash"

BAKED_URL = "https://baked.example.com"

# Mirrors the real installer's contract (bash script, argv in, exit code out):
# one ARG: line per argv entry (order preserved) plus SELF: with its own path,
# so tests can recover the bootstrap's temp dir and check it was cleaned up.
# --stub-exit-7 exercises exit-code propagation via a plain passthrough arg.
STUB_INSTALLER = """\
#!/usr/bin/env bash
printf 'SELF:%s\\n' "$0"
for a in "$@"; do printf 'ARG:%s\\n' "$a"; done
for a in "$@"; do [ "$a" = --stub-exit-7 ] && exit 7; done
exit 0
"""


class BootstrapTestCase(unittest.TestCase):
    """Shared fixture: a fake codeload tarball + a runnable bootstrap copy."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="tokenfold-bootstrap-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.tarball = self._build_tarball()

    def _build_tarball(self):
        # codeload layout: a single <repo>-<ref>/ top-level dir.
        client_dir = self.tmp / "tree" / "tokenfold-main" / "client"
        client_dir.mkdir(parents=True)
        stub = client_dir / "install-tokenfold-hook.sh"
        stub.write_text(STUB_INSTALLER)
        stub.chmod(0o755)
        out = self.tmp / "tokenfold.tar.gz"
        with tarfile.open(out, "w:gz") as tf:
            tf.add(self.tmp / "tree" / "tokenfold-main", arcname="tokenfold-main")
        return out

    def _bootstrap(self, baked=None):
        """Copy bootstrap.sh, optionally simulating the server-side URL bake.

        baked=None keeps the raw __TOKENFOLD_URL__ placeholder — the
        "fetched straight from GitHub, unsubstituted" case.
        """
        text = BOOTSTRAP_SRC.read_text()
        if baked is not None:
            text = text.replace("__TOKENFOLD_URL__", baked)
        script = self.tmp / "bootstrap.sh"
        script.write_text(text)
        return script

    def _run(self, script, *args, env_extra=None):
        env = dict(os.environ)
        env["TOKENFOLD_BOOTSTRAP_TARBALL"] = self.tarball.as_uri()
        env.update(env_extra or {})
        return subprocess.run(
            [BASH, str(script), *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )

    @staticmethod
    def _stub_args(stdout):
        return [ln[len("ARG:"):] for ln in stdout.splitlines() if ln.startswith("ARG:")]

    @staticmethod
    def _stub_tmpdir(stdout):
        """Recover the bootstrap's mktemp dir from the stub's SELF: line."""
        for ln in stdout.splitlines():
            if ln.startswith("SELF:"):
                # <tmpd>/tokenfold-main/client/install-tokenfold-hook.sh
                return pathlib.Path(ln[len("SELF:"):]).parent.parent.parent
        return None


class UrlResolutionTest(BootstrapTestCase):
    def test_baked_url_used_when_no_url_arg(self):
        proc = self._run(self._bootstrap(baked=BAKED_URL))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self._stub_args(proc.stdout)[:2], ["--url", BAKED_URL])

    def test_url_arg_overrides_baked_default(self):
        proc = self._run(
            self._bootstrap(baked=BAKED_URL), "--url", "https://cli.example.com"
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        args = self._stub_args(proc.stdout)
        # The installer's parser is last-occurrence-wins, so the user's URL
        # must be the FINAL --url value the installer sees.
        url_values = [args[i + 1] for i, a in enumerate(args) if a == "--url"]
        self.assertEqual(url_values[-1], "https://cli.example.com")

    def test_unsubstituted_placeholder_treated_as_unset(self):
        proc = self._run(self._bootstrap(baked=None))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--url", proc.stderr)
        # Installer must never run without a resolvable URL.
        self.assertNotIn("ARG:", proc.stdout)

    def test_unsubstituted_placeholder_with_explicit_url_works(self):
        proc = self._run(
            self._bootstrap(baked=None), "--url", "https://cli.example.com"
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(
            self._stub_args(proc.stdout)[:2], ["--url", "https://cli.example.com"]
        )

    def test_url_flag_without_value_dies(self):
        proc = self._run(self._bootstrap(baked=BAKED_URL), "--url")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--url", proc.stderr)


class PassthroughTest(BootstrapTestCase):
    def test_token_and_flags_pass_through_in_order(self):
        proc = self._run(
            self._bootstrap(baked=BAKED_URL),
            "--no-push", "--token", "tk_secret'quote", "--keep-legacy",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(
            self._stub_args(proc.stdout),
            ["--url", BAKED_URL, "--no-push", "--token", "tk_secret'quote", "--keep-legacy"],
        )

    def test_token_may_be_omitted(self):
        # Installer resolves $TOKENFOLD_API_KEY / ~/.config itself.
        proc = self._run(self._bootstrap(baked=BAKED_URL))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("--token", self._stub_args(proc.stdout))


class ExecutionTest(BootstrapTestCase):
    def test_exits_with_installer_code_and_cleans_tmpdir(self):
        proc = self._run(self._bootstrap(baked=BAKED_URL), "--stub-exit-7")
        self.assertEqual(proc.returncode, 7)
        tmpd = self._stub_tmpdir(proc.stdout)
        self.assertIsNotNone(tmpd)
        self.assertFalse(tmpd.exists(), "tmpdir not cleaned up after installer failure: %s" % tmpd)

    def test_tmpdir_cleaned_on_success(self):
        proc = self._run(self._bootstrap(baked=BAKED_URL))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        tmpd = self._stub_tmpdir(proc.stdout)
        self.assertIsNotNone(tmpd)
        self.assertFalse(tmpd.exists(), "tmpdir not cleaned up: %s" % tmpd)

    def test_progress_lines_are_bootstrap_prefixed(self):
        proc = self._run(self._bootstrap(baked=BAKED_URL))
        own = [
            ln for ln in proc.stdout.splitlines()
            if ln.strip() and not ln.startswith(("ARG:", "SELF:"))
        ]
        self.assertTrue(own, "expected [bootstrap] progress output")
        for ln in own:
            self.assertTrue(ln.startswith("[bootstrap]"), ln)


class FailureModeTest(BootstrapTestCase):
    def test_download_failure_dies_with_stderr_message(self):
        missing = (self.tmp / "does-not-exist.tar.gz").as_uri()
        proc = self._run(
            self._bootstrap(baked=BAKED_URL),
            env_extra={"TOKENFOLD_BOOTSTRAP_TARBALL": missing},
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("[bootstrap]", proc.stderr)
        self.assertNotIn("ARG:", proc.stdout)

    def test_preflight_names_missing_tool(self):
        # PATH with tar/python3 but no curl: preflight must fail fast, naming
        # the missing tool, before any download is attempted.
        bindir = self.tmp / "bin"
        bindir.mkdir()
        for tool in ("tar", "python3"):
            src = shutil.which(tool)
            self.assertIsNotNone(src, "test host missing %s" % tool)
            os.symlink(src, bindir / tool)
        proc = self._run(
            self._bootstrap(baked=BAKED_URL), env_extra={"PATH": str(bindir)}
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("curl", proc.stderr)
        self.assertNotIn("ARG:", proc.stdout)


class ScriptContentTest(unittest.TestCase):
    def test_contract_strings_present(self):
        # /install.sh's server tests assert on the codeload URL, and
        # app/install.py substitutes the placeholder — both are load-bearing.
        text = BOOTSTRAP_SRC.read_text()
        self.assertIn("codeload.github.com/jaedync/tokenfold/tar.gz/refs/heads/main", text)
        self.assertIn('TOKENFOLD_URL_DEFAULT="__TOKENFOLD_URL__"', text)

    def test_bootstrap_is_executable(self):
        self.assertTrue(os.access(BOOTSTRAP_SRC, os.X_OK))


if __name__ == "__main__":
    unittest.main()
