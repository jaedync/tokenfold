"""The notify token is a bearer secret: startup must never print it."""
import contextlib
import io

from app.tests._support import TempDBTestCase


class NotifyTokenLogTest(TempDBTestCase):
    def _init_and_capture(self):
        import app.notify as notify
        saved_env = notify.NOTIFY_TOKEN
        notify.NOTIFY_TOKEN = ""
        self.addCleanup(setattr, notify, "NOTIFY_TOKEN", saved_env)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            notify.init_notify_token()
        return notify._notify_token, out.getvalue()

    def test_generated_token_is_not_printed(self):
        token, output = self._init_and_capture()
        self.assertTrue(token)
        self.assertNotIn(token, output)

    def test_token_from_db_is_not_printed(self):
        self._init_and_capture()  # stores a generated token in meta
        token, output = self._init_and_capture()  # second start reads it back
        self.assertTrue(token)
        self.assertNotIn(token, output)
