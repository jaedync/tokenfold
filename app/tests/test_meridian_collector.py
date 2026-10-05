"""Always-on Meridian quota collector: config, wire behavior, loop logging."""
import json
import os
import signal
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from app import meridian_collector as mc
from app.tests.test_meridian_quota import FIXTURE, OBSERVED_MS, quota

KEY = "collector-test-key"


class _Handler(BaseHTTPRequestHandler):
    routes = {}
    seen = []

    def log_message(self, *args):
        pass

    def _send(self, status, body=b"", headers=()):
        self.send_response(status)
        for name, value in headers:
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _raw(self):
        """Malformed responses the normal route table cannot express."""
        if self.path == "/garbage":
            self.wfile.write(b"garbage\r\n\r\n")
        elif self.path == "/big-nolength":  # HTTP/1.0: body ends at close
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'"' + b"x" * (mc.MAX_BODY_BYTES + 10) + b'"')
        elif self.path == "/drip":
            self.send_response(200)
            self.end_headers()
            for _ in range(10):
                self.wfile.write(b" ")
                self.wfile.flush()
                time.sleep(0.3)
        elif self.path == "/truncated":
            self.send_response(200)
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.write(b'{"a": ')
        else:
            return False
        return True

    def do_GET(self):
        self.seen.append(("GET", self.path, dict(self.headers), None))
        if not self._raw():
            self._send(*self.routes.get(self.path, (404, b"")))

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.seen.append(("POST", self.path, dict(self.headers), body))
        if not self._raw():
            self._send(*self.routes.get(self.path, (404, b"")))


class WireTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        _Handler.seen = []
        _Handler.routes = {
            "/ok": (200, b'{"a": 1}'),
            "/redirect": (302, b"", [("Location", "/ok")]),
            "/big": (200, b'"' + b"x" * mc.MAX_BODY_BYTES + b'"'),
            "/notjson": (200, b"not json"),
            "/down": (503, b'{"a": 1}'),
            "/api/usage/claude": (200, b'{"status": "ok", "updated_at": "t"}'),
            "/denied": (401, b'{"detail": "Invalid API key"}'),
        }

    def test_get_returns_json_without_credentials(self):
        self.assertEqual(mc.fetch_json(self.base + "/ok"), {"a": 1})
        headers = {k.lower() for k in _Handler.seen[0][2]}
        self.assertNotIn("x-api-key", headers)
        self.assertNotIn("authorization", headers)

    def test_get_refuses_redirect_oversize_non_json_and_errors(self):
        for path in ("/redirect", "/big", "/notjson", "/down", "/missing"):
            self.assertIsNone(mc.fetch_json(self.base + path), path)
        self.assertEqual([s[1] for s in _Handler.seen].count("/ok"), 0)

    def test_malformed_responses_are_none_not_exceptions(self):
        for path in ("/garbage", "/big-nolength", "/truncated"):
            self.assertIsNone(mc.fetch_json(self.base + path), path)
            self.assertEqual(mc.post_json(self.base + path, FIXTURE, KEY)[1], None, path)
        self.assertEqual(mc.post_json(self.base + "/garbage", FIXTURE, KEY), (None, None))

    def test_slow_drip_body_stops_at_the_overall_deadline(self):
        started = time.monotonic()
        self.assertIsNone(mc.fetch_json(self.base + "/drip", timeout=1))
        self.assertLess(time.monotonic() - started, 2.5)

    def test_unreachable_host_is_none_not_an_exception(self):
        self.server_closed = mc.fetch_json("http://127.0.0.1:9/ok", timeout=1)
        self.assertIsNone(self.server_closed)

    def test_environment_proxies_are_never_used(self):
        bogus = "http://127.0.0.1:9"
        with patch.dict(os.environ, {"http_proxy": bogus, "HTTP_PROXY": bogus,
                                     "no_proxy": "", "NO_PROXY": ""}):
            self.assertEqual(mc.fetch_json(self.base + "/ok"), {"a": 1})

    def test_post_sends_key_and_strict_json(self):
        code, body = mc.post_json(self.base + "/api/usage/claude", FIXTURE, KEY)
        self.assertEqual((code, body["status"]), (200, "ok"))
        method, path, headers, raw = _Handler.seen[0]
        headers = {k.lower(): v for k, v in headers.items()}  # names are case-insensitive
        self.assertEqual((method, path), ("POST", "/api/usage/claude"))
        self.assertEqual(headers["x-api-key"], KEY)
        self.assertEqual(headers["content-type"], "application/json")
        self.assertEqual(json.loads(raw), FIXTURE)
        self.assertEqual(mc.post_json(self.base + "/denied", FIXTURE, KEY)[0], 401)
        self.assertEqual(mc.post_json("http://127.0.0.1:9/x", FIXTURE, KEY, timeout=1),
                         (None, None))
        with self.assertRaises(ValueError):
            mc.post_json(self.base + "/api/usage/claude", {"pct": float("nan")}, KEY)


def _cfg(**overrides):
    values = dict(meridian="http://127.0.0.1:3456", tokenfold="http://127.0.0.1:5055",
                  api_key=KEY, machine="test-personal", interval_s=mc.DEFAULT_INTERVAL_S)
    values.update(overrides)
    return mc.Config(**values)


class CollectOnceTest(unittest.TestCase):
    def fake_fetch(self, health, body):
        calls = []

        def fetch(url, timeout=mc.REQUEST_TIMEOUT_S):
            calls.append(url)
            return health if url.endswith("/health") else body
        return fetch, calls

    def fake_post(self, code=200, status="ok"):
        posts = []

        def post(url, payload, api_key, timeout=mc.REQUEST_TIMEOUT_S):
            posts.append((url, payload, api_key))
            return code, ({"status": status} if code == 200 else None)
        return post, posts

    def collect(self, health, body, **post_kwargs):
        fetch, calls = self.fake_fetch(health, body)
        post, posts = self.fake_post(**post_kwargs)
        status = mc.collect_once(_cfg(), fetch=fetch, post=post,
                                 now=lambda: OBSERVED_MS / 1000)
        return status, calls, posts

    def test_pro_and_max_post_the_parsed_payload(self):
        for sub in ("pro", "max"):
            status, calls, posts = self.collect({"auth": {"subscriptionType": sub}}, quota())
            self.assertEqual(status, "posted")
            self.assertEqual(calls, ["http://127.0.0.1:3456/health",
                                     "http://127.0.0.1:3456/v1/usage/quota"])
            self.assertEqual(posts, [("http://127.0.0.1:5055/api/usage/claude",
                                      FIXTURE, KEY)])

    def test_other_subscriptions_never_read_quota_or_post(self):
        for health in ({"auth": {"subscriptionType": "enterprise"}}, {"auth": None}, {}, []):
            status, calls, posts = self.collect(health, quota())
            self.assertEqual((status, len(calls), posts), ("not_personal", 1, []), health)

    def test_unavailable_invalid_and_rejected_statuses(self):
        ok = {"auth": {"subscriptionType": "max"}}
        self.assertEqual(self.collect(None, quota())[0], "unavailable")
        self.assertEqual(self.collect(ok, None)[0], "unavailable")
        self.assertEqual(self.collect(ok, {"profile": "work"})[0], "invalid")
        self.assertEqual(self.collect(ok, quota(), code=401)[0], "rejected_401")
        self.assertEqual(self.collect(ok, quota(), code=None)[0], "send_failed")
        self.assertEqual(self.collect(ok, quota(), status="ignored_stale")[0], "stale")


class ConfigTest(unittest.TestCase):
    def test_defaults_and_overrides(self):
        cfg = mc.load_config({"STATS_API_KEY": KEY, "COLLECTOR_MACHINE": "ms01"})
        self.assertEqual(cfg, _cfg(machine="ms01"))
        cfg = mc.load_config({"STATS_API_KEY": KEY, "COLLECTOR_MACHINE": "ms01",
                              "MERIDIAN_URL": "http://localhost:4000/v1",
                              "TOKENFOLD_URL": "https://usage.example/",
                              "COLLECTOR_INTERVAL_S": "120"})
        self.assertEqual((cfg.meridian, cfg.tokenfold, cfg.interval_s),
                         ("http://localhost:4000", "https://usage.example", 120))

    def test_machine_defaults_to_hostname(self):
        with patch("app.meridian_collector.socket.gethostname", return_value="ms01"):
            self.assertEqual(mc.load_config({"STATS_API_KEY": KEY}).machine, "ms01")

    def test_meridian_must_stay_on_loopback(self):
        for url in ("https://api.anthropic.com", "http://example.com",
                    "http://127.0.0.1.evil", "file:///tmp/x", "http://u:p@localhost",
                    "http://localhost?profile=work", "http://localhost#work", ""):
            self.assertIsNone(mc.local_meridian_base(url), url)
        for host in ("localhost", "127.0.0.1", "[::1]"):
            self.assertEqual(mc.local_meridian_base(f"http://{host}:1234/v1"),
                             f"http://{host}:1234")

    def test_invalid_config_fails_with_a_friendly_message(self):
        base = {"STATS_API_KEY": KEY, "COLLECTOR_MACHINE": "ms01"}
        for env in ({"COLLECTOR_MACHINE": "ms01"},
                    {**base, "STATS_API_KEY": ""},
                    {**base, "MERIDIAN_URL": "http://example.com"},
                    {**base, "TOKENFOLD_URL": "ftp://x"},
                    {**base, "TOKENFOLD_URL": "http://u:p@x"},
                    {**base, "TOKENFOLD_URL": "http://usage.example"},
                    {**base, "TOKENFOLD_URL": "http://[bad"},
                    {**base, "TOKENFOLD_URL": "http://127.0.0.1:abc"},
                    {**base, "STATS_API_KEY": "k\u00e9y\u2603"},
                    {**base, "STATS_API_KEY": "key\n"},
                    {**base, "COLLECTOR_INTERVAL_S": "5"},
                    {**base, "COLLECTOR_INTERVAL_S": "soon"},
                    {**base, "COLLECTOR_MACHINE": "a\nb"}):
            with self.assertRaises(ValueError) as ctx:
                mc.load_config(env)
            self.assertNotIn(KEY, str(ctx.exception))


class RunLoopTest(unittest.TestCase):
    def test_logs_only_health_transitions_and_never_the_key(self):
        statuses = iter(["posted", "stale", "posted", "unavailable", "unavailable",
                         "posted"])
        stop = threading.Event()
        logs = []

        def collect(cfg):
            try:
                return next(statuses)
            except StopIteration:
                stop.set()
                return "posted"

        mc.run(_cfg(interval_s=0), stop, collect=collect, log=logs.append)
        self.assertEqual(len(logs), 3, logs)
        self.assertIn("healthy", logs[0])
        self.assertIn("unavailable", logs[1])
        self.assertIn("healthy", logs[2])
        self.assertFalse(any(KEY in line for line in logs))

    def test_stale_replies_turn_degraded_after_no_new_observation(self):
        statuses = iter(["posted"] + ["stale"] * 5 + ["posted"])
        stop, logs, now = threading.Event(), [], [0.0]

        def collect(cfg):
            now[0] += 300
            status = next(statuses, None)
            if status is None:
                stop.set()
                return "posted"
            return status

        mc.run(_cfg(interval_s=0), stop, collect=collect, log=logs.append,
               clock=lambda: now[0])
        self.assertEqual(logs, ["quota collection healthy",
                                "quota collection degraded: no_new_observation",
                                "quota collection healthy"])

    def test_unexpected_exception_does_not_stop_the_loop(self):
        stop = threading.Event()
        logs, calls = [], []

        def collect(cfg):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError(KEY)
            stop.set()
            return "posted"

        mc.run(_cfg(interval_s=0), stop, collect=collect, log=logs.append)
        self.assertEqual(len(calls), 2)
        self.assertIn("RuntimeError", logs[0])
        self.assertFalse(any(KEY in line for line in logs))

    def test_stop_interrupts_the_wait(self):
        stop = threading.Event()

        def collect(cfg):
            stop.set()
            return "posted"
        worker = threading.Thread(target=mc.run, args=(_cfg(interval_s=3600), stop),
                                  kwargs={"collect": collect, "log": lambda s: None})
        worker.start()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())

    def _main(self, run):
        env = {"STATS_API_KEY": KEY, "COLLECTOR_MACHINE": "test-personal"}
        with patch.dict(os.environ, env), patch("app.meridian_collector.run", run), \
                patch("app.meridian_collector.log"):
            code = mc.main()
        blocked = signal.pthread_sigmask(signal.SIG_BLOCK, [])
        self.assertNotIn(signal.SIGTERM, blocked)  # mask restored
        return code

    def test_main_stops_cleanly_on_sigterm(self):
        caller, seen = threading.get_ident(), []

        def run(cfg, stop):
            signal.pthread_kill(caller, signal.SIGTERM)
            seen.append(stop.wait(5))
        self.assertEqual(self._main(run), 0)
        self.assertEqual(seen, [True])

    def test_main_exits_nonzero_when_the_loop_crashes(self):
        def run(cfg, stop):
            raise RuntimeError(KEY)
        self.assertEqual(self._main(run), 1)

    def test_main_rejects_bad_config_without_running(self):
        with patch.dict(os.environ, {"STATS_API_KEY": ""}, clear=False), \
                patch("app.meridian_collector.run") as run, \
                patch("app.meridian_collector.log") as log:
            self.assertEqual(mc.main(), 2)
            run.assert_not_called()
            log.assert_called_once()


if __name__ == "__main__":
    unittest.main()
