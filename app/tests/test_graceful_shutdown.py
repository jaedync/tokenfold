"""SIGTERM stops the server cleanly while a dashboard holds its SSE stream open.

uvicorn waits for open connections before the lifespan shutdown, and the
stats stream never ends by itself, so without a graceful timeout a stop
waited until Docker killed the process: the lifespan shutdown (thread joins,
closing the database) never ran. The test starts the server with the
Dockerfile's own command.
"""
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def dockerfile_cmd():
    match = re.search(r"^CMD\s+(\[.*\])\s*$", (ROOT / "Dockerfile").read_text(), re.M)
    return json.loads(match.group(1))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def compose_seconds(text, key):
    match = re.search(rf"^\s+{key}:\s*(\d+)s\s*$", text, re.M)
    return int(match.group(1)) if match else None


class GracefulShutdownTest(unittest.TestCase):
    def test_sigterm_with_an_open_stream_runs_the_lifespan_shutdown(self):
        port = free_port()
        cmd = dockerfile_cmd()
        self.assertEqual(cmd[0], "uvicorn")
        args = [a for a in cmd[1:]]
        args[args.index("--host") + 1] = "127.0.0.1"
        args[args.index("--port") + 1] = str(port)
        tmp = tempfile.mkdtemp()
        env = dict(os.environ, DB_PATH=os.path.join(tmp, "t.db"), STATS_API_KEY="x",
                   TOKENFOLD_PREWARM="0", TOKENFOLD_PRICING_FETCH="0", DASHBOARD_PASSWORD="")
        log = open(os.path.join(tmp, "server.log"), "w+")
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", *args], cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1).read()
                break
            except OSError:
                time.sleep(0.2)
        stream = socket.create_connection(("127.0.0.1", port))
        self.addCleanup(stream.close)
        stream.sendall(b"GET /api/stats/stream HTTP/1.1\r\nHost: t\r\nAccept: text/event-stream\r\n\r\n")
        stream.settimeout(10)
        self.assertIn(b"200", stream.recv(64))

        started = time.time()
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.fail("the server still runs 20 s after SIGTERM")
        log.seek(0)
        out = log.read()
        self.assertIn("Application shutdown complete", out)
        # Docker must wait longer than the graceful timeout plus the lifespan's
        # thread-join deadline, or it kills the shutdown half way.
        graceful = int(args[args.index("--timeout-graceful-shutdown") + 1])
        grace = compose_seconds((ROOT / "docker-compose.yml").read_text(), "stop_grace_period")
        self.assertIsNotNone(grace, "docker-compose.yml sets stop_grace_period for tokenfold")
        self.assertGreaterEqual(grace, graceful + 10 + 5)
        self.assertLess(time.time() - started, 20)


if __name__ == "__main__":
    unittest.main()
