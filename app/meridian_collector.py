"""Always-on Claude quota collector for a host-local Meridian.

The dotfleet Pi reporter pushes quota only on Pi session events, so the
dashboard fell minutes to hours behind whenever Pi was idle or another client
spent the quota. This polls the same Meridian route on a fixed cadence and
posts through the same validated endpoint, so ordering, history and caching
stay in one server code path.

Run as `python -m app.meridian_collector` (the `meridian` compose profile).
"""
import http.client
import json
import os
import signal
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import NamedTuple
from urllib.parse import urlsplit

from .meridian_quota import parse_meridian_quota

DEFAULT_MERIDIAN_URL = "http://127.0.0.1:3456"
DEFAULT_TOKENFOLD_URL = "http://127.0.0.1:5055"
# Anthropic rate-limits the OAuth usage endpoint per token. Measured on ms01
# (2026-10-04): a 60 s poll got HTTP 429 on every second request, so a fresh
# observation arrived every 120 s either way. 120 s keeps that freshness
# without the rejected requests.
DEFAULT_INTERVAL_S = 120
MIN_INTERVAL_S = 30
MAX_INTERVAL_S = 3600
REQUEST_TIMEOUT_S = 5
MAX_BODY_BYTES = 65536
# Meridian serves a stale OAuth answer for up to 15 min; past that, a run of
# "not newer" replies means nothing is refreshing the quota.
NO_NEW_OBSERVATION_S = 900
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
HEALTHY = {"posted", "stale"}
SHUTDOWN_SIGNALS = {signal.SIGTERM, signal.SIGINT}


class Config(NamedTuple):
    meridian: str
    tokenfold: str
    api_key: str
    machine: str
    interval_s: int


def log(message):
    print(f"[meridian_collector] {message}", flush=True)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None  # urllib then raises HTTPError for the 3xx


# Explicit empty ProxyHandler: environment proxies must never see this traffic.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)


def _read_json(response, deadline):
    """Bounded in size and in total time, not only per socket operation."""
    declared = response.headers.get("Content-Length")
    if declared is not None and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        return None
    chunks, size = [], 0
    while True:
        if time.monotonic() > deadline:
            return None
        chunk = response.read1(16384)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            return None
        chunks.append(chunk)
    try:
        return json.loads(b"".join(chunks))
    except (ValueError, RecursionError):  # 64 KiB of '[' nests past the limit
        return None


def fetch_json(url, timeout=REQUEST_TIMEOUT_S):
    """Bounded credential-free GET; None for any failure, never an exception."""
    deadline = time.monotonic() + timeout
    try:
        with _OPENER.open(urllib.request.Request(url, method="GET"), timeout=timeout) as resp:
            return _read_json(resp, deadline) if resp.status == 200 else None
    except urllib.error.HTTPError as error:
        error.close()
        return None
    # URLError is an OSError; HTTPException covers bad status lines and
    # truncated bodies; ValueError covers malformed URLs.
    except (OSError, http.client.HTTPException, ValueError):
        return None


def post_json(url, payload, api_key, timeout=REQUEST_TIMEOUT_S):
    """POST strict JSON. Returns (status, body) or (None, None) on transport failure."""
    data = json.dumps(payload, allow_nan=False).encode()
    request = urllib.request.Request(url, data=data, method="POST", headers={
        "Content-Type": "application/json", "X-API-Key": api_key})
    deadline = time.monotonic() + timeout
    try:
        with _OPENER.open(request, timeout=timeout) as resp:
            return resp.status, _read_json(resp, deadline)
    except urllib.error.HTTPError as error:
        error.close()
        return error.code, None
    except (OSError, http.client.HTTPException):
        return None, None


def local_meridian_base(url):
    """Loopback origin only: the collector must never contact a remote proxy."""
    if not isinstance(url, str) or "?" in url or "#" in url:
        return None
    try:
        parts = urlsplit(url)
        host, _ = parts.hostname, parts.port  # .port raises on a bad port
    except ValueError:
        return None
    if (parts.scheme not in ("http", "https") or parts.username or parts.password
            or host not in LOOPBACK_HOSTS):
        return None
    return f"{parts.scheme}://{parts.netloc}"


def collect_once(cfg, *, fetch=fetch_json, post=post_json, now=time.time):
    """One poll. Returns a short status; never includes response content."""
    health = fetch(cfg.meridian + "/health")
    if health is None:
        return "unavailable"
    auth = health.get("auth") if isinstance(health, dict) else None
    if not isinstance(auth, dict) or auth.get("subscriptionType") not in ("pro", "max"):
        return "not_personal"
    body = fetch(cfg.meridian + "/v1/usage/quota")
    if body is None:
        return "unavailable"
    payload = parse_meridian_quota(body, cfg.machine, now())
    if payload is None:
        return "invalid"
    code, reply = post(cfg.tokenfold + "/api/usage/claude", payload, cfg.api_key)
    if code is None:
        return "send_failed"
    if code != 200:
        return f"rejected_{code}"
    return "posted" if isinstance(reply, dict) and reply.get("status") == "ok" else "stale"


def _tokenfold_base(url):
    try:
        parts = urlsplit(url)
        host, _ = parts.hostname, parts.port  # .port raises on a bad port
    except ValueError:
        host = None
    if (not host or parts.scheme not in ("http", "https") or parts.username
            or parts.password or parts.query or parts.fragment):
        raise ValueError("TOKENFOLD_URL must be a plain http(s) URL")
    if parts.scheme == "http" and host not in LOOPBACK_HOSTS:
        raise ValueError("TOKENFOLD_URL must use https unless it is loopback")
    return url.rstrip("/")


def load_config(env):
    """Validate the environment. Error messages never echo the API key."""
    api_key = env.get("STATS_API_KEY", "")
    if not api_key:
        raise ValueError("STATS_API_KEY is required")
    # http.client sends header values as latin-1; anything else fails every poll.
    if not api_key.isascii() or not api_key.isprintable():
        raise ValueError("STATS_API_KEY must be printable ASCII")
    meridian = local_meridian_base(env.get("MERIDIAN_URL", DEFAULT_MERIDIAN_URL))
    if meridian is None:
        raise ValueError("MERIDIAN_URL must be a loopback http(s) URL")
    tokenfold = _tokenfold_base(env.get("TOKENFOLD_URL", DEFAULT_TOKENFOLD_URL))
    machine = env.get("COLLECTOR_MACHINE") or socket.gethostname()
    if not machine.strip() or len(machine) > 128 or not machine.isprintable():
        raise ValueError("COLLECTOR_MACHINE must be 1-128 printable characters")
    raw_interval = env.get("COLLECTOR_INTERVAL_S") or str(DEFAULT_INTERVAL_S)
    if not raw_interval.isdigit() or not MIN_INTERVAL_S <= int(raw_interval) <= MAX_INTERVAL_S:
        raise ValueError(f"COLLECTOR_INTERVAL_S must be {MIN_INTERVAL_S}-{MAX_INTERVAL_S}")
    return Config(meridian, tokenfold, api_key, machine, int(raw_interval))


def _state(status, since_new_s):
    if status not in HEALTHY:
        return status
    # "stale" alone is normal: another reporter may have posted a newer sample.
    return "healthy" if since_new_s < NO_NEW_OBSERVATION_S else "no_new_observation"


def run(cfg, stop, *, collect=collect_once, log=log, clock=time.monotonic):
    """Poll until ``stop`` is set. Logs only health transitions."""
    last, last_new = None, clock()
    while not stop.is_set():
        started = clock()
        try:
            status = collect(cfg)
        except Exception as error:  # the loop must outlive any single poll
            status = f"error_{type(error).__name__}"
        if status == "posted":
            last_new = clock()
        state = _state(status, clock() - last_new)
        if state != last:
            log("quota collection healthy" if state == "healthy"
                else f"quota collection degraded: {state}")
            last = state
        stop.wait(max(0.0, cfg.interval_s - (clock() - started)))


def _worker(cfg, stop, crashed, main_ident):
    try:
        run(cfg, stop)
    except BaseException as error:
        log(f"collector loop crashed: {type(error).__name__}")
    finally:
        if not stop.is_set():
            crashed.set()
            # Wake main's sigwait so the process exits and Docker restarts it.
            signal.pthread_kill(main_ident, signal.SIGTERM)


def main():
    try:
        cfg = load_config(os.environ)
    except ValueError as error:
        log(f"not starting: {error}")
        return 2
    # sigwait instead of a handler: a handler calling Event.set() can deadlock
    # if the signal lands while the main thread holds the Event's lock.
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, SHUTDOWN_SIGNALS)
    stop, crashed = threading.Event(), threading.Event()
    try:
        log(f"collecting every {cfg.interval_s}s from {cfg.meridian} as {cfg.machine}")
        worker = threading.Thread(target=_worker, name="collector", daemon=True,
                                  args=(cfg, stop, crashed, threading.get_ident()))
        worker.start()  # inherits the blocked mask
        signal.sigwait(SHUTDOWN_SIGNALS)
        stop.set()
        worker.join(timeout=REQUEST_TIMEOUT_S * 3)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)
    return 1 if crashed.is_set() else 0


if __name__ == "__main__":
    sys.exit(main())
