#!/usr/bin/env python3
"""Send Antigravity model counters and quota snapshots without transcript text."""
from __future__ import annotations

import fcntl
import json
import os
import re
import socket
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import UUID

TRANSCRIPT = ".system_generated/logs/transcript_full.jsonl"
BATCH_SIZE = 500
CURSOR_VERSION = 1
COUNTERS = ("input_tokens", "cache_read_tokens", "output_tokens")
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
ISO_TIME = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,9})?(?:Z|[+-]\d\d:\d\d)", re.ASCII)


class PushError(RuntimeError):
    """A report was not acknowledged."""


def _uuid(value):
    try:
        return str(UUID(value))
    except (TypeError, ValueError, AttributeError):
        return None


def _epoch(value):
    if not isinstance(value, str) or not ISO_TIME.fullmatch(value):
        return None
    try:
        # Python 3.9 only accepts 3 or 6 fractional digits. Keep microseconds.
        normalized = re.sub(r"\.(\d+)", lambda m: "." + m[1][:6].ljust(6, "0"), value)
        epoch = datetime.fromisoformat(normalized.replace("Z", "+00:00")).timestamp()
    except (ValueError, OverflowError):
        return None
    return epoch if 0 <= epoch <= 10**11 else None


def _count(value):
    return type(value) is int and 0 <= value <= 10**12


def parse_transcript_step(raw: bytes, conversation_id: str) -> dict | None:
    """Allowlist only the counters from completed MODEL steps."""
    try:
        doc = json.loads(raw)
    except (ValueError, RecursionError):
        return None
    if not isinstance(doc, dict) or doc.get("source") != "MODEL" or doc.get("status") != "DONE":
        return None
    conversation = _uuid(conversation_id)
    index, stamp, model = (doc.get(k) for k in ("step_index", "created_at", "model"))
    if (conversation is None or not _count(index) or _epoch(stamp) is None
            or not isinstance(model, str) or not IDENTIFIER.fullmatch(model)):
        return None
    counters = {key: doc.get(key) for key in COUNTERS}
    if not all(_count(value) for value in counters.values()):
        return None
    return {"conversation_id": conversation, "step_index": index,
            "timestamp": stamp, "model": model, **counters}


def _signature(stat):
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def sync_transcripts(root: Path, cursors: dict, post) -> None:
    """Scan changed files, including in-place rewrites; commit only acknowledged IDs.

    The set of steps survives rotation. Unlike a high-water mark, it allows a
    previously incomplete step to become final behind a later step.
    """
    pending, steps, signatures = [], {}, {}
    complete = False

    def flush():
        nonlocal pending
        if pending:
            post(pending)
            pending = []

    try:
        for path in sorted(root.glob(f"*/{TRANSCRIPT}")):
            cid = _uuid(path.parents[2].name)
            if cid is None:
                continue
            prev = cursors.get(cid, {})
            try:
                stream = path.open("rb")
            except FileNotFoundError:
                continue  # Rotated before open. Check it again next run.
            with stream:
                signature = _signature(os.fstat(stream.fileno()))
                if signature == prev.get("signature"):
                    continue
                seen = steps.setdefault(cid, set(prev.get("steps", ())))
                for raw in stream:
                    if not raw.endswith(b"\n"):
                        break  # The final line can still be in progress.
                    event = parse_transcript_step(raw, cid)
                    if event is None or event["step_index"] in seen:
                        continue
                    seen.add(event["step_index"])
                    pending.append(event)
                    if len(pending) >= BATCH_SIZE:
                        flush()
                if signature == _signature(os.fstat(stream.fileno())):
                    signatures[cid] = signature
        flush()
        complete = True
    finally:
        for event in pending:
            steps[event["conversation_id"]].remove(event["step_index"])
        for cid, seen in steps.items():
            cursors[cid] = {"steps": sorted(seen),
                            "signature": signatures.get(cid) if complete else None}


def load_cursors(path: Path) -> dict:
    try:
        doc = json.loads(path.read_bytes())
        if isinstance(doc, dict) and doc.get("version") == CURSOR_VERSION:
            cursors = doc.get("conversations")
            if isinstance(cursors, dict) and all(
                _uuid(cid) == cid and isinstance(c, dict) and isinstance(c.get("steps"), list)
                and all(_count(i) for i in c["steps"]) for cid, c in cursors.items()
            ):
                return cursors
    except FileNotFoundError:
        return {}
    except (ValueError, RecursionError):
        print("Invalid Antigravity cursor; retrying from transcripts", file=sys.stderr)
        return {}
    print("Invalid Antigravity cursor; retrying from transcripts", file=sys.stderr)
    return {}


def save_cursors(path: Path, cursors: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".cursor-", delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump({"version": CURSOR_VERSION, "conversations": cursors},
                      stream, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _post(url: str, key: str, endpoint: str, body: dict) -> dict:
    req = Request(url + endpoint, data=json.dumps(body).encode(), method="POST",
                  headers={"Content-Type": "application/json", "X-API-Key": key})
    try:
        with build_opener(_NoRedirect()).open(req, timeout=60) as response:
            if response.status != 200:
                raise PushError(f"{endpoint}: HTTP {response.status}")
            result = json.loads(response.read(65536))
    except HTTPError as error:
        error.close()
        raise PushError(f"{endpoint}: HTTP {error.code}") from None
    except (URLError, OSError, ValueError, RecursionError):
        raise PushError(f"{endpoint}: no valid response") from None
    if not isinstance(result, dict):
        raise PushError(f"{endpoint}: invalid response")
    return result


def _setting(env: str, *files: Path) -> str:
    value = os.environ.get(env, "").strip()
    if value:
        return value
    for path in files:
        try:
            value = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue
        if value:
            return value
    return ""


def sync_spend(root, state_path, send, machine):
    cursors = load_cursors(state_path)
    before = cursors.copy()

    def post(events):
        reply = send("/api/ingest/antigravity", {"machine": machine, "events": events})
        accepted, duplicates = reply.get("accepted"), reply.get("duplicates")
        if not (_count(accepted) and _count(duplicates) and accepted + duplicates == len(events)):
            raise PushError("Antigravity ingest did not acknowledge the batch")

    try:
        sync_transcripts(root, cursors, post)
    finally:
        if cursors != before:
            save_cursors(state_path, cursors)


def main() -> int:
    home = Path.home()
    config = home / ".config"
    state = Path(os.environ.get("XDG_STATE_HOME") or home / ".local/state") / "tokenfold-antigravity"
    try:
        url = _setting("TOKENFOLD_URL", config / "notify-relay-url").rstrip("/")
        key = _setting("TOKENFOLD_API_KEY", config / "tokenfold-api-key", config / "notify-relay-token")
        machine = os.environ.get("TOKENFOLD_MACHINE") or socket.gethostname()
        parsed = urlsplit(url)
        if (not key or not parsed.hostname or parsed.username is not None
                or parsed.query or parsed.fragment
                or not IDENTIFIER.fullmatch(machine)
                or not (parsed.scheme == "https" or parsed.scheme == "http"
                        and parsed.hostname in {"localhost", "127.0.0.1", "::1"})):
            raise PushError("Set a Tokenfold URL, ingest key, and valid machine name")
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (state / "push.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return 0

            def send(endpoint, body):
                return _post(url, key, endpoint, body)

            sync_spend(home / ".gemini/antigravity-cli/brain", state / "cursor.json", send, machine)
    except PushError as error:
        print(error, file=sys.stderr)
        return 1
    except (OSError, ValueError):
        print("Cannot read Antigravity input or write the push cursor", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
