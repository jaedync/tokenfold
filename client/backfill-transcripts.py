#!/usr/bin/env python3
"""Backfill tokenfold history from this machine's local Claude Code transcripts.

Repairs four things the server couldn't know at original ingest time:
  1. cache-tier split (5m vs 1h ephemeral cache writes) — events ingested
     before the nested `usage.cache_creation` shape was parsed have it as 0
     and were billed at the cheaper 5m rate
  2. server-tool request counts (usage.server_tool_use) — web search bills
     $10/1k requests; events ingested before capture have the counts as 0
  3. AI session titles — `ai-title` transcript records predating server capture
  4. thinking-block signature headers (sig_version / sig_header / sig_cipher_len),
     which reveal the model that actually served each block; events ingested
     before the client split the blob have them unset

Performance notes (transcript trees run to GBs):
  - files are scanned in PARALLEL (one process per CPU) and lines are
    prefiltered as BYTES — json decoding only happens on candidate lines
  - a skip-cache (~/.tokenfold-backfill-cache.json) remembers files already
    scanned AND submitted (by mtime+size), so re-runs only touch new/changed
    files; pass --full to rescan everything
  - day re-rolls happen ONCE server-side (batches defer; a final request
    carries the union of touched days)

Reads every transcript under ~/.claude/projects (recursively — includes
subagent and workflow transcripts) and POSTs batches to /api/backfill.
The server only fills values that are currently unset, so re-running is safe.

Config (same resolution as the push hook):
  TOKENFOLD_URL / ~/.config/notify-relay-url
  TOKENFOLD_API_KEY / ~/.config/tokenfold-api-key / ~/.config/notify-relay-token

Usage: python3 backfill-transcripts.py [--dry-run] [--full] [--serial]
"""

import base64
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

BATCH = 20_000  # server-side cap per request
CACHE_PATH = Path.home() / ".tokenfold-backfill-cache.json"
# Bump whenever harvest_file gains a NEW field: a cache written before the
# field existed marks files "done" that were never scanned for it, silently
# skipping all history on incremental re-runs. v2 = server_tool_use counts,
# v3 = thinking-block signature headers.
CACHE_VERSION = 3

# bytes-level prefilters: a line without any marker can't contribute
_B_CACHE = b'"cache_creation"'
_B_TITLE = b'"ai-title"'
_B_STU = b'"server_tool_use"'
_B_SIG = b'"signature"'


def split_signature(b64):
    """Split a thinking-block signature blob into (version, header_b64, cipher_len).

    The blob is protobuf: top level f1 = varint format version (absent means 0),
    f2 = envelope; the envelope's f1 is the plaintext header and f5 is the
    ciphertext. Returns the header re-encoded as standard base64, or None when
    the blob has no envelope/header.

    Deliberately tolerant: the header format changed four times in six weeks,
    so an unreadable signature must degrade to (0, None, 0) rather than raise.
    Losing an event because its signature is a shape we have not seen would be
    far worse than losing the signature.

    Kept self-contained (nested helpers, no module-level dependencies beyond
    base64) because an identical copy lives in the server's app/sigheader.py
    and a test asserts the two sources match byte for byte.
    """
    try:
        def read_varint(buf, i):
            """Return (value, index just past the varint)."""
            value = shift = 0
            while True:
                byte = buf[i]
                i += 1
                value |= (byte & 0x7F) << shift
                if not byte & 0x80:
                    return value, i
                shift += 7
                if shift > 63:
                    raise ValueError("varint too long")

        def walk(buf):
            """Yield (field_number, value) for one protobuf message.

            Varints come back as int and length-delimited fields as bytes;
            fixed-width fields are skipped since no field we want uses them.
            """
            i, end = 0, len(buf)
            while i < end:
                tag, i = read_varint(buf, i)
                field, wire = tag >> 3, tag & 7
                if field == 0:
                    raise ValueError("field number 0 is not valid protobuf")
                if wire == 0:
                    value, i = read_varint(buf, i)
                    yield field, value
                elif wire == 2:
                    size, i = read_varint(buf, i)
                    stop = i + size
                    if stop > end:
                        raise ValueError("length-delimited field overruns buffer")
                    yield field, buf[i:stop]
                    i = stop
                elif wire == 5:
                    i += 4
                elif wire == 1:
                    i += 8
                else:
                    raise ValueError("unsupported wire type")
                if i > end:
                    raise ValueError("field overruns buffer")

        # Transcripts store the blob unpadded often enough to matter.
        raw = base64.b64decode(b64 + "=" * (-len(b64) % 4))
        version, envelope = 0, None
        for field, value in walk(raw):
            if field == 1 and isinstance(value, int):
                version = value
            elif field == 2 and isinstance(value, bytes):
                envelope = value
        if envelope is None:
            return version, None, 0
        header, cipher_len = None, 0
        for field, value in walk(envelope):
            if field == 1 and isinstance(value, bytes):
                header = value
            elif field == 5 and isinstance(value, bytes):
                cipher_len = len(value)
        if header is None:
            return version, None, 0
        return version, base64.b64encode(header).decode("ascii"), cipher_len
    except Exception:
        return 0, None, 0


def read_config():
    url = os.environ.get("TOKENFOLD_URL")
    if not url:
        f = Path.home() / ".config" / "notify-relay-url"
        url = f.read_text().strip() if f.exists() else None
    key = os.environ.get("TOKENFOLD_API_KEY")
    if not key:
        for name in ("tokenfold-api-key", "notify-relay-token"):
            f = Path.home() / ".config" / name
            if f.exists():
                key = f.read_text().strip()
                break
    if not url or not key:
        sys.exit("config missing: need TOKENFOLD_URL + TOKENFOLD_API_KEY "
                 "(or ~/.config/notify-relay-url + tokenfold-api-key)")
    return url.rstrip("/"), key


def _first_thinking_signature(rec):
    """Return the signature of a record's first thinking block, or None.

    Claude Code writes one thinking block per assistant record, but the format
    permits several; the first is the one the server's ingest path keeps, so
    match it here or the two would disagree.
    """
    content = (rec.get("message") or {}).get("content")
    if not isinstance(content, list):
        return None
    for blk in content:
        if not isinstance(blk, dict) or blk.get("type") != "thinking":
            continue
        sig = blk.get("signature")
        if isinstance(sig, str) and sig:
            return sig
    return None


def harvest_file(path_str):
    """Scan ONE transcript. Returns (cache_tiers, server_tools, titles, sig_headers).
    Top-level function so it pickles for the process pool (Windows spawn)."""
    cache_tiers = {}
    server_tools = {}
    titles = {}
    sig_headers = {}
    try:
        with open(path_str, "rb") as fh:
            for line in fh:  # binary iteration: no decode cost on skipped lines
                has_cache = _B_CACHE in line
                has_title = _B_TITLE in line
                has_stu = _B_STU in line
                has_sig = _B_SIG in line
                if not (has_cache or has_title or has_stu or has_sig):
                    continue
                try:
                    rec = json.loads(line)
                except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
                    continue
                rtype = rec.get("type")
                if has_title and rtype == "ai-title":
                    sid, t = rec.get("sessionId"), rec.get("aiTitle")
                    if isinstance(sid, str) and sid and isinstance(t, str) and t:
                        titles[sid] = t[:256]  # last one per session wins
                elif rtype == "assistant":
                    uuid = rec.get("uuid")
                    if not uuid:
                        continue
                    usage = (rec.get("message") or {}).get("usage") or {}
                    if has_cache:
                        cc = usage.get("cache_creation")
                        if isinstance(cc, dict):
                            c5m = cc.get("ephemeral_5m_input_tokens", 0)
                            c1h = cc.get("ephemeral_1h_input_tokens", 0)
                            if (isinstance(c5m, int) and isinstance(c1h, int)
                                    and (c5m or c1h)):
                                cache_tiers[uuid] = [c5m, c1h]
                    if has_stu:
                        stu = usage.get("server_tool_use")
                        if isinstance(stu, dict):
                            ws = stu.get("web_search_requests", 0)
                            wf = stu.get("web_fetch_requests", 0)
                            # all-zero is the common case — no repair value,
                            # and the server treats 0 as unset anyway
                            if (isinstance(ws, int) and isinstance(wf, int)
                                    and (ws or wf)):
                                server_tools[uuid] = [ws, wf]
                    if has_sig:
                        sig = _first_thinking_signature(rec)
                        if sig:
                            version, header, cipher_len = split_signature(sig)
                            # header None means the blob is a shape we cannot
                            # read: nothing worth sending, and the server would
                            # reject it anyway
                            if header:
                                sig_headers[uuid] = [version, header, cipher_len]
    except OSError:
        pass  # unreadable file: skip; it stays out of the cache and retries next run
    return cache_tiers, server_tools, titles, sig_headers


def _load_cache(full):
    if full or not CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(CACHE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    # Legacy flat {path: sig} caches and older versions force a full rescan.
    if not isinstance(data, dict) or data.get("v") != CACHE_VERSION:
        return {}
    files = data.get("files")
    return files if isinstance(files, dict) else {}


def _save_cache(sigs):
    try:
        CACHE_PATH.write_text(json.dumps({"v": CACHE_VERSION, "files": sigs}))
    except OSError:
        pass


def _file_sig(st):
    return [int(st.st_mtime), st.st_size]


def harvest(full=False, serial=False):
    """Walk all transcripts (parallel);
    return (cache_tiers, server_tools, titles, sig_headers, scanned_sigs)."""
    root = Path.home() / ".claude" / "projects"
    cache = _load_cache(full)
    todo, skipped, sigs = [], 0, {}
    for path in sorted(root.rglob("*.jsonl")):
        try:
            sig = _file_sig(path.stat())
        except OSError:
            continue
        key = str(path)
        sigs[key] = sig
        if cache.get(key) == sig:
            skipped += 1
            continue
        todo.append(key)
    print(f"{len(todo)} files to scan ({skipped} unchanged, skipped via cache)",
          flush=True)

    cache_tiers, server_tools, titles, sig_headers = {}, {}, {}, {}
    started = time.time()
    done = 0

    def progress():
        if done % 100 == 0:
            print(f"  …{done}/{len(todo)} files ({len(cache_tiers)} splits, "
                  f"{len(server_tools)} server-tool, {len(titles)} titles, "
                  f"{len(sig_headers)} sig headers, "
                  f"{time.time()-started:.0f}s)", flush=True)

    if serial or len(todo) < 8:
        for key in todo:
            ct, st, tt, sh = harvest_file(key)
            cache_tiers.update(ct)
            server_tools.update(st)
            titles.update(tt)
            sig_headers.update(sh)
            done += 1
            progress()
    else:
        with ProcessPoolExecutor() as pool:
            futures = {pool.submit(harvest_file, key): key for key in todo}
            for fut in as_completed(futures):
                ct, st, tt, sh = fut.result()
                cache_tiers.update(ct)
                server_tools.update(st)
                titles.update(tt)
                sig_headers.update(sh)
                done += 1
                progress()
    print(f"scan finished in {time.time()-started:.1f}s", flush=True)
    return cache_tiers, server_tools, titles, sig_headers, sigs


def post(url, key, payload):
    req = urllib.request.Request(
        url + "/api/backfill",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-API-Key": key},
        method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def main():
    dry = "--dry-run" in sys.argv
    full = "--full" in sys.argv
    serial = "--serial" in sys.argv
    url, key = read_config()
    cache_tiers, server_tools, titles, sig_headers, sigs = harvest(
        full=full, serial=serial)
    print(f"harvested: {len(cache_tiers)} events with cache split, "
          f"{len(server_tools)} with server-tool counts, "
          f"{len(sig_headers)} with signature headers, "
          f"{len(titles)} session titles", flush=True)
    if dry:
        print("dry run — nothing sent")
        return

    items = list(cache_tiers.items())
    st_items = list(server_tools.items())
    sig_items = list(sig_headers.items())
    tot_events = tot_titles = tot_st = tot_sig = 0
    days = set()
    # Data batches defer the expensive server-side day re-roll (reroll=false);
    # titles ride along with the first batch (they're small).
    first = True
    for i in range(0, max(len(items), 1), BATCH):
        chunk = dict(items[i:i + BATCH])
        payload = {"cache_tiers": chunk, "titles": titles if first else {},
                   "reroll": False}
        first = False
        try:
            r = post(url, key, payload)
        except Exception as e:
            sys.exit(f"batch failed at offset {i}: {e}")
        tot_events += r["updated_events"]
        tot_titles += r["updated_titles"]
        days.update(r["touched_days"])
        print(f"  batch {i // BATCH + 1}: +{r['updated_events']} events, "
              f"+{r['updated_titles']} titles", flush=True)

    # Server-tool counts in their own batches (same deferred-re-roll protocol).
    # .get(): a server that predates this field ignores it and won't echo it.
    for i in range(0, len(st_items), BATCH):
        chunk = dict(st_items[i:i + BATCH])
        try:
            r = post(url, key, {"server_tools": chunk, "reroll": False})
        except Exception as e:
            sys.exit(f"server-tool batch failed at offset {i}: {e}")
        tot_st += r.get("updated_server_tools", 0)
        days.update(r["touched_days"])
        print(f"  server-tool batch {i // BATCH + 1}: "
              f"+{r.get('updated_server_tools', 0)} events", flush=True)

    # Signature headers in their own batches, same deferred-re-roll protocol.
    # A server that predates this field silently ignores it (pydantic drops
    # unknown keys) and does not echo updated_sig_headers. That must NOT count
    # as done: the fleet guard writes its done-marker on exit 0 and the
    # skip-cache below would mark these files processed, so an old server
    # would permanently swallow this machine's history. Track the ack.
    sig_acked = True
    for i in range(0, len(sig_items), BATCH):
        chunk = dict(sig_items[i:i + BATCH])
        try:
            r = post(url, key, {"sig_headers": chunk, "reroll": False})
        except Exception as e:
            sys.exit(f"sig-header batch failed at offset {i}: {e}")
        if "updated_sig_headers" not in r:
            sig_acked = False
        tot_sig += r.get("updated_sig_headers", 0)
        days.update(r["touched_days"])
        print(f"  sig-header batch {i // BATCH + 1}: "
              f"+{r.get('updated_sig_headers', 0)} events", flush=True)

    if days:
        print(f"re-rolling {len(days)} affected days (single final pass)…", flush=True)
        try:
            post(url, key, {"cache_tiers": {}, "titles": {},
                            "reroll_days": sorted(days)})
        except Exception as e:
            sys.exit(f"final re-roll failed: {e} — re-run to retry")

    if sig_items and not sig_acked:
        print("server did not acknowledge sig_headers (predates this field): "
              "skip-cache NOT saved, re-run after the server is upgraded",
              file=sys.stderr, flush=True)
        sys.exit(3)

    # only now is it safe to remember these files as fully processed
    _save_cache(sigs)
    print(f"done: {tot_events} events repaired, {tot_st} server-tool counts, "
          f"{tot_sig} signature headers, {tot_titles} titles added, "
          f"{len(days)} days re-rolled")


if __name__ == "__main__":
    main()
