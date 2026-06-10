#!/usr/bin/env python3
"""Backfill tokenfold history from this machine's local Claude Code transcripts.

Repairs three things the server couldn't know at original ingest time:
  1. cache-tier split (5m vs 1h ephemeral cache writes) — events ingested
     before the nested `usage.cache_creation` shape was parsed have it as 0
     and were billed at the cheaper 5m rate
  2. server-tool request counts (usage.server_tool_use) — web search bills
     $10/1k requests; events ingested before capture have the counts as 0
  3. AI session titles — `ai-title` transcript records predating server capture

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
# skipping all history on incremental re-runs. v2 = server_tool_use counts.
CACHE_VERSION = 2

# bytes-level prefilters: a line without any marker can't contribute
_B_CACHE = b'"cache_creation"'
_B_TITLE = b'"ai-title"'
_B_STU = b'"server_tool_use"'


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


def harvest_file(path_str):
    """Scan ONE transcript. Returns (cache_tiers, server_tools, titles).
    Top-level function so it pickles for the process pool (Windows spawn)."""
    cache_tiers = {}
    server_tools = {}
    titles = {}
    try:
        with open(path_str, "rb") as fh:
            for line in fh:  # binary iteration: no decode cost on skipped lines
                has_cache = _B_CACHE in line
                has_title = _B_TITLE in line
                has_stu = _B_STU in line
                if not (has_cache or has_title or has_stu):
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
    except OSError:
        pass  # unreadable file: skip; it stays out of the cache and retries next run
    return cache_tiers, server_tools, titles


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
    return (cache_tiers, server_tools, titles, scanned_sigs)."""
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

    cache_tiers, server_tools, titles = {}, {}, {}
    started = time.time()
    done = 0

    def progress():
        if done % 100 == 0:
            print(f"  …{done}/{len(todo)} files ({len(cache_tiers)} splits, "
                  f"{len(server_tools)} server-tool, {len(titles)} titles, "
                  f"{time.time()-started:.0f}s)", flush=True)

    if serial or len(todo) < 8:
        for key in todo:
            ct, st, tt = harvest_file(key)
            cache_tiers.update(ct)
            server_tools.update(st)
            titles.update(tt)
            done += 1
            progress()
    else:
        with ProcessPoolExecutor() as pool:
            futures = {pool.submit(harvest_file, key): key for key in todo}
            for fut in as_completed(futures):
                ct, st, tt = fut.result()
                cache_tiers.update(ct)
                server_tools.update(st)
                titles.update(tt)
                done += 1
                progress()
    print(f"scan finished in {time.time()-started:.1f}s", flush=True)
    return cache_tiers, server_tools, titles, sigs


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
    cache_tiers, server_tools, titles, sigs = harvest(full=full, serial=serial)
    print(f"harvested: {len(cache_tiers)} events with cache split, "
          f"{len(server_tools)} with server-tool counts, "
          f"{len(titles)} session titles", flush=True)
    if dry:
        print("dry run — nothing sent")
        return

    items = list(cache_tiers.items())
    st_items = list(server_tools.items())
    tot_events = tot_titles = tot_st = 0
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

    if days:
        print(f"re-rolling {len(days)} affected days (single final pass)…", flush=True)
        try:
            post(url, key, {"cache_tiers": {}, "titles": {},
                            "reroll_days": sorted(days)})
        except Exception as e:
            sys.exit(f"final re-roll failed: {e} — re-run to retry")

    # only now is it safe to remember these files as fully processed
    _save_cache(sigs)
    print(f"done: {tot_events} events repaired, {tot_st} server-tool counts, "
          f"{tot_titles} titles added, {len(days)} days re-rolled")


if __name__ == "__main__":
    main()
