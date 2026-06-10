#!/usr/bin/env python3
"""Backfill tokenfold history from this machine's local Claude Code transcripts.

Repairs two things the server couldn't know at original ingest time:
  1. cache-tier split (5m vs 1h ephemeral cache writes) — events ingested
     before the nested `usage.cache_creation` shape was parsed have it as 0
     and were billed at the cheaper 5m rate
  2. AI session titles — `ai-title` transcript records predating server capture

Reads every transcript under ~/.claude/projects (recursively — includes
subagent and workflow transcripts) and POSTs batches to /api/backfill.
The server only fills values that are currently unset, so re-running is safe.

Config (same resolution as the push hook):
  TOKENFOLD_URL / ~/.config/notify-relay-url
  TOKENFOLD_API_KEY / ~/.config/tokenfold-api-key / ~/.config/notify-relay-token

Usage: python3 backfill-transcripts.py [--dry-run]
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

BATCH = 20_000  # server-side cap per request


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


def harvest():
    """Walk all transcripts; return (cache_tiers, titles)."""
    cache_tiers: dict = {}
    titles: dict = {}
    root = Path.home() / ".claude" / "projects"
    files = sorted(root.rglob("*.jsonl"))
    print(f"scanning {len(files)} transcript files under {root}")
    for path in files:
        try:
            with open(path, errors="ignore") as fh:
                for line in fh:
                    # cheap pre-filters before paying for json.loads
                    if '"cache_creation"' not in line and '"ai-title"' not in line:
                        continue
                    try:
                        rec = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    rtype = rec.get("type")
                    if rtype == "ai-title":
                        sid, t = rec.get("sessionId"), rec.get("aiTitle")
                        if isinstance(sid, str) and sid and isinstance(t, str) and t:
                            titles[sid] = t[:256]  # last one per session wins
                    elif rtype == "assistant":
                        uuid = rec.get("uuid")
                        usage = (rec.get("message") or {}).get("usage") or {}
                        cc = usage.get("cache_creation")
                        if not (uuid and isinstance(cc, dict)):
                            continue
                        c5m = cc.get("ephemeral_5m_input_tokens", 0)
                        c1h = cc.get("ephemeral_1h_input_tokens", 0)
                        if isinstance(c5m, int) and isinstance(c1h, int) and (c5m or c1h):
                            cache_tiers[uuid] = [c5m, c1h]
        except OSError as e:
            print(f"  skip {path.name}: {e}", file=sys.stderr)
    return cache_tiers, titles


def post(url, key, payload):
    req = urllib.request.Request(
        url + "/api/backfill",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-API-Key": key},
        method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def main():
    dry = "--dry-run" in sys.argv
    url, key = read_config()
    cache_tiers, titles = harvest()
    print(f"harvested: {len(cache_tiers)} events with cache split, {len(titles)} session titles")
    if dry:
        print("dry run — nothing sent")
        return

    items = list(cache_tiers.items())
    tot_events = tot_titles = 0
    days: set = set()
    # titles ride along with the first batch (they're small)
    first = True
    for i in range(0, max(len(items), 1), BATCH):
        chunk = dict(items[i:i + BATCH])
        payload = {"cache_tiers": chunk, "titles": titles if first else {}}
        first = False
        try:
            r = post(url, key, payload)
        except Exception as e:
            sys.exit(f"batch failed at offset {i}: {e}")
        tot_events += r["updated_events"]
        tot_titles += r["updated_titles"]
        days.update(r["touched_days"])
        print(f"  batch {i // BATCH + 1}: +{r['updated_events']} events, "
              f"+{r['updated_titles']} titles")
    print(f"done: {tot_events} events repaired, {tot_titles} titles added, "
          f"{len(days)} days re-rolled")


if __name__ == "__main__":
    main()
