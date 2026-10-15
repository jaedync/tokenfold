"""Write the Ledger fixtures: ledger.json, session.json, ledger-sessions/<id>.json.

Run: .venv/bin/python scripts/make_next_fixtures_ledger.py

One invented world from a fixed seed, shaped as /api/ledger and
/api/sessions/{id} (docs/ux-2026-10/BUILD.md). Every number agrees: a run's
cost is its tokens at list price, a session's cost is its threads, and the
daily spend is the sum of the sessions. Like production it holds many tiny
"quiet" sessions (one call, no subagents), Claude Code runs without a label
and Pi runs with nested children. Every name is a placeholder.
"""

import json
import random
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "static" / "next" / "fixtures"
TZ = ZoneInfo("America/Chicago")
H, D, MIN = 3600, 86400, 60
# Same clock as scripts/make_next_fixtures.py (the shell's mock "now").
NOW = datetime(2026, 10, 15, 14, 44, 10, tzinfo=TZ).timestamp()
DAYS = 30
TODAY0 = datetime.fromtimestamp(NOW, TZ).replace(hour=0, minute=0, second=0).timestamp()
START = datetime.fromtimestamp(TODAY0 - (DAYS - 1) * D + 3 * H, TZ).replace(hour=0).timestamp()
rng = random.Random(20261015)

# $ per MTok: input, output, cache write, cache read
PRICES = {
    "Fable 5.1": (10, 50, 12.5, 1.0), "Opus 5.5": (5, 25, 6.25, 0.5),
    "Opus 4.8": (5, 25, 6.25, 0.5), "Sonnet 5.5": (3, 15, 3.75, 0.3),
    "Haiku 5.5": (1, 5, 1.25, 0.1), "GPT-6 Luna": (2.5, 15, 0, 0.25),
    "DeepSeek V4.1 Flash": (0.27, 1.1, 0, 0.07), "GLM-5.3": (0.6, 2.2, 0, 0.11),
    "Qwen3.8 local": (0, 0, 0, 0),
}
PROVIDERS = {"Fable 5.1": "Anthropic", "Opus 5.5": "Anthropic", "Opus 4.8": "Anthropic",
             "Sonnet 5.5": "Anthropic", "Haiku 5.5": "Anthropic", "GPT-6 Luna": "OpenAI Codex",
             "DeepSeek V4.1 Flash": "OpenCode Go", "GLM-5.3": "OpenCode Go",
             "Qwen3.8 local": "Local"}
MACHINES = {  # sessions a weekday, Pi share, projects
    "laptop": (3, 0.0, ["api-server", "site", "dashboard", "notes"]),
    "devbox": (5, 0.3, ["api-server", "infra", "agent-tools", "dashboard"]),
    "homelab": (3, 0.8, ["infra", "agent-tools", "notes"]),
    "gpu-box": (1, 1.0, ["bench"]),
    "vm-01": (1, 0.5, ["site", "api-server"]),
}
TITLES = ["Fix flaky ingest test", "Add pagination to sessions", "Refactor auth middleware",
          "Speed up stats query", "Handle 429 from upstream", "Rework landing hero",
          "Rotate backup timers", "Harden firewall rules", "Decode throughput sweep",
          "Add session drawer", "Pivot table sorting", "Trim tool output", "Summarize the week"]
LABELS = ["explore-callers", "run-tests", "review-diff", "docs-search", "draft-migration",
          "profile-query", "check-layout", "bench-decode", "audit-errors", "write-tests"]
TOOLS_CC = {"Bash": 30, "Read": 22, "Edit": 15, "Grep": 10, "Glob": 4, "Write": 4, "WebFetch": 2}
TOOLS_PI = {"bash": 40, "read": 25, "edit": 18, "write": 5, "message": 5, "web_search": 3}


def wpick(weights):
    keys = list(weights)
    return rng.choices(keys, [max(0.0, weights[k]) for k in keys])[0]


def main_model(machine, t):
    if machine == "homelab":
        return wpick({"DeepSeek V4.1 Flash": 3, "GLM-5.3": 2, "Opus 5.5": 3})
    if machine == "gpu-box":
        return wpick({"Qwen3.8 local": 5, "Opus 5.5": 3})
    if machine == "vm-01":
        return wpick({"GPT-6 Luna": 6, "Sonnet 5.5": 2})
    # Opus 4.8 gives way to Opus 5.5 in the first third; Fable grows late.
    return wpick({"Opus 4.8": max(0, 1 - t / 0.35) * 3, "Opus 5.5": 0.2 + min(1, t / 0.3) * 4,
                  "Fable 5.1": max(0, t - 0.6) * 4, "Sonnet 5.5": 0.5})


def sub_model(machine, main):
    if machine == "homelab":
        return wpick({"DeepSeek V4.1 Flash": 4, "GLM-5.3": 3, "Haiku 5.5": 1})
    if machine == "gpu-box":
        return wpick({"Qwen3.8 local": 4, "Haiku 5.5": 1})
    return wpick({"Sonnet 5.5": 4, "Haiku 5.5": 2, main: 3})


def thread(model, minutes, sub):
    p = PRICES[model]
    tok = {"input": int(minutes * rng.uniform(100, 600)),
           "output": int(minutes * (rng.uniform(3000, 6000) if sub else rng.uniform(1200, 4000))),
           "cache_write": int(minutes * (rng.uniform(16e3, 36e3) if sub else rng.uniform(2e3, 8e3))),
           "cache_read": int(minutes * rng.uniform(0.4e6, 1.0e6))}
    cost = (tok["input"] * p[0] + tok["output"] * p[1] + tok["cache_write"] * p[2]
            + tok["cache_read"] * p[3]) / 1e6
    return tok, cost


def tool_mix(pi, n):
    out = {}
    for _ in range(n):
        k = wpick(TOOLS_PI if pi else TOOLS_CC)
        out[k] = out.get(k, 0) + 1
    return out


def add(acc, more):
    for k, v in more.items():
        acc[k] = acc.get(k, 0) + v


def make_runs(sid, machine, model, pi, start, end, n):
    runs, tokens, tools = [], {}, {}
    for i in range(n):
        s0 = start + rng.uniform(0, 0.85) * (end - start)
        e0 = min(end, s0 + rng.uniform(3, 30) * MIN)
        active = (e0 - s0) * rng.uniform(0.75, 0.95)
        rmodel = sub_model(machine, model)
        tok, cost = thread(rmodel, active / 60, True)
        calls = tool_mix(pi, int(active / 60 * rng.uniform(2, 5)))
        rid = f"{sid}-r{i:02d}"
        runs.append({"id": rid, "parent": None, "model": rmodel, "start": round(s0),
                     "end": round(e0), "cost_usd": cost, "tools": sum(calls.values()),
                     # Claude Code drops the Task description at ingest: no label.
                     "label": rng.choice(LABELS) if pi else None, "active_s": active})
        add(tokens, tok)
        add(tools, calls)
        # Pi workers can start workers of their own: a second level in the tree.
        if pi and rng.random() < 0.25:
            c0 = s0 + (e0 - s0) * rng.uniform(0.1, 0.5)
            c1 = min(e0, c0 + rng.uniform(2, 12) * MIN)
            ctok, ccost = thread(sub_model(machine, model), (c1 - c0) / 60, True)
            ccalls = tool_mix(pi, int((c1 - c0) / 60 * 3))
            runs.append({"id": f"{rid}c", "parent": rid, "model": sub_model(machine, model),
                         "start": round(c0), "end": round(c1), "cost_usd": ccost,
                         "tools": sum(ccalls.values()), "label": rng.choice(LABELS),
                         "active_s": (c1 - c0) * 0.9})
            add(tokens, ctok)
            add(tools, ccalls)
    runs.sort(key=lambda r: (r["start"], r["id"]))
    return runs, tokens, tools


def session(idx, machine, start, end, t, live=False, quiet=False):
    _n, pi_share, projects = MACHINES[machine]
    pi = rng.random() < pi_share
    model = main_model(machine, t) if not quiet else "Haiku 5.5"
    sid = f"s-{idx:04x}"
    active = (end - start) * rng.uniform(0.4, 0.8)
    tok, orch = thread(model, max(0.2, active / 60), False)
    tools = tool_mix(pi, 0 if quiet else int(active / 60 * rng.uniform(1.5, 3)))
    n_runs = 0 if quiet else min(14, int(rng.random() ** 1.3 * ((end - start) / H * 5 + 2)))
    runs, rtok, rtools = make_runs(sid, machine, model, pi, start, end, n_runs)
    add(tok, rtok)
    add(tools, rtools)
    if runs and not pi:
        tools["Task"] = sum(1 for r in runs if r["parent"] is None)
    agent = sum(r["cost_usd"] for r in runs)
    title = None if quiet or rng.random() < 0.25 else rng.choice(TITLES)
    return {"id": sid, "machine": machine, "project": rng.choice(projects), "title": title,
            "client": "pi-agent" if pi else "claude-code", "start": round(start),
            "end": round(end), "active_s": active,
            "prompts": rng.randint(0, 1) if quiet else max(1, int(active / 60 / 8 * rng.uniform(0.5, 1.5))),
            "model": model, "cost_usd": orch + agent, "orch_cost_usd": orch,
            "agent_cost_usd": agent, "runs": runs, "live": live,
            "orch_models": {model: orch}, "tool_calls": tools, "tokens": tok}


def world():
    sessions, idx = [], 0
    for d in range(DAYS):
        day0 = START + d * D
        weekend = datetime.fromtimestamp(day0 + 3 * H, TZ).weekday() >= 5
        t = d / (DAYS - 1)
        for machine, (n, _pi, _p) in MACHINES.items():
            for _ in range(rng.randint(0, 1) if weekend else rng.randint(max(0, n - 2), n)):
                s = day0 + rng.uniform(8, 21) * H
                e = s + min(6 * H, max(6 * MIN, rng.lognormvariate(4.1, 0.75) * MIN))
                if e > NOW - 3 * MIN:
                    continue
                sessions.append(session(idx, machine, s, e, t))
                idx += 1
        # A relay's own calls: a few seconds each, about $0.01 (the "quiet" ones).
        for _ in range(rng.randint(6, 14)):
            s = day0 + rng.uniform(0, 23.9) * H
            if s < NOW - MIN:
                sessions.append(session(idx, "laptop", s, s + rng.uniform(2, 40), t, quiet=True))
                idx += 1
    for machine, ago in (("devbox", 128), ("laptop", 46), ("homelab", 31)):
        sessions.append(session(idx, machine, NOW - ago * MIN, NOW - 20, 1.0, live=True))
        idx += 1
    sessions.sort(key=lambda s: (-s["end"], s["id"]))
    return sessions


def local_day(epoch):
    return datetime.fromtimestamp(epoch, TZ).strftime("%Y-%m-%d")


def daily(sessions):
    days = {local_day(START + d * D + 3 * H): {} for d in range(DAYS)}
    for s in sessions:
        day = days.setdefault(local_day(s["start"]), {})
        add(day, s["orch_models"])
        for r in s["runs"]:
            add(day, {r["model"]: r["cost_usd"]})
    return [{"day": k, "by_model": {m: round(v, 4) for m, v in sorted(by.items()) if v > 0}}
            for k, by in sorted(days.items())]


def ledger(sessions):
    runs = [r for s in sessions for r in s["runs"]]
    totals = {"cost_usd": sum(s["cost_usd"] for s in sessions), "sessions": len(sessions),
              "runs": len(runs), "active_s": sum(s["active_s"] for s in sessions),
              "agent_active_s": sum(r["active_s"] for r in runs),
              "prompts": sum(s["prompts"] for s in sessions)}
    return {"from": START, "to": NOW, "model_providers": PROVIDERS, "daily": daily(sessions),
            "sessions": sessions, "totals": totals}


def detail(s):
    models = dict(s["orch_models"])
    for r in s["runs"]:
        add(models, {r["model"]: r["cost_usd"]})
    total = s["cost_usd"]
    prompts = sorted(rng.uniform(s["start"], s["end"]) for _ in range(s["prompts"]))
    blocks = int(s["active_s"] / 30)
    anthropic = PROVIDERS[s["model"]] == "Anthropic"
    roll = rng.random()
    served = ({"state": "unknown", "blocks": 0} if not anthropic or not blocks
              else {"state": "rerouted" if roll < 0.06 else "partly_hidden" if roll < 0.16 else "as_asked",
                    "blocks": blocks})
    edits = sum(v for k, v in s["tool_calls"].items() if k.lower() in ("edit", "write"))
    return {**s, "prompt_times": prompts,
            "models": [{"model": m, "cost_usd": v, "share": v / total if total else None}
                       for m, v in sorted(models.items(), key=lambda kv: -kv[1])],
            "tools": [{"name": k, "calls": v, "errors": None}
                      for k, v in sorted(s["tool_calls"].items(), key=lambda kv: -kv[1])],
            "file_edits": edits, "served_model": served}


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def main():
    sessions = world()
    write(OUT / "ledger.json", ledger(sessions))
    shown = [s for s in sessions if s["runs"] or s["cost_usd"] >= 0.5 or s["prompts"] > 1]
    target = OUT / "ledger-sessions"
    for old in target.glob("*.json"):
        old.unlink()
    for s in shown:
        write(target / f"{s['id']}.json", detail(s))
    biggest = max(shown, key=lambda s: s["cost_usd"])
    write(OUT / "session.json", detail(biggest))
    print(f"wrote ledger.json ({len(sessions)} sessions, {len(shown)} details)", file=sys.stderr)


if __name__ == "__main__":
    main()
