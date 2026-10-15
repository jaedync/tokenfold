"""Write static/next/fixtures/*.json: one invented, internally consistent world.

Run: .venv/bin/python scripts/make_next_fixtures.py

The fixtures feed /next?mock=1 and the browser test. They follow the endpoint
shapes in docs/ux-2026-10/BUILD.md. Pace blocks come from app.usage_pace, so
the mock verdicts are the ones the server would give for the same numbers.
Every name is a placeholder; no production data is read.
"""

import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.usage_pace import tight_epoch, usage_pace  # noqa: E402

OUT = ROOT / "static" / "next" / "fixtures"
TZ = ZoneInfo("America/Chicago")
H, D = 3600, 86400
NOW = datetime(2026, 10, 15, 14, 44, 10, tzinfo=TZ).timestamp()
rng = random.Random(20261015)


def at(month, day, hour=0, minute=0, tz=TZ):
    return datetime(2026, month, day, hour, minute, tzinfo=tz).timestamp()


def floor_min(t):
    return (t // 60) * 60.0


def write(name, payload):
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} B)")


# ---------------------------------------------------------------- runway
def window(provider, key, label, kind, window_s, pct, resets_at, h1, h6, usd_spent, age_s=60):
    pace = usage_pace(pct, resets_at, window_s, NOW) or {}
    pace = {k: pace.get(k) for k in ("pace_state", "expected_pct", "projected_pct",
                                    "reaches_100_epoch", "elapsed_fraction")}
    relevant = h1 if kind == "short" else h6
    eta = floor_min(NOW + (100 - pct) / relevant * H) if relevant and relevant > 0 else None
    tight = tight_epoch(pace.get("pace_state"), eta, resets_at)
    burn = {"h1": h1, "h6": h6}
    if provider != "claude":
        # Provider snapshots keep no reading history: no burn, no TIGHT (be-runway).
        tight, burn = None, None
    reaches = pace.get("reaches_100_epoch")
    per_point = usd_spent / pct if usd_spent is not None and pct else None
    if reaches is not None and reaches < resets_at:
        verdict = "runs_out"
    elif tight is not None:
        verdict = "tight"
    elif pace.get("pace_state") == "too_early":
        verdict = "too_early"
    else:
        verdict = "clear"
    waitable = ((kind == "short" and resets_at - NOW <= H)
                or (kind == "long" and reaches is not None and resets_at < reaches))
    return {"id": f"{provider}:{key}", "provider": provider, "key": key, "label": label,
            "kind": kind, "window_s": window_s, "pct": pct, "resets_at": resets_at,
            "window_start": resets_at - window_s, "observed_at": floor_min(NOW - age_s),
            "stale": age_s > 900, "age_s": age_s, "pace": pace, "tight_epoch": tight,
            "burn_pct_per_h": burn, "usd_spent": usd_spent,
            "usd_per_point": per_point,
            "usd_left": (100 - pct) * per_point if per_point is not None else None,
            "waitable": waitable, "verdict": verdict}


SEVEN_RESET = at(10, 16, 3)
PER_POINT_7D = 24.6
personal_windows = [
    window("claude", "seven_day", "Claude · 7-day", "long", 7 * D, 92.0, SEVEN_RESET,
           0.80, 0.70, 92.0 * PER_POINT_7D),
    window("claude", "scoped:fable", "Claude · Fable 7-day", "long", 7 * D, 41.0, SEVEN_RESET,
           0.0, 0.12, 41.0 * 12.4),
    window("claude", "five_hour", "Claude · 5-hour", "short", 5 * H, 35.0, at(10, 15, 15),
           4.2, 3.1, 35.0 * 3.7),
    window("opencode-go", "five_hour", "OpenCode Go · 5-hour", "short", 5 * H, 22.0,
           NOW + 2 * H + 49 * 60, 3.2, 2.4, 22.0 * 0.66, age_s=180),
    window("opencode-go", "weekly", "OpenCode Go · Weekly", "long", 7 * D, 31.0, at(10, 19),
           0.4, 0.38, 31.0 * 5.5, age_s=180),
    window("opencode-go", "monthly", "OpenCode Go · Monthly", "long", 31 * D, 38.0, at(11, 1),
           0.15, 0.11, 38.0 * 18.0, age_s=180),
    window("codex", "primary", "Codex · Weekly", "long", 7 * D, 12.0, at(10, 17, 9),
           0.2, 0.1, 12.0 * 11.3, age_s=1400),
]
BINDING = personal_windows[0]


def workdays(month_start, month_end):
    days, t = [], month_start
    while t < month_end:
        if datetime.fromtimestamp(t, timezone.utc).weekday() < 5:
            days.append(t)
        t += D
    return days


MONTH_START = datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()
MONTH_END = datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp()
WORKDAYS = workdays(MONTH_START, MONTH_END)


def workday_fraction(t):
    done = sum(min(1.0, max(0.0, (t - d) / D)) for d in WORKDAYS)
    return done / len(WORKDAYS)


BUDGET, SPENT = 3000.0, 1650.0


def month_window():
    frac = workday_fraction(NOW)
    projected = SPENT / frac
    pct = SPENT / BUDGET * 100
    # First moment the workday-paced spend reaches the budget.
    reaches = next((floor_min(t) for t in range(int(NOW), int(MONTH_END), 1800)
                    if SPENT / frac * workday_fraction(t) >= BUDGET), None)
    return {"id": "claude:month", "provider": "claude", "key": "month",
            "label": "Claude · October bill", "kind": "month",
            "window_s": MONTH_END - MONTH_START, "pct": pct, "resets_at": MONTH_END,
            "window_start": MONTH_START, "observed_at": floor_min(NOW - 180), "stale": False,
            "age_s": 180,
            "pace": {"pace_state": "above", "expected_pct": round(frac * 100, 1),
                     "projected_pct": round(projected / BUDGET * 100, 1),
                     "reaches_100_epoch": reaches, "elapsed_fraction": frac},
            "tight_epoch": None, "burn_pct_per_h": {"h1": None, "h6": None},
            "usd_spent": SPENT, "usd_per_point": BUDGET / 100, "usd_left": BUDGET - SPENT,
            "waitable": False, "verdict": "runs_out"}, projected, frac


def runway_payloads():
    personal = {"generated_at": NOW, "scope": "personal", "binding": BINDING["id"],
                "windows": personal_windows, "budget": None}
    month, projected, frac = month_window()
    left_days = len(WORKDAYS) * (1 - frac)
    work = {"generated_at": NOW, "scope": "enterprise", "binding": month["id"],
            "windows": [month],
            "budget": {"budget_usd": BUDGET, "spent_usd": SPENT, "projected_usd": projected,
                       "pace_state": "above",
                       "per_workday_left_usd": (BUDGET - SPENT) / left_days}}
    return personal, work


# ---------------------------------------------------------------- trajectory
ACTIVE = [(9, 8.5, 23.6), (10, 9, 25.2), (11, 11, 18), (12, 13, 22), (13, 8, 24.5),
          (14, 8.6, 23.8), (15, 5, 15)]


def is_active(t):
    return any(at(10, d) + a * H <= t < at(10, d) + b * H for d, a, b in ACTIVE)


def personal_trajectory():
    w = BINDING
    start, step = w["window_start"], 30 * 60
    raw = []
    t = start
    while t <= NOW:
        raw.append((t, (0.4 + rng.random() * 1.2) * (2.4 if rng.random() < 0.15 else 1)
                    if is_active(t) else 0.0))
        t += step
    total = sum(v for _, v in raw) or 1
    acc, points = 0.0, []
    for t, v in raw:
        acc += v / total * w["pct"]
        points.append([floor_min(t), round(acc, 1)])
    points.append([floor_min(NOW), w["pct"]])
    return {"points": points, "resets": [], "start": start, "end": w["resets_at"]}


def work_trajectory():
    # Billed meter readings as percent of the budget (app/month_trajectory.py).
    # Its own random stream, so the other fixtures keep their values.
    local = random.Random(7)
    points, acc, t = [], 0.0, MONTH_START
    while t <= NOW:
        dt = datetime.fromtimestamp(t, TZ)
        if dt.weekday() < 5 and 8 <= dt.hour < 19:
            acc += 0.7 + local.random() * 0.6
        points.append([t, acc])
        t += 2 * H
    scale = (SPENT / BUDGET * 100) / (acc or 1)
    points = [[p[0], round(p[1] * scale, 2)] for p in points] + [[floor_min(NOW), SPENT / BUDGET * 100]]
    return {"points": points, "resets": [], "start": MONTH_START, "end": MONTH_END}


# ---------------------------------------------------------------- now
MODEL = {"opus": "Opus 5.5", "fable": "Fable 5.1", "sonnet": "Sonnet 5.5", "haiku": "Haiku 5.5",
         "deepseek": "DeepSeek V4", "glm": "GLM-5", "gpt": "GPT-6"}
CLAUDE_FAMILIES = {"opus", "fable", "sonnet", "haiku"}


def session(sid, machine, project, family, state, since_s, cost, burn=0.0, subs=0,
            reason=None, client="claude-code", title=None):
    share = (burn / PER_POINT_7D) if family in CLAUDE_FAMILIES and burn else (
        0.0 if family in CLAUDE_FAMILIES else None)
    return {"id": sid, "machine": machine, "project": project, "title": title, "state": state,
            "state_since": floor_min(NOW - since_s) + since_s % 60, "waiting_reason": reason,
            "model": MODEL[family], "family": family, "client": client, "subagents": subs,
            "cost_usd": cost, "burn_usd_per_h": burn, "share_of_binding_pct_per_h": share}


def now_payload(scope):
    if scope == "personal":
        sessions = [
            session("s-a41f", "laptop", "site", "fable", "waiting", 260, 6.12, reason="question",
                    title="Fix the footer layout"),
            session("s-b207", "vm-01", "dashboard", "glm", "waiting", 65, 0.88, reason="permission",
                    client="pi-agent"),
            session("s-d913", "homelab", "infra", "opus", "working", 72 * 60, 31.75, 9.0, 5),
            session("s-f5c2", "gpu-box", "agent-tools", "deepseek", "working", 41 * 60, 2.20, 2.1,
                    client="pi-agent"),
            session("s-c388", "devbox", "api-server", "opus", "working", 23 * 60, 18.40, 7.5, 3),
            session("s-g0e1", "laptop", "api-server", "gpt", "working", 12 * 60, 1.30, 1.2,
                    client="pi-agent"),
            session("s-e6b4", "devbox", "bench", "sonnet", "working", 8 * 60, 1.90, 3.2, 1),
            session("s-h772", "homelab", "notes", "haiku", "idle", 26 * 60, 0.41),
        ]
        offline = [{"name": "nas", "last_event": NOW - 3 * D, "last_ingest": NOW - 3 * D,
                    "working": 0, "waiting": 0}]
    else:
        sessions = [
            session("w-1a20", "vm-01", "infra", "sonnet", "waiting", 160, 3.85, reason="permission"),
            session("w-2b31", "laptop", "api-server", "opus", "working", 47 * 60, 22.60, 19.0, 2),
            session("w-3c42", "laptop", "dashboard", "haiku", "idle", 34 * 60, 0.62),
        ]
        offline = []
    machines = {}
    for s in sessions:
        m = machines.setdefault(s["machine"], {"name": s["machine"], "last_event": NOW - 8,
                                              "last_ingest": NOW - 40, "working": 0, "waiting": 0})
        if s["state"] in ("working", "waiting"):
            m[s["state"]] += 1
    count = lambda st: sum(1 for s in sessions if s["state"] == st)  # noqa: E731
    burn = sum(s["burn_usd_per_h"] for s in sessions)
    live = {"working": count("working"), "waiting": count("waiting"), "idle": count("idle"),
            "subagents": sum(s["subagents"] for s in sessions),
            "machines_online": len(machines), "machines_total": len(machines) + len(offline),
            "newest_data_age_s": 6,
            "burn_usd_per_h": {"m30": round(burn, 2), "h1": round(burn * 0.93, 2)}}
    return {"generated_at": NOW, "scope": scope, "live": live, "sessions": sessions,
            "machines": list(machines.values()) + offline}


# ---------------------------------------------------------------- today
def today_payload(scope):
    if scope == "personal":
        models = [("opus", "anthropic", 236.0), ("fable", "anthropic", 98.0),
                  ("sonnet", "anthropic", 41.0), ("deepseek", "opencode-go", 14.0),
                  ("glm", "opencode-go", 9.0), ("gpt", "openai-codex", 8.3),
                  ("haiku", "anthropic", 6.0)]
        machines = [("devbox", 171.0, 2), ("laptop", 118.0, 1), ("homelab", 74.0, 1),
                    ("gpu-box", 38.0, 1), ("vm-01", 11.3, 0)]
        typical = (268.0, 365.0, 84, 20000)
    else:
        models = [("opus", "anthropic", 118.4), ("sonnet", "anthropic", 21.7),
                  ("haiku", "anthropic", 2.9)]
        machines = [("laptop", 126.5, 1), ("vm-01", 16.5, 0)]
        typical = (92.0, 147.0, 31, 9400)
    spend = round(sum(m[2] for m in models), 2)
    return {"day": "2026-10-15", "spend_usd": spend, "typical_by_now_usd": typical[0],
            "typical_day_usd": typical[1], "prompts": typical[2], "active_s": typical[3],
            "by_model": [{"model": MODEL[f], "family": f, "provider": p, "usd": u,
                          "tokens": int(u * 41000)} for f, p, u in models],
            "by_machine": [{"machine": m, "usd": u, "working": w} for m, u, w in machines]}


# ---------------------------------------------------------------- limit rate
PEAK_TZ = ZoneInfo("America/Los_Angeles")


def peak_hours():
    hours = []
    for dow in range(7):  # 0 = Monday, as Python's weekday()
        for hour in range(24):
            if dow < 5 and 5 <= hour <= 10:
                factor = {5: 1.23, 10: 1.22}.get(hour, 1.33)
            else:
                factor = 1.0
            hours.append({"dow": dow, "hour": hour, "factor": factor})
    return hours


def limit_rate_payload():
    granted = datetime(2026, 10, 4, 22, tzinfo=timezone.utc).timestamp()
    return {"generated_at": floor_min(NOW - 1500), "period_start": granted,
            "fit_quality": {"r2": 0.95, "intervals": 118, "status": "ok", "note": None},
            "usd_per_point": {"seven_day": {"value": 24.4, "lo": 23.7, "hi": 25.6},
                              "five_hour": {"value": 3.7, "lo": 3.5, "hi": 3.9}},
            "steps": [
                {"at": datetime(2026, 9, 1, 20, tzinfo=timezone.utc).timestamp(),
                 "bucket": "seven_day", "from_usd_per_point": 23.5, "to_usd_per_point": 5.3,
                 "cause": "rate_change"},
                {"at": datetime(2026, 9, 22, 15, tzinfo=timezone.utc).timestamp(),
                 "bucket": "seven_day", "from_usd_per_point": 5.3, "to_usd_per_point": 12.9,
                 "cause": "rate_change"},
                {"at": granted, "bucket": "seven_day", "from_usd_per_point": 12.9,
                 "to_usd_per_point": 24.4, "cause": "granted_reset"},
                {"at": granted, "bucket": "five_hour", "from_usd_per_point": 1.01,
                 "to_usd_per_point": 3.7, "cause": "granted_reset"}],
            "families": [
                {"family": "opus", "usd_per_point": 24.7, "lo": 23.9, "hi": 25.6, "spend_usd": 2100.0},
                {"family": "sonnet", "usd_per_point": 19.2, "lo": 11.9, "hi": 40.1, "spend_usd": 182.0},
                {"family": "fable", "usd_per_point": 13.6, "lo": 12.0, "hi": 25.8, "spend_usd": 131.0},
                {"family": "haiku", "usd_per_point": None, "lo": None, "hi": None, "spend_usd": 18.0,
                 "note": "not enough data"}],
            "peak": {"active": False, "factor": 1.37, "until": None,
                     "next_start": datetime(2026, 10, 16, 5, tzinfo=PEAK_TZ).timestamp(),
                     "tz": "America/Los_Angeles",
                     "hours": peak_hours()},
            "cache_read_weight": {"value": 0.47, "lo": 0.40, "hi": 0.64}}


def limit_rate_unavailable():
    """Work scope: no subscription windows, so the server sends an all-null fit."""
    return {"generated_at": floor_min(NOW - 1500), "period_start": None,
            "fit_quality": {"r2": None, "intervals": 0, "status": "not_available_for_scope",
                            "note": "Work scope has no subscription windows."},
            "usd_per_point": {"seven_day": None, "five_hour": None}, "steps": [],
            "families": [], "peak": None, "cache_read_weight": None}


# ---------------------------------------------------------------- history
def window_history_payload():
    windows, end = [], SEVEN_RESET - 7 * D
    peaks = [71, 84, 96, 100, 88, 79, 92, 100, 97, 85, 94]
    for i, peak in enumerate(reversed(peaks)):
        start = end - 7 * D
        per_point = 24.4 if start >= at(10, 4) else (12.9 if start >= at(9, 22) else 5.3)
        granted = [at(10, 4, 17)] if start <= at(10, 4, 17) < end else []
        windows.append({"start": start, "end": end, "peak_pct": peak, "hit_limit": peak >= 100,
                        "granted_resets": granted, "usd": round(peak * per_point, 1),
                        "usd_per_point": per_point})
        end = start
    return {"windows": list(reversed(windows))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    personal, work = runway_payloads()
    write("runway", personal)
    write("runway-work", work)
    write("runway-trajectory", personal_trajectory())
    write("runway-trajectory-work", work_trajectory())
    write("now", now_payload("personal"))
    write("now-work", now_payload("enterprise"))
    write("today", today_payload("personal"))
    write("today-work", today_payload("enterprise"))
    write("limit-rate", limit_rate_payload())
    write("limit-rate-work", limit_rate_unavailable())
    write("window-history", window_history_payload())
    # ledger.json and session.json: scripts/make_next_fixtures_ledger.py
    assert not math.isnan(personal["windows"][0]["usd_left"])


if __name__ == "__main__":
    main()
