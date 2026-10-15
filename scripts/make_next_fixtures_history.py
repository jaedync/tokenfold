"""Write the History and Lab fixtures of /next?mock=1 (static/next/fixtures/).

Run: .venv/bin/python scripts/make_next_fixtures_history.py

Same invented world as scripts/make_next_fixtures.py (its clock, its 7-day
reset, its model names). The endpoints here are the classic dashboard's:
/api/window-history (5-hour bucket), /api/spend-history, /api/stats (only the
fields History and Lab read), /api/served-models/timeline?sessions=0 and
/api/enterprise-budget. Every name is a placeholder; no production data is read.
"""

import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.water import compute_energy_wh, compute_water_ml  # noqa: E402
from scripts.make_next_fixtures import H, NOW, TZ, at, floor_min, write  # noqa: E402

rng = random.Random(20261016)
FIVE_H = 5 * H

# ---------------------------------------------------------------- 5-hour windows
def five_hour_history():
    """Ended 5 h windows of the last ~9 days: work hours only, a few at 100%."""
    windows, t = [], floor_min(at(10, 6, 7))
    while t + FIVE_H < NOW - FIVE_H:
        hour = datetime.fromtimestamp(t, TZ).hour
        if 7 <= hour <= 20 and rng.random() < 0.8:
            peak = min(100, max(4, round(rng.gauss(58, 26))))
            usd = round(peak * rng.uniform(3.3, 4.1), 1)
            granted = [floor_min(t + 3 * H)] if rng.random() < 0.04 else []
            windows.append({"start": t, "end": t + FIVE_H, "peak_pct": peak, "hit_limit": peak >= 100,
                            "granted_resets": granted, "usd": usd,
                            "usd_per_point": round(usd / peak, 2)})
            t += FIVE_H
        else:
            t += 2 * H
    return {"windows": windows}


# ---------------------------------------------------------------- monthly spend
# (key in by_model, share of the month). Claude Code rows are bare names; Pi
# rows carry "Provider / Model", as the real payload does.
MIX = {
    "early": [("Opus 4.8", .58), ("Sonnet 5", .16), ("Haiku 4.5", .02), ("Opus 4.7", .14),
              ("OpenAI Codex / GPT-5.6 Sol", .07), ("OpenCode Go / GLM-5.3", .03)],
    "mid": [("Fable 5", .46), ("Opus 5", .24), ("Sonnet 5", .05), ("Haiku 4.5", .01),
            ("Anthropic / Opus 5", .08), ("OpenAI Codex / GPT-5.6 Sol", .09),
            ("OpenCode Go / GLM-5.3", .03), ("OpenCode Go / DeepSeek V4 Flash", .02),
            ("OpenCode Go / Kimi K3", .004), ("OpenRouter / Gemini 3.8 Flash", .006)],
    "late": [("Opus 5.5", .47), ("Anthropic / Opus 5.5", .18), ("Fable 5.1", .12),
             ("Sonnet 5.5", .05), ("Anthropic / Sonnet 5.5", .02), ("Haiku 5.5", .006),
             ("OpenAI Codex / GPT-6 Astra", .08), ("OpenAI Codex / GPT-6.1 Sol", .03),
             ("OpenCode Go / DeepSeek V4.1 Flash", .012), ("OpenCode Go / GLM-5.3", .01),
             ("Opencode / GLM-5.3", .002)],
}
PERSONAL_TOTALS = [("2025-11", 2380, "early"), ("2025-12", 3120, "early"), ("2026-01", 3960, "early"),
                   ("2026-02", 4410, "early"), ("2026-03", 5230, "mid"), ("2026-04", 4870, "mid"),
                   ("2026-05", 6100, "mid"), ("2026-06", 7520, "mid"), ("2026-07", 9100, "mid"),
                   ("2026-08", 8740, "late"), ("2026-09", 10230, "late"), ("2026-10", 5010, "late")]
WORK_MIX = [("Anthropic / Opus 5.5", .62), ("Anthropic / Sonnet 5.5", .2), ("Opus 5.5", .09),
            ("Anthropic / Haiku 5.5", .01), ("OpenRouter / DeepSeek V4.1 Flash", .05),
            ("OpenRouter / GLM-5.3", .03)]
WORK_TOTALS = [("2026-06", 640), ("2026-07", 1480), ("2026-08", 1720), ("2026-09", 1964), ("2026-10", 960)]


def month(label, total, mix):
    by_model = {}
    for name, share in mix:
        by_model[name] = round(total * share * rng.uniform(.85, 1.15), 2)
    return {"month": label, "total": round(sum(by_model.values()), 2),
            "by_model": dict(sorted(by_model.items())), "partial": label == "2026-10"}


def spend_history(scope):
    if scope == "personal":
        return {"months": [month(m, t, MIX[era]) for m, t, era in PERSONAL_TOTALS]}
    return {"months": [month(m, t, WORK_MIX) for m, t in WORK_TOTALS]}


# ---------------------------------------------------------------- /api/stats subset
PRICES = {  # USD per MTok: input, output, cache write 5 m, cache read
    "Opus 5.5": (4, 20, 5, .2), "Fable 5.1": (10, 50, 12.5, .25), "Sonnet 5.5": (2, 10, 2.5, .1),
    "Haiku 5.5": (.8, 4, 1, .08), "Opus 4.8": (5, 25, 6.25, .5), "Sonnet 5": (3, 15, 3.75, .3),
    "Haiku 4.5": (1, 5, 1.25, .1),
}
# (model, input tok, output tok, cost, active h, 14 d share, output tok/s); the
# zero-priced rows are Pi models whose provider reports the cost.
USAGE = [
    ("Opus 5.5", 210e6, 61e6, 10345.5, 648, .86, 71.8), ("Fable 5.1", 9e6, 2.6e6, 713.2, 20.2, .3, 61.4),
    ("Sonnet 5.5", 14e6, 5.2e6, 274.5, 27.3, .95, 97.2), ("Haiku 5.5", 4e6, 1.1e6, 7.8, 5.6, 1, 164.7),
    ("Opus 4.8", 160e6, 52e6, 12472.1, 453.9, 0, 66.6), ("Sonnet 5", 7e6, 2.3e6, 245.9, 15.3, 0, 86.5),
    ("Haiku 4.5", 9e6, 4.1e6, 307.2, 31.5, .01, 80.0), ("GPT-6 Astra", 41e6, 13e6, 2071.8, 116.3, .27, 24.2),
    ("DeepSeek V4.1 Flash", 30e6, 6.3e6, 5.6, 15.4, .5, 112.9), ("GLM-5.3", 3e6, .7e6, 15.1, 2.5, .2, 52.3),
]
BENCH = {"Opus 4.8": {"SWE-bench": 82.4, "Terminal-Bench": 68.1, "OSWorld": 74.0, "ARC-AGI-2": 71.2},
         "Sonnet 5": {"SWE-bench": 80.3, "Terminal-Bench": 61.5, "OSWorld": 73.3, "ARC-AGI-2": 60.1},
         "Haiku 4.5": {"SWE-bench": 73.3, "Terminal-Bench": 40.2, "OSWorld": 50.7}}


def model_row(name, inp, out, cost, hours, recent, tps):
    r_inp, r_out, r_h = round(inp * recent), round(out * recent), round(hours * recent, 1)
    return {"model": name, "cost": cost, "recent_cost": round(cost * recent, 2),
            "has_reported_cost": name not in PRICES, "unpriced": False,
            "active_hours": hours, "recent_active_hours": r_h,
            "all_cost_per_hour": round(cost / hours, 2) if hours else None,
            "recent_cost_per_hour": round(cost * recent / r_h, 2) if r_h else None,
            "all_output_tok_per_s": tps, "recent_output_tok_per_s": tps if recent else None,
            "input": round(inp), "output": round(out), "recent_input": r_inp, "recent_output": r_out,
            "energy_wh": round(compute_energy_wh(name, inp, out), 1),
            "water_ml": round(compute_water_ml(name, inp, out), 1),
            "recent_energy_wh": round(compute_energy_wh(name, r_inp, r_out), 1),
            "recent_water_ml": round(compute_water_ml(name, r_inp, r_out), 1)}


def pricing(name):
    i, o, cw, cr = PRICES.get(name, (0, 0, 0, 0))
    return {"input": i, "output": o, "cache_write": cw, "cache_write_1h": i * 2, "cache_read": cr}


def meter_daily():
    days, out = [at(10, d, 12, tz=timezone.utc) for d in range(1, 16)], []
    for t in days:
        if datetime.fromtimestamp(t, timezone.utc).weekday() >= 5:
            continue
        official = round(rng.uniform(40, 160), 2)
        measured = round(official * rng.uniform(.93, 1.0), 2)
        out.append({"day": datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d"),
                    "official_usd": official, "measured_usd": measured,
                    "unaccounted_usd": round(official - measured, 2)})
    return out


def stats(scope):
    usage = USAGE if scope == "personal" else [u for u in USAGE if u[0] in ("Opus 5.5", "Sonnet 5.5", "Haiku 5.5")]
    body = {"scope": scope, "version": 1,
            "model_breakdown": [model_row(*u) for u in usage],
            "model_pricing": {u[0]: pricing(u[0]) for u in usage},
            "benchmarks": {k: v for k, v in BENCH.items() if any(u[0] == k for u in usage)},
            "meter": None, "month_hero": {"value": 5010.0, "source": "estimate", "measured_usd": 5010.0,
                                          "unaccounted_usd": None, "limit_usd": None,
                                          "remaining_usd": None, "utilization": None}}
    if scope == "enterprise":
        daily = meter_daily()
        used = round(sum(d["official_usd"] for d in daily), 2)
        measured = round(sum(d["measured_usd"] for d in daily), 2)
        body["meter"] = {"used_usd": used, "limit_usd": 2000.0, "utilization": None,
                         "fetched_epoch": NOW - 7 * 60, "machine": "work-laptop", "fresh": True,
                         "anchored": None, "remembered_limit_usd": 2000.0, "daily": daily}
        body["month_hero"] = {"value": used, "source": "meter", "measured_usd": measured,
                              "unaccounted_usd": round(used - measured, 2), "limit_usd": 2000.0,
                              "remaining_usd": round(2000 - used, 2), "utilization": round(used / 20, 1)}
    return body


# ---------------------------------------------------------------- served-model timeline
def served_timeline():
    """30 days in 6 h cells: Opus is hidden by the v4 header for the last 26
    days; Fable flips to a named other model for two short runs."""
    bin_s, since = 21600, at(9, 16)
    bins = []
    for model, base in (("Opus 5.5", 420), ("Fable 5.1", 60), ("Sonnet 5.5", 30)):
        for k in range(int((NOW - since) // bin_s)):
            start = int(since + k * bin_s)
            hour = datetime.fromtimestamp(start, TZ).hour
            if hour < 6 or rng.random() < .25:
                continue
            n = max(1, round(base * rng.uniform(.3, 1.4)))
            if model == "Opus 5.5":
                bins.append([model, start, "self" if start < at(9, 20) else "hidden", n])
            elif model == "Fable 5.1" and (at(9, 27) <= start < at(9, 28, 12) or at(10, 8) <= start < at(10, 9)):
                bins.append([model, start, "self", max(1, n // 3)])
                bins.append([model, start, "kettle-e2c95a10-v2", n - n // 3])
            else:
                bins.append([model, start, "self", n])
    models = ["Opus 5.5", "Fable 5.1"]
    ledger = [
        {"model": "Opus 5.5", "state": "self", "served_model": "claude-opus-5-5", "sig_version": 2,
         "sig_fields": "1,3,5,6,7,8,10,11,14,17", "first_seen": at(9, 16, 9, 12), "last_seen": at(9, 19, 23, 40),
         "blocks": 4120, "sessions": 61, "machines": ["devbox", "laptop"]},
        {"model": "Opus 5.5", "state": "hidden", "served_model": None, "sig_version": 4,
         "sig_fields": "1,3,7,8", "first_seen": at(9, 20, 8, 3), "last_seen": NOW - 600,
         "blocks": 31840, "sessions": 402, "machines": ["devbox", "gpu-box", "homelab", "laptop"]},
        {"model": "Fable 5.1", "state": "kettle-e2c95a10-v2", "served_model": "claude-kettle-e2c95a10-v2-prod",
         "sig_version": 2, "sig_fields": "1,3,5,6,7,8,10,11,14,17", "first_seen": at(9, 27, 10, 15),
         "last_seen": at(10, 8, 22, 51), "blocks": 287, "sessions": 9, "machines": ["devbox"]},
    ]
    bins = [b for b in bins if b[0] in models]
    return {"days": 30, "bin_seconds": bin_s, "since_epoch": since, "models": models, "bins": bins,
            "ledger": ledger, "latest": {"Opus 5.5": {"state": "hidden", "since": at(9, 20, 8, 3), "blocks": 31840},
                                         "Fable 5.1": {"state": "self", "since": at(10, 9, 1, 2), "blocks": 812}}}


def main():
    write("window-history-5h", five_hour_history())
    write("spend-history", spend_history("personal"))
    write("spend-history-work", spend_history("enterprise"))
    write("stats", stats("personal"))
    write("stats-work", stats("enterprise"))
    write("served-timeline", served_timeline())
    write("enterprise-budget", {"budget_usd": 2000.0})


if __name__ == "__main__":
    main()
