"""GET / - serves rendered HTML dashboard."""

import html
import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .aggregator import build_dashboard_data
from .config import STATS_OWNER

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))


def _fmt_num(n):
    n = int(n)
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n:,}"
    return str(n)


def _fmt_time(s):
    s = int(s)
    h, rem = divmod(s, 3600)
    m = rem // 60
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m"


def _fmt_cost(c):
    if c >= 1:
        return f"${c:,.2f}"
    if c >= 0.01:
        return f"${c:.2f}"
    if c == 0:
        return "$0.00"
    return f"${c:.3f}"


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    data = build_dashboard_data()
    c = data["cards"]

    CARD_CLASSES = [
        "stat-card stat-card--blue stat-card--geo-circle",   # Sessions
        "stat-card stat-card--geo-rect",                      # Human Prompts
        "stat-card stat-card--red stat-card--geo-circle",     # Total Tokens
        "stat-card stat-card--black",                         # Active Time
        "stat-card",                                           # Tool Calls
        "stat-card stat-card--yellow",                         # Models Used
        "stat-card",                                           # Avg Prompts/Day
        "stat-card stat-card--geo-rect",                       # Avg Active/Day
    ]
    card_items = [
        ("Sessions", _fmt_num(c["sessions"])),
        ("Human Prompts", _fmt_num(c["human_prompts"])),
        ("Total Tokens", _fmt_num(c["total_tokens"])),
        ("Active Time", _fmt_time(c["active_time_s"])),
        ("Tool Calls", _fmt_num(c["tool_calls"])),
        ("Models Used", str(c["models_used"])),
        ("Avg Prompts/Day", _fmt_num(c["avg_prompts_day"])),
        ("Avg Active/Day", _fmt_time(c["avg_active_day_s"])),
    ]
    cards_html = "\n".join(
        f'<div class="{cls}"><div class="stat-label">{lab}</div>'
        f'<div class="stat-value">{val}</div></div>'
        for (lab, val), cls in zip(card_items, CARD_CLASSES)
    )

    max_day_cost = max((d["cost"] for d in data["daily"] if d["cost"] > 0), default=1.0)
    HM_COLORS = ["", "rgba(230,51,41,0.08)", "rgba(230,51,41,0.16)",
                 "rgba(230,51,41,0.26)", "rgba(230,51,41,0.38)"]

    def row_hm_style(cost):
        if not cost:
            return ""
        p = cost / max_day_cost
        lvl = 1 if p < 0.25 else 2 if p < 0.55 else 3 if p < 0.80 else 4
        return f' style="background:{HM_COLORS[lvl]}"'

    rows = []
    for d in reversed(data["daily"]):
        cost_color = 'var(--red)' if d["cost"] > 0 else 'var(--gray-dim)'
        rows.append(
            f'<tr{row_hm_style(d["cost"])}><td>{d["date"]}</td><td>{d["sessions"]}</td>'
            f'<td>{d["prompts"]}</td><td>{d["tool_calls"]}</td>'
            f'<td>{_fmt_num(d["input_tokens"])}</td><td>{_fmt_num(d["output_tokens"])}</td>'
            f'<td>{_fmt_num(d["cache_read_tokens"])}</td>'
            f'<td>{_fmt_time(d["active_minutes"] * 60)}</td>'
            f'<td style="color:{cost_color}">{_fmt_cost(d["cost"])}</td></tr>')
    table_rows = "\n".join(rows)

    machines_list = data.get("machines", [])
    machine_last_active = data.get("machine_last_active", {})
    import time as _time
    _now_epoch = _time.time()
    if machines_list:
        pills = []
        for m in machines_list:
            last = machine_last_active.get(m, 0)
            active = (_now_epoch - last) < 900  # 15 minutes
            cls = "machine-pill machine-pill--active" if active else "machine-pill"
            pills.append(f'<span class="{cls}">{html.escape(m)}</span>')
        machines_pills = "".join(pills)
    else:
        machines_pills = '<span class="machine-pill" style="color:var(--gray-dim)">no machines</span>'

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "data_json": json.dumps(data),
        "cards_html": cards_html,
        "table_rows": table_rows,
        "gen_time": data["generation_time"],
        "data_range": data["data_range"],
        "machines_pills": machines_pills,
        "owner": STATS_OWNER,
    })
