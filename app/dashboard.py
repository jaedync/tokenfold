"""GET / - serves rendered HTML dashboard."""

import html
import json
import shlex
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .aggregator import build_dashboard_data
from . import config
from .auth import require_dashboard_auth
from .config import DEFAULT_SCOPE, STATS_OWNER, VALID_SCOPES
from .install import external_base_url  # single source of truth for the base URL

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))


def _fmt_num(n):
    """One notation per magnitude: 1.2K, 492K, 2.4M, 1.08B — mirrors fN() in
    the template (comma-thousands like '492,000' next to '2.4M' read as two
    different units). Trailing zeros are trimmed."""
    n = int(n)

    def unit(v, digits, suffix):
        s = f"{v:.{digits}f}".rstrip("0").rstrip(".")
        return s + suffix

    if n >= 999_500_000:  # 999.5M+ rounds up: promote to B
        return unit(n / 1_000_000_000, 2, "B")
    if n >= 999_950:      # 999,950+ rounds up: promote to M
        return unit(n / 1_000_000, 1, "M")
    if n >= 1_000:
        return unit(n / 1_000, 1, "K")
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


def build_install_command(base_url, key):
    """The one-command onboarding line the footer button copies: pipe the
    server's own /install.sh into bash with the ingest key inline. shlex.quote
    keeps a key with shell metacharacters (spaces, quotes) from breaking out of
    the --token argument when the command is pasted into a terminal."""
    return f"curl -fsSL {base_url}/install.sh | bash -s -- --token {shlex.quote(key)}"


@router.get("/", response_class=HTMLResponse, dependencies=[Depends(require_dashboard_auth)])
def dashboard(request: Request, scope: Optional[str] = None):
    # Soft-fail: a bookmarked bad/forbidden ?scope= shouldn't 403 the whole page;
    # it just serves the allowed scope. API routes do hard-fail (400/403).
    # Scope precedence: lock > ?scope= > tf_scope cookie > default. The cookie
    # lets a bare / render the user's saved scope in ONE load (previously the
    # client redirected after render — a full double page-load on every visit).
    requested = scope
    cookie_scope = request.cookies.get("tf_scope")
    locked = config.LOCKED_SCOPE
    if locked and locked in VALID_SCOPES:
        effective = locked
    elif requested in VALID_SCOPES:
        effective = requested
    elif cookie_scope in VALID_SCOPES:
        effective = cookie_scope
    else:
        effective = DEFAULT_SCOPE

    data = build_dashboard_data(effective)
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
    # The zero-width-space sub-detail reserves its line box server-side so the
    # card doesn't grow (layout shift) when JS fills the subtitle in.
    cards_html = "\n".join(
        f'<div class="{cls}"><div class="stat-label">{lab}</div>'
        f'<div class="stat-value">{val}</div>'
        f'<div class="sub-detail">​</div></div>'
        for (lab, val), cls in zip(card_items, CARD_CLASSES)
    )

    max_day_cost = max((d["cost"] for d in data["daily"] if d["cost"] > 0), default=1.0)
    # Tint capped at 0.12 alpha so high-cost rows stay legible (UX P2-25);
    # mirrors HM_ROW in the template's rebuildDailyTable().
    HM_COLORS = ["", "rgba(230,51,41,0.03)", "rgba(230,51,41,0.06)",
                 "rgba(230,51,41,0.09)", "rgba(230,51,41,0.12)"]

    def row_hm_style(cost):
        if not cost:
            return ""
        p = cost / max_day_cost
        lvl = 1 if p < 0.25 else 2 if p < 0.55 else 3 if p < 0.80 else 4
        return f' style="background:{HM_COLORS[lvl]}"'

    # Red is reserved for warning-level days (> 1.5x the nonzero-day average);
    # ordinary costs render black (UX P1-8). Mirrors rebuildDailyTable().
    _nz_costs = [d["cost"] for d in data["daily"] if d["cost"] > 0]
    _hot_cost = (sum(_nz_costs) / len(_nz_costs)) * 1.5 if _nz_costs else 0.0

    def _cost_color(cost):
        if not cost:
            return "var(--gray-dim)"
        if _hot_cost > 0 and cost > _hot_cost:
            return "var(--red)"
        return "var(--black)"

    rows = []
    for d in reversed(data["daily"]):
        cost_color = _cost_color(d["cost"])
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

    # Reserve one cost-meta line per costed model (default 14d window) so the
    # JS-rendered per-model lines don't push the page down on slow networks.
    _meta_lines = sum(1 for m in data.get("model_breakdown", [])
                      if m.get("recent_cost", 0) > 0)
    # 0.8rem vertical padding x2 + line-height 1.8 per model line
    cost_meta_minh = f"calc(1.6rem + {_meta_lines} * 1.8em)" if _meta_lines else "0"

    return templates.TemplateResponse(request, "dashboard.html", {
        # Encode every '<' as its JSON unicode escape so no '</script', '<!--',
        # or '<script' token can form inside the embedded <script> block. A bare
        # '</'->'<\/' replace is insufficient: '<!--<script>' has no '</' and
        # flips the HTML script-data tokenizer into the "double escaped" state.
        # '<' is valid JSON string content and JSON.parse restores it to '<'.
        "data_json": json.dumps(data).replace("<", "\\u003c"),
        "cards_html": cards_html,
        "table_rows": table_rows,
        "cost_meta_minh": cost_meta_minh,
        "gen_time": data["generation_time"],
        "data_range": data["data_range"],
        "machines_pills": machines_pills,
        "owner": STATS_OWNER,
        "scope_label": effective.upper(),
        "scope": effective,
        "scope_locked": bool(config.LOCKED_SCOPE),
        # Ingest key for client onboarding (footer click-to-reveal). Fail-closed:
        # only embedded when the dashboard itself is behind Basic auth — an open
        # dashboard must never leak the machine-ingest key.
        "ingest_key": config.STATS_API_KEY if config.DASHBOARD_PASSWORD else "",
        # One-command install line for the footer copy button. Same fail-closed
        # gate as ingest_key, AND requires a key to embed: the command inlines
        # the ingest key, so an open dashboard (no DASHBOARD_PASSWORD) or a
        # keyless instance must never render it.
        "install_cmd": (
            build_install_command(external_base_url(request), config.STATS_API_KEY)
            if config.DASHBOARD_PASSWORD and config.STATS_API_KEY else ""
        ),
        # Billing-reading writes are human actions behind Basic auth; an open
        # dashboard renders the history read-only (the server enforces too).
        "readings_writable": bool(config.DASHBOARD_PASSWORD),
    })
