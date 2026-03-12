# Monthly Activity Chart

## Overview

Add a Monthly Activity panel to the dashboard, placed directly after the existing Daily Activity chart. The panel displays a Chart.js bar chart with a metric toggle (Cost, Time, Sessions, Prompts, Tokens) and a range selector (6 Mo, 1 Year, All) defaulting to 6 months.

## Backend — aggregator.py

Add a `monthly[]` array to the `build_dashboard_data()` output, computed during cache rebuild alongside the existing `daily[]` array.

**Data shape** — roll up daily buckets into monthly buckets:

```python
# Each entry in monthly[]:
{
    "month": "2026-03",        # YYYY-MM
    "cost": 42.15,
    "active_minutes": 305.3,   # sum of daily active_minutes (matches daily_list field)
    "sessions": 47,
    "prompts": 312,
    "input_tokens": 2150000,
    "output_tokens": 890000,
    "cache_creation_tokens": 500000,
    "cache_read_tokens": 1400000,
    "tool_calls": 1023,
}
```

Implementation: iterate the already-computed `daily_list`, group by `date[:7]`, sum each numeric field. No new SQL queries needed. The `active_minutes` field is summed from daily entries (already rounded integers in `daily_list`; cumulative rounding error is negligible for chart display). The backend emits all months from the first month with activity through the current month, including zero-activity months as zero-value entries. The frontend range toggle filters this list.

## Frontend — dashboard.html

### Panel placement

New `<div class="panel">` inserted after the Daily Activity panel (`<!-- ======== DAILY ACTIVITY ========>`), before the Token Breakdown panel.

### Controls

Two toggle rows in a flex container, matching the existing `.mode-toggle` / `.mode-btn` pattern from the date bar:

- **Left — metric toggle**: Cost | Time | Sessions | Prompts | Tokens — "Tokens" shows total tokens (input + output + cache creation + cache read) as a single bar
- **Right — range toggle**: 6 Mo | 1 Year | All (default: 6 Mo)

Both reuse the `.mode-btn` CSS styling but under scoped containers (`.monthly-metric-toggle` and `.monthly-range-toggle`) to avoid conflicts with the global `.mode-btn` selectors used by `setMode()`. The monthly toggle click handlers manage active state within their own container only.

### Chart

- Chart.js bar chart on a `<canvas>` element
- Bar color: `var(--blue)` (`#1a4b8c`)
- Current (partial) month: rendered with reduced opacity (0.5) and dashed border to visually distinguish from complete months
- X-axis labels: abbreviated month names (e.g., "Oct", "Nov", "Mar")
- Y-axis: auto-formatted per metric — `$X.XX` for cost, `Xh Xm` for time, `X,XXX` for counts under 1M, `X.XM` for 1M+, `X.XB` for 1B+
- Tooltip on hover shows exact value

### JavaScript

- New `renderMonthlyChart()` function called from `renderMode()` (the existing orchestrator that `setMode()` delegates to)
- Reads `D.monthly` from the data JSON
- Filters by selected range (last 6 calendar months, last 12, or all)
- Switches displayed dataset based on active metric button
- Chart instance stored and updated (not destroyed/recreated) on toggle

## Files changed

| File | Change |
|------|--------|
| `app/aggregator.py` | Add monthly rollup logic, include `monthly` key in output dict |
| `templates/dashboard.html` | New panel HTML, CSS for metric toggle, JS for `renderMonthlyChart()` |

## Out of scope

- Monthly breakdown by machine (could be added later)
- Monthly table view (bars only for now)
- Export/download of monthly data
