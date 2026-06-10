# Tokenfold Dashboard — UX Audit & Master Implementation Plan

Audited: 2026-06-10 · Inputs: `templates/dashboard.html` (4,161 lines), `app/dashboard.py`, `app/aggregator.py`, full-page screenshot (~8,800px), live site (personal scope).

**Out of scope here (already being fixed by main session):** number-format consistency, expandable sessions rows, sessions section moved up, machine-colored session borders, heatmap size/negative space, Caddy compression + scope double-load.

---

## Proposed Page Order (top → bottom)

| # | Section | Rationale |
|---|---------|-----------|
| 1 | Header + date bar (scope/mode controls, status) | Identity + controls stay first; they set context for everything below. |
| 2 | **Cost hero** with explicit period label + orch/agent split | The headline number — but must say *which window* it covers (see P1-5). |
| 3 | **Cost & Limits** (MTD counter, weekly limit, 5-hr window, extra usage — merged pace) | "Can I keep working / am I on pace?" is the most actionable info; today it's buried under 8 stat cards. |
| 4 | Stat cards (sessions, prompts, tokens, active time) | Secondary KPIs; supporting detail, not the lede. |
| 5 | **Recent Sessions** (per existing plan: moved up, expandable, machine borders) | "What did I just do and what did it cost" is the #1 repeat-visit question. |
| 6 | Activity (enlarged calendar heatmap + hourly strip as one section) | Rhythm/when — one combined "Activity" section instead of three small ones. |
| 7 | Trends: Daily Activity + Token Breakdown | Time-series detail after the at-a-glance layers. |
| 8 | Models: ONE Cost-by-Model stacked bar + Model Breakdown table (donuts deleted) | All model questions answered in one place; kills 3 redundant charts. |
| 9 | Machines: one cost donut + Daily Cost by Machine + Machine table | Consolidated from 4 sections to one block. |
| 10 | Tools + Projects | Interesting but low-frequency; fine deep on the page. |
| 11 | Daily Breakdown table | Reference/archive material. |
| 12 | Reference (collapsed accordion): Pricing per MTok, Benchmarks, Cost/Hr vs Throughput | Static Anthropic data isn't *your* telemetry — demote behind a disclosure. |
| 13 | Footer (ingest key, generated time) | Unchanged. |

Add a slim sticky jump-nav (anchors per section) — the page is ~8,800px with no in-page navigation (see P2-22).

---

## P1 — Fix now

- [x] **P1-1 Chart.js is render-blocking and a single point of failure.** It loads synchronously in `<head>` from jsDelivr (`dashboard.html:34`); if the CDN stalls/fails, the *entire* dashboard dies — sessions table, heatmap, daily table rebuild are all in the same script that throws at `Chart.defaults` (`:2400`). Fix: self-host chart.js under `/static/`, load with `defer`, and guard `typeof Chart !== 'undefined'` so DOM-only renders (tables, heatmaps) survive without it.
- [x] **P1-2 Interval & listener leaks — rate-limits poller multiplies on every render.** The `/api/rate-limits` fetch + `setInterval(...,60000)` lives *inside* `renderMode()` (`:3685-3700`), which runs on every mode click and every 30s live update → one extra permanent poller per render. Also leaking per render: env-equiv cycling `setInterval` (`:2988`), document-level `keydown` handlers (`:3012`, `:3678`), and `renderHeatmap`/`renderHourly` re-attach `mouseover/mousemove/mouseleave` on the same wrappers each call (`:2641-2648`, `:2684-2691`). Fix: hoist the rate-limits poller out of `renderMode` (run once at boot); store/clear interval ids; attach heatmap listeners once via delegation.
- [x] **P1-3 Chart animations replay on every refresh and break captures.** All charts are destroyed + recreated in `renderMode()` (`:2734`), so every 30s data bump replays ~1s grow-from-zero animations (the audit screenshot caught Daily Activity, Daily Cost by Machine, Tool Usage, and Projects as *blank* mid-animation). Fix: `Chart.defaults.animation = false` (or 200ms max), and prefer `chart.data = ...; chart.update('none')` over destroy/recreate for live updates (see P2-26).
- [x] **P1-4 Two sections are literally both titled "Cost by Model", plus a third encoding.** Stacked bar (`:2117-2124`), donut (`:2168-2174`), and the Model Breakdown table cost column all show the same fact; Model Usage tokens donut (`:2161-2167`) is visually near-identical to the cost donut. Fix: keep the stacked bar (it shows composition) + table; delete both donuts (`cModels` `:2903`, `cCostDonut` `:2918`) or fold a single tiny donut into the table panel.
- [x] **P1-5 Three large unlabeled cost figures contradict each other above the fold.** Hero `$3,192.64` (active-mode window), `$2,510.35 MONTH TO DATE`, `$2,050.52 spent this week` — a 14-day figure *larger* than MTD reads as a bug. Fix: hero label must name the window — "ESTIMATED API COST · LAST 14 DAYS" (update `#costSplit`/label in `renderMode` `:2933`); in personal scope add a "API-equivalent — not billed" tag (user is on a subscription; `dashboard.py` knows scope).
- [x] **P1-6 Contrast failures.** Yellow `#f5c518` used as *text/number* color on cream: extra-usage gauge pct (`:3406`), Minutes axis ticks + title (`:2868`), card sub-detail accents (`:2795` etc. on light cards). 7px type: `.rate-gauge-stat-label` 0.44rem (`:791`), dayline labels 0.42rem (`:668`), marker label 0.5rem (`:643`). Fix: add `--yellow-text: #9a7a00` token for yellow-on-light text; floor all labels at 0.55rem.
- [x] **P1-7 Machine identity fragmentation — same Mac counted as 3 machines.** Live payload has `macbook-pro`, `jaedyns-macbook-pro`, and `jaedyns-macbook-pro.tailedc58.ts.net` as separate machines (donut legends even truncate the FQDN). Fix in `aggregator.py`: normalize hostnames at aggregation (strip `.ts.net`/domain suffix, lowercase, alias map in config) so charts, pills, and tables show one canonical machine; keep raw name in tooltip.
- [x] **P1-8 Red means everything, so it means nothing.** Every nonzero cost is red: daily table (`dashboard.py:112`, `dashboard.html:2710`), machine table (`:3905`), sessions (`:4043`). Red should be reserved for warning states (over pace, over budget, stale). Fix: normal costs in `--black`; red only when a row/figure exceeds a threshold (e.g., > 1.5× daily average, projection > 100%).

## P2 — Soon

- [x] **P2-9 "Monthly Cost" header mislabels its content.** The section contains weekly limit, 5-hour window, extra usage, 7-day pace (`:2027-2047`). Rename to "Cost & Limits"; the right-side label switches from "month to date" to oauth staleness text (`:3432`) — give staleness its own anchor near the gauges.
- [x] **P2-10 Sparse "7-Day Pace" panel duplicates the weekly gauge stats.** After fetch it shows only "57h 5m active · 7d" + a chart button (`:3129-3146`) — the same number appears again as "active this week" inside Weekly Limit stats (`:3345`). Merge: move the pace-chart trigger button onto the Weekly Limit gauge and delete `#weekPacePanel`; promotes the OAuth gauges up one row.
- [x] **P2-11 Weekly gauge reads "0% · 100% remaining" beside "$2,050.52 spent this week".** Right after a window reset this looks broken. Fix: when `wPct < 1` but week stats are nonzero, caption the gauge "window just reset" and visually separate rolling-7d stats from the reset-window bar (`:3290-3354`).
- [x] **P2-12 Daily Activity chart has THREE y-axes** (tool calls left, minutes + prompts both right, `:2857-2875`) — uninterpretable, and the two right axes collide. Fix: two axes max — prompts + active minutes; move tool calls into the tooltip only.
- [x] **P2-13 Token Breakdown is flattened by cache reads.** With 99.78% cache hit, one 900M cache-read day makes input/output invisible (`:2878-2895`). Fix: default to I/O + cache-creation only with a "include cache reads" toggle, or log-scale option.
- [x] **P2-14 Project Activity uses dual top/bottom x-axes** (cost top, minutes bottom, same row bars, `:3836-3852`) — two units overlaid per row is misread as one scale. Fix: cost-only bars; minutes in tooltip (or a small "h" suffix column after the bar).
- [x] **P2-15 Tool labels blow out the chart gutter.** `mcp__plugin_playwright_playwright__browser_take_screenshot` etc. consume ~half the Tool Usage panel width (`:3818`). Fix in `aggregator.py` tool naming: display-name MCP tools as `server · tool` (strip `mcp__` prefix and dedupe repeated server tokens); full name in tooltip.
- [x] **P2-16 Static reference content sits mid-page.** Pricing per MTok + Benchmarks (`:2135-2157`) are Anthropic constants, not telemetry, and currently outrank the user's own model/session data. Move to a collapsed "Reference" accordion at the bottom (with Cost/Hr vs Throughput as a candidate third tab).
- [x] **P2-17 Four machine sections → one block.** Usage by Machine donut + Prompts by Machine donut (near-identical shapes) + Daily Cost by Machine + Machine table (`:2186-2226`). Keep ONE donut (cost), the daily stacked bar, and the table; drop the prompts donut (prompts are a table column). Also rename "Usage by Machine" → "Cost by Machine".
- [x] **P2-18 Mode toggle's scope of effect is invisible.** Today/14d/All re-renders charts+cards, but heatmap, hourly strip, sessions, and Cost & Limits ignore it. Fix: a small period chip in each section header ("14 days" / "all time" / "rolling 7d") so each block self-describes its window.
- [x] **P2-19 env-bar looks static but is a modal trigger** (`:1945`, click handler `:3006`). Add a visible ⓘ / "methodology" affordance; also `cursor:pointer` alone isn't discoverable on touch.
- [x] **P2-20 Stale/contradictory footnotes.** Sessions: "Titled rows come from Claude Desktop" (`:2279`) — false; titles come from the summarizer for Code sessions too. Rewrite: "Titles are AI-generated summaries; untitled sessions show their project." Cache-tier sentence duplicated under Pricing (`:2147`) and Model Breakdown (`:2264`) — keep once. Verify "sub-agent output understated" note (`:2264`) is still true for current CC versions. *(note kept as-is — not re-verified against current CC; everything else in this item done)*
- [x] **P2-21 Silent error swallowing.** `/api/rate-limits` and live-update fetches `.catch(()=>{})` (`:3691`, `:4122`, `:4129`) — pace panel just stays empty forever. Show a quiet inline state: "limits unavailable — retrying" (also satisfies house error-handling rules).
- [x] **P2-22 No in-page navigation on an 8,800px page.** Add a slim sticky jump-nav (Cost · Limits · Sessions · Activity · Models · Machines · Tools · Daily) or anchor links in the date bar.
- [x] **P2-23 Scope toggle and mode toggle look like one control.** Identical 3px-border segmented styles, adjacent (`:1911-1922`). Differentiate: micro-label "SCOPE" / "PERIOD" above each, larger gap; add `aria-pressed` to buttons; give scope switch a pressed/loading state (it triggers a full reload, `:2389`).
- [x] **P2-24 Live refresh repaints everything.** `fetchAndApply` (`:4093`) calls `renderMode` + full innerHTML rebuilds of heatmap/hourly/sessions — visual reflash every data bump, scroll position lost inside `.tbl-wrap`, charts replay (see P1-3). Fix: in-place `chart.update()`, and skip DOM rebuild when the rendered values are unchanged.
- [x] **P2-25 Daily table heat tint hurts legibility.** Row backgrounds to rgba(red, 0.38) + red cost text (`:2700`, `dashboard.py:100`) make high-cost rows the *hardest* to read. Cap tint at ~0.12 alpha, or replace with a 3px left heat bar per row.
- [x] **P2-26 Heatmap scroll position + legend.** Calendar starts scrolled to January (oldest) in its overflow container (`:2079`, `:2637`); after render set `scrollLeft = scrollWidth` so "now" is visible. Add a less→more legend and explain the blue "peak day" cell (single blue cell inside a red ramp is read as a different category, `:2583`).
- [x] **P2-27 Hourly strip color code is unexplained.** Yellow = previous 12h, blue = current period, hatch = upcoming (`:2659-2672`) — there's no legend, and at ≤480px all hour labels are hidden (`:1852`) leaving anonymous colored squares with hover-only tooltips. Add a 3-swatch legend; on mobile show every 3rd label instead of none.
- [x] **P2-28 Mobile tables.** Sessions (8 cols) and Daily (9 cols) with `white-space:nowrap` force long horizontal scrolls with no affordance. Hide secondary columns under 768px (sessions: machine→left-border color per existing plan, $/hr; daily: cache read, tool calls) and add a right-edge scroll shadow on `.tbl-wrap`.
- [x] **P2-29 Accessibility batch.** No `aria-label`/`role="img"` on chart canvases or heatmap SVG; `th[data-info]` tooltips are hover-only (touch/keyboard inaccessible — `:1510`); modals lack `role="dialog"`, `aria-modal`, focus trap (`:1962`, `:2050`); status square is color-only (`:1902`) — pair with text it already has, but give machine-pill active state a non-color cue too (`:324`).

## P3 — Polish

- [x] **P3-30 Brand mismatch.** `<title>` says TOKENFOLD, `<h1>` says CLAUDE CODE, footer says Tokenfold (`:6`, `:1875`, `:2308`). Pick one lockup (e.g., h1 TOKENFOLD, red band "Claude Code usage"); also `og:image` is SVG (`:19`) which Discord/Twitter won't render — export a PNG.
- [x] **P3-31 Four duration formatters + two number formatters** (`fT :2444`, `fmtHM :3123`, `fmtHMg :3327`, `dur :4030`, `rel :4024` rounds *up* hours; server `_fmt_*` in `dashboard.py:21-48`). Single shared util each side; this is the root of the formatting-consistency complaint.
- [x] **P3-32 Initial daily-table double render.** Server renders rows (`dashboard.py:110-120`), then `rebuildDailyTable()` immediately rebuilds identical DOM via `renderMode` (`:4003`). Skip the first rebuild unless mode ≠ all.
- [x] **P3-33 Decorative real estate.** Cost-hero side panel is 240px of pure decoration (`:514-553`) and the hero block is ~280px min — fine for the art direction, but consider slimming both ~30% so Limits move above the fold on laptops.
- [x] **P3-34 Empty chart states.** Charts with no data in window render bare axes (Today mode early morning). Render "no activity in this window" text instead of an empty canvas.
- [x] **P3-35 Benchmarks chart value.** ARC-AGI-2/OSWorld scores for models you don't run is marketing data; if kept (P2-16 accordion), filter to models present in `model_breakdown`. *(already true: aggregator builds benchmarks only for models in model_stats)*
- [x] **P3-36 "DATA UPDATED" indicator** (`:1908`) flashes 2.5s in the date bar where nobody is looking. Tie it to the status square (pulse) or briefly highlight changed cards instead.
- [x] **P3-37 Footer timestamp** "Generated 2026-06-10 11:38" — ambiguous TZ; add zone or use relative ("2m ago", it already live-updates). *(generation_time already carries %Z; verified)*
- [x] **P3-38 Heatmap/hourly tooltips are mouse-only**; add `<title>` children to SVG rects as a free fallback for touch/AT.
- [x] **P3-39 Session `rel()` rounding**: `Math.round(s/3600)` shows "2h ago" at 1h31m; use `Math.floor` for ages.
- [x] **P3-40 Pricing table 1h-cache fallback** computes `p.input*2` client-side when `cache_write_1h` is null (`:2524`) — move fallback into `aggregator.py` so the client never invents prices.

---

### Notes on root causes worth knowing
- `renderMode()` is a 1,280-line god function (`:2733-4010`) containing chart builds, card text, env modal wiring, the rate-limit poller, and the pace modal. Most P1/P2 leak fixes are mechanical once it's split into `renderCards / renderCharts / initLimits (once) / renderLimits` — aligns with the 200-400-line file/function house style.
- The screenshot-blank-charts symptom (P1-3) is the same root cause as the "page feels slow" complaint: every visit pays entrance clip-paths + 1.2s count-ups + ~1s chart grows before the data is readable.

---

## R — Table/animation regression batch (2026-06-10, interaction-deep verification)

All reproduced in a real browser first (Playwright: scrolling inside every `.tbl-wrap`, expanding rows, simulated live refreshes, throttled-network loads), then fixed, then re-verified interactively.

- [x] **R-1 Layout shifts on load** (CLS 0.16 fast / 1.14 throttled → 0.00 / 0.01). Causes: `display:none → JS reveal` for Cost & Limits / env bar / OAuth gauges / hero note, costMeta filled post-paint, boot gated on deferred Chart.js. Fix: two-phase boot (DOM boot synchronous pre-paint; charts attach at DOMContentLoaded into pre-sized containers), statically visible sections, server-reserved OAuth-gauge placeholder (scope known server-side), per-model costMeta min-height, zero-width-space sub-detail line reservation.
- [x] **R-2 Sticky thead broken.** `border-collapse:collapse` painted the 4px header border at the table layer (it scrolled away with rows) → `border-collapse:separate`. `th[data-info]{position:relative}` overrode `position:sticky` ($/hr header scrolled away) → removed; tooltip now drops *below* the header (above was clipped by the scroll container). First-column `padding-left:0` → 0.85rem; machine-color border moved from `tr` (doesn't paint under `separate`) to `td:first-child` via `--mc`.
- [x] **R-3 Sortable headers** on all 5 `.tbl-wrap` tables: click/Enter/Space, ▲▼ indicator, `aria-sort`, numeric columns start descending, `—`/`·` sort last, detail rows travel with parents, sort re-applied after every tbody rebuild. Parser handles $, K/M/B, durations, "Xh ago", ISO dates.
- [x] **R-4 Session detail models** now a vertical mb-stat list (model left, cost right) instead of a ' · '-joined string.
- [x] **R-5 Number notation unified** — fN()/server `_fmt_num` both render 1.2K / 492K / 2.4M / 1.08B (no more comma-thousands next to M-notation); round-up promotion at unit boundaries.
- [x] **R-6 Live refresh preserves UI state.** Expanded rows keyed by session_id / model name (Sets), `.tbl-wrap` scrollTop restored, model & machine tables skip rebuild when data unchanged (render keys), sort re-applied.
- [x] **R-7 Extra finds (same pass):** jump-nav wrapped to 3 rows on phones so anchors landed 10px under it (`flex-wrap:nowrap`, it already horizontal-scrolls); session/model rows were mouse-only (now tabbable, Enter/Space toggles, `aria-expanded`, focus ring, expand arrow added to session rows); long values (project paths) in detail groups jammed into their labels (ellipsis).

### Deferred (noted, not fixed)
- **Font-swap CLS on cold cache:** Google Fonts load with `display=swap`; on a first uncached visit the Archivo Black/DM Sans swap can still nudge text metrics (~0.01 CLS observed). Fixing properly means self-hosting the fonts (like chart.js) or `size-adjust` fallback overrides — worth doing if the fonts ever get flagged again.
- **costMeta reservation is heuristic:** min-height reserves one line per costed model in the default 14d window; a localStorage mode of `today`/`all` with a *different* costed-model count can still shift a line's worth of height on slow networks.
