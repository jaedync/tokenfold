# Tokenfold `/next`: build contract (2026-10-15)

This file is the shared contract for the parallel build of the new dashboard. Read
`docs/ux-2026-10/BRAINSTORM.md` for the why. The approved mockups are the visual spec:
`.ux/mockups/runway/index.html` and `.ux/mockups/ledger/index.html` (open them; they are
static HTML with a simulator). The quality bar is the Kiln dashboard
(`~/dev/kiln/src/kiln/web/static/`: ES modules, `css/kiln.css`, `js/panels/*.js`, uPlot).

## Owner decisions (fixed)

- Views: **Runway** (landing), **Ledger**, **History**, **Lab**. Built at `/next`, next to `/`.
- One forecast everywhere: the server pace block (`app/usage_pace.py`, measured from a granted
  reset). TIGHT only from `trend[bucket].tight_epoch`. Never compute another projection or
  time-to-limit in the browser.
- Phone order on Runway: live bar, hero, Now (waiting first), other windows, chart, Today.
- Light theme only for now (dark later). Bauhaus/De Stijl system: warm paper, ink rules,
  red/blue/yellow, Archivo Black + DM Sans (self-hosted woff2, copy from Kiln `fonts/`),
  uppercase tracked micro-labels, tabular numerals. WCAG AA: no yellow text on paper (ochre).
- Honest numbers: dollars are "API-equivalent at list price, not billed" (say it once per
  view); `≈` marks every estimate; stale data greys out with its age.
- Served-model check, energy/water, price table, benchmarks move to **Lab**.

## Front end layout (no build step)

```
templates/next.html            Jinja shell: masthead, live bar, <main id="view">, footer.
                               Passes scope, owner, default view, static version as data-*.
static/next/css/tokenfold.css  design tokens + atoms + layout (port from the mockups)
static/next/fonts/             archivo-black-latin-400.woff2, dm-sans-latin-wght.woff2 (+ OFL txt)
static/next/vendor/uplot/      uPlot.iife.min.js + uPlot.min.css (copy from Kiln vendor/)
static/next/js/main.js         boot: read data-*, router (#runway | #ledger | #history | #lab),
                               mount views, start polling; each panel mounts in try/catch
static/next/js/api.js          fetch helpers (same-origin, credentials, JSON, AbortController,
                               backoff); ?mock=1 loads static/next/fixtures/*.json instead
static/next/js/store.js        tiny immutable store: get(), set(key, value) -> new state, subscribe
static/next/js/format.js       money (≈$), pct, durations, clock (local tz), relative age, esc()
static/next/js/views/{runway,ledger,history,lab}.js     compose panels per view
static/next/js/panels/*.js     one module per panel: export default {id, mount(el, ctx)};
                               mount returns {update(state), destroy()}
static/next/fixtures/*.json    one fixture per endpoint below, shaped exactly as specified
```

- Polling: `/api/now` every 5 s; `/api/runway` every 30 s; `/api/today` every 60 s; the rest on
  view open and on the existing SSE version bump (`/api/stats/stream`). Pause when the tab is
  hidden (reuse the idea of `static/dashboard-refresh.js`).
- Tab title as status: `93% 7d · 2 waiting · Tokenfold`. Favicon stays.
- Escape every server string (`esc`). No inline event handlers. No CDN.

## New endpoints (all `Depends(require_dashboard_auth)`, `scope` query like `/api/rate-limits`)

All times are epoch seconds (float), minute-floored where they come from limit data. Money is
USD float, unrounded is fine (the browser rounds). Unknown = `null`, never a made-up number.

### `GET /api/now?scope=` (live, cheap, polled every 5 s)
```json
{"generated_at": 0, "scope": "personal",
 "live": {"working": 4, "waiting": 2, "idle": 1, "subagents": 7,
          "machines_online": 5, "machines_total": 9, "newest_data_age_s": 12,
          "burn_usd_per_h": {"m30": 33.1, "h1": 29.4}},
 "sessions": [{"id": "sid", "machine": "laptop", "project": "api-server", "title": null,
               "state": "waiting", "state_since": 0, "waiting_reason": "question|permission|attention|null",
               "model": "Opus 5.5", "family": "opus", "client": "claude-code|pi-agent",
               "subagents": 3, "cost_usd": 12.4, "burn_usd_per_h": 21.0,
               "share_of_binding_pct_per_h": 0.42}],
 "machines": [{"name": "laptop", "last_event": 0, "last_ingest": 0, "working": 1, "waiting": 0}]}
```
Source: `agent_state.snapshot()` (in memory) joined to events for cost/model/title, canonical
machine names (`machine` normalization used by `/api/stats`), `state_since` (add to
agent_state), waiting reason (from the notify event kind). Sessions sorted waiting (longest
first), working, idle. Scope: the agent state is fleet-wide; filter sessions to the scope's
account class where the session's events say so; say in the code what is not filterable.

### `GET /api/runway?scope=` (polled every 30 s)
```json
{"generated_at": 0, "scope": "personal", "binding": "claude:seven_day",
 "windows": [{"id": "claude:seven_day", "provider": "claude|opencode-go|codex",
   "key": "seven_day", "label": "Claude · 7-day", "kind": "long|short|month",
   "window_s": 604800, "pct": 92.0, "resets_at": 0, "window_start": 0,
   "observed_at": 0, "stale": false, "age_s": 60,
   "pace": {"pace_state": "on", "expected_pct": 92.7, "projected_pct": 99.2,
            "reaches_100_epoch": null, "elapsed_fraction": 0.93},
   "tight_epoch": null, "burn_pct_per_h": {"h1": 0.8, "h6": 0.57},
   "usd_spent": 3527.7, "usd_per_point": 38.3, "usd_left": 306.0,
   "waitable": false, "verdict": "clear|tight|runs_out|too_early"}],
 "budget": null}
```
- Build from the existing `_build_rate_limits` result (do not duplicate its SQL). `binding`
  (`app/runway_rules.py`): when no long window is at risk (no `runs_out`/`tight` verdict and
  every `projected_pct` < 90), `claude:seven_day` binds if present (work scope: `claude:month`).
  Otherwise the long window with the highest `projected_pct` (fallback per window: its pct)
  binds. Owner decision 2026-10-15: right after a weekly reset the Claude window is too early
  to project, and another provider's small projection must not take the hero. Short windows
  (≤ 1 day, e.g. 5 h) never bind; `waitable` = short window that resets within 60 min, or a
  long window whose reset comes within 60 min of its 100% time (pace `reaches_100_epoch`,
  else `tight_epoch`).
- `verdict`: `runs_out` when `reaches_100_epoch` < `resets_at`; `tight` when `tight_epoch` is set;
  `too_early` when pace says so; else `clear`.
- Work scope: `windows` holds the monthly billing meter as one `kind: "month"` window and
  `budget` = `{budget_usd, spent_usd, projected_usd, pace_state, per_workday_left_usd}` from
  `monthly_budget`.
- `GET /api/runway/trajectory?scope=personal&window=claude:seven_day` →
  `{"points": [[t, pct], ...], "resets": [t, ...], "start": 0, "end": 0}` from `limit_readings`
  (existing `/api/limit-history` logic), downsampled to ≤ 400 points.

### `GET /api/today?scope=` (polled every 60 s)
```json
{"day": "2026-10-15", "spend_usd": 412.0, "typical_by_now_usd": 268.0, "typical_day_usd": 365.0,
 "prompts": 84, "active_s": 20000,
 "by_model": [{"model": "Opus 5.5", "family": "opus", "provider": "anthropic", "usd": 236.0, "tokens": 0}],
 "by_machine": [{"machine": "devbox", "usd": 171.0, "working": 2}]}
```
Typical = median over the last 28 days of cumulative spend at this local hour (and full day).

### `GET /api/limit-rate?scope=personal` (Exchange rate; cached, recomputed hourly)
```json
{"generated_at": 0, "period_start": 0, "fit_quality": {"r2": 0.95, "intervals": 120},
 "usd_per_point": {"seven_day": {"value": 24.4, "lo": 23.7, "hi": 25.6},
                   "five_hour": {"value": 3.7, "lo": 3.5, "hi": 3.9}},
 "steps": [{"at": 0, "bucket": "seven_day", "from_usd_per_point": 12.9, "to_usd_per_point": 24.4, "cause": "granted_reset|rate_change"}],
 "families": [{"family": "opus", "usd_per_point": 24.7, "lo": 23.9, "hi": 25.6, "spend_usd": 2100.0},
              {"family": "fable", "usd_per_point": null, "lo": null, "hi": null, "spend_usd": 40.0, "note": "not enough data"}],
 "peak": {"active": true, "factor": 1.37, "until": 0, "tz": "America/Los_Angeles",
          "hours": [{"dow": 1, "hour": 5, "factor": 1.23}]},
 "cache_read_weight": {"value": 0.5, "lo": 0.4, "hi": 0.65}}
```
Port the method in `~/.cache/limit-xray/` (read `REPORT.md`, `lib.py`, `fit.py`): deduped
requests (the canonical predicate now does this), intervals between readings, NNLS by family,
day-block bootstrap bands, change points at granted resets or ±35% rate moves, peak factor from
5 h residuals. No numpy in the image unless you add it to `requirements.txt` (state the cost);
pure-Python NNLS for ≤ 8 features is fine. Show a family only with ≥ $100 spend in the period.

### `GET /api/ledger?scope=&from=&to=` and `GET /api/sessions/{id}?scope=`
```json
{"from": 0, "to": 0,
 "daily": [{"day": "2026-10-01", "by_model": {"Opus 5.5": 120.0}}],
 "sessions": [{"id": "sid", "machine": "devbox", "project": "dashboard", "title": "Add session drawer",
   "client": "claude-code|pi-agent", "start": 0, "end": 0, "active_s": 0, "prompts": 16,
   "model": "Fable 5.1", "cost_usd": 284.0, "orch_cost_usd": 200.0, "agent_cost_usd": 84.0,
   "runs": [{"id": "agentId", "parent": null, "model": "Sonnet 5.5", "start": 0, "end": 0,
             "cost_usd": 4.5, "tools": 32, "label": "Bench decode speed"}],
   "live": false}],
 "totals": {"cost_usd": 0, "sessions": 0, "runs": 0, "active_s": 0, "agent_active_s": 0, "prompts": 0}}
```
`/api/sessions/{id}` adds: `prompt_times[]`, `models[{model, cost_usd, share}]`,
`tools[{name, calls, errors}]`, `tokens{input, output, cache_read, cache_write}`,
`file_edits`, `served_model{state: "as_asked|partly_hidden|rerouted|unknown"}`.
Pivots (project/model/machine/tool/day) are computed in the browser from `/api/ledger`. If the
payload for 30 days is over ~2 MB or ~1 s on the production copy, add a `/api/ledger/pivot`
and say so. Twins are already excluded by the canonical predicates; use them.

### `GET /api/window-history?scope=personal&bucket=seven_day`
```json
{"windows": [{"start": 0, "end": 0, "peak_pct": 97, "hit_limit": false,
              "granted_resets": [0], "usd": 2440.0, "usd_per_point": 24.4}]}
```

## Engineering rules

- TDD: a failing test first for each endpoint and each pure function. Tests use the existing
  unittest style (`app/tests/_support.py` `TempDBTestCase`). Run the full suite with
  `TOKENFOLD_PLAYWRIGHT="$PWD/.ux/node_modules/playwright" .venv/bin/python -m unittest discover -s app/tests -t .`
  Known failure on main, not yours: `test_cache_invalidation ... prints_no_traceback`.
- Performance on real data: a scrubbed production copy (OAuth credentials removed) is at
  `~/.cache/tokenfold-prod/prod-20261015.db` (read-only). Copy it to your own scratch path
  (`cp --reflink=auto`) and point `DB_PATH` at the copy. Never write to the original. Targets:
  `/api/now` < 150 ms, `/api/runway` < 1.5 s, `/api/today` < 500 ms, `/api/ledger` 30 d < 1.5 s
  (cold, on this copy). The first start on a copy may rebuild rollups (about 2 min).
- Keep functions small, files 200 to 400 lines (800 max), immutable data, explicit errors
  (log and degrade one panel; never 500 the whole view).
- New routers: one module per feature (`app/now.py`, `app/runway.py`, `app/today.py`,
  `app/limit_rate.py`, `app/ledger.py`), registered in `app/main.py`.
- Do not change `/` or the old template, except the route registration. Do not push.
