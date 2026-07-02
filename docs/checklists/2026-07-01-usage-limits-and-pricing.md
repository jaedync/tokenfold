# Usage-Limits & Pricing Checklist — 2026-07-01

Tracking doc for four related problems raised 2026-07-01. Produced by a 5-area
audit (limit capture, projection math, bucket handling, pricing engine, and a
read-only probe of the live ms01 database), each area adversarially verified.
Every claim below is grounded in code or production data as of `main@218c05e`.

Status legend: `[ ]` todo · `[x]` done · `[~]` in progress. IDs (A1, B2, …) are
stable — reference them when picking up work.

---

## Production evidence snapshot (2026-07-01)

- `meta.oauth_usage` live blob: `five_hour` 51%, `seven_day` 11% still populate
  the legacy keys, but **every legacy per-model bucket (`seven_day_sonnet`,
  `seven_day_opus`) is null and there is no `seven_day_fable` key**. The Fable
  weekly limit (17%, resets 2026-07-02T07:59:59Z) exists ONLY in a new
  `data.limits[]` array: `kind=weekly_scoped`, `scope.model.display_name="Fable"`,
  `scope.model.id=null`. Eight opaque null bucket keys also present
  (`seven_day_cowork`, `tangelo`, `iguana_necktie`, …) — the payload shape is
  fluid; parsers must ignore null/unknown entries instead of hardcoding names.
- **Sonnet 5 is overcharged 1.5x right now**: no static entry exists, so the
  LiteLLM-fed `pricing_cache` supplies standard rates (3, 15, 3.75, 0.30)
  instead of intro rates (2, 10, 2.50, 0.20). Verified to reproduce all five
  `daily_summary` "Sonnet 5" cost fields to <1e-6. Recorded 2026-06-30..07-01:
  **$26.8133; correct: $17.8756 ($8.94 overstated** across 754 events, 3
  machines).
- **No utilization history exists anywhere.** Both writers (server poller every
  600s, client `POST /api/usage` on hook fires) `INSERT OR REPLACE` the single
  `meta.oauth_usage` row. `billing_readings` historizes only the manual
  enterprise MTD dollar figure (stale since 2026-06-17). A mid-window limit
  reset is destroyed by the next poll.

## Answers to the four questions

1. **Are we historizing the usage limit?** No. Latest snapshot only; resets are
   unrecoverable after ~10 minutes. → Workstream C.
2. **Better trend analysis between integer percent steps?** Nothing exists —
   the only projection is a client-side whole-window linear pace
   (`calcExpectedPct`). Utilization arrives integer-quantized, so real burn
   rate needs step-crossing interpolation over historized readings. → D.
3. **Does the dashboard support the Fable limit?** No. The Sonnet gauge
   silently vanished (null bucket → `sonnet_pct=null` → row hidden) and the
   Fable limit is invisible because it lives in `limits[]`, which nothing
   parses. → B.
4. **Sonnet 5 pricing?** Wrong today (1.5x over), and it gets worse: the daily
   full sweep re-summarizes ALL days every ~24h, so after 2026-09-01 it would
   silently reprice August history at standard rates unless era selection keys
   on the EVENT timestamp. → A.

Recommended order: **A (money is wrong today) → B (gauge invisible today) →
C → D (D depends on C) → E anytime**.

---

## Workstream A — Sonnet 5 pricing + date-effective eras (P0)

Key mechanics discovered: costs are baked into `daily_summary` at summarize
time, but `_run_periodic_sweep` re-rolls the last 7 days hourly and
`_run_full_sweep` re-rolls ALL days every ~24h (`aggregator.py:1238-1267`,
armed in `main.py:46`). So **no manual backfill mechanism is needed — history
self-heals within ~1h of deploying a fix** — but the same sweeps are the
corruption vector that makes event-time eras mandatory before September.

- [x] **A1** (P0/M) Date-effective pricing eras in `pricing.py`, keyed on event
  time. `MODEL_PRICING` values become either a plain 4-tuple (all existing
  entries unchanged) or an ascending list of `(effective_from_epoch_utc,
  4-tuple)`. `get_pricing(model, ts_epoch=None)` and `compute_cost(...,
  ts_epoch=None)` select the last era with `effective_from <= ts_epoch`
  (`None` → `time.time()`). Boundaries built via
  `datetime(..., tzinfo=timezone.utc).timestamp()`, no magic epochs.
  Files: `app/pricing.py`, `app/tests/test_pricing_eras.py`.
  Accept: constant-tuple models identical with/without `ts_epoch`; era-list
  models flip at the boundary; existing `test_pricing.py` /
  `test_pricing_unknown.py` pass unchanged.
- [x] **A2** (P0/S) Add Sonnet 5 entries. `MODEL_DISPLAY['claude-sonnet-5'] =
  'Sonnet 5'` (explicit, not munging-dependent); `MODEL_PRICING['Sonnet 5']` =
  intro `(2.00, 10.00, 2.50, 0.20)` from epoch 0, then `(3.00, 15.00, 3.75,
  0.30)` from 2026-09-01T00:00:00Z — with a comment flagging 3.75/0.30 as an
  ASSUMED 1.25x/0.1x of the $3 base (Anthropic only published intro cache
  rates). Insert the key between the Opus block and Sonnet 4.6 so
  `MODEL_ORDER` sorts it correctly.
  Accept: `display_model('claude-sonnet-5') == display_model('claude-sonnet-5-20260930') == 'Sonnet 5'`;
  era lookups return the right tuples; `is_priced('Sonnet 5')` true with
  dynamic pricing cleared; sorts above Sonnet 4.6.
- [x] **A3** (P0/S) Era-listed static entries win over LiteLLM. `get_pricing`
  checks `_dynamic_pricing` FIRST (`pricing.py:201-204`), so the poisoned
  `pricing_cache` would override A2. Flip precedence ONLY for models whose
  static value is an era list (plain-tuple models keep dynamic-first so
  LiteLLM can still correct drift for the other 11 models).
  Accept: seeded `_dynamic_pricing['Sonnet 5'] = (3.0, 15.0, 3.75, 0.30)`
  still yields intro rates for an August `ts_epoch`, no cache invalidation
  needed; existing dynamic-override test for plain-tuple models still passes.
- [x] **A4** (P0/M) Thread event timestamps through all cost call sites.
  `summarizer.py` passes `r['first_ts']` (already selected, line 54);
  `cost_windows.py` prices at per-request granularity so a window straddling
  2026-08-31/09-01 sums both eras correctly (its outer
  `GROUP BY model,speed,geo` currently discards timestamps); `aggregator.py`
  sites: recent-sessions (~313, 323-330), today-panel (~600, 634), all-time
  parts (~959, 1007-1009), 48h hourly (~555), `api.py:145` pace chart;
  `notify.py:100` deliberately stays wall-clock (live events) with a comment.
  Accept: seeded Sonnet 5 requests at Aug 31 23:00Z + Sep 1 01:00Z price as
  2.00·M_aug + 3.00·M_sep through `compute_window_cost`; `summarize_days` on
  an August day with mocked wall-clock in September still stores intro cost.
- [x] **A5** (P1/S) Boundary + modifier guard tests. Both sides of the era
  flip for input/output/cache-write/cache-read; 1h-cache premium = 2x the
  era's base input ($4 intro vs $6 standard); `GEO_US_MULT` 1.1x multiplies
  era rates (preserve current behavior — prod applies it to vertech rows
  today). Test docstrings state which rates are assumptions.
- [x] **A6** (P1/S) Regression: unknown-model fallback unchanged. Unknown
  models still bill $0 + `unpriced` flag + one rate-limited forced LiteLLM
  refresh, with or without `ts_epoch` (this contract is what prevented a
  repeat of the Fable 5 undercount); web-search fee still charged for
  unpriced models.
- [x] **A7** (P0/S) Verify healing on prod after deploy. Within ~1h (periodic
  sweep) the five `model_json` "Sonnet 5" cost fields for 2026-06-30/07-01
  drop to exactly 2/3 of current values (total 26.8133 → 17.8756 ± 0.001,
  e.g. vertech 2026-07-01: 13.70926 → 9.13951); no other model's fields
  change. One-shot read-only SQL on ms01 + dashboard check.
- [x] **A8** (P2/S) Housekeeping: `MODEL_BENCHMARKS` entry only if
  Anthropic-published scores exist (never fabricate); `water.py` explicit
  `'Sonnet 5'` Sonnet-class energy entry; aggregator model-table renders a
  Sonnet 5 row without KeyError.

Open: standard-period cache rates unconfirmed (assumed 1.25x/0.1x); billing
cutover timezone assumed UTC; does Sonnet 5 get a fast tier later
(`FAST_OPUS_BASE` is Opus-only — silent standard-rate billing if one appears)?

## Workstream B — Fable limit gauge via `limits[]` (P0)

The audit's code-only reading suggested a generic `seven_day_*` pass-through;
production data overrules it: per-model limits now arrive ONLY in
`data.limits[]`. Legacy keys must still be tolerated (other plans/accounts may
populate them), but `limits[]` is the primary source.

- [x] **B1** (P0/M) Normalization layer: one helper (new module or `api.py`)
  `normalize_usage_buckets(usage_dict) -> [{key, label, utilization,
  resets_at}]` merging (a) non-null legacy dict buckets (`five_hour`,
  `seven_day`, `seven_day_opus`, …) and (b) `limits[]` entries
  (`session`→five_hour, `weekly_all`→seven_day, `weekly_scoped`→keyed by
  `scope.model.display_name`, tolerating null `model.id`). Null/unknown noise
  entries (`tangelo`, `iguana_necktie`, …) dropped without error. Reused by
  B2, C1, and B5 — single source of truth.
  Accept: unit test feeds the EXACT prod payload shape (limits array +
  null legacy buckets + 8 noise keys) → exactly three buckets out
  (five_hour 51, seven_day 11, Fable 17 w/ its resets_at); dedupe rule when
  both legacy and limits[] report the same window is defined and tested.
- [x] **B2** (P0/S) `/api/rate-limits` emits `oauth.buckets` (list of
  normalized buckets, `resets_at` minute-scrubbed via `_scrub_to_minute`).
  Remove `sonnet_pct`/`opus_pct` and migrate the template in the same change
  (grep says the template is the sole runtime consumer; no dual sources).
  Update the endpoint docstring (`api.py:75-87`).
  Accept: fable bucket flows end-to-end; enterprise scope emits no `oauth`
  key and `"buckets"` joins the forbidden-strings list in
  `test_enterprise_only.py`.
- [x] **B3** (P0/M) Dynamic gauge rendering in `dashboard.html` (replaces the
  hardcoded Opus/Sonnet rows at ~4663-4684). Label map (`Fable · 7-Day` etc.)
  with a derived fallback label for unknown buckets; per-bucket `resets_at`
  shown via `fmtReset` (legacy rows never had it). Written test-first: seed
  a fable-only payload → exactly one Fable row, zero Sonnet/Opus rows.
  **Visual verification in a real browser before calling it done.**
- [x] **B4** (P2/S) Deterministic color strategy: opus=var(--black),
  sonnet=var(--blue), fable=distinct (yellow + --yellow-text, or one new
  token); unknown buckets draw from a fixed cycle indexed by sorted key so
  colors are stable across the 60s poll; **var(--red) never assigned**
  (reserved for over-pace overflow). Verify visually with 3+ buckets seeded.
- [x] **B5** (P1/M) `/api/ha` gains `model_buckets` built from the same
  normalizer: `{pct_used, resets_at, resets_in_s}` per scoped bucket,
  `_truncate_to_minute` scrub preserved (anti-fingerprinting), absent on
  enterprise-locked instances exactly like `five_hour`/`weekly`. Update
  `scripts/ha-smoke.sh` + the 2026-04-15 HA spec note (line ~146).

Open: should scoped gauges get full `buildGauge` treatment (expected-pace
marker) now that `resets_at` is available? (Cheap once B3 lands.)

## Workstream C — Historize limit readings + reset detection (P1)

- [x] **C1** (P1/S+M) Append-only `limit_readings` table + shared writer.
  Schema block in `db.py` SCHEMA (CREATE TABLE IF NOT EXISTS — no `_migrate`
  change needed): `id PK, fetched_epoch REAL NOT NULL, source TEXT
  ('server'|'client'), bucket TEXT, utilization REAL, resets_at TEXT (raw,
  pre-scrub), resets_at_epoch REAL`, index `(bucket, fetched_epoch)`. New
  `app/limit_readings.py` exposes `record_limit_readings(conn, usage_dict,
  fetched_epoch, source)` using the B1 normalizer; validates utilization is
  finite numeric (prod sends floats: 51.0), skips+logs bad buckets, never
  raises into callers. **Every-poll writes, NOT dedupe-on-change**: "still
  N% at T" rows are what bound each integer step-crossing to one poll
  interval for D1 (~720 rows/day — trivial). Wire BOTH writers:
  `usage_fetcher._fetch_usage` and `ingest.store_usage`.
  **Compliance gate (verifier catch): `store_usage` has NO `LOCKED_SCOPE`
  gate today — the new write must skip when `LOCKED_SCOPE=='enterprise'`**,
  matching `usage_fetcher.should_run`; assert via the
  `test_enterprise_only.py` pattern.
- [x] **C2** (P1/M) Reset detection, derived on read (tunable without
  re-migration). `detect_resets(rows)`: for consecutive same-bucket readings,
  reset iff `prev.resets_at_epoch > prev.fetched_epoch` (window still active
  — rules out natural expiry) AND (`utilization` drops ≥ RESET_DROP_PTS
  (start: 10) OR `resets_at_epoch` jumps > 2x poll interval). Named
  constants. Tests: natural 5h expiry (63→0, resets_at in past) → NO event;
  mid-window 63→2 → event; resets_at +26h w/ flat utilization → event;
  1-2pt jitter → NO event.
- [x] **C3** (P1/M) `GET /api/limit-history?bucket=&hours=` behind
  `require_dashboard_auth`, router registered like `billing_readings`
  (`main.py:99,109`). Returns readings (minute-scrubbed `resets_at`) +
  detected resets. 403/absent on enterprise scope or locked instance —
  mirror `test_enterprise_only.py`. Validate bucket `[a-z0-9_]{1,64}`,
  clamp hours ≤ 2160.
- [x] **C4** (P1/M) Dashboard reset visibility: utilization series + vertical
  reset markers on the existing pace-chart modal (~4742+), plus a
  "reset <time>" annotation on the weekly gauge when a reset falls inside
  the current window. Seed a synthetic mid-window reset; template test +
  real-browser visual check.
- [x] **C5** (P2/S) Retention: `DELETE FROM limit_readings WHERE
  fetched_epoch < now - RETENTION_DAYS(90)` at most daily, in the fetcher
  maintenance loop (fine once C1's ingest gate exists — locked instances
  accumulate no rows). Test: 91d-old rows pruned, 89d survive.
- [x] **C6** (P2/S) `POST /api/usage` bucket-level validation: keep raw-dict
  storage for back-compat (non-dict already 400s — `ingest.py:505`), delegate
  per-bucket validation to `record_limit_readings` (skip+log invalid, record
  valid).

Open: can utilization legitimately dip 1-2 pts mid-window (argues for the
10-pt threshold)? Cross-dedupe server/client readings in the same minute, or
let analysis collapse runs? Should HA push a "reset detected" boolean (P2
follow-on)?

## Workstream D — Sub-window burn rate & projections (P1, needs C)

- [x] **D1** (P1/M) `app/limit_trends.py`: `compute_burn(conn, bucket, now,
  window_s)` → `{pct_per_hr, samples, resets_in_window}`. Math: load window
  + one straddling reading each side; split at reset boundaries (C2), keep
  trailing post-reset segment; piecewise-linear interpolation through
  `(fetched_epoch, utilization)` points (integer step from u to u+k between
  polls → crossings placed by the linear segment = unbiased midpoint under
  ±poll-interval uncertainty); burn = `(û(now) − û(now−window_s)) /
  (window_s/3600)`; `None` under 2 readings or <15 min span. Exact-value
  unit tests with frozen `now`.
  AMENDED at review: the denominator is the OBSERVED span
  `now - max(boundary, first_reading)` rather than the full `window_s`,
  so a reset-trimmed or cold-start segment reports its true rate instead
  of a diluted one (the spec formula overstated time-to-limit up to 2x
  right after a reset).
- [x] **D2** (P1/M) Trend block in `/api/rate-limits`:
  `oauth.trend[bucket] = {burn_1h_pct_per_hr, burn_6h_pct_per_hr,
  eta_100_epoch, pace}` for every bucket in readings (bucket-generic — a new
  scoped bucket appears with zero code change). `eta = now +
  (100−pct)/burn` (6h burn for weekly, 1h for five_hour), null when burn ≤ 0;
  `pace ∈ {under,on,over}` vs even-drain `100/window_hours` ± 10% deadband.
  Enterprise invariant re-asserted.
- [x] **D3** (P1/M) Gauge rendering: "≈X%/hr (last 6h)" + "limit ~<relative
  time>" + under/on/over label; existing whole-window projection relabeled
  "window avg"; 5-Hour gauge gets the same treatment (today it has zero pace
  analysis); graceful degradation when trend absent (renders exactly as
  today — asserted).
- [x] **D4** (P1/L) Utilization-over-window chart: server-downsampled series
  (≤200 pts) in `oauth.trend[bucket].series` riding the existing 60s
  `/api/rate-limits` poll (no new endpoint, no #tf-data bloat); even-drain
  reference diagonal; NOW-line plugin reuse (~4835); reset markers from C2.
  Real-browser visual check.
- [x] **D5** (P1/S) Fix the mixed-window "budget left" estimate — visible bug
  today: `dashboard.html:4634-4641` divides ROLLING-7d cost by LIMIT-window
  pct, so right after any reset it shows absurd numbers. Either compute over
  the actual limit window `[resets_at−7d, now]` (server has
  `compute_window_cost`) or suppress when a reset was detected in the last
  24h / `wExpected` below a floor.
- [x] **D6** (P2/S) Clamp/hide the "expected" pace marker when `resets_at <=
  now` (stale snapshot currently pins the marker at 100% against old
  utilization during idle periods between 5h windows).

Open: 10-min polling bounds crossing precision to ±5 min — tighten
`USAGE_FETCH_INTERVAL_S` to 300s (mind the 429 backoff) or carry the
uncertainty? Expose burn/ETA via `/api/ha` too? 5-hour bucket: windowed
regression vs crossing-pair instantaneous rates (only ~6 readings/hr)?

## Workstream E — Test hygiene (independent)

- [x] **E1** (P1/S) Fix the 10 `test_session_burn.py` reds. Root cause
  verified: hardcoded `NOW=1781000000` (2026-06-09, line 15) drifted outside
  `_build_recent_sessions`' `now − RECENCY_DAYS(14)` cutoff → empty
  `recent_sessions`, 8 ERROR + 2 FAIL. These test the per-session $/hr
  feature (`aggregator.py:267-383`), NOT limit-percent burn — do not
  conflate with D. Fix: derive NOW from `time.time()` (or patch the
  aggregator clock) so the suite is date-independent. Accept: 16/16 pass on
  any date, no production code changes, **zero allowed-red exceptions
  remain** in the full suite.

---

## Cross-cutting notes

- **Deploy**: normal path — push GitHub main → box `git pull` →
  `docker compose build` + `up -d` from `/home/jaedy/services`. A7 needs one
  post-deploy prod verification pass.
- **Deadline pressure**: A must ship (or at least A1/A2/A4) **well before
  2026-09-01**, or the daily full sweep auto-corrupts August Sonnet 5 history
  at standard rates.
- **Compliance invariant** (applies to B2/B5/C1/C3/D2): enterprise scope and
  enterprise-locked instances must never emit or persist personal Max limit
  data — every new surface re-asserts the `test_enterprise_only.py` pattern.
- **Privacy invariant**: `resets_at` is stored raw but minute-scrubbed at
  every read boundary (`api.py:22-38`, `ha.py` `_truncate_to_minute`) — new
  endpoints included.

---

## Deployed 2026-07-02

Pushed `59de3f5..9f79506`, box fast-forwarded, image rebuilt, container
recreated. Verified live: 401/200 auth gates; `oauth.buckets` carries
`scoped:fable` with real data; `limit_readings` historizing (first poll 72s
after boot); `/api/ha` `model_buckets.fable` minute-scrubbed; **A7: Sonnet 5
cost for 06-30+07-01 re-rolled $33.85 -> $22.57, ratio 0.6667 — exactly the
standard->intro 2/3 repricing** (totals grew past the audit's $26.81 snapshot
as 07-01 usage accrued; the ratio is the proof).

---

## Workstream F — Follow-on 2026-07-02: window-true spend + look-back charts

User report: weekly subpanel shows `spent · rolling 7d` (never resets) where
the current-limit-window spend belongs; granted mid-window resets must also
truncate the window. Plus two new look-back charts. Design approved
2026-07-02 with one amendment: window-history entries are per reset-bounded
SEGMENT (a granted reset splits a week → multiple entries per week), never
one-per-calendar-week.

- [x] **F1** (P0/S) `limit_window.start_epoch` respects granted resets:
  `max(weekly_resets_at − 7d, last detect_resets boundary in-window)`,
  minute-floored. Lazy import (api ← limit_readings cycle).
- [x] **F2** (P0/S) Weekly subpanel rows become `spent · this window` /
  `active · this window` fed from `oauth.limit_window`; rolling-7d rows
  remain only as fallback when `limit_window` absent; carry-over note gated
  on the fallback path only.
- [x] **F3** (P1/M) `compute_window_cost_by_model()` in cost_windows.py —
  same dedupe/era/geo semantics, returns {display_model: cost};
  `compute_window_cost` delegates (sum of values, behavior identical).
- [x] **F4** (P1/M) `app/spend_history.py`: weekly window segmentation —
  natural boundaries from observed `resets_at_epoch` transitions, granted
  boundaries from `detect_resets` (merge within 3600s, double-fire safe),
  pre-historization boundaries inferred by 7d back-stepping (flagged
  `inferred`), ongoing segment last. Per segment: cost, peak_pct (null
  pre-historization), end_kind natural|granted|ongoing. All emitted epochs
  minute-floored.
- [x] **F5** (P1/M) UTC monthly costs per model from raw events (NOT local-TZ
  day summaries), oldest event → now.
- [x] **F6** (P1/M) `GET /api/spend-history?scope=` — months always
  (scope-filtered); `windows` key ONLY personal scope + not
  enterprise-locked + oauth_usage present (mirrors rate-limits oauth
  gating). Dashboard auth.
- [x] **F7** (P2/S) `limit_readings` RETENTION_DAYS 90 → 400;
  `/api/limit-history` HOURS_MAX matches.
- [x] **F8** (P1/L) New dashboard section "Spend history": (a) limit-window
  chart — bars = $ per segment labeled by end date (multiple bars per week
  possible), peak-% dots on 0–100 right axis, granted/ongoing visually
  distinct, inferred windows marked; (b) monthly UTC chart — stacked by
  model, current month partial. Personal scope only for (a).
- [x] **F9** (P1/M) Tests: segmentation (multi-reset week, double-fire
  merge, inferred stepping, minute flooring), by_model delegation equality,
  UTC month boundary vs local TZ, endpoint gating + auth, F1 truncation,
  template labels.
- [x] **F10** (P1/M) Playwright visual pass with extended seed (granted
  reset mid-week → 2+ bars in one week; multi-month events).

### Workstream F review + fix record (2026-07-02)

Adversarial review (opus): FIX_FIRST with 1 HIGH — all addressed, suite
606/606 OK:

- **H1 (fixed)**: unbounded look-back from unvalidated ancient ts_epoch
  (epoch-0 event => ~24k month queries / ~105k segment builds per dashboard
  load). Fix: `_oldest_event_epoch` clamped (segments: MAX_SEGMENTS+2
  windows; months: MAX_MONTHS=72 x 32d), loop caps as belt-and-braces,
  months wrapped in its own narrow try/except (was the one blast-radius
  gap). Ingest-side ts_epoch range validation deliberately NOT touched in
  this batch (shared writer; legit legacy imports carry old timestamps) —
  read-side clamps fully neutralize the DoS. Open item if ever revisited.
- **M1 (fixed)**: stale/out-of-order client snapshot (55->40->55) read as a
  granted reset, moving limit_window.start_epoch (headline undercount) and
  painting a fake red bar. Fix: `persistent_resets()` in limit_readings
  (drop events whose NEXT reading recovered above before-10pts; trailing
  events kept provisionally); F1 uses it; `_pair_boundary` requires drop
  persistence + only forward anchor jumps count. Burn/trend paths keep raw
  detect_resets (spurious cut is benign there).
- **L1 (fixed)**: ongoing segment inherited inferred=True on fresh installs
  -> live window rendered faded. Ongoing is now never inferred.
- **L2 (won't fix, rationale)**: windows chart labels use browser-local
  dates while monthly chart is UTC. Deliberate: window boundaries are
  moments (match the gauges' local "resets Sat 8:57a" convention); months
  are UTC billing periods. Different kinds of time.
- Bonus fix surfaced by visual pass: MODEL_COLORS predated the Claude 5
  family — Sonnet 5 hash-collided to Sonnet 4.6's exact blue (ambiguous
  stacked charts on PROD today), Fable 5 to gray, Opus 4.8 to red. Added
  explicit family-hue/era-shade entries (Fable teal #0f6b5c, Opus 4.8
  #b8231a, Opus 4.7 #e08a1e, Sonnet 5 #4f83bd).
- Visual pass (Playwright, seeded multi-window scenario): F2 rows
  window-true, 19 window bars (inferred faded blue / granted red /
  ongoing gray + peak dots), monthly stack shows model-mix shift with
  distinct colors, "Mar '26" month labels, zero page errors, no
  horizontal scroll at 1440/960/768/390.

---

## Workstream F deployed 2026-07-02

Pushed `bc9784c..6a4a6f4`, box fast-forwarded, image rebuilt, container
recreated. Live verification: 401/200 auth gates on /api/spend-history;
personal months Jan $148.83 -> Jun $9,219.33 (7 true-UTC months); 26 window
segments (pre-historization windows inferred with null peak, ongoing window
$225.32 at peak 11%); all emitted epochs minute-floored; enterprise scope
gets NO windows key and a scope-filtered 3-month axis (first enterprise
events mid-May); `limit_window` = {start 2026-07-02T08:00Z (the natural
rollover), $224.51, 3h05m active} feeding the new "spent · this window"
subpanel rows; spendHistorySection + Fable 5 color entries live in the
served template.
