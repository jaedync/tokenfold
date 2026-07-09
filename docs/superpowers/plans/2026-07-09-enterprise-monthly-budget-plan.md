# Enterprise Monthly Budget Pacing — Implementation Plan

Approved design (2026-07-09): mirror the Personal weekly-limit pacing on the Enterprise
scope, driven by a user-set monthly dollar budget instead of Anthropic's OAuth
utilization data (which is deliberately withheld from enterprise scope).

## Global Constraints

- **Compliance invariant (must not regress):** the `oauth` key must NEVER appear in any
  response when effective scope is enterprise or `LOCKED_SCOPE == "enterprise"`. The new
  `monthly_budget` block must NEVER appear when effective scope is personal.
- Budget stored in the `meta` table, key `enterprise_monthly_budget_usd`. No env var.
  No auto-seeding migration — the value is set post-deploy via the API/UI.
- Month boundaries are **UTC calendar months**, exactly matching
  `spend_history.monthly_costs`.
- MTD spend = tokenfold's computed event costs for enterprise scope (same cost pipeline
  as the monthly spend chart, including the US 1.1x geo override) — NOT `billing_readings`.
- Pace verdict: `ratio = (mtd_cost / elapsed_fraction) / budget_usd`;
  `under` if ratio < 0.90, `over` if ratio > 1.10, else `on` (same ±10% deadband as
  `limit_trends.bucket_trend`).
- Early-month suppression: when `elapsed_fraction < 0.05`, `pace` and
  `projected_eom_usd` are `null` (gauge + expected marker still render; verdict and
  projection text suppressed client-side).
- `projected_eom_usd = mtd_cost / elapsed_fraction` (window average — same projection
  style as the personal "projected by reset (window avg)").
- All new write input validated: budget must be a finite number, > 0, ≤ 1,000,000;
  `null` clears it. Invalid → HTTP 400 with a clear message.
- Jython does not apply here (FastAPI/CPython app): 4-space indent per existing app/ style;
  template JS matches existing dashboard.html idioms.
- Follow existing code style; NO new frameworks/deps. Tests use the existing pytest suite
  patterns in `app/tests/`.
- Do not push to any remote. Commit locally on `feat/enterprise-monthly-budget` with
  conventional-commit messages (no attribution footer).

## Task 1 — Backend: budget storage, pacing math, API surface

**Files:** new `app/monthly_budget.py`, edits to `app/api.py` (and router registration if
endpoints live in a new module), new `app/tests/test_monthly_budget.py`.

1. `app/monthly_budget.py` (new, small, focused):
   - `get_budget(conn) -> float | None` — read meta key `enterprise_monthly_budget_usd`;
     tolerate missing/garbage values by returning None (log at debug/warn, never raise).
   - `set_budget(conn, value: float | None)` — validate (finite, > 0, ≤ 1_000_000);
     `None` deletes the key. Raise `ValueError` with a message on invalid input.
   - `monthly_budget_block(conn, now: float | None = None) -> dict | None` — returns
     `None` if no budget set, else:
     ```json
     {
       "budget_usd": 1000.0,
       "month": "2026-07",
       "month_end_epoch": 1785542400,
       "mtd_cost": 123.45,
       "elapsed_fraction": 0.27,   // 4 dp is fine
       "expected_usd": 270.0,      // elapsed_fraction * budget
       "projected_eom_usd": 457.2, // mtd/elapsed; null when elapsed < 0.05
       "pace": "under" | "on" | "over" | null
     }
     ```
   - UTC month start/end via `datetime`/`calendar` (no local TZ anywhere). MTD cost:
     reuse the existing cost computation used for enterprise monthly spend — inspect
     `app/spend_history.py` (`monthly_costs`, `compute_window_cost_by_model` /
     `compute_window_cost`) and reuse the cheapest correct one, scope="enterprise",
     window `[month_start, now)`. Do not duplicate pricing logic.
   - Pace + suppression per Global Constraints. Round dollar fields to 2 dp,
     `elapsed_fraction` to 4 dp.

2. Wire into `/api/rate-limits` (`app/api.py::rate_limits`): when the **effective** scope
   is enterprise, attach `weekly_budget["monthly_budget"] = monthly_budget_block(...)`
   only when the block is not None. Personal scope: never attached. Keep the oauth
   gating code untouched.

3. New endpoints (place them following the pattern of the billing-readings endpoints —
   find where those routes live and use the same auth dependency so budget writes carry
   the exact same Basic-auth protection as the dashboard/readings writes; do NOT
   restrict the write to personal scope — this is an enterprise-scope setting):
   - `GET /api/enterprise-budget` → `{"budget_usd": float | null}`
   - `POST /api/enterprise-budget` body `{"budget_usd": number | null}` → 200 with the
     stored value, 400 on validation failure. Validate with the schema style already
     used by neighboring POST endpoints (pydantic model if that's the local pattern).

4. Tests (`app/tests/test_monthly_budget.py`, follow fixture patterns of neighboring
   test files — they build a temp sqlite DB and hit the FastAPI test client):
   - set/get/clear budget roundtrip; garbage meta value → get returns None.
   - Validation: 0, negative, NaN, inf, 2e6, "abc" → ValueError / HTTP 400; null clears.
   - Pace math: mid-month fixtures for under (<0.9), on (0.9–1.1 inclusive edges per the
     comparison operators chosen — pin the boundary behavior in tests), over (>1.1).
   - Early-month: elapsed_fraction < 0.05 → pace null, projected null, expected_usd
     still present.
   - Zero spend mid-month → pace "under", projected 0.
   - No budget set → `monthly_budget_block` returns None AND `/api/rate-limits?scope=enterprise`
     has no `monthly_budget` key.
   - Scope gating: enterprise response has `monthly_budget` (when set) and NO `oauth`;
     personal response has NO `monthly_budget` (even with budget set).
   - Month boundary: verify month/month_end_epoch for a fixture date like
     2026-07-09T12:00Z (month_end = 2026-08-01T00:00:00Z) and a December fixture
     (year rollover).
   - Run the full suite (`pytest app/tests`) and report the count; pre-existing failures
     (if any) must be reported as pre-existing with evidence they fail on the base commit.

## Task 2 — Frontend: enterprise monthly gauge card + inline budget edit

**Files:** `templates/dashboard.html` only (single-file template holding HTML+CSS+JS).

Context from Task 1: `/api/rate-limits?scope=enterprise` now returns
`weekly_budget.monthly_budget` per the JSON shape above (absent when no budget set),
and `GET/POST /api/enterprise-budget` reads/writes `{"budget_usd": number|null}`
(same-origin, cookies/Basic auth already in place for all dashboard fetches).

1. In the rate-limits section, render a **"Monthly usage limit"** gauge card when the
   fetched rate-limits payload contains `monthly_budget` (i.e. enterprise scope with a
   budget set). Hide/skip entirely otherwise. Reuse the existing gauge card DOM
   structure and CSS classes used by the personal weekly gauge — do not invent a new
   visual language:
   - Bar fill = `mtd_cost / budget_usd * 100`, colored with the existing
     `barColor(pct, expectedPct)` helper where `expectedPct = elapsed_fraction * 100`.
   - "expected" vertical marker at `expectedPct` using the existing marker markup, and
     make sure the card participates in the existing post-render marker-overlap hiding
     pass (Workstream G) so labels never collide at narrow widths.
   - Verdict line reusing the exact wording style: "under pace" / "on pace" /
     "<strong class=proj-warn>over pace</strong>". Suppress when `pace` is null.
   - Projection line: `Projected by month end: $X (window avg)` with `.proj-warn` when
     `projected_eom_usd > budget_usd`. Suppress when null.
   - Detail cells matching the weekly card's cell style: spent MTD (`$mtd_cost`),
     budget (`$budget_usd`), budget left (`$budget - mtd`, floored at $0 display),
     and the month label + "resets <month_end>" style countdown using existing
     time-formatting helpers.
2. Inline budget editing: clicking the budget value turns it into a small numeric input
   (or uses the same interaction pattern as billing-readings editing if one exists —
   check first and match it). On commit: `POST /api/enterprise-budget`; on success
   re-fetch rate-limits and re-render; on failure show the error non-destructively
   (match existing error display patterns). Escape/blur cancels. Also support clearing
   (empty input → `null`).
3. When enterprise scope has NO budget set, render a minimal one-line affordance in the
   same section ("Set monthly budget…" click-to-edit) instead of dead space, so the
   feature is discoverable. Personal scope: nothing new renders at all.
4. Numbers formatted with the existing currency/number helpers (`fC`/`fN` or whatever
   the template uses — match neighbors).
5. Tests: this template has server-side render tests in the suite (they check markers /
   parse `#tf-data` etc.) — add coverage consistent with how existing template features
   are tested (e.g., marker strings present in served HTML, endpoint wiring). Run the
   full suite and report counts.

## Task 3 — End-to-end verification (controller-run, not a subagent implementer task)

- Full pytest suite green.
- Playwright render checks at the usual 4 widths (mobile → desktop) on BOTH scopes
  against a local dev server: enterprise shows the gauge (with a budget set via POST),
  personal shows no trace; zero JS console errors; no horizontal scroll at mobile.
- Verify the compliance invariant live: enterprise response has no `oauth`; personal
  has no `monthly_budget`.
