# Enterprise billing readings — design

**Date:** 2026-06-11
**Decisions (Jaedyn):** dashboard input · Δofficial vs Δmeasured + coverage % per interval.

## Problem

Tokenfold measures enterprise cost from transcripts; the Claude org billing
page is ground truth but is only a point-in-time number with no history.
Recording it periodically historizes it: the FIRST reading proves little, but
the delta between consecutive readings vs our measured cost over the exact
same window quantifies measurement coverage going forward (sidecar calls,
non-Claude-Code usage, list-vs-billed pricing).

## Data

New first-class table (not derived; never rebuilt from events):

    billing_readings(id INTEGER PK AUTOINCREMENT, scope TEXT DEFAULT 'enterprise',
                     amount_usd REAL NOT NULL, measured_usd REAL,
                     month TEXT NOT NULL,           -- UTC 'YYYY-MM'
                     recorded_at TEXT, recorded_epoch REAL, note TEXT)
    + index (scope, month, recorded_epoch)

`measured_usd` = frozen snapshot of our UTC-MTD (same `compute_window_cost`
path as the month hero) at the instant of recording — "what we knew then."

## API (app/billing_readings.py, new router)

- `POST /api/billing-readings {amount_usd, note?}` — Basic auth AND
  fail-closed: 403 when `DASHBOARD_PASSWORD` unset (require_dashboard_auth is
  open in that case by design, so the route re-checks). Validates amount
  (finite, 0 ≤ x < 10^7), note ≤ 256 chars. Server stamps UTC time + month,
  snapshots measured_usd, triggers eager rebuild. Returns the stored row.
- `DELETE /api/billing-readings/{id}` — same guards; 404 on unknown id.

## Payload (aggregator, enterprise scope only)

`billing_readings`: newest-first rows (capped 50). For consecutive same-UTC-month
pairs, the LATER row carries `delta_official`, `delta_measured`
(recomputed live over [t_prev, t_cur] — backfills sharpen history), and
`coverage_pct` (None when Δofficial ≤ $0.005). The newest same-month row also
carries `measured_since` (live measured cost from its epoch to now — official
counterpart is unknowable until the next reading). Personal scope: absent/empty.

## UI (enterprise view, under the month hero)

Input + Record button (rendered only when `readings_writable` — i.e.
DASHBOARD_PASSWORD set — via a template var like `ingest_key`), history table
(when, official $, measured-then $, Δoff, Δmeas, coverage %), per-row delete
with confirm. Values via `esc()`/`fC()`. POST/DELETE ride the page's Basic
auth; on success the existing live-refresh path repaints.

## Tests

`app/tests/test_billing_readings.py`: auth 401 / fail-closed 403, POST
happy-path with snapshot correctness, validation rejects, delete + 404,
interval math (incl. cross-month exclusion and zero-Δofficial), personal-scope
absence, note XSS posture, template wiring (ids + renderer present).
