# Task brief: served-model timeline endpoint + chip semantics (backend)

Repo: ~/tokenfold (FastAPI + SQLite). Run tests with `~/tokenfold/.venv/bin/python -m pytest app/tests -q`
(python 3.12 venv; stock python3 is 3.9 and cannot run the suite). Read
`app/served_models.py`, `app/sigheader.py`, `app/tests/test_served_models_api.py`,
`app/tests/test_served_model_chip.py`, `docs/superpowers/specs/2026-08-18-served-model-signature.md`
first. Follow the module's existing style (docstrings that explain WHY, gate
enterprise scope with 404, clamp days, never raise on odd data). No em dashes or
en dashes anywhere in files (use a plain hyphen or a comma). Do NOT touch
`templates/dashboard.html` (someone else is editing it concurrently) or `app/aggregator.py`
beyond what is strictly needed. Commit on the current branch when done with a
conventional message (`feat: served-model timeline API and latest-state chip`); no
Claude attribution in the message. Do not push.

## Background

`events` rows with `sig_header IS NOT NULL` carry `served_model` (header f6, NULL when
the format hides it), `sig_version`, `sig_fields`, plus `model` (requested), `ts_epoch`,
`day`, `session_id`, `source_machine`. Reroutes happen in RUNS (sticky for a stretch,
then flip), e.g. per session `fable x104 -> kettle x124`, and fleet-wide the picture
changes hour to hour. The current chip ("3% kettle-e2c95a10-v2", share of the range)
hides that. We want the data served as runs and transitions.

State of a block, everywhere below: `"self"` when served_model == model, `"hidden"`
when served_model IS NULL, else the display slug via `slug()` (existing helper).
Requested models are reported by DISPLAY name via `display_model()` (as the chip does).

## 1. `GET /api/served-models/timeline?days=N&scope=` (new, in app/served_models.py)

Same auth/scope gate and days clamp as `/api/served-models` (factor the gate into a
helper both endpoints use). Query rows in range: `sig_header IS NOT NULL AND day >= since_day AND {pred}`,
selecting model, served_model, sig_version, sig_fields, ts_epoch, session_id, source_machine.
Do the shaping in Python (rows are tens of thousands at most).

Response:

```
{
  "days": N,
  "bin_seconds": B,          # server picks: days <= 2 -> 1800, <= 14 -> 3600, <= 90 -> 21600, else 86400
  "since_epoch": float,      # local midnight (TZ_NAME) starting the first day of the window
  "models": ["Fable 5", ...],  # display names that have at least one NON-self block in range, by blocks desc
  "bins": [["Fable 5", bin_start_epoch, state, blocks], ...],   # only for models in "models"; only non-empty cells; bin_start = floor(ts_epoch / B) * B
  "sessions": {session_id: {"machine": str, "runs": [[model_display, state, t0_epoch, t1_epoch, blocks], ...]}},
                             # only sessions with at least one non-self block in range; runs = consecutive same (model,state) in ts order
  "ledger": [ {"model": display, "state": state, "served_model": raw|null, "sig_version": int|null,
               "sig_fields": str|null, "first_seen": epoch, "last_seen": epoch, "blocks": n,
               "sessions": n, "machines": [sorted distinct], "first_session": sid, "first_machine": str}, ...],
                             # ALL combos incl. self, one row per (model, served_model|null, sig_version, sig_fields), sorted by first_seen asc
  "latest": {model_display: {"state": state, "since": epoch, "blocks": n}}   # for models in "models": state of the most recent block (by ts_epoch across all sessions) and the start of that final fleet-wide run
}
```

Keep the field order and names exactly; the dashboard is being written against this
contract. `since_epoch` uses `TZ_NAME` from app/config like `_since_day` does.

## 2. Chip semantics (served_model_chips in app/served_models.py)

Change the chip text from a range share to a state + when. For each model and mode
window, one entry per NON-self state present in the window (foreign slug or hidden),
ordered by blocks desc:

- if that state was seen within the last 24 h of the window's data (i.e. its last_seen
  is >= max(ts_epoch of any signed block of that model in the window) - 86400):
  `kettle-e2c95a10-v2 since Aug 17`
- otherwise: `kettle-e2c95a10-v2 Aug 17-18` (first-last seen, local dates; single
  day -> `Aug 17`)
- hidden renders as `hidden (v4) since Aug 19` (version = the most common sig_version
  among the hidden blocks).

Joined with " · " like today. Also return, per model, a `title` with the old share
text (`3% kettle-e2c95a10-v2 · 1% hidden`, denominator = all signed blocks of the
model, as `_chip_label` does today, now including hidden). Shape becomes
`{"all": {"Fable 5": {"text": "...", "title": "..."}}, "14d": {...}, "today": {...}}`.
Dates use TZ_NAME and `%b %-d`. Update `app/tests/test_served_model_chip.py` (and
whatever asserts the old shape, grep for `served_models` in app/tests) accordingly.
The aggregator passes the dict through untouched, verify with grep, do not restructure it.

## 3. Tests (add app/tests/test_served_model_timeline.py, extend the chip tests)

Use the existing fixtures/patterns in `test_served_models_api.py` (how it seeds
events and calls the app). Cover: bin size selection per days; bins only for models
with a non-self block and no empty cells; run compression per session (order by
ts_epoch, a flip mid-session yields two runs, a self-only session is absent);
ledger rows keyed by (model, served, version, fields) with first/last/machines/
first_session; latest = most recent block's state and the start of that run;
enterprise 404 and days clamp; chip text `since` vs date-range vs `hidden (v4)`, and
the title share. Run the whole suite; leave it green (pre-existing flaky failures in
client/test_push_perf.py::SkipCacheTest and client/test_cursor_atomic.py are known and
not yours).

Report back: the commit hash, files touched, test counts, and any contract deviation
you had to make (there should be none; if the contract is impossible as written, say
so instead of silently changing it).
