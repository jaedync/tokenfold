# Realtime Pipeline Plan — 2026-07-10

Goal: keystroke-to-dashboard latency of ~2–4 seconds (from today's ~45–90s), with
three surgical changes: coalesced eager rebuilds (server), SSE version stream
(server), EventSource live refresh with poll fallback (dashboard), plus a resident
client watcher daemon. Approved design in conversation 2026-07-09/10.

## Global Constraints

- Jaedyn's explicit requirement, verbatim: "If the backend rebuild takes longer
  than events take to land we do need to be careful there to make sure things get
  queued up properly and not over-stacked. efficient!"
- Client (`client/claude-stats-push.py`) is **stdlib-only** Python, compatible with
  Python 3.9 (`from __future__ import annotations` pattern). No pip dependencies.
- Server tests: stdlib `unittest`, run via `.venv/bin/python -m unittest discover -s app/tests`.
  Do NOT use pytest (it segfaults in this environment). Bare `python` is not on
  PATH — always `.venv/bin/python`. The suite is currently 707 tests, all passing;
  it must stay green.
- Client tests live at `client/test_*.py`, also stdlib unittest.
- The Anthropic OAuth usage fetch cadence must NOT change: `USAGE_FETCH_MIN_INTERVAL = 300`
  inside the client gates it, regardless of how often push cycles run.
- TDD mandatory: write failing tests first (RED), then implement (GREEN). Include
  RED/GREEN evidence in the report.
- Conventional commits (`feat:`, `fix:`, `test:` ...), NO Co-Authored-By or any
  attribution footers.
- Follow existing code style in each file (Python: 4-space, snake_case, module
  docstrings explaining design intent; dashboard JS: the existing IIFE + `var`/`const`
  mix, comments explain WHY).
- Never print/log/echo secrets: `STATS_API_KEY`, `DASHBOARD_PASSWORD`,
  `notify_token`, OAuth tokens. Tests must not embed real tokens.
- SSE endpoint must be auth-gated exactly like the dashboard
  (`Depends(require_dashboard_auth)` from `app/auth.py`).
- Dashboard behavior must not regress when SSE is unavailable (old browser, proxy
  buffering, endpoint 500): the existing 30s version poll is the fallback.

## Task 1: Coalesced rebuild drain loop (server)

**File:** `app/aggregator.py` (lines ~194–236, `trigger_eager_rebuild`).
**Tests:** extend `app/tests/test_cache_invalidation.py`.

### Current behavior and the gap

`trigger_eager_rebuild()` bumps `_cache_gen` + `_cache_version` and clears
`_cached_data` under `_cache_lock`, then spawns at most one worker thread
(`_rebuilding` flag). The worker builds `_build_dashboard_data_inner(DEFAULT_SCOPE)`
and stores the result only if `_cache_gen` is unchanged (generation check).

**The gap:** when an invalidation lands mid-build, the in-flight result is
discarded and NO new rebuild is scheduled. The cache stays cold until the next
HTTP request pays for a synchronous build. Under ~1s-cadence realtime ingest with
a rebuild that takes longer than the ingest interval, eager rebuilds would be
discarded almost every time — precisely the over-stacking/starvation hazard
Jaedyn called out.

### Required change

Convert the worker into a **drain loop**: after each build, under `_cache_lock`,
if `_cache_gen` still equals the generation the build started with, store the
result, clear `_rebuilding`, and exit. Otherwise adopt the new generation and
build again. Invariants (each needs a test):

1. **At most one worker thread ever runs**, no matter how many invalidations
   arrive (spy on thread creation or on concurrent entry into the build fn).
2. **No lost rebuilds:** an invalidation during a build always results in one
   more build; the final cached data is built from the final generation.
3. **Coalescing:** N invalidations arriving during one build cause exactly ONE
   follow-up build, not N (e.g. 5 invalidations mid-build → 2 total builds).
4. **Exception safety:** if `_build_dashboard_data_inner` raises, `_rebuilding`
   is cleared (try/finally) so future invalidations can rebuild; the exception
   must not leave the flag stuck True.
5. **Version semantics unchanged:** `_cache_version` bumps once per
   invalidation call (the SSE stream and `/api/stats/version` rely on it);
   `get_cache_version()` untouched.

### Test approach

In tests, patch `app.aggregator._build_dashboard_data_inner` with a stub using
`threading.Event` gates to deterministically hold a build "in flight" while the
test fires more invalidations — no sleeps for synchronization (a short
`join(timeout=...)` on completion events is fine). Count builds and assert the
invariants above. Also assert existing behavior still holds: result stored under
correct scope key, stale result never overwrites newer cache.

### Commit

One commit, e.g. `fix(aggregator): drain-loop rebuild — coalesce stacked invalidations without dropping the last one`.

## Task 2: SSE version stream endpoint (server)

**Files:** new `app/stream.py`; register router in `app/main.py` (follow the
existing `from .monthly_budget import router as monthly_budget_router` /
`app.include_router(...)` pattern). **Tests:** new `app/tests/test_stream.py`.

### Endpoint spec

`GET /api/stats/stream`, `dependencies=[Depends(require_dashboard_auth)]`
(import from `.auth`, same as `app/monthly_budget.py` does).

Returns `StreamingResponse` (starlette), `media_type="text/event-stream"`,
headers: `Cache-Control: no-cache`, `X-Accel-Buffering: no`, `Connection: keep-alive`.

Async generator behavior:

1. Immediately emit the current version so a (re)connecting client can detect
   a missed update: `data: {"version": <int>}\n\n` (JSON payload).
2. Loop: `await asyncio.sleep(POLL_S)` with `POLL_S = 0.3`, read
   `get_cache_version()` (thread-safe, cheap); when it differs from the last
   emitted value, emit a new `data:` event. This deliberately avoids
   cross-thread asyncio signaling from the rebuild worker: a 300ms in-process
   check is free and cannot deadlock or leak loop references. Document this
   choice in the module docstring.
3. Emit a keepalive SSE comment `: keepalive\n\n` when nothing has been sent
   for `KEEPALIVE_S = 20` seconds (defeats proxy idle timeouts).
4. Exit cleanly on client disconnect: check `await request.is_disconnected()`
   each loop iteration OR rely on cancellation of the generator — either is
   acceptable, but there must be no unbounded resource growth per disconnected
   client and no traceback noise in server logs on normal disconnect
   (`asyncio.CancelledError` must not be swallowed into a log spew).

Constants `POLL_S` and `KEEPALIVE_S` at module top (no magic numbers inline).

### Test approach

Use FastAPI's `TestClient` with `client.stream("GET", "/api/stats/stream")` (or
equivalent) via the existing test support (`app/tests/_support.py` — read it
first and reuse its app/client fixtures and auth helpers). To keep tests
deterministic and fast, make the generator's poll/keepalive intervals injectable
(module constants patched in tests to small values). Tests:

1. Unauthenticated request → 401 (when auth is configured in the fixture).
2. First event arrives promptly and carries the current version as JSON.
3. After `invalidate_cache()` / version bump, a new event with the new version
   arrives.
4. Keepalive comment appears when idle (with patched small `KEEPALIVE_S`).
5. Response headers correct (`text/event-stream`, no-cache).

Read events by iterating the streamed lines with a hard cap / timeout so a
regression can't hang the suite.

### Commit

`feat(server): SSE version stream at /api/stats/stream`.

## Task 3: Dashboard EventSource live refresh with poll fallback

**File:** `templates/dashboard.html`, the live-refresh IIFE (search for
`const POLL_INTERVAL = 30000;`, currently ~line 6145–6222).
**Tests:** extend `app/tests/test_dashboard_template.py` (the established
pattern there is source-grep assertions against the rendered template — read a
few existing tests first and match their style).

### Required behavior

Keep everything that exists (`fetchAndApply`, `checkForUpdate`, `startPolling`,
`stopPolling`, `showRefreshIndicator`, the `isFetching` coalescing guard, scope
pinning). Add an SSE layer on top:

1. `startStream()`: create `new EventSource('/api/stats/stream')` (same-origin;
   browser attaches Basic-auth credentials automatically). Guard with
   `if (typeof EventSource === 'undefined') { startPolling(); return; }`.
2. `onmessage`: parse `JSON.parse(e.data).version`; if it differs from
   `knownVersion`, call `fetchAndApply()` (which already sets `knownVersion`
   from the payload and coalesces via `isFetching`). Malformed payloads are
   ignored silently (a keepalive comment never reaches onmessage, but be
   defensive with try/catch around the parse).
3. `onopen`: SSE is live → `stopPolling()` (no double-driving).
4. `onerror`: close the EventSource, `startPolling()` immediately (fallback),
   and schedule a single reconnect attempt via `setTimeout(startStream, STREAM_RETRY_MS)`
   with `STREAM_RETRY_MS = 60000`. Ensure repeated errors don't stack multiple
   timers or leak EventSource instances (track the instance + timer in closure
   vars; `stopStream()` clears both).
5. `visibilitychange`: hidden → `stopStream()` + `stopPolling()`;
   visible → `checkForUpdate()` (immediate catch-up) + `startStream()`.
6. Initial load: `startStream()` instead of `startPolling()` (polling starts
   only as fallback).

Comment the WHY at the layer boundary: SSE is the primary update signal, the
version poll is the fallback, and `fetchAndApply` remains the single render
path.

### Test approach (TDD)

Source-grep tests in `test_dashboard_template.py`: rendered template contains
`new EventSource('/api/stats/stream')`, an `EventSource` capability guard, a
`stopStream` that closes and clears the retry timer, `visibilitychange` wiring
that calls `stopStream`, and that `startPolling` still exists (fallback not
removed). RED first (assert before editing template), then implement.

**Do NOT run browsers or Playwright yourself** — the controller does visual
verification after your task.

### Commit

`feat(dashboard): EventSource live refresh, version poll demoted to fallback`.

## Task 4: Client watcher daemon (--watch) + single-instance lock

**File:** `client/claude-stats-push.py` (currently 638 lines; keep the growth
modest and factored). New launchd template `client/com.jaedynchilton.tokenfold-watch.plist`.
Installer wiring in `install.sh` (read it first; add an opt-in flag, keep default
behavior identical). **Tests:** new `client/test_watch.py`.

### Part A — single-instance lock (fixes an observed prod bug)

Concurrent hook-fired pushes race on the cursor file: observed repeated
`FileNotFoundError: ~/.tokenfold-cursor.json.tmp -> ~/.tokenfold-cursor.json`
in `~/.tokenfold-push.log` (two processes both write the same tmp, first
`os.replace` wins, second raises).

Add an exclusive advisory lock around the whole push cycle: `fcntl.flock` (import
`fcntl` lazily/guarded so the module still imports on Windows, which is not a
target) on `~/.tokenfold-push.lock`, `LOCK_EX | LOCK_NB`. If the lock is held,
log one line and exit 0 — the holder will pick up the new events (server dedups
by UUID regardless). The lock must be held for the entire read→push→save-cursors
cycle in both one-shot and watch modes.

### Part B — watch mode

`--watch` argv flag (simple `sys.argv` check or minimal argparse — match the
file's existing style, it currently has no arg parsing). Behavior:

- Acquire the flock ONCE for the process lifetime (a second `--watch` instance
  must refuse to start: log + exit 0). Hook-fired one-shot pushes while the
  daemon runs simply skip (lock held) — hooks need no config change.
- Refactor `main()`'s scan-and-push body into a callable `run_push_cycle(cursors)`
  (or equivalent) used by both one-shot and watch modes. Pure refactor for the
  one-shot path: identical behavior, existing log lines preserved.
- Loop, every `HOT_POLL_S = 1.0` seconds: stat only the **hot set** — session
  files whose mtime is within `HOT_WINDOW_S = 2 * 3600` of now (tracked from
  the last full scan). If any hot file's `(size, mtime)` signature changed →
  run a push cycle.
- Every `RESCAN_S = 60` seconds: full `find_session_files()` glob rescan to
  discover brand-new session files and refresh the hot set; also run a push
  cycle if the rescan finds anything new.
- Call `_fetch_and_push_usage()` on every push cycle AND at least once per
  `RESCAN_S` tick even when idle — its internal 300s stamp gate keeps the
  external cadence unchanged; this preserves today's meter freshness during
  idle periods.
- Efficiency: an idle tick must be O(hot set) stats, no file opens, no globbing.
  Exceptions inside the loop are caught + logged with context and the loop
  continues (a transient error must not kill the daemon); KeyboardInterrupt
  exits cleanly.
- Cursor safety: cursors are loaded once per push cycle from disk (not held
  stale across cycles) OR held in memory and saved each cycle — pick ONE,
  document why, and make sure a crashed cycle can't wind the cursor backwards
  past data the server already has (server dedups, so replays are safe;
  skipping ahead without pushing is NOT safe).

### Part C — launchd template + installer

`client/com.jaedynchilton.tokenfold-watch.plist`: Label
`com.jaedynchilton.tokenfold-watch`, ProgramArguments
`[/usr/bin/python3, ~PLACEHOLDER~/tokenfold-push.py, --watch]`, `KeepAlive`
true, `RunAtLoad` true, stdout/err to `~/.tokenfold-push.log`. Use an obvious
placeholder token the installer substitutes (mirror however `install.sh`
handles paths today — read it first).

`install.sh`: add an opt-in `--watch` mode that installs the plist into
`~/Library/LaunchAgents/` (path-substituted) and `launchctl bootstrap`s it
(with `bootout` first if already loaded, tolerant of "not loaded"). Non-macOS:
print a clear message that watch-mode install is macOS-only for now and exit
nonzero. Default (no flag) installer behavior must be byte-identical in effect
to today.

### Test approach (TDD)

`client/test_watch.py`, stdlib unittest, no network (stub `push_batch` /
`urlopen`-level functions):

1. flock exclusivity: first acquire wins; a second process/attempt logs and
   exits 0 (test via two `fcntl.flock` attempts on the same file, or by
   invoking the acquire helper twice with separate fds).
2. Hot-set selection: files with mtime older than `HOT_WINDOW_S` are excluded;
   newer included.
3. Change detection: signature change on a hot file triggers exactly one push
   cycle; unchanged tick triggers none.
4. Rescan discovers a newly created session file within one `RESCAN_S` tick
   (patch the interval small).
5. One-shot refactor regression: `run_push_cycle` invoked via the classic path
   produces the same cursor-file writes as before (fixture JSONL → cursors
   advance; reuse patterns from existing client tests if any cover this).
6. Usage-fetch gating untouched: watch loop calls `_fetch_and_push_usage` and
   the 300s stamp logic is what throttles (assert call happens; do not test
   Anthropic itself).

For `install.sh` changes, extend `app/tests/test_install_sh.py` following its
existing pattern (it greps/executes the script — read it first).

### Commit(s)

Up to three commits: `fix(client): single-instance flock around push cycle`,
`feat(client): --watch resident daemon (1s hot-set poll, 60s rescan)`,
`feat(install): opt-in launchd watch-mode install`.
