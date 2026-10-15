// Data feeds: one per endpoint, shared by every panel that reads it.
// A view says which feeds it wants and how often; a feed fetches while the
// tab is visible, backs off after failures and writes its result into the
// store as {status, data, error, at}. status: 'loading' | 'ok' | 'error'.
// Polling pauses in a hidden tab and catches up when the tab comes back.
//
// A scope switch keeps the old content on screen: the store says
// `switching: true` (main.js holds the panels and marks the view busy) until
// every feed in use has an answer for the new scope, or SWITCH_HOLD_MS ends.
// Data of the old scope never shows as the new scope's data.

import { backoffMs } from './api.js';

const BUMP_MIN_GAP_MS = 10000;  // SSE bumps arrive per ingest; one refetch per 10 s is enough
// The server answers at once with a body built before the newest ingest
// while it builds the next one (X-Tokenfold-Stale). Read it again this soon,
// once per bump, instead of at the next poll.
const STALE_RETRY_MS = 4000;
// A remounted feed with no poll interval reuses data younger than this.
const FRESH_MS = 30000;
// Longest time a scope switch keeps the old content before it shows "Loading".
const SWITCH_HOLD_MS = 5000;

export const LOADING = Object.freeze({ status: 'loading', data: null, error: null, at: null });

export function createFeeds({ store, api, mock, env = window, switchHoldMs = SWITCH_HOLD_MS,
  staleRetryMs = STALE_RETRY_MS, bumpGapMs = BUMP_MIN_GAP_MS }) {
  const feeds = new Map();
  const doc = env.document;
  let gen = 0;          // scope generation: reset() starts a new one
  let holdTimer = null;

  const visible = () => !doc.hidden;

  function feedFor(key, spec) {
    let f = feeds.get(key);
    if (!f) {
      // gen: the scope generation of the data in the store. doneGen/doneKey:
      // the generation and params of the last answer (data or error).
      // staleRead: a stale answer was read again already (cleared by a bump or a fresh answer).
      f = { key, spec, users: 0, timer: null, ctrl: null, failures: 0, last: 0, paramsKey: null, pending: false,
        once: false, gen: -1, doneGen: -1, doneKey: null, staleRead: false, due: 0 };
      feeds.set(key, f);
    }
    return f;
  }

  function params(f) {
    return f.spec.params ? f.spec.params(store.get()) : {};
  }

  function schedule(f, delayMs) {
    clearTimeout(f.timer);
    f.due = Date.now() + Math.max(0, delayMs);
    f.timer = setTimeout(() => run(f), Math.max(0, delayMs));
  }

  // Schedule a read, but keep a timer that fires sooner (a poll or a stale
  // read-again is not pushed back by a bump).
  function scheduleBy(f, delayMs) {
    if (f.timer && f.due <= Date.now() + delayMs) return;
    schedule(f, delayMs);
  }

  async function run(f) {
    f.timer = null;
    if (!f.users && !f.once) return;
    if (!visible()) { f.pending = true; return; }
    const p = params(f);
    // Not ready: a dependency (e.g. the binding window) is unknown. No read
    // has these params now, so the store watch below reads once they come
    // (an aborted read could have left the same params key).
    if (p === null) { f.once = false; f.paramsKey = null; return; }
    if (f.ctrl) f.ctrl.abort();
    const ctrl = new AbortController();
    f.ctrl = ctrl;
    f.pending = false;
    f.last = Date.now();
    f.paramsKey = JSON.stringify(p);
    const g = gen;
    // Only a feed that never answered says "Loading": a refetch keeps what is on screen.
    if (!store.get()[f.key]) store.set(f.key, LOADING);
    const meta = { stale: false };
    try {
      const data = await api.getJSON(f.spec.endpoint, p, { scope: store.get().scope, mock, signal: ctrl.signal, meta });
      if (ctrl.signal.aborted) return;
      f.failures = 0;
      f.gen = g;
      done(f, g);
      store.set(f.key, { status: 'ok', data, error: null, at: Date.now() / 1000 });
      if (f.spec.intervalMs && f.users) schedule(f, f.spec.intervalMs);
      if (!meta.stale) f.staleRead = false;
      else if (!f.staleRead && f.users) { f.staleRead = true; schedule(f, staleRetryMs); }
    } catch (err) {
      if (ctrl.signal.aborted) return;
      f.failures += 1;
      done(f, g);
      // Keep the last good data so a blip greys the panel out instead of
      // blanking it, but never data of another scope.
      const old = f.gen === gen ? store.get()[f.key] : null;
      store.set(f.key, { status: 'error', data: old ? old.data : null, error: { status: err.status ?? 0, message: err.message }, at: old ? old.at : null });
      if (f.users) schedule(f, Math.max(f.spec.intervalMs || 0, backoffMs(f.failures, err.status)));
    } finally {
      if (f.ctrl === ctrl) { f.ctrl = null; f.once = false; }
      settle();
    }
  }

  function done(f, g) {
    f.doneGen = g;
    f.doneKey = f.paramsKey;
  }

  // ms until a feed's data needs a refetch: 0 for none, old, another scope's
  // or other params.
  function freshFor(f) {
    const cur = store.get()[f.key];
    if (!cur || cur.status !== 'ok' || f.gen !== gen || !f.last) return 0;
    const p = params(f);
    if (p === null || JSON.stringify(p) !== f.paramsKey) return 0;
    return Math.max(0, f.last + (f.spec.intervalMs || FRESH_MS) - Date.now());
  }

  // End a scope switch once every feed in use answered for the new scope
  // with its current params.
  function settle() {
    if (!store.get().switching) return;
    for (const f of feeds.values()) {
      if (!f.users) continue;
      const p = params(f);
      if (p === null) continue;
      if (f.doneGen !== gen || f.doneKey !== JSON.stringify(p)) return;
    }
    finishSwitch();
  }

  // Data of an older scope generation leaves the store: a feed with no
  // users keeps nothing on screen, so its old data could only show later,
  // under the new scope.
  function dropOld(f) {
    const cur = store.get()[f.key];
    if (f.gen !== gen && cur && cur.data !== null) store.set(f.key, LOADING);
  }

  // After the hold, a feed still without new-scope data says "Loading". Every
  // feed of an older generation drops its data, with or without users.
  function finishSwitch() {
    clearTimeout(holdTimer);
    holdTimer = null;
    if (!store.get().switching) return;
    for (const f of feeds.values()) {
      if (f.users && f.doneGen !== gen) store.set(f.key, LOADING);
      else dropOld(f);
    }
    store.set('switching', false);
  }

  // want('now', {endpoint, intervalMs, params(state), bump}) -> release()
  function want(key, spec) {
    const f = feedFor(key, spec);
    f.spec = spec;
    f.users += 1;
    if (!store.get().switching) dropOld(f);
    // A read in flight (a prefetch) carries on and starts the polling.
    if (f.users === 1 && !f.ctrl) schedule(f, freshFor(f));
    return () => {
      f.users -= 1;
      if (f.users > 0) return;
      clearTimeout(f.timer);
      if (f.ctrl) f.ctrl.abort();
      f.ctrl = null;
      f.timer = null;
      dropOld(f);
      settle();
    };
  }

  // One read of a feed no view uses yet (idle prefetch), unless its data is
  // fresh. It does not poll; a later want() reuses the data or the read.
  function prefetch(key, spec) {
    const f = feedFor(key, spec);
    if (f.users || f.ctrl) return;
    f.spec = spec;
    if (freshFor(f) > 0) return;
    f.once = true;
    schedule(f, 0);
  }

  function refresh(key) {
    const f = feeds.get(key);
    if (f && f.users) schedule(f, 0);
  }

  // The SSE version bump: refetch feeds that follow it, at most once per gap.
  function bump() {
    for (const f of feeds.values()) {
      if (!f.users || !f.spec.bump) continue;
      f.staleRead = false;
      scheduleBy(f, Math.max(0, f.last + bumpGapMs - Date.now()));
    }
  }

  // Scope change: fetch every feed in use again for the new scope and keep
  // its old data on screen meanwhile (switching). Feeds not in use, and feeds
  // with no read in the new scope, drop their data.
  function reset() {
    gen += 1;
    let wait = false;
    for (const f of feeds.values()) {
      if (f.ctrl) f.ctrl.abort();
      f.ctrl = null;
      f.once = false;
      f.failures = 0;
      f.paramsKey = null;
      const inUse = f.users > 0 && params(f) !== null;
      if (!inUse) store.set(f.key, LOADING);
      if (f.users) schedule(f, 0);
      wait = wait || inUse;
    }
    clearTimeout(holdTimer);
    holdTimer = null;
    if (!wait) { store.set('switching', false); return; }
    store.set('switching', true);
    holdTimer = setTimeout(finishSwitch, switchHoldMs);
  }

  // A feed whose params depend on another feed refetches when they change.
  store.subscribe(() => {
    for (const f of feeds.values()) {
      if (!f.users || !f.spec.params || f.ctrl) continue;
      const p = params(f);
      if (p !== null && JSON.stringify(p) !== f.paramsKey) schedule(f, 0);
    }
  });

  doc.addEventListener('visibilitychange', () => {
    if (!visible()) return;
    for (const f of feeds.values()) {
      if (!f.users) continue;
      const overdue = f.spec.intervalMs && Date.now() - f.last >= f.spec.intervalMs;
      if (f.pending || overdue) schedule(f, 0);
    }
  });

  return { want, refresh, bump, reset, prefetch };
}
