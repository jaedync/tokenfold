// Data feeds: one per endpoint, shared by every panel that reads it.
// A view says which feeds it wants and how often; a feed fetches while the
// tab is visible, backs off after failures and writes its result into the
// store as {status, data, error, at}. status: 'loading' | 'ok' | 'error'.
// Polling pauses in a hidden tab and catches up when the tab comes back.

import { backoffMs } from './api.js';

const BUMP_MIN_GAP_MS = 10000;  // SSE bumps arrive per ingest; one refetch per 10 s is enough

export const LOADING = Object.freeze({ status: 'loading', data: null, error: null, at: null });

export function createFeeds({ store, api, mock, env = window }) {
  const feeds = new Map();
  const doc = env.document;

  const visible = () => !doc.hidden;

  function feedFor(key, spec) {
    let f = feeds.get(key);
    if (!f) {
      f = { key, spec, users: 0, timer: null, ctrl: null, failures: 0, last: 0, paramsKey: null, pending: false };
      feeds.set(key, f);
    }
    return f;
  }

  function params(f) {
    return f.spec.params ? f.spec.params(store.get()) : {};
  }

  function schedule(f, delayMs) {
    clearTimeout(f.timer);
    f.timer = setTimeout(() => run(f), Math.max(0, delayMs));
  }

  async function run(f) {
    f.timer = null;
    if (!f.users) return;
    if (!visible()) { f.pending = true; return; }
    const p = params(f);
    if (p === null) return;  // not ready: a dependency (e.g. the binding window) is unknown
    if (f.ctrl) f.ctrl.abort();
    const ctrl = new AbortController();
    f.ctrl = ctrl;
    f.pending = false;
    f.last = Date.now();
    f.paramsKey = JSON.stringify(p);
    const prev = store.get()[f.key];
    if (!prev || prev.status !== 'ok') store.set(f.key, LOADING);
    try {
      const data = await api.getJSON(f.spec.endpoint, p, { scope: store.get().scope, mock, signal: ctrl.signal });
      if (ctrl.signal.aborted) return;
      f.failures = 0;
      store.set(f.key, { status: 'ok', data, error: null, at: Date.now() / 1000 });
      if (f.spec.intervalMs) schedule(f, f.spec.intervalMs);
    } catch (err) {
      if (ctrl.signal.aborted) return;
      f.failures += 1;
      const old = store.get()[f.key];
      // Keep the last good data so a blip greys the panel out instead of blanking it.
      store.set(f.key, { status: 'error', data: old ? old.data : null, error: { status: err.status ?? 0, message: err.message }, at: old ? old.at : null });
      schedule(f, Math.max(f.spec.intervalMs || 0, backoffMs(f.failures, err.status)));
    } finally {
      if (f.ctrl === ctrl) f.ctrl = null;
    }
  }

  // want('now', {endpoint, intervalMs, params(state), bump}) -> release()
  function want(key, spec) {
    const f = feedFor(key, spec);
    f.spec = spec;
    f.users += 1;
    if (f.users === 1) schedule(f, 0);
    return () => {
      f.users -= 1;
      if (f.users > 0) return;
      clearTimeout(f.timer);
      if (f.ctrl) f.ctrl.abort();
      f.timer = null;
    };
  }

  function refresh(key) {
    const f = feeds.get(key);
    if (f && f.users) schedule(f, 0);
  }

  // The SSE version bump: refetch feeds that follow it, at most once per gap.
  function bump() {
    for (const f of feeds.values()) {
      if (!f.users || !f.spec.bump) continue;
      schedule(f, Math.max(0, f.last + BUMP_MIN_GAP_MS - Date.now()));
    }
  }

  // Scope change: drop every result and fetch again for the new scope.
  function reset() {
    for (const f of feeds.values()) {
      if (f.ctrl) f.ctrl.abort();
      f.failures = 0;
      f.paramsKey = null;
      store.set(f.key, LOADING);
      if (f.users) schedule(f, 0);
    }
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

  return { want, refresh, bump, reset };
}
