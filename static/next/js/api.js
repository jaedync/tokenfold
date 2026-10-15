// Fetch helpers for the /next endpoints (docs/ux-2026-10/BUILD.md).
// Same origin, Basic auth credentials ride along, JSON only, abortable.
// With ?mock=1 every call reads static/next/fixtures/*.json instead.

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

// name -> real path and fixture file. `work`: a separate fixture exists for
// the work (enterprise) scope, named <fixture>-work.json. `timeoutMs`: a
// longer wait than TIMEOUT_MS for a read that can be slow when cold.
const TIMEOUT_MS = 20000;
const SLOW_TIMEOUT_MS = 90000;

export const ENDPOINTS = Object.freeze({
  now: { path: '/api/now', fixture: 'now', work: true },
  runway: { path: '/api/runway', fixture: 'runway', work: true },
  trajectory: { path: '/api/runway/trajectory', fixture: 'runway-trajectory', work: true },
  today: { path: '/api/today', fixture: 'today', work: true },
  limitRate: { path: '/api/limit-rate', fixture: 'limit-rate', work: true },
  ledger: { path: '/api/ledger', fixture: 'ledger' },
  session: { path: (p) => `/api/sessions/${encodeURIComponent(p.id)}`, fixture: 'session' },
  // Cold after a restart until the prewarm has run: one week of events per
  // ended window. Wait for it instead of failing at TIMEOUT_MS.
  windowHistory: { path: '/api/window-history', fixture: 'window-history', timeoutMs: SLOW_TIMEOUT_MS },
  // History and Lab (existing endpoints of the classic dashboard).
  windowHistory5h: { path: '/api/window-history', fixture: 'window-history-5h', timeoutMs: SLOW_TIMEOUT_MS },
  spendHistory: { path: '/api/spend-history', fixture: 'spend-history', work: true },
  stats: { path: '/api/stats', fixture: 'stats', work: true },
  servedTimeline: { path: '/api/served-models/timeline', fixture: 'served-timeline' },
  enterpriseBudget: { path: '/api/enterprise-budget', fixture: 'enterprise-budget' },
});

const FIXTURES = new URL('../fixtures/', import.meta.url);

// How long a request to this endpoint may take before it fails.
export function timeoutFor(name) {
  return (ENDPOINTS[name] && ENDPOINTS[name].timeoutMs) || TIMEOUT_MS;
}

function url(name, params, { mock, scope }) {
  const ep = ENDPOINTS[name];
  if (!ep) throw new ApiError(0, `unknown endpoint ${name}`);
  if (mock) {
    const file = ep.fixture + (ep.work && scope === 'enterprise' ? '-work' : '');
    return new URL(`${file}.json`, FIXTURES).href;
  }
  const path = typeof ep.path === 'function' ? ep.path(params) : ep.path;
  const query = { scope, ...params };
  if (typeof ep.path === 'function') delete query.id;
  const qs = new URLSearchParams(
    Object.entries(query).filter(([, v]) => v !== null && v !== undefined && v !== ''));
  return path + (qs.toString() ? `?${qs}` : '');
}

// Early reads (js/early.js): the shell starts the first feeds while the rest
// of the module graph loads. getJSON takes each response once, if it is
// younger than EARLY_MAX_AGE_MS. After the first getJSON call no early read
// starts: the feeds run already.
const EARLY_MAX_AGE_MS = 10000;
const early = new Map();  // url -> {at, promise}
let booted = false;

export function startEarly(name, params, { scope, mock = false } = {}) {
  if (booted) return;
  const u = url(name, params, { mock, scope });
  if (early.has(u)) return;
  const promise = fetch(u, {
    headers: { Accept: 'application/json' }, credentials: 'same-origin', signal: AbortSignal.timeout(timeoutFor(name)),
  });
  promise.catch(() => {});  // a failed early read is read again by its feed
  early.set(u, { at: Date.now(), promise });
}

// Early reads that no feed took (the boot checks expect none).
export const pendingEarly = () => [...early.keys()];

function takeEarly(u) {
  const e = early.get(u);
  if (!e) return null;
  early.delete(u);
  return Date.now() - e.at <= EARLY_MAX_AGE_MS ? e.promise : null;
}

function untilAbort(promise, signal) {
  if (!signal) return promise;
  if (signal.aborted) return Promise.reject(new DOMException('Aborted', 'AbortError'));
  return Promise.race([promise, new Promise((_, reject) => {
    signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), { once: true });
  })]);
}

// getJSON('runway', {}, {scope, mock, signal}) -> parsed JSON or ApiError.
export async function getJSON(name, params = {}, { scope, mock = false, signal } = {}) {
  booted = true;
  const timeout = AbortSignal.timeout(timeoutFor(name));
  const combined = signal && AbortSignal.any ? AbortSignal.any([signal, timeout]) : (signal || timeout);
  const u = url(name, params, { mock, scope });
  let res = null;
  const pending = takeEarly(u);
  if (pending) {
    try {
      res = await untilAbort(pending, signal);
    } catch (err) {
      if (err.name === 'AbortError' && signal && signal.aborted) throw err;
      res = null;  // the early read failed: read again
    }
  }
  try {
    if (!res) {
      res = await fetch(u, {
        headers: { Accept: 'application/json' }, credentials: 'same-origin', signal: combined,
      });
    }
  } catch (err) {
    if (err.name === 'AbortError' && signal && signal.aborted) throw err;
    throw new ApiError(0, 'The server did not answer.');
  }
  const text = await res.text();
  if (!res.ok) {
    let detail = '';
    try { detail = (JSON.parse(text) || {}).detail || ''; } catch { /* not JSON: keep the status */ }
    throw new ApiError(res.status, typeof detail === 'string' && detail ? detail : res.statusText);
  }
  try {
    return JSON.parse(text);
  } catch {
    throw new ApiError(res.status, 'The server sent data that is not JSON.');
  }
}

// Retry delay after `failures` failures in a row: 2 s, 4 s, ... up to 2 min.
// A missing endpoint (404) waits longest: it will not appear in seconds.
export function backoffMs(failures, status) {
  const cap = status === 404 ? 300000 : 120000;
  return Math.min(cap, 1000 * 2 ** Math.max(1, failures));
}
