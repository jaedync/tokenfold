// Boot: read the shell's data-*, start the shared feeds, mount the live bar,
// route between views (#runway | #ledger | #history | #lab) and keep the tab
// title as a status line. Every panel mounts in try/catch (panel.js).

import { createStore } from './store.js';
import * as api from './api.js';
import { createFeeds } from './feeds.js';
import { mountInto, mountPanel, feedStatus } from './panel.js';
import { createScroll } from './scroll.js';
import { createReserve } from './reserve.js';
import { bindingWindow, dataOf, nowFrom } from './select.js';
import { windowTag } from './palette.js';
import livebar from './panels/livebar.js';
import runway from './views/runway.js';
import ledger from './views/ledger.js';
import historyView from './views/history.js';
import lab from './views/lab.js';

const VIEWS = { runway, ledger, history: historyView, lab };
const body = document.body;
const query = new URLSearchParams(location.search);
const mock = query.get('mock') === '1';
const SKEW_TOLERANCE_S = 2;
// The other views' feeds are read once this long after the first view mounts.
const PREFETCH_DELAY_MS = 1500;

const store = createStore({
  scope: body.dataset.scope || 'personal',
  scopeLocked: body.dataset.scopeLocked === '1',
  owner: body.dataset.owner || '',
  view: null,
  skew: 0,
  mock,
});
const feeds = createFeeds({ store, api, mock });
const announcer = document.getElementById('announce');
const viewRoot = document.getElementById('view');

const ctx = {
  store, api, feeds, mock,
  now: () => nowFrom(store.get()),
  // Polite screen-reader message; clearing first makes a repeat message speak again.
  announce(text) {
    announcer.textContent = '';
    requestAnimationFrame(() => { announcer.textContent = text; });
  },
  navigate(id, focusPanel) {
    // pushState, not location.hash: the old entry keeps its scroll position
    // (saved first) and the new view opens at the top, with no hashchange.
    if (location.hash !== `#${id}`) {
      history.pushState(scroll.leave(), '', `#${id}`);
      show(id, true);
      scroll.restore();
    }
    if (focusPanel) {
      requestAnimationFrame(() => {
        const target = viewRoot.querySelector(`[data-panel="${focusPanel}"]`);
        if (!target) return;
        target.scrollIntoView({ block: 'start' });
        const heading = target.querySelector('h2');
        if (heading) { heading.tabIndex = -1; heading.focus({ preventScroll: true }); }
      });
    }
  },
};

// The live bar and the tab title need these on every view.
feeds.want('now', { endpoint: 'now', intervalMs: 5000 });
feeds.want('runway', { endpoint: 'runway', intervalMs: 30000, bump: true });

// Server clock: relative times ("in 12 h") use the server's now. In mock mode
// the fixtures' generated_at fixes the clock once, so the mock world holds still.
let skewFixed = false;
store.subscribe((s) => {
  const gen = s.now && s.now.status === 'ok' && s.now.data ? s.now.data.generated_at : null;
  if (typeof gen !== 'number' || (mock && skewFixed)) return;
  const skew = gen - Date.now() / 1000;
  if (Math.abs(skew - s.skew) > SKEW_TOLERANCE_S) store.set('skew', skew);
  skewFixed = true;
});

// ---------------------------------------------------------------- views
let active = null;

function mountView(view) {
  const releases = Object.entries(view.feeds || {}).map(([key, spec]) => feeds.want(key, spec));
  const root = view.layout === 'grid' || !view.layout ? 'grid' : view.layout;
  const container = document.createElement('div');
  container.className = root;
  container.dataset.view = view.id;
  viewRoot.replaceChildren(container);
  const panels = (view.panels || []).map((p) => mountPanel(p, container, ctx));
  reserve.apply(view.id, store.get().scope, container);
  // During a scope switch the panels keep the old content (feeds.js reset).
  const update = (s) => { if (s.switching) return; for (const p of panels) p.update(s); };
  const unsubscribe = store.subscribe(update);
  update(store.get());
  return {
    id: view.id,
    container,
    destroy() {
      unsubscribe();
      for (const release of releases) release();
      for (const p of panels) p.destroy();
    },
  };
}

function show(id, userAction) {
  const view = Object.hasOwn(VIEWS, id) ? VIEWS[id] : null;
  if (!view || (active && active.id === id)) return;
  if (active) active.destroy();
  active = mountView(view);
  store.set('view', id);
  for (const a of document.querySelectorAll('[data-view-link]')) {
    if (a.dataset.viewLink === id) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  }
  if (userAction) viewRoot.focus({ preventScroll: true });
}

// Own keys only: a hash like #constructor names no view.
const hashView = () => (Object.hasOwn(VIEWS, location.hash.slice(1)) ? location.hash.slice(1) : null);
// Back, forward or a hash typed in the address bar: the entry's own scroll
// position. The event comes after the entry changed, but the window still
// shows the old view, so its position is kept first.
// An entry with no view hash (the page as opened, or the skip link's #view)
// shows the default view, as at boot.
window.addEventListener('hashchange', () => {
  if (location.hash && !hashView() && document.getElementById(location.hash.slice(1))) return;
  const id = hashView() || body.dataset.view || 'runway';
  scroll.remember();
  show(id, true);
  scroll.restore();
});
// A plain click on a view tab goes through navigate(); a click with a
// modifier (new tab, new window) stays a normal link.
for (const a of document.querySelectorAll('[data-view-link]')) {
  a.addEventListener('click', (ev) => {
    if (ev.button !== 0 || ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.altKey) return;
    ev.preventDefault();
    ctx.navigate(a.dataset.viewLink);
  });
}

// ---------------------------------------------------------------- scroll
// The visible view is ready for a scroll restore when each of its feeds
// answered (data or error). A feed with no read in this state (no params,
// e.g. the limit rate in the work scope) does not count.
function viewReady() {
  const s = store.get();
  if (!active || s.switching) return false;
  const view = VIEWS[active.id];
  return Object.entries(view.feeds || {}).every(([key, spec]) => {
    if (spec.params && spec.params(s) === null) return true;
    return feedStatus(s[key]) !== 'loading';
  });
}
const scroll = createScroll({ store, root: viewRoot, isReady: viewReady });
const reserve = createReserve({
  store, root: viewRoot, isReady: viewReady,
  where: () => (active ? { view: active.id, scope: store.get().scope, container: active.container } : null),
});

// ---------------------------------------------------------------- scope
function setScope(scope) {
  if (scope === store.get().scope || store.get().scopeLocked) return;
  for (const b of document.querySelectorAll('button[data-scope]')) b.setAttribute('aria-pressed', String(b.dataset.scope === scope));
  // Same cookie as the classic dashboard, so / and /next open on the same scope.
  document.cookie = `tf_scope=${scope};path=/;max-age=31536000;SameSite=Lax`;
  try { localStorage.setItem('tf_scope', scope); } catch { /* private mode: the cookie is enough */ }
  const next = new URLSearchParams(location.search);
  next.set('scope', scope);
  history.replaceState(history.state, '', `${location.pathname}?${next}${location.hash}`);
  store.set('scope', scope);
  // A panel with no data for the new scope reserves that scope's height.
  if (active) reserve.apply(active.id, scope, active.container);
  skewFixed = false;
  feeds.reset();
}
// aria-busy, and the bar in tokenfold.css, while the new scope loads.
let wasSwitching = false;
store.subscribe((s) => {
  const switching = Boolean(s.switching);
  if (switching === wasSwitching) return;
  wasSwitching = switching;
  if (switching) viewRoot.setAttribute('aria-busy', 'true'); else viewRoot.removeAttribute('aria-busy');
  if (!switching) prefetchOthers();
});
for (const b of document.querySelectorAll('button[data-scope]')) b.addEventListener('click', () => setScope(b.dataset.scope));

// ---------------------------------------------------------------- idle prefetch
// Read every other view's feeds once while the page is idle, so a view
// switch shows content at once. Again after each scope switch.
let prefetchTimer = null;
function prefetchOthers() {
  clearTimeout(prefetchTimer);
  prefetchTimer = setTimeout(() => {
    const idle = window.requestIdleCallback || ((fn) => setTimeout(fn, 0));
    idle(() => {
      if (document.hidden) return;
      for (const view of Object.values(VIEWS)) {
        if (active && view.id === active.id) continue;
        for (const [key, spec] of Object.entries(view.feeds || {})) feeds.prefetch(key, spec);
      }
    }, { timeout: 3000 });
  }, PREFETCH_DELAY_MS);
}

// ---------------------------------------------------------------- tab title
const baseTitle = document.title;
store.subscribe((s) => {
  const w = bindingWindow(dataOf(s.runway));
  const live = dataOf(s.now) && s.now.data.live;
  const parts = [];
  if (w && typeof w.pct === 'number') parts.push(`${Math.floor(w.pct)}% ${windowTag(w)}`);
  // Always say the count once the live feed answers, so the title keeps one shape.
  if (live && typeof live.waiting === 'number') parts.push(`${live.waiting} waiting`);
  parts.push(baseTitle);
  const title = parts.join(' · ');
  if (document.title !== title) document.title = title;
});

// ---------------------------------------------------------------- SSE version bump
let stream = null;
let streamVersion = null;
function openStream() {
  if (mock || stream || typeof EventSource !== 'function') return;
  stream = new EventSource('/api/stats/stream');
  stream.onmessage = (ev) => {
    let version = null;
    try { version = JSON.parse(ev.data).version; } catch { return; }
    // The first frame after (re)connect is the current version, not a change.
    if (streamVersion !== null && version !== streamVersion) feeds.bump();
    streamVersion = version;
  };
}
function closeStream() {
  if (stream) stream.close();
  stream = null;
}
document.addEventListener('visibilitychange', () => (document.hidden ? closeStream() : openStream()));

// ---------------------------------------------------------------- boot
const bar = mountInto(livebar, document.getElementById('livebar'), ctx);
store.subscribe((s) => { if (!s.switching) bar.update(s); });
bar.update(store.get());
show(hashView() || body.dataset.view || 'runway', false);
scroll.restore();
openStream();
prefetchOthers();
