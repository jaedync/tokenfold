// Shared Ledger helpers: UI state actions, colours per payload, SVG and
// tooltip builders. Text always goes in through textContent (h(), svg()).

import { h, listen, patch } from '../dom.js';
import * as f from '../format.js';
import { feedStatus, loadingLine } from '../panel.js';
import { DEFAULT_UI, uiOf, hasFilter } from './ledger-model.js';
import { modelColors } from './ledger-colors.js';

// ---------------------------------------------------------------- UI state
// The Ledger's shared UI state is one frozen store key, `ledgerUi`.
export function setUi(store, patch) {
  store.set('ledgerUi', { ...uiOf(store.get()), ...patch });
}

export function setPeriod(store, pkey, from = null, to = null) {
  setUi(store, { pkey, from, to });
}

export function toggleFilter(store, dim, val) {
  const { filters } = uiOf(store.get());
  const next = hasFilter(filters, dim, val)
    ? filters.filter((x) => !(x.dim === dim && x.val === val))
    : [...filters, { dim, val }];
  setUi(store, { filters: next });
}

export const clearFilters = (store) => setUi(store, { filters: DEFAULT_UI.filters });

// The open session lives in the store (`ledgerDrawer`); the element to give
// focus back to cannot (store values are frozen), so it stays here.
let returnFocus = null;
export function openSession(store, id, from) {
  if (from) returnFocus = { el: from, sid: id };
  store.set('ledgerDrawer', id);
}
export function closeSession(store) {
  store.set('ledgerDrawer', null);
  const back = returnFocus;
  returnFocus = null;
  if (!back) return;
  let el = back.el;
  if (!el || !document.contains(el)) el = document.querySelector(`[data-sid="${CSS.escape(back.sid)}"]`);
  if (el) el.focus({ preventScroll: false });
}

// ---------------------------------------------------------------- colours
const colorMemo = new WeakMap();
// Colours for a derive() result: every model in the 30 days and the period.
export function colorsFor(d) {
  let c = colorMemo.get(d);
  if (c) return c;
  const costs = {};
  for (const day of d.daily30 || []) {
    for (const [m, v] of Object.entries(day.by_model || {})) costs[m] = (costs[m] || 0) + v;
  }
  // Models without daily spend (local, $0) still need a colour.
  const extra = {};
  for (const list of [d.sessions30, d.sessions]) {
    for (const s of list) for (const p of s.parts) if (!(p.model in costs)) extra[p.model] = (extra[p.model] || 0) + p.cost;
  }
  c = modelColors({ ...costs, ...extra });
  colorMemo.set(d, c);
  return c;
}

export const swatch = (color, extra = {}) => h('i', { class: 'swatch', style: { background: color }, 'aria-hidden': 'true', ...extra });

// ---------------------------------------------------------------- SVG
const NS = 'http://www.w3.org/2000/svg';
export function svg(tag, attrs, ...children) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'text') el.textContent = String(v);
    // listen(), as h() does: patch() hands a kept node the newest handler.
    else if (k.startsWith('on') && typeof v === 'function') listen(el, k.slice(2), v);
    else el.setAttribute(k, typeof v === 'number' ? String(Math.round(v * 10) / 10) : String(v));
  }
  for (const c of children.flat()) if (c) el.append(c);
  return el;
}

// Sparkline: area + line + end dot. max: fixed top (1 for shares) or auto.
export function spark(vals, w, hgt, color, max) {
  const n = vals.length;
  const top = max || Math.max(1e-9, ...vals);
  const x = (i) => (n === 1 ? w / 2 : 1 + (i / (n - 1)) * (w - 4));
  const y = (v) => hgt - 2 - (v / top) * (hgt - 5);
  const pts = vals.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  return svg('svg', { class: 'spark', width: w, height: hgt, viewBox: `0 0 ${w} ${hgt}`, 'aria-hidden': 'true' },
    svg('polygon', { points: `${x(0)},${hgt - 1} ${pts} ${x(n - 1)},${hgt - 1}`, fill: color, opacity: 0.2 }),
    svg('polyline', { points: pts, fill: 'none', stroke: color, 'stroke-width': 2, 'stroke-linejoin': 'round' }),
    svg('line', { x1: 0, x2: w, y1: hgt - 1, y2: hgt - 1, stroke: '#161616', 'stroke-width': 1.5 }),
    n ? svg('circle', { cx: x(n - 1), cy: y(vals[n - 1]), r: 2.6, fill: '#161616' }) : null);
}

// ---------------------------------------------------------------- tooltip
// One floating tooltip per panel; mouse only (touch opens the drawer instead).
export function tooltip() {
  const el = h('div', { class: 'lg-tip', hidden: true, role: 'presentation' });
  document.body.append(el);
  return {
    show(ev, ...children) {
      patch(el, ...children);
      el.hidden = false;
      const r = el.getBoundingClientRect();
      let x = ev.clientX + 14, y = ev.clientY + 14;
      if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - 14;
      if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - 14;
      el.style.left = `${Math.max(8, x)}px`;
      el.style.top = `${Math.max(8, y)}px`;
    },
    hide() { el.hidden = true; },
    destroy() { el.remove(); },
  };
}

export const tipRow = (left, right, color) => h('div', { class: 'tt-row' },
  h('span', null, color ? h('i', { style: { background: color } }) : null, left), h('b', null, right));

// ---------------------------------------------------------------- text
export const plural = (n, one, many = `${one}s`) => `${f.int(n)} ${n === 1 ? one : many}`;
export const money = (v) => (v > 0 && v < 0.01 ? '<$0.01' : f.usd(v));
export const pct0 = (v) => (v >= 0.995 ? '100%' : v > 0 && v < 0.005 ? '<1%' : `${Math.round(v * 100)}%`);
export function tokens(n) {
  if (f.missing(n)) return f.DASH;
  if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e8 ? 0 : 1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(n >= 1e5 ? 0 : 1)}K`;
  return String(Math.round(n));
}
const dowDate = new Intl.DateTimeFormat('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
export const dayLabel = (e) => dowDate.format(new Date(e * 1000));

// Re-focus the same session after a list is rebuilt.
export function keepFocus(root, rebuild) {
  const a = document.activeElement;
  const sid = a && root.contains(a) && a.dataset ? a.dataset.sid : null;
  rebuild();
  if (!sid) return;
  const again = root.querySelector(`[data-sid="${CSS.escape(sid)}"]`);
  if (again) again.focus({ preventScroll: true });
}

// A panel with no ledger data yet: one loading line while the 30-day read
// runs (the panel keeps its reserved height), nothing after an error (the
// summary panel says the ledger is unavailable).
export function ledgerWait(state) {
  return feedStatus(state.ledger) === 'loading' ? loadingLine('Loading…') : null;
}
// The feed status, for a render key: without data derive() stays the same
// object while the feed moves from loading to error.
export const ledgerStatus = (state) => (state.ledger ? state.ledger.status : null);
