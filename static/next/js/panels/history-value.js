// Value: API-equivalent spend per month as a line. With a plan price saved in
// this browser (localStorage), it also shows the price line and how many
// times the price each month is worth. Personal scope only.
import { h, patch, setText, approx, panelHead } from '../dom.js';
import * as f from '../format.js';
import { dataOf, nowFrom } from '../select.js';
import { feedLine, staleNote, stateLine } from '../panel.js';
import { monthLabel, niceScale, changed } from './history-common.js';

const KEY = 'tf_plan_price_usd';
const MONTHS = 12;
const SVG = 'http://www.w3.org/2000/svg';
const H = 200, PAD = { l: 46, r: 14, t: 14, b: 24 };

function readPrice() {
  try {
    const v = Number(localStorage.getItem(KEY));
    return Number.isFinite(v) && v > 0 ? v : null;
  } catch { return null; }
}
function writePrice(v) {
  try {
    if (v) localStorage.setItem(KEY, String(v)); else localStorage.removeItem(KEY);
  } catch { /* private mode: the price lives until the page closes */ }
}

function svg(tag, attrs, ...kids) {
  const el = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v !== null && v !== undefined) el.setAttribute(k, String(v));
  for (const k of kids.flat()) if (k) el.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return el;
}

// Drawn at the real pixel width (redrawn on resize), so text never scales.
function lineChart(months, price, W) {
  const max = Math.max(price || 0, ...months.map((m) => m.total || 0));
  const { top, step } = niceScale(max * 1.05);
  const x = (i) => PAD.l + (months.length < 2 ? (W - PAD.l - PAD.r) / 2 : i * (W - PAD.l - PAD.r) / (months.length - 1));
  const y = (v) => PAD.t + (1 - v / top) * (H - PAD.t - PAD.b);
  const grid = [];
  for (let v = 0; v <= top + 1e-9; v += step) {
    grid.push(svg('line', { class: 'vg', x1: PAD.l, x2: W - PAD.r, y1: y(v), y2: y(v) }),
      svg('text', { class: 'vt', x: PAD.l - 6, y: y(v) + 4, 'text-anchor': 'end' }, f.usdShort(v)));
  }
  const pts = months.map((m, i) => `${x(i).toFixed(1)},${y(m.total || 0).toFixed(1)}`).join(' ');
  return svg('svg', { class: 'vchart', viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img',
    'aria-label': `API-equivalent per month: ${months.map((m) => `${monthLabel(m.month).long} ${f.usd(m.total, { cents: false })}`).join(', ')}` },
  grid,
  price ? svg('line', { class: 'vprice', x1: PAD.l, x2: W - PAD.r, y1: y(price), y2: y(price) }) : null,
  svg('polyline', { class: 'vline', points: pts }),
  months.map((m, i) => svg('circle', { class: m.partial ? 'vdot open' : 'vdot', cx: x(i), cy: y(m.total || 0), r: 4.5 },
    svg('title', null, `${monthLabel(m.month).long}${m.partial ? ' (to date)' : ''}: ${f.usd(m.total, { cents: false })}`))),
  months.map((m, i) => ((W < 520 || months.length > 8) && (months.length - 1 - i) % 2 ? null
    : svg('text', { class: 'vx', x: x(i), y: H - 6, 'text-anchor': 'middle' }, monthLabel(m.month).short))));
}

function headline(months, price) {
  const full = months.filter((m) => !m.partial);
  const last = full[full.length - 1] || months[months.length - 1];
  if (!last) return null;
  const label = monthLabel(last.month).long;
  return h('div', { class: 'v-head' },
    h('div', { class: 'v-big' }, approx(), f.usd(last.total, { cents: false })),
    h('p', { class: 'v-sub' }, `API-equivalent in ${label}${last.partial ? ' (to date)' : ''}.`,
      price ? h('b', null, ` ≈${(last.total / price).toFixed(1)}× your ${f.usd(price, { cents: false })} plan price.`) : null));
}

function ratios(months, price) {
  if (!price) return null;
  return h('ol', { class: 'v-ratio', 'aria-label': 'Value per month as times the plan price' }, months.slice(-6).map((m) => h('li', { 'data-key': m.month },
    h('span', { class: 'kicker' }, monthLabel(m.month).short),
    h('b', null, m.partial ? 'to date' : `${(m.total / price).toFixed(1)}×`))));
}

export default {
  id: 'history-value',
  label: 'Value',
  mount(el, ctx) {
    const head = panelHead('Value');
    el.setAttribute('aria-labelledby', head.id);
    const lead = h('p', { class: 'panel-sub hl-lead' }, 'What your use would cost at API list prices, per month. Save your plan price to compare the two. The price stays in this browser.');
    const state = h('div');
    const body = h('div', { class: 'v-body' });
    const input = h('input', { type: 'number', id: 'v-price', min: '1', max: '100000', step: '1', inputmode: 'decimal', placeholder: 'e.g. 200' });
    const msg = h('span', { class: 'v-msg', role: 'status' });
    const clear = h('button', { type: 'button', class: 'btn' }, 'Clear');
    const form = h('form', { class: 'v-form', novalidate: '' },
      h('label', { for: 'v-price' }, 'Plan price per month (USD)'),
      h('div', { class: 'v-row' }, input, h('button', { type: 'submit', class: 'btn' }, 'Save'), clear), msg);
    el.append(head.el, lead, state, body, form);

    let price = readPrice();
    let seen = null;
    let width = 0;
    const ro = new ResizeObserver(() => {
      const w = Math.floor(body.clientWidth);
      if (w > 0 && Math.abs(w - width) > 8) { seen = null; update(ctx.store.get()); }
    });
    ro.observe(body);
    if (price) input.value = String(price);
    form.addEventListener('submit', (ev) => {
      ev.preventDefault();
      const v = Number(input.value);
      if (!Number.isFinite(v) || v <= 0 || v > 100000) { setText(msg, 'Type a price from 1 to 100,000.'); return; }
      price = Math.round(v * 100) / 100;
      writePrice(price);
      setText(msg, 'Saved in this browser.');
      seen = null; update(ctx.store.get());
    });
    clear.addEventListener('click', () => {
      price = null; input.value = ''; writePrice(null);
      setText(msg, 'Price cleared.');
      seen = null; update(ctx.store.get());
    });

    function update(s) {
      const key = [s.spendHistory, s.scope, price, Math.floor(body.clientWidth)];
      if (!changed(seen, key)) return;
      seen = key;
      // The work scope pays per use against a budget: the meter panel covers it.
      el.hidden = s.scope === 'enterprise';
      if (el.hidden) return;
      const line = feedLine(s.spendHistory, 'the monthly spend');
      patch(state, line, staleNote(s.spendHistory, nowFrom(s)));
      if (line) { patch(body); return; }
      const months = (dataOf(s.spendHistory).months || []).filter((m) => m && m.month).slice(-MONTHS);
      if (!months.length) { patch(body, stateLine('No spend is recorded yet.')); return; }
      width = Math.max(260, Math.floor(body.clientWidth) || 600);
      patch(body, headline(months, price), lineChart(months, price, width),
        h('div', { class: 'v-keys' },
          h('span', null, h('i', { class: 'k-line' }), 'API-equivalent'),
          price ? h('span', null, h('i', { class: 'k-price' }), `plan price ${f.usd(price, { cents: false })}`) : null,
          h('span', null, h('i', { class: 'k-open' }), 'month to date')),
        ratios(months, price));
    }
    return { update, destroy() { ro.disconnect(); } };
  },
};
