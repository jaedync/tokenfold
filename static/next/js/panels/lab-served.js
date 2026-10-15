// Served-model check (/api/served-models/timeline?sessions=0): which model
// actually wrote each signed thinking block, per time cell, for the models
// that were ever answered by something else. Personal scope only.
import { h, fill, panelHead, segmented } from '../dom.js';
import * as f from '../format.js';
import { dataOf, nowFrom } from '../select.js';
import { changed } from './history-common.js';
import { feedLine, staleNote, stateLine } from '../panel.js';

export const SERVED_DAYS = [['7', '7 d'], ['30', '30 d'], ['90', '90 d']];
const FOREIGN = ['var(--red)', 'var(--blue)', 'var(--m-deepseek)', 'var(--ochre)', 'var(--m-fable)'];
const MAX_TICKS = 7;

// Colour per state: self light, hidden dark and hatched, other models in order of first sight.
function palette(bins) {
  const map = new Map([['self', 'color-mix(in srgb, var(--muted) 30%, var(--paper))'], ['hidden', 'var(--muted)']]);
  for (const b of bins) if (!map.has(b[2])) map.set(b[2], FOREIGN[(map.size - 2) % FOREIGN.length]);
  return map;
}

const stateName = (state) => (state === 'self' ? 'the model asked for' : state === 'hidden' ? 'model id hidden by the header' : state);

function cellTitle(t0, t1, bin, states) {
  const when = bin >= 86400 ? f.dateShort(t0) : `${f.dateShort(t0)} ${f.clock(t0)} to ${f.clock(t1)}`;
  return `${when} · ${states.map(([k, n]) => `${stateName(k)} ${f.int(n)}`).join(' · ')}`;
}

// Non-self states get a floor height, so 5 rerouted blocks in a busy cell still show.
function shares(states) {
  const total = states.reduce((a, [, n]) => a + n, 0) || 1;
  const raw = states.map(([k, n]) => (k !== 'self' ? Math.max(0.25, n / total) : n / total));
  const sum = raw.reduce((a, v) => a + v, 0) || 1;
  return raw.map((v) => v / sum * 100);
}

function ticks(t0, now) {
  const out = [];
  const d = new Date(t0 * 1000);
  d.setHours(0, 0, 0, 0);
  d.setDate(d.getDate() + 1);
  const days = [];
  while (d.getTime() / 1000 < now) { days.push(d.getTime() / 1000); d.setDate(d.getDate() + 1); }
  const every = Math.max(1, Math.ceil(days.length / MAX_TICKS));
  days.forEach((t, i) => { if (i % every === 0) out.push(t); });
  return out;
}

function strips(data, now, colors) {
  const bin = data.bin_seconds || 3600;
  const t0 = data.since_epoch;
  const n = Math.max(1, Math.ceil((now - t0) / bin));
  const grid = new Map();
  for (const [model, start, state, blocks] of data.bins || []) {
    const i = Math.floor((start - t0) / bin);
    if (i < 0 || i >= n) continue;
    const row = grid.get(model) || new Map();
    const cell = row.get(i) || new Map();
    cell.set(state, (cell.get(state) || 0) + blocks);
    row.set(i, cell);
    grid.set(model, row);
  }
  const span = now - t0;
  return h('div', { class: 'sv-strips' },
    (data.models || []).map((model) => {
      const row = grid.get(model) || new Map();
      const cells = [];
      for (let i = 0; i < n; i += 1) {
        const cell = row.get(i);
        if (!cell) { cells.push(h('span', { class: 'sv-c' })); continue; }
        const states = [...cell.entries()].sort((a, b) => b[1] - a[1]);
        const hs = shares(states);
        cells.push(h('span', { class: 'sv-c', title: cellTitle(t0 + i * bin, t0 + (i + 1) * bin, bin, states) },
          states.map(([k], j) => h('i', { class: k === 'hidden' ? 'hatch' : null, style: { height: `${hs[j]}%`, background: colors.get(k) } }))));
      }
      return [h('div', { class: 'sv-name' }, model), h('div', { class: 'sv-row', role: 'img', 'aria-label': `${model}: ${(data.latest || {})[model] ? stateName(data.latest[model].state) : ''}` }, cells)];
    }),
    h('div', { class: 'sv-axis', 'aria-hidden': 'true' }, ticks(t0, now).map((t) =>
      h('span', { style: { left: `${(t - t0) / span * 100}%` } }, f.dateShort(t)))));
}

function latest(data) {
  return h('ul', { class: 'sv-now' }, (data.models || []).map((m) => {
    const l = (data.latest || {})[m];
    if (!l) return null;
    const text = l.state === 'self' ? 'served as asked' : l.state === 'hidden' ? 'header hides the model id' : `served by ${l.state}`;
    return h('li', { 'data-state': l.state === 'self' ? 'self' : 'other' }, h('b', null, m), ` · ${text} since ${f.dateShort(l.since)}`);
  }));
}

function changeLog(data) {
  const rows = data.ledger || [];
  if (!rows.length) return null;
  return h('details', { class: 'sv-log' },
    h('summary', null, `Change log · ${rows.length} combinations`),
    h('div', { class: 'tbl-scroll' }, h('table', { class: 'lab-tbl' },
      h('thead', null, h('tr', null, ['First seen', 'Asked for', 'Served by', 'Header', 'Last seen', 'Blocks', 'Sessions'].map((t, i) => h('th', { class: i >= 5 ? 'r' : null, scope: 'col' }, t)))),
      h('tbody', null, rows.map((r) => h('tr', { 'data-self': r.state === 'self' ? '1' : null },
        h('td', null, `${f.dateShort(r.first_seen)} ${f.clock(r.first_seen)}`),
        h('td', null, r.model),
        h('td', { title: r.served_model || '' }, r.state === 'self' ? 'same model' : r.state === 'hidden' ? 'hidden' : r.state),
        h('td', null, `v${r.sig_version ?? '?'}`, h('small', null, ` fields ${r.sig_fields || 'none'}`)),
        h('td', null, `${f.dateShort(r.last_seen)} ${f.clock(r.last_seen)}`),
        h('td', { class: 'r' }, f.int(r.blocks)),
        h('td', { class: 'r' }, f.int(r.sessions))))))));
}

export default {
  id: 'lab-served',
  label: 'Served-model check',
  cls: 'p-wide',
  mount(el, ctx) {
    const head = panelHead('Served-model check');
    el.setAttribute('aria-labelledby', head.id);
    const days = segmented('Period', SERVED_DAYS, ctx.store.get().labServedDays || '30', (v) => ctx.store.set('labServedDays', v));
    head.el.append(days.el);
    const lead = h('p', { class: 'panel-sub hl-lead' }, 'Each thinking block carries a signed header that names the model that wrote it. This check counts the blocks in each time cell and shows when a different model answered, or when the header hides the model id.');
    const state = h('div');
    const body = h('div', { class: 'sv-body' });
    el.append(head.el, lead, state, body);
    let seen = null;
    function update(s) {
      const key = [s.servedTimeline, s.scope, s.labServedDays];
      if (!changed(seen, key)) return;
      seen = key;
      days.select(s.labServedDays || '30');
      // Served-model capture is personal (Max) data: the work scope hides
      // the panel, as it hides the other personal-only panels.
      el.hidden = s.scope === 'enterprise';
      if (el.hidden) return;
      const line = feedLine(s.servedTimeline, 'the served-model check');
      fill(state, line, staleNote(s.servedTimeline, nowFrom(s)));
      if (line) { fill(body); return; }
      const data = dataOf(s.servedTimeline);
      if (!(data.models || []).length) {
        fill(body, stateLine('Every signed block in this period came from the model that was asked for.'), changeLog(data));
        return;
      }
      const colors = palette(data.bins || []);
      const bin = data.bin_seconds || 3600;
      fill(body, latest(data), strips(data, nowFrom(s), colors),
        h('div', { class: 'sv-keys' }, [...colors.entries()].filter(([k]) => (data.bins || []).some((b) => b[2] === k)).map(([k, c]) =>
          h('span', null, h('i', { class: k === 'hidden' ? 'hatch' : null, style: { background: c } }), stateName(k))),
        h('span', null, `one cell = ${bin >= 86400 ? '1 day' : bin >= 3600 ? `${bin / 3600} h` : `${bin / 60} min`}`)),
        changeLog(data));
    }
    return { update };
  },
};
