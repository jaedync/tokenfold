// Breakdown: one table, five ways to cut the period (project, model,
// machine, tool, day), computed in the browser from /api/ledger. Columns
// sort; a row click filters every panel (a day row sets the period).
import { h, fill, setText, segmented, panelHead } from '../dom.js';
import * as f from '../format.js';
import { derive, matched, dayStart, daysBetween, hasFilter, projectLabel } from './ledger-model.js';
import { providerGroup } from './ledger-colors.js';
import { colorsFor, swatch, spark, money, pct0, toggleFilter, setPeriod, dayLabel } from './ledger-kit.js';

const DIMS = [['project', 'Project'], ['model', 'Model'], ['machine', 'Machine'], ['tool', 'Tool'], ['day', 'Day']];
const LABEL = Object.fromEntries(DIMS);
// Real data has a long tail (temp folders, test runs): show the top rows and
// fold the rest behind a button. Days are never folded.
const MIN_TREND_DAYS = 7, FEW_ROWS = 12;
const SORTABLE = new Set(['key', 'cost', 'calls', 'sessions', 'runs', 'subp', 'act', 'prompts']);
const toolLabel = (k) => (k === 'Task' ? 'Subagent (Task)' : k);

function newRow(key, n) {
  return { key, cost: 0, sub: 0, ids: new Set(), runs: 0, act: 0, prompts: 0, calls: 0, models: {}, trend: new Array(n).fill(0) };
}

// Day index (into trend days) of a timestamp, or -1.
const indexer = (days) => (t) => {
  for (let i = days.length - 1; i >= 0; i -= 1) if (t >= days[i]) return t < dayStart(days[i], 1) ? i : -1;
  return -1;
};

function keysOf(dim, s, p) {
  if (dim === 'project') return [s.project];
  if (dim === 'machine') return [s.machine];
  if (dim === 'model') return [p.model];
  return [];
}

function addPeriod(rows, dim, d, n, dayIx, days) {
  const get = (k) => { if (!rows.has(k)) rows.set(k, newRow(k, n)); return rows.get(k); };
  const counts = (s) => !s.quiet || d.ui.showQuiet;
  for (const m of matched(d.sessions, d.ui.filters, dim === 'day' ? null : dim)) {
    const { s } = m;
    if (dim === 'tool') {
      for (const [k, calls] of Object.entries(s.tools)) {
        const row = get(k);
        row.calls += calls;
        row.cost += m.cost;
        if (counts(s)) row.ids.add(s.id);
        row.runs += s.runs.length;
      }
      continue;
    }
    for (const p of m.parts) {
      const keys = dim === 'day' ? [days[dayIx(p.start)]].filter((x) => x !== undefined) : keysOf(dim, s, p);
      for (const k of keys) {
        const row = get(k);
        row.cost += p.cost;
        if (p.kind === 'run') { row.sub += p.cost; row.runs += 1; }
        if (counts(s)) row.ids.add(s.id);
        row.act += p.active;
        row.prompts += p.prompts;
        row.models[p.model] = (row.models[p.model] || 0) + p.cost;
      }
    }
  }
}

// Trend lines come from the 30-day data, so a short period still has a shape.
function addTrend(rows, dim, d, tdays) {
  const ix = indexer(tdays);
  for (const m of matched(d.sessions30, d.ui.filters, dim)) {
    if (dim === 'tool') {
      const i = ix(m.s.start);
      if (i < 0) continue;
      for (const [k, calls] of Object.entries(m.s.tools)) if (rows.has(k)) rows.get(k).trend[i] += calls;
      continue;
    }
    for (const p of m.parts) {
      const i = ix(p.start);
      if (i < 0) continue;
      for (const k of keysOf(dim, m.s, p)) if (rows.has(k)) rows.get(k).trend[i] += p.cost;
    }
  }
}

export function pivotRows(d, dim) {
  const days = daysBetween(d.range.from, d.range.to);
  const end = days[days.length - 1] ?? dayStart(d.range.from);
  const tdays = [];
  for (let t = Math.min(days[0] ?? end, dayStart(end, -(MIN_TREND_DAYS - 1))); t <= end; t = dayStart(t, 1)) tdays.push(t);
  const rows = new Map();
  if (dim === 'day') for (const t of days) rows.set(t, newRow(t, 0));
  addPeriod(rows, dim, d, tdays.length, indexer(days), days);
  if (dim !== 'day') addTrend(rows, dim, d, tdays);
  const list = [...rows.values()].map((r) => ({ ...r, sessions: r.ids.size }))
    .filter((r) => dim === 'day' || r.cost > 0 || r.calls > 0 || r.sessions > 0);
  return { rows: list, trendDays: tdays.length };
}

function cell(k, r, ctx) {
  const { dim, metric, total, maxV, c, d } = ctx;
  if (k === 'key') {
    const on = dim !== 'day' && hasFilter(d.ui.filters, dim, r.key);
    const name = dim === 'model'
      ? [swatch(c.color(r.key).fill), h('span', { class: 't' }, r.key), h('span', { class: 'prov' }, providerGroup(r.key, d.providers))]
      : dim === 'day' ? h('span', { class: 't' }, `${dayLabel(r.key)}${r.key === dayStart(d.now) ? ' · today' : ''}`)
        : h('span', { class: 't', title: r.key }, dim === 'tool' ? toolLabel(r.key) : dim === 'project' ? projectLabel(r.key) : r.key);
    return h('td', null, h('button', { type: 'button', class: 'pv-name', 'data-key': String(r.key), 'aria-pressed': dim === 'day' ? null : String(on) }, name));
  }
  if (k === 'cost') return h('td', { class: 'pv-r c-cost' }, money(r.cost));
  if (k === 'calls') return h('td', { class: 'pv-r c-cost' }, f.int(r.calls));
  if (k === 'share') {
    const sh = r[metric] / (dim === 'day' ? maxV : total);
    return h('td', null, h('div', { class: 'share' }, h('span', { class: 'pbar' }, h('i', { style: { width: `${(sh * 100).toFixed(1)}%` } })), h('small', null, dim === 'day' ? '' : pct0(sh))));
  }
  if (k === 'sessions' || k === 'runs' || k === 'prompts') return h('td', { class: 'pv-r c-opt' }, f.int(r[k]));
  if (k === 'act') return h('td', { class: 'pv-r c-opt' }, f.shortDur(r.act));
  if (k === 'subp') {
    const v = r.cost ? r.sub / r.cost : 0;
    return h('td', { class: 'pv-r c-opt' }, h('span', { class: 'sub-pct' }, pct0(v), h('i', null, h('u', { style: { width: `${v * 100}%` } }))));
  }
  if (k === 'trend') return h('td', { class: 'c-trend' }, spark(r.trend, 92, 24, dim === 'model' ? c.color(r.key).line : '#161616'));
  if (k === 'top') {
    const top = Object.keys(r.models).sort((a, b) => r.models[b] - r.models[a])[0];
    return h('td', null, top ? h('span', { class: 'pv-top' }, swatch(c.color(top).fill), h('span', { class: 't' }, top)) : h('span', { class: 'muted' }, f.DASH));
  }
  return h('td');
}

export default {
  id: 'ledger-pivot',
  label: 'Breakdown',
  mount(el, ctx) {
    const { store } = ctx;
    let dim = 'project', sort = { key: 'cost', dir: -1 }, all = false;
    const head = panelHead('Breakdown');
    el.setAttribute('aria-labelledby', head.id);
    const seg = segmented('Rows by', DIMS, dim, (v) => {
      dim = v; all = false;
      sort = v === 'day' ? { key: 'key', dir: -1 } : { key: v === 'tool' ? 'calls' : 'cost', dir: -1 };
      render(true);
    });
    const note = h('span', { class: 'pv-note' });
    const table = h('table', { class: 'pv' });
    const more = h('button', { type: 'button', class: 'btn pv-more', hidden: true, onclick: () => { all = !all; render(true); more.focus(); } });
    el.append(head.el, h('div', { class: 'pv-head' }, h('span', { class: 'kicker' }, 'Rows by'), seg.el, note), h('div', { class: 'scroll-x' }, table), more);

    table.addEventListener('click', (ev) => {
      const s = ev.target.closest('[data-sort]');
      if (s) {
        const k = s.dataset.sort;
        sort = sort.key === k ? { key: k, dir: -sort.dir } : { key: k, dir: k === 'key' ? 1 : -1 };
        render(true);
        table.querySelector(`[data-sort="${k}"]`).focus();
        return;
      }
      const n = ev.target.closest('[data-key]');
      if (!n) return;
      if (dim === 'day') {
        const t = Number(n.dataset.key);
        const d = derive(store.get(), ctx.now());
        if (t === dayStart(d.now)) setPeriod(store, 'today'); else setPeriod(store, 'custom', t, dayStart(t, 1));
        return;
      }
      toggleFilter(store, dim, n.dataset.key);
    });

    let seen = null;
    function render(force) {
      const d = derive(store.get(), ctx.now());
      if (!force && d === seen) return;
      seen = d;
      if (!d.hasData) { fill(table); setText(note, ''); more.hidden = true; return; }
      const c = colorsFor(d);
      const { rows: every, trendDays } = pivotRows(d, dim);
      const isTool = dim === 'tool', isDay = dim === 'day';
      const metric = isTool ? 'calls' : 'cost';
      const val = (r, k) => (k === 'subp' ? r.sub / (r.cost || 1) : r[k]);
      every.sort((a, b) => {
        const va = val(a, sort.key), vb = val(b, sort.key);
        return (typeof va === 'string' ? va.localeCompare(vb) : va - vb) * sort.dir || b[metric] - a[metric];
      });
      const nAll = every.length, cap = !isDay && !all && nAll > FEW_ROWS + 2;
      const rows = cap ? every.slice(0, FEW_ROWS) : every;
      const total = every.reduce((a, r) => a + r[metric], 0) || 1;
      const maxV = Math.max(1e-9, ...every.map((r) => r[metric]));
      const cols = isTool
        ? [['key', 'Tool', ''], ['calls', 'Calls', 'pv-r'], ['share', 'Share of calls', ''], ['sessions', 'Sessions', 'pv-r c-opt'], ['runs', 'Subagents', 'pv-r c-opt'], ['trend', `Calls a day · ${trendDays}d`, 'c-trend']]
        : [['key', LABEL[dim], ''], ['cost', '≈ Cost', 'pv-r'], ['share', isDay ? 'Vs busiest day' : 'Share', ''], ['sessions', 'Sessions', 'pv-r c-opt'],
          ['subp', 'Subagent', 'pv-r c-opt'], ['act', 'Active', 'pv-r c-opt'], ['prompts', 'Prompts', 'pv-r c-opt'],
          [isDay ? 'top' : 'trend', isDay ? 'Top model' : `Cost a day · ${trendDays}d`, isDay ? '' : 'c-trend']];
      const cctx = { dim, metric, total, maxV, c, d };
      const thead = h('thead', null, h('tr', null, cols.map(([k, label, cls]) => h('th', {
        class: cls, scope: 'col', 'aria-sort': sort.key === k ? (sort.dir < 0 ? 'descending' : 'ascending') : null,
      }, SORTABLE.has(k) ? h('button', { type: 'button', 'data-sort': k }, label) : label))));
      const tbody = h('tbody', null, rows.length ? rows.map((r) => h('tr', {
        class: `${dim !== 'day' && hasFilter(d.ui.filters, dim, r.key) ? 'is-on' : ''}${r[metric] === 0 ? ' is-zero' : ''}`,
      }, cols.map(([k]) => cell(k, r, cctx)))) : h('tr', null, h('td', { colspan: cols.length, class: 'pv-empty' },
        isTool && !d.sessions.some((x) => x.hasTools) ? 'This server does not send tool counts.' : 'Nothing matches.')));
      const ids = new Set();
      for (const r of every) for (const x of r.ids) ids.add(x);
      const t = every.reduce((a, r) => ({ cost: a.cost + r.cost, calls: a.calls + r.calls, sub: a.sub + r.sub, act: a.act + r.act, prompts: a.prompts + r.prompts }), { cost: 0, calls: 0, sub: 0, act: 0, prompts: 0 });
      // With a filter on this dimension the table still lists every row (to add more), so the total says so.
      const own = d.ui.filters.some((x) => x.dim === dim);
      const totLabel = own ? `All ${LABEL[dim].toLowerCase()}s` : 'Total';
      const foot = rows.length && !isTool ? h('tfoot', null, h('tr', null,
        h('td', null, totLabel), h('td', { class: 'pv-r' }, money(t.cost)), h('td'), h('td', { class: 'pv-r c-opt' }, f.int(ids.size)),
        h('td', { class: 'pv-r c-opt' }, pct0(t.sub / (t.cost || 1))), h('td', { class: 'pv-r c-opt' }, f.shortDur(t.act)),
        h('td', { class: 'pv-r c-opt' }, f.int(t.prompts)), h('td', { class: isDay ? '' : 'c-trend' })))
        : rows.length ? h('tfoot', null, h('tr', null, h('td', null, totLabel), h('td', { class: 'pv-r' }, f.int(t.calls)), h('td'),
          h('td', { class: 'pv-r c-opt' }, f.int(ids.size)), h('td', { class: 'c-opt' }), h('td', { class: 'c-trend' }))) : null;
      fill(table, thead, tbody, foot);
      more.hidden = isDay || nAll <= FEW_ROWS + 2;
      setText(more, all ? 'Show fewer' : `Show all ${f.int(nAll)} ${LABEL[dim].toLowerCase()}s`);
      setText(note, isDay ? 'Each part of a session counts on the day it started. Click a day to make it the period.'
        : isTool ? 'A tool has no cost of its own. Click a tool to see the sessions that used it.'
          : dim === 'model' ? 'Active time counts each run, so subagent time adds on. Click a model to filter.'
            : 'Click a row to filter every panel to it.');
    }
    return { update() { render(false); } };
  },
};
