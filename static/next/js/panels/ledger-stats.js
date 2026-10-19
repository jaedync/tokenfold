// Ledger stat strip: cost, sessions, subagents, active time and prompts for
// the period and the filters. The one place in the view that says what the
// dollars are: API-equivalent at list price, not billed.
import { h, patch, setText, approx } from '../dom.js';
import * as f from '../format.js';
import { feedLine, staleNote } from '../panel.js';
import { derive, matched, daysBetween } from './ledger-model.js';
import { money, dayLabel } from './ledger-kit.js';

const dateFmt = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric' });

export function periodLabel(d) {
  const { pkey } = d.ui;
  if (pkey === 'today') return 'Today so far';
  if (pkey === '7d') return 'Last 7 days';
  if (pkey === '30d') return 'Last 30 days';
  const a = d.range.from, b = d.range.to - 1;
  if (daysBetween(a, d.range.to).length === 1) return dayLabel(a);
  const da = new Date(a * 1000), db = new Date(b * 1000);
  return da.getMonth() === db.getMonth() ? `${dateFmt.format(da)}–${db.getDate()}` : `${dateFmt.format(da)}–${dateFmt.format(db)}`;
}

export function sums(d) {
  const all = matched(d.sessions, d.ui.filters);
  const out = { cost: 0, sessions: 0, live: 0, runs: 0, active: 0, agentActive: 0, prompts: 0 };
  for (const m of all) {
    out.cost += m.cost;
    if (!m.s.quiet || d.ui.showQuiet) out.sessions += 1;
    if (m.s.live) out.live += 1;
    for (const p of m.parts) {
      if (p.kind === 'run') { out.runs += 1; out.agentActive += p.active; } else out.active += p.active;
      out.prompts += p.prompts;
    }
  }
  return out;
}

function stat(key, value, sub) {
  return h('div', { class: 'ls' }, h('span', { class: 'kicker' }, key), h('span', { class: 'ls-v' }, value), h('span', { class: 'ls-sub' }, sub));
}

export default {
  id: 'ledger-stats',
  label: 'Period summary',
  cls: 'p-wide',
  mount(el, ctx) {
    const title = h('h2', { class: 'ls-title', id: 'ledger-title' }, 'Ledger');
    const period = h('span', { class: 'ls-period' });
    const busy = h('span', { class: 'ls-busy', role: 'status' });
    const state = h('div');
    const row = h('div', { class: 'ls-row' });
    el.setAttribute('aria-labelledby', 'ledger-title');
    el.append(h('div', { class: 'ls-head' }, title, h('div', { class: 'ls-meta' }, h('div', { class: 'ls-line' }, period, busy))), state, row);

    let seen = null;
    return {
      update(s) {
        const d = derive(s, ctx.now());
        const key = [d, s.ledger && s.ledger.status, s.ledgerPeriod && s.ledgerPeriod.status];
        if (seen && key.every((k, i) => k === seen[i])) return;
        seen = key;
        const line = d.hasData ? null : feedLine(s.ledger, 'the ledger');
        patch(state, line, staleNote(s.ledger, ctx.now()));
        row.hidden = Boolean(line);
        const filtered = d.ui.filters.length > 0;
        setText(period, `${periodLabel(d)} · ${filtered ? 'filtered' : 'all machines'}`);
        const loading = d.hasData && !d.exact;
        setText(busy, loading ? 'Loading this period' : '');
        busy.title = loading ? 'The figures below are from the 30-day data until this period loads.' : '';
        el.classList.toggle('is-loading', loading);
        if (line) return;
        const t = sums(d);
        const days = Math.max(1, daysBetween(d.range.from, d.range.to).length);
        patch(row,
          stat('Cost', [approx(), money(t.cost)], 'API-equivalent at list price, not billed'),
          stat('Sessions', f.int(t.sessions), days > 1 ? `${(t.sessions / days).toFixed(1)} a day` : `${f.int(t.live)} live now`),
          stat('Subagents', f.int(t.runs), t.sessions ? `${(t.runs / t.sessions).toFixed(1)} runs a session` : f.DASH),
          stat('Active', f.shortDur(t.active), `+ ${f.shortDur(t.agentActive)} in subagents`),
          stat('Prompts', f.int(t.prompts), t.prompts ? `${money(t.cost / t.prompts)} each` : f.DASH));
      },
    };
  },
};
