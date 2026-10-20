// Ledger stat strip: cost, sessions, subagents, active time and prompts for
// the period and the filters. The one place in the view that says what the
// dollars are: API-equivalent at list price, not billed.
import { h, patch, setText, approx } from '../dom.js';
import * as f from '../format.js';
import { feedLine, staleNote } from '../panel.js';
import { derive, matched, daysBetween, dayStart } from './ledger-model.js';
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

const BUSY_TEXT = 'Loading this period';
const BUSY_WHY = 'The figures below are from the 30-day data until this period loads.';
const epochOfDay = (day) => { const [y, m, dd] = day.split('-').map(Number); return new Date(y, m - 1, dd).getTime() / 1000; };
// The meter bills every call on the work account; calls that never reached
// Tokenfold show up only as this difference. It covers every work machine,
// so a filtered view hides it.
const GAP_DATES = 3;
function gapText(d) {
  if (d.ui.filters.length) return null;
  const from = dayStart(d.range.from);
  const days = d.meterGaps.filter((g) => { const t = epochOfDay(g.day); return t >= from && t < d.range.to; });
  if (!days.length) return null;
  const usd = days.reduce((a, g) => a + g.billed - g.recorded, 0);
  const dates = days.slice(0, GAP_DATES).map((g) => f.dateShort(epochOfDay(g.day)));
  const more = days.length - dates.length;
  const when = more ? `${dates.join(', ')} and ${more} more ${more === 1 ? 'day' : 'days'}` : dates.join(', ');
  return `The billing meter shows ${money(usd)} more than Tokenfold recorded on ${when}.`;
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
    const gap = h('p', { class: 'ls-gap' });
    el.setAttribute('aria-labelledby', 'ledger-title');
    el.append(h('div', { class: 'ls-head' }, title, h('div', { class: 'ls-meta' }, h('div', { class: 'ls-line' }, period, busy))), state, row, gap);

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
        // Short on screen; the reason goes to screen readers and the tooltip.
        patch(busy, loading ? [BUSY_TEXT, h('span', { class: 'sr-only' }, `. ${BUSY_WHY}`)] : null);
        busy.title = loading ? BUSY_WHY : '';
        el.classList.toggle('is-loading', loading);
        patch(gap, line ? null : gapText(d));
        gap.hidden = !gap.firstChild;
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
