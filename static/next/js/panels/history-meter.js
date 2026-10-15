// Billing meter (work scope): Anthropic's own month-to-date figure against
// the budget, and the daily meter increases next to what Tokenfold measured.
// Data: /api/stats `meter` and `month_hero` (cached there), budget fallback
// /api/enterprise-budget. Hidden in the personal scope.
import { h, fill, approx, panelHead } from '../dom.js';
import * as f from '../format.js';
import { dataOf, nowFrom } from '../select.js';
import { feedLine, staleNote, stateLine } from '../panel.js';
import { niceScale, changed, monthLabel } from './history-common.js';

// "2026-09-15" -> "Sep 15", the date style of the other panels.
const dayLabel = (day) => `${monthLabel(day).short} ${Number(String(day).slice(8, 10)) || ''}`.trim();

const SOURCE = {
  meter: 'Billed figure from the meter.',
  meter_anchored: 'Last meter reading this month, plus the spend measured after it.',
  estimate: 'No fresh meter reading: this is the Tokenfold estimate.',
};

function daily(rows) {
  if (!rows.length) return null;
  const max = Math.max(...rows.map((r) => Math.max(r.official_usd || 0, r.measured_usd || 0)));
  const { top } = niceScale(max * 1.05);
  return h('div', { class: 'md' },
    h('div', { class: 'kicker sub-k' }, 'Daily meter increase vs measured (last 30 days)'),
    h('div', { class: 'md-plot', role: 'img', 'aria-label': `Daily billed increase for ${rows.length} days, with the measured spend as a mark.` },
      h('span', { class: 'md-top' }, f.usdShort(top)),
      rows.map((r) => h('div', { class: 'md-col', title: `${r.day}: billed ${f.usd(r.official_usd)} · measured ${f.usd(r.measured_usd)}` },
        h('i', { style: { height: `${Math.max(0, r.official_usd || 0) / top * 100}%` } }),
        h('b', { style: { bottom: `${Math.max(0, r.measured_usd || 0) / top * 100}%` } })))),
    h('div', { class: 'md-x' }, h('span', null, dayLabel(rows[0].day)), h('span', null, dayLabel(rows[rows.length - 1].day))),
    h('div', { class: 'cmp-keys' },
      h('span', null, h('i', { style: { background: 'var(--ink)', width: '10px' } }), 'billed increase (meter)'),
      h('span', null, h('i', { style: { background: 'var(--blue)', height: '3px', width: '12px', 'vertical-align': '3px' } }), 'measured by Tokenfold (estimate)')));
}

function body(stats, budgetUsd, now) {
  const meter = stats.meter;
  const hero = stats.month_hero || {};
  const budget = !f.missing(hero.limit_usd) && hero.limit_usd > 0 ? hero.limit_usd : budgetUsd;
  if (!meter && !budget) {
    return stateLine('No billing meter readings yet.', 'The meter appears after a work machine reports its usage.');
  }
  const value = hero.value;
  const isEstimate = hero.source === 'estimate';
  const used = budget && !f.missing(value) ? value / budget * 100 : null;
  const age = meter && !f.missing(meter.fetched_epoch) ? now - meter.fetched_epoch : null;
  const fact = (k, v, sub) => h('div', null, h('dt', null, k), h('dd', null, h('span', { class: 'fv' }, v), sub ? h('span', { class: 'fs' }, sub) : null));
  return [
    h('div', { class: 'mt-head' },
      h('div', { class: 'mt-big' }, isEstimate ? approx() : null, f.usd(value, { cents: false })),
      h('div', { class: 'mt-of' }, budget ? `of ${f.usd(budget, { cents: false })} budget · ${f.pct(used)} used` : 'no budget set',
        h('span', null, SOURCE[hero.source] || ''))),
    budget ? h('div', { class: 'meter mt-bar', role: 'img', 'aria-label': `${Math.round(used || 0)} percent of the budget used` },
      h('i', { style: { width: `${Math.min(100, used || 0)}%` } })) : null,
    budget ? h('div', { class: 'meter-scale' }, h('span', null, '$0'), h('span', null, f.usd(budget, { cents: false }))) : null,
    h('dl', { class: 'facts mt-facts' },
      fact('Measured', [approx(), f.usd(hero.measured_usd, { cents: false })], 'events Tokenfold saw'),
      fact('Not seen', f.missing(hero.unaccounted_usd) ? f.DASH : f.usd(hero.unaccounted_usd, { cents: false }), 'billed minus measured'),
      fact('Left', f.usd(hero.remaining_usd, { cents: false }), budget ? 'before the budget' : null),
      fact('Meter read', age === null ? f.DASH : f.age(age), meter && !meter.fresh ? 'stale: older than 48 h' : null)),
    daily((meter && meter.daily) || []),
  ];
}

export default {
  id: 'history-meter',
  label: 'Billing meter',
  cls: 'p-wide',
  mount(el) {
    const head = panelHead('Billing meter');
    el.setAttribute('aria-labelledby', head.id);
    const lead = h('p', { class: 'panel-sub hl-lead' }, 'Anthropic\'s own billing meter for this month, against the monthly budget. The meter is the billed amount; the measured figure is the Tokenfold estimate at list price.');
    const state = h('div');
    const out = h('div', { class: 'mt-body' });
    el.append(head.el, lead, state, out);
    let seen = null;
    function update(s) {
      el.hidden = s.scope !== 'enterprise';
      const key = [s.stats, s.enterpriseBudget, s.scope, Math.floor(nowFrom(s) / 60)];
      if (el.hidden || !changed(seen, key)) return;
      seen = key;
      const line = feedLine(s.stats, 'the billing meter');
      fill(state, line, staleNote(s.stats, nowFrom(s)));
      if (line) { fill(out); return; }
      const b = dataOf(s.enterpriseBudget);
      fill(out, body(dataOf(s.stats), b && typeof b.budget_usd === 'number' ? b.budget_usd : null, nowFrom(s)));
    }
    return { update };
  },
};
