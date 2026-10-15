// Benchmarks and cost per hour: published benchmark scores per model (static
// reference data) next to what an active hour of each model cost you and how
// fast it wrote. Data: /api/stats benchmarks and model_breakdown.
import { h, patch, approx, panelHead, segmented } from '../dom.js';
import * as f from '../format.js';
import { dataOf, nowFrom } from '../select.js';
import { feedLine, staleNote, stateLine } from '../panel.js';
import { familyColor, familyOf } from '../palette.js';
import { PERIODS } from './lab-energy.js';
import { API_NOTE, changed } from './history-common.js';

const BENCH = [['SWE-bench', 'var(--blue)'], ['Terminal-Bench', 'var(--red)'], ['OSWorld', 'var(--yellow)'], ['ARC-AGI-2', 'var(--ink)']];
const MIN_HOURS = 0.5;  // a model needs this much active time for a fair hourly rate

function benchBlock(bench) {
  const models = Object.keys(bench || {}).filter((m) => bench[m] && Object.keys(bench[m]).length).sort();
  if (!models.length) return stateLine('No model in this scope has published scores here.');
  return [
    h('div', { class: 'bn-keys' }, BENCH.map(([name, c]) => h('span', null, h('i', { style: { background: c } }), name))),
    h('ul', { class: 'bn-list' }, models.map((m) => h('li', { 'data-key': m },
      h('div', { class: 'bn-m' }, h('span', { class: 'swatch', style: { background: familyColor(familyOf(m)) } }), m),
      BENCH.map(([name, c]) => {
        const v = bench[m][name];
        return f.missing(v) ? null : h('div', { class: 'bn-row', title: `${m} · ${name} ${v}%` },
          h('span', { class: 'bn-k' }, name),
          h('span', { class: 'bn-b' }, h('i', { style: { width: `${Math.min(100, v)}%`, background: c } })),
          h('span', { class: 'bn-v' }, `${v.toFixed(1)}%`));
      })))),
  ];
}

function hourBlock(stats, period) {
  const cph = period === 'all' ? 'all_cost_per_hour' : 'recent_cost_per_hour';
  const tps = period === 'all' ? 'all_output_tok_per_s' : 'recent_output_tok_per_s';
  const hrs = period === 'all' ? 'active_hours' : 'recent_active_hours';
  const list = (stats.model_breakdown || []).filter((m) => (m[hrs] || 0) >= MIN_HOURS && (m[cph] || 0) > 0)
    .sort((a, b) => b[cph] - a[cph]);
  if (!list.length) return stateLine('No model has enough active time in this period.');
  const max = list[0][cph];
  return h('div', { class: 'tbl-scroll' }, h('table', { class: 'lab-tbl hr-tbl' },
    h('thead', null, h('tr', null, h('th', { scope: 'col' }, 'Model'), h('th', { scope: 'col' }, h('span', { class: 'sr-only' }, 'Bar')),
      h('th', { class: 'r', scope: 'col' }, '$ per active h'), h('th', { class: 'r', scope: 'col' }, 'Output tok/s'), h('th', { class: 'r hr-act', scope: 'col' }, 'Active'))),
    h('tbody', null, list.map((m) => h('tr', { 'data-key': m.model },
      h('th', { scope: 'row', title: m.model }, h('span', { class: 'swatch', style: { background: familyColor(familyOf(m.model)) } }), m.model),
      h('td', { class: 'hr-bar' }, h('i', { style: { width: `${m[cph] / max * 100}%` } })),
      h('td', { class: 'r strong' }, approx(), f.usd(m[cph], { cents: true })),
      h('td', { class: 'r' }, f.missing(m[tps]) ? f.DASH : f.fixed(m[tps], 0)),
      h('td', { class: 'r hr-act' }, `${f.fixed(m[hrs], 1)} h`))))));
}

export default {
  id: 'lab-bench',
  label: 'Benchmarks and cost per hour',
  cls: 'p-wide',
  mount(el, ctx) {
    const head = panelHead('Benchmarks and cost per hour');
    el.setAttribute('aria-labelledby', head.id);
    const period = segmented('Period', PERIODS, ctx.store.get().labPeriod || 'recent', (v) => ctx.store.set('labPeriod', v));
    head.el.append(period.el);
    const bench = h('div', { class: 'bn' });
    const hour = h('div', { class: 'hr' });
    const state = h('div');
    el.append(head.el, state, h('div', { class: 'bh-body' },
      h('section', { class: 'bh-col', 'aria-label': 'Benchmarks' },
        h('h3', { class: 'kicker sub-k' }, 'Published benchmarks'),
        h('p', { class: 'panel-sub bh-lead' }, 'Scores that Anthropic publishes for each model. This is reference data, not your use.'),
        bench),
      h('section', { class: 'bh-col', 'aria-label': 'Cost per active hour' },
        h('h3', { class: 'kicker sub-k' }, 'Cost per active hour'),
        h('p', { class: 'panel-sub bh-lead' }, `What one active hour of each model cost, and how fast it wrote output, from your sessions. Models need ${MIN_HOURS} h of active time. ${API_NOTE}`),
        hour)));
    let seen = null;
    function update(s) {
      const p = s.labPeriod || 'recent';
      const key = [s.stats, s.scope, p];
      if (!changed(seen, key)) return;
      seen = key;
      period.select(p);
      const line = feedLine(s.stats, 'the model reference');
      patch(state, line, staleNote(s.stats, nowFrom(s)));
      const d = dataOf(s.stats);
      if (!d) { patch(bench); patch(hour); return; }
      patch(bench, benchBlock(d.benchmarks));
      patch(hour, hourBlock(d, p));
    }
    return { update };
  },
};
