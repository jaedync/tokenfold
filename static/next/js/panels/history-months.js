// Monthly spend (/api/spend-history): one stacked bar per UTC month, by
// model, with a legend grouped by provider. In the work scope a line marks
// the monthly budget. The browser only adds and draws server values.
import { h, fill, approx, panelHead, segmented } from '../dom.js';
import * as f from '../format.js';
import { dataOf, nowFrom } from '../select.js';
import { feedLine, staleNote, stateLine } from '../panel.js';
import { buildSeries, niceScale, API_NOTE, changed } from './history-common.js';

const RECENT = 12;

function budgetOf(s) {
  if (s.scope !== 'enterprise') return null;
  const stats = dataOf(s.stats);
  const hero = stats && stats.month_hero;
  if (hero && typeof hero.limit_usd === 'number' && hero.limit_usd > 0) return hero.limit_usd;
  const b = dataOf(s.enterpriseBudget);
  return b && typeof b.budget_usd === 'number' && b.budget_usd > 0 ? b.budget_usd : null;
}

function legend(providers) {
  return h('div', { class: 'mo-legend', 'aria-label': 'Models by provider' }, providers.map((p) =>
    h('div', { class: 'mo-lg' },
      h('div', { class: 'mo-lg-p' }, p.name, h('span', null, [' ', approx(), f.usdShort(p.total)])),
      h('ul', null, p.series.map((s) => h('li', null,
        h('span', { class: `swatch${s.hatch ? ' hatch' : ''}`, style: { background: s.color } }),
        h('span', { class: 'mo-lg-n' }, s.model)))))));
}

function readout(row, series, budget) {
  if (!row) return h('div', { class: 'mo-read' });
  const items = series.map((s) => ({ s, v: row.values.get(s.id) || 0 })).filter((x) => x.v > 0)
    .sort((a, b) => b.v - a.v);
  const sum = items.reduce((a, x) => a + x.v, 0) || 1;
  return h('div', { class: 'mo-read', 'aria-live': 'polite' },
    h('div', { class: 'kicker' }, row.partial ? `${row.label.long} · to date` : row.label.long),
    h('div', { class: 'mo-read-v' }, approx(), f.usd(row.total, { cents: false })),
    budget ? h('p', { class: 'mo-read-sub' }, `${Math.round(row.total / budget * 100)}% of the ${f.usd(budget, { cents: false })} budget`) : null,
    h('ul', { class: 'mo-read-list' }, items.slice(0, 8).map(({ s, v }) => h('li', null,
      h('span', { class: `swatch${s.hatch ? ' hatch' : ''}`, style: { background: s.color } }),
      h('span', { class: 'rn', title: `${s.provider} · ${s.model}` }, s.other ? `${s.provider} · other` : s.model),
      h('span', { class: 'rv' }, f.usd(v, { cents: false })),
      h('span', { class: 'rp' }, `${Math.round(v / sum * 100)}%`)))),
    items.length > 8 ? h('p', { class: 'mo-read-sub' }, `${items.length - 8} more models · ${f.usd(items.slice(8).reduce((a, x) => a + x.v, 0), { cents: false })}`) : null);
}

function chart(rows, series, budget, picked, onPick) {
  const max = Math.max(budget || 0, ...rows.map((r) => r.total || 0));
  const { top, step } = niceScale(max * 1.04);
  const grid = [];
  for (let v = 0; v <= top + 1e-9; v += step) grid.push(h('span', { class: 'mo-gl', style: { bottom: `${v / top * 100}%` } }, h('em', null, f.usdShort(v))));
  const cols = rows.map((r, i) => {
    const segs = series.map((s) => ({ s, v: r.values.get(s.id) || 0 })).filter((x) => x.v > 0);
    return h('button', {
      type: 'button', class: 'mo-col', 'aria-pressed': String(i === picked), 'data-partial': r.partial ? '1' : null,
      'aria-label': `${r.label.long}${r.partial ? ', to date' : ''}: about ${f.usd(r.total, { cents: false })}`,
      onclick: () => onPick(i),
    },
    h('span', { class: 'mo-stack', style: { height: `${(r.total || 0) / top * 100}%` } },
      h('span', { class: 'mo-total' }, f.usdShort(r.total)),
      segs.map(({ s, v }) => h('i', { class: s.hatch ? 'hatch' : null, style: { flex: `${v} 1 0`, background: s.color } }))),
    h('span', { class: 'mo-x' }, r.label.short, r.partial ? h('small', null, 'to date') : null));
  });
  return h('div', { class: 'mo-chart' },
    h('div', { class: 'mo-plot' }, grid,
      budget ? h('span', { class: 'mo-budget', style: { bottom: `${budget / top * 100}%` } }, h('em', null, `Budget ${f.usdShort(budget)}`)) : null,
      h('div', { class: 'mo-cols', 'data-dense': rows.length > 7 ? '1' : null, style: { '--n': String(rows.length) } }, cols)));
}

export default {
  id: 'history-months',
  label: 'Spend by month',
  mount(el, ctx) {
    const head = panelHead('Spend by month');
    el.setAttribute('aria-labelledby', head.id);
    const range = segmented('Months shown', [['recent', `${RECENT} months`], ['all', 'All']], 'recent', (v) => { span = v; picked = null; seen = null; update(ctx.store.get()); });
    head.el.append(range.el);
    const lead = h('p', { class: 'panel-sub hl-lead' });
    const state = h('div');
    const body = h('div', { class: 'mo-body' });
    el.append(head.el, lead, state, body);
    let span = 'recent', picked = null, seen = null;

    function update(s) {
      const budget = budgetOf(s);
      const key = [s.spendHistory, s.scope, budget, span, picked];
      if (!changed(seen, key)) return;
      seen = key;
      const work = s.scope === 'enterprise';
      fill(lead, `API-equivalent spend per calendar month (UTC), stacked by model. ${API_NOTE}`,
        work ? ' The billing meter has the billed amount.' : '');
      const line = feedLine(s.spendHistory, 'the monthly spend');
      fill(state, line, staleNote(s.spendHistory, nowFrom(s)));
      if (line) { fill(body); return; }
      const months = (dataOf(s.spendHistory).months || []).filter((m) => m && m.month);
      if (!months.length) { fill(body, stateLine('No spend is recorded yet.')); return; }
      range.el.hidden = months.length <= RECENT;
      const shown = span === 'all' ? months : months.slice(-RECENT);
      const { series, providers, rows } = buildSeries(shown);
      // Default pick: the newest full month (the current one is still running).
      const def = rows.length > 1 && rows[rows.length - 1].partial ? rows.length - 2 : rows.length - 1;
      const pick = picked === null || picked >= rows.length ? def : picked;
      fill(body,
        h('div', { class: 'mo-main' }, chart(rows, series, budget, pick, (i) => {
          picked = i;
          update(ctx.store.get());
          // The redraw replaced the buttons: keep the keyboard on the picked bar.
          const btn = body.querySelectorAll('.mo-col')[i];
          if (btn) btn.focus();
        }),
          readout(rows[pick], series, budget)),
        legend(providers),
        h('p', { class: 'hl-note' }, rows.some((r) => r.partial)
          ? 'The newest month is not over: its bar shows the spend to date, not a forecast. Select a bar to see its models.'
          : 'Select a bar to see its models.'));
    }
    return { update };
  },
};
