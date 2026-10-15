// Energy and water: an estimate of the inference energy of your tokens and
// the water that cools it. Values come from /api/stats model_breakdown
// (energy_wh, water_ml), which app/water.py computes on the server.
import { h, patch, approx, panelHead, segmented } from '../dom.js';
import * as f from '../format.js';
import { dataOf, nowFrom } from '../select.js';
import { changed } from './history-common.js';
import { feedLine, staleNote, stateLine } from '../panel.js';
import { familyColor, familyOf } from '../palette.js';

export const PERIODS = [['recent', '14 days'], ['all', 'All time']];
const TOP = 6;
// Municipal water per household task, in mL (USGS, EPA), as the classic dashboard.
const COMPARE = [
  ['ten-minute showers', 75000], ['washing machine loads', 50000], ['toilet flushes', 6000], ['glasses of water', 250],
];

function energyText(wh) {
  if (f.missing(wh)) return [f.DASH, ''];
  return wh >= 1000 ? [(wh / 1000).toFixed(wh >= 100000 ? 0 : 1), 'kWh'] : [String(Math.round(wh)), 'Wh'];
}
function waterText(ml) {
  if (f.missing(ml)) return [f.DASH, ''];
  if (ml >= 1e6) return [(ml / 1e6).toFixed(1), 'm³'];
  return ml >= 1000 ? [(ml / 1000).toFixed(ml >= 100000 ? 0 : 1), 'L'] : [String(Math.round(ml)), 'mL'];
}

function rows(stats, period) {
  const ek = period === 'all' ? 'energy_wh' : 'recent_energy_wh';
  const wk = period === 'all' ? 'water_ml' : 'recent_water_ml';
  return (stats.model_breakdown || []).map((m) => ({ model: m.model, wh: m[ek] || 0, ml: m[wk] || 0 }))
    .filter((m) => m.wh > 0).sort((a, b) => b.wh - a.wh);
}

function methodology() {
  const li = (...t) => h('li', null, ...t);
  return h('details', { class: 'lab-method' },
    h('summary', null, 'How the estimate works'),
    h('div', null,
      h('p', null, 'Energy = tokens × watt-hours per million tokens. Output tokens use about 5× the energy of input tokens. Cache reads and writes are not counted.'),
      h('ul', null,
        li(h('b', null, 'Opus class'), ': 2,800 Wh per million output tokens, 560 per million input.'),
        li(h('b', null, 'Sonnet class'), ': 1,950 and 390. Models without their own figure use these values.'),
        li(h('b', null, 'Haiku class'), ': 650 and 130.')),
      h('p', null, 'Water (mL) = energy (Wh) × PUE 1.12 × (on-site 0.36 + grid 0.50 L/kWh) ≈ energy × 0.96.'),
      h('ul', null,
        li('PUE 1.12: a blend of AWS (1.15) and Google (1.09) sustainability reports.'),
        li('On-site 0.36 L/kWh: liquid-cooled AI racks at AWS, Google and Azure, blended equally (0.08, 0.30, 0.70).'),
        li('Grid 0.50 L/kWh: the US grid average of 1.8, lowered for renewable supply.')),
      h('p', null, 'Only inference is counted. Training, tool time, network and idle time are not.'),
      h('p', { class: 'lab-src' }, 'Sources: Jegham et al., "How Hungry is AI?" (2025); Epoch AI (2024); Couch, "Electricity Use of AI Coding Agents" (2026); Li et al., "Making AI Less Thirsty" (2025); AWS and Google 2024 reports; USGS.')));
}

function view(list) {
  const wh = list.reduce((a, m) => a + m.wh, 0);
  const ml = list.reduce((a, m) => a + m.ml, 0);
  const [ev, eu] = energyText(wh);
  const [wv, wu] = waterText(ml);
  const top = list.slice(0, TOP);
  const rest = list.slice(TOP).reduce((a, m) => a + m.wh, 0);
  const max = top.length ? top[0].wh : 1;
  const cmp = COMPARE.map(([name, per]) => ({ name, n: ml / per })).filter((c) => c.n >= 0.1).slice(0, 3);
  return [
    h('div', { class: 'en-big' },
      h('div', null, h('div', { class: 'kicker' }, 'Energy'), h('div', { class: 'en-v' }, approx(), ev, h('small', null, eu))),
      h('div', null, h('div', { class: 'kicker' }, 'Water'), h('div', { class: 'en-v' }, approx(), wv, h('small', null, wu)))),
    cmp.length ? h('p', { class: 'en-cmp' }, 'About the water of ', cmp.map((c, i) => [i ? ', ' : '', h('b', null, `${c.n >= 10 ? Math.round(c.n) : c.n.toFixed(1)} ${c.name}`)]), '.') : null,
    h('ul', { class: 'en-list', 'aria-label': 'Energy by model' }, top.map((m) => h('li', { 'data-key': m.model },
      h('span', { class: 'swatch', style: { background: familyColor(familyOf(m.model)) } }),
      h('span', { class: 'en-n', title: m.model }, m.model),
      h('span', { class: 'en-b' }, h('i', { style: { width: `${m.wh / max * 100}%` } })),
      h('span', { class: 'en-x' }, energyText(m.wh).join(' ')))),
    rest > 0 ? h('li', { class: 'en-rest' }, h('span'), h('span', { class: 'en-n' }, `${list.length - TOP} other models`), h('span'), h('span', { class: 'en-x' }, energyText(rest).join(' '))) : null),
  ];
}

export default {
  id: 'lab-energy',
  label: 'Energy and water',
  mount(el, ctx) {
    const head = panelHead('Energy and water');
    el.setAttribute('aria-labelledby', head.id);
    const period = segmented('Period', PERIODS, ctx.store.get().labPeriod || 'recent', (v) => ctx.store.set('labPeriod', v));
    head.el.append(period.el);
    const lead = h('p', { class: 'panel-sub hl-lead' }, 'The energy that the inference of your tokens used, and the water that cooled it. Tokenfold estimates both from published per-token figures; they are not measurements.');
    const state = h('div');
    const body = h('div', { class: 'en-body' });
    el.append(head.el, lead, state, body, methodology());
    let seen = null;
    function update(s) {
      const p = s.labPeriod || 'recent';
      const key = [s.stats, s.scope, p];
      if (!changed(seen, key)) return;
      seen = key;
      period.select(p);
      const line = feedLine(s.stats, 'the usage totals');
      patch(state, line, staleNote(s.stats, nowFrom(s)));
      if (line) { patch(body); return; }
      const list = rows(dataOf(s.stats), p);
      patch(body, list.length ? view(list) : stateLine('No tokens in this period.'));
    }
    return { update };
  },
};
