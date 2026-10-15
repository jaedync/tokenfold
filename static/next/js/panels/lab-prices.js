// Price table: the list prices per million tokens that Tokenfold uses now to
// price your use (/api/stats model_pricing). Models whose provider reports
// the cost (Pi) or that run locally have no list price and are only counted.
import { h, patch, panelHead } from '../dom.js';
import * as f from '../format.js';
import { dataOf, nowFrom } from '../select.js';
import { changed } from './history-common.js';
import { feedLine, staleNote, stateLine } from '../panel.js';
import { familyColor, familyOf, FAMILY_ORDER } from '../palette.js';

// [key, long header, short header for a narrow panel]
const COLS = [['input', 'Input', 'In'], ['output', 'Output', 'Out'], ['cache_write', 'Cache write 5 min', 'Write 5m'],
  ['cache_write_1h', 'Cache write 1 h', 'Write 1h'], ['cache_read', 'Cache read', 'Read']];
const rank = (fam) => { const i = FAMILY_ORDER.indexOf(fam); return i < 0 ? FAMILY_ORDER.length : i; };
// "Opus 5.5" -> [5, 5]: newest version first inside a family.
const version = (name) => (String(name).match(/\d+(\.\d+)?/g) || []).map(Number);
function byVersionDesc(a, b) {
  const va = version(a), vb = version(b);
  for (let i = 0; i < Math.max(va.length, vb.length); i += 1) {
    const d = (vb[i] || 0) - (va[i] || 0);
    if (d) return d;
  }
  return a.localeCompare(b);
}

const money = (v) => (f.missing(v) ? f.DASH : `$${v < 1 ? v.toFixed(v < 0.1 ? 3 : 2) : v.toFixed(2).replace(/\.00$/, '')}`);

export default {
  id: 'lab-prices',
  label: 'Price table',
  mount(el) {
    const head = panelHead('Price table');
    el.setAttribute('aria-labelledby', head.id);
    const lead = h('p', { class: 'panel-sub hl-lead' }, 'List prices in US dollars per million tokens, as Tokenfold uses them now to price your use. A 1-hour cache write costs 2× the input price.');
    const state = h('div');
    const body = h('div');
    el.append(head.el, lead, state, body);
    let seen = null;
    function update(s) {
      const key = [s.stats, s.scope];
      if (!changed(seen, key)) return;
      seen = key;
      const line = feedLine(s.stats, 'the price table');
      patch(state, line, staleNote(s.stats, nowFrom(s)));
      if (line) { patch(body); return; }
      const pricing = dataOf(s.stats).model_pricing || {};
      const all = Object.entries(pricing).filter(([, p]) => p && typeof p === 'object');
      const priced = all.filter(([, p]) => (p.input || 0) > 0 || (p.output || 0) > 0)
        .map(([name, p]) => ({ name, p, fam: familyOf(name) }))
        .sort((a, b) => rank(a.fam) - rank(b.fam) || byVersionDesc(a.name, b.name));
      const unpriced = all.length - priced.length;
      if (!priced.length) { patch(body, stateLine('No model in this scope has a list price.')); return; }
      patch(body,
        h('div', { class: 'tbl-scroll' }, h('table', { class: 'lab-tbl pr-tbl' },
          h('caption', { class: 'sr-only' }, 'US dollars per million tokens'),
          h('thead', null, h('tr', null, h('th', { scope: 'col' }, 'Model'), COLS.map(([, t, short]) => h('th', { class: 'r', scope: 'col', title: t }, h('span', { class: 'th-long' }, t), h('span', { class: 'th-short', 'aria-hidden': 'true' }, short))))),
          h('tbody', null, priced.map(({ name, p, fam }) => h('tr', { 'data-key': name },
            h('th', { scope: 'row' }, h('span', { class: 'swatch', style: { background: familyColor(fam) } }), name),
            COLS.map(([k]) => h('td', { class: k === 'output' ? 'r strong' : 'r' }, money(p[k])))))))),
        unpriced ? h('p', { class: 'hl-note' }, `${unpriced} more ${unpriced === 1 ? 'model has' : 'models have'} no list price here: the provider reports the cost (Pi), or the model runs locally.`) : null);
    }
    return { update };
  },
};
