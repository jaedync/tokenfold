// "Who spent it": the subagent share of the period's cost, cost by provider,
// the biggest change in the model mix over 30 days, dollars per prompt and
// the share of input read from cache.
import { h, fill, panelHead } from '../dom.js';
import * as f from '../format.js';
import { derive, matched, peakParallel, filterSets } from './ledger-model.js';
import { providerGroup, groupRank, GROUP_COLOR } from './ledger-colors.js';
import { colorsFor, swatch, spark, money, pct0, tokens, plural } from './ledger-kit.js';

const EDGE_DAYS = 5;

function totals(d) {
  const t = { orch: 0, sub: 0, runs: 0, sessions: 0, prompts: 0, peak: 0, prov: new Map(), cr: 0, inTok: 0, out: 0, tokSessions: 0 };
  for (const m of matched(d.sessions, d.ui.filters)) {
    t.sessions += 1;
    for (const p of m.parts) {
      if (p.kind === 'run') { t.sub += p.cost; t.runs += 1; } else { t.orch += p.cost; t.prompts += p.prompts; }
      const g = providerGroup(p.model, d.providers);
      t.prov.set(g, (t.prov.get(g) || 0) + p.cost);
    }
    t.peak = Math.max(t.peak, peakParallel(m.parts.filter((p) => p.kind === 'run').map((p) => p.run)));
    // Tokens are per session: they count when the whole session matches (no model filter).
    if (m.s.tokens && m.parts.length === m.s.parts.length) {
      const k = m.s.tokens;
      t.cr += k.cache_read || 0;
      t.inTok += (k.input || 0) + (k.cache_write || 0) + (k.cache_read || 0);
      t.out += k.output || 0;
      t.tokSessions += 1;
    }
  }
  return t;
}

// The largest gain in share of daily cost, first 5 days against the last 5.
// It ignores a model filter (one model is always 100% of itself).
function shift(d, c) {
  const days = d.daily30 || [];
  if (days.length < EDGE_DAYS * 2) return null;
  const win = (a, b) => {
    const o = {};
    let tot = 0;
    for (const x of days.slice(a, b)) for (const [m, v] of Object.entries(x.by_model || {})) { o[m] = (o[m] || 0) + v; tot += v; }
    for (const m of Object.keys(o)) o[m] /= tot || 1;
    return o;
  };
  const early = win(0, EDGE_DAYS), late = win(days.length - EDGE_DAYS, days.length);
  let best = null, bd = 0, lost = null, ld = 0;
  for (const m of c.order) { const dl = (late[m] || 0) - (early[m] || 0); if (dl > bd) { bd = dl; best = m; } }
  if (!best || bd < 0.05) return { none: true };
  for (const m of c.order) {
    const dl = (late[m] || 0) - (early[m] || 0);
    if (m !== best && dl < 0 && Math.abs(dl) > Math.abs(ld)) { ld = dl; lost = m; }
  }
  const series = days.map((x) => { const tot = Object.values(x.by_model || {}).reduce((a, b) => a + b, 0); return tot ? (x.by_model[best] || 0) / tot : 0; });
  return { best, lost, early, late, series, n: days.length };
}

function shiftBlock(d, c) {
  const s = shift(d, c);
  if (!s) return null;
  if (s.none) return [h('div', { class: 'kicker ws-k' }, 'Model mix'), h('p', { class: 'ws-shift-text' }, 'No large change in the model mix in 30 days.')];
  const text = [swatch(c.color(s.best).fill), ' ', h('b', null, s.best),
    ` went from ${pct0(s.early[s.best] || 0)} to ${pct0(s.late[s.best] || 0)} of daily cost in ${s.n} days`,
    s.lost ? `. It took over from ${s.lost}, which fell from ${pct0(s.early[s.lost] || 0)} to ${pct0(s.late[s.lost] || 0)}.` : '.'];
  return [h('div', { class: 'kicker ws-k' }, 'Biggest model shift'),
    h('div', { class: 'ws-shift' }, h('p', { class: 'ws-shift-text' }, text), spark(s.series, 96, 34, c.color(s.best).line, 1))];
}

export default {
  id: 'ledger-split',
  label: 'Who spent it',
  mount(el, ctx) {
    const head = panelHead('Who spent it');
    el.setAttribute('aria-labelledby', head.id);
    const body = h('div');
    el.append(head.el, body);
    let seen = null;
    return {
      update(s) {
        const d = derive(s, ctx.now());
        if (d === seen) return;
        seen = d;
        if (!d.hasData) { fill(body); return; }
        const t = totals(d), c = colorsFor(d);
        const tot = t.orch + t.sub;
        if (!t.sessions) { fill(body, h('p', { class: 'ws-empty' }, 'Nothing matches this period and these filters.')); return; }
        const subShare = tot ? t.sub / tot : 0;
        const prov = [...t.prov.entries()].sort((a, b) => groupRank(a[0]) - groupRank(b[0]));
        const modelFiltered = Boolean(filterSets(d.ui.filters).model);
        // Tokens come per session, not per model: a model filter cannot split them.
        const tokNote = modelFiltered ? 'Tokens do not split by model. Remove the model filter to see them.' : null;
        fill(body,
          h('div', { class: 'ws-big' }, h('span', { class: 'ws-pct' }, pct0(subShare)), h('span', { class: 'ws-cap' }, 'of the cost was subagent work')),
          h('div', { class: 'ws-bar', role: 'img', 'aria-label': `Orchestrators ${money(t.orch)}, subagents ${money(t.sub)}` },
            h('i', { class: 'sb-orch', style: { width: `${(1 - subShare) * 100}%` } }), subShare > 0 ? h('i', { class: 'sb-sub', style: { width: `${subShare * 100}%` } }) : null),
          h('div', { class: 'ws-legend' }, h('span', null, 'Orchestrators ', h('b', null, money(t.orch))), h('span', null, 'Subagents ', h('b', null, money(t.sub)))),
          h('p', { class: 'panel-sub ws-note' }, `${plural(t.runs, 'subagent run')} in ${plural(t.sessions, 'session')}. Peak fan-out: ${t.peak} at once.`),
          h('div', { class: 'ws-sec' }, h('div', { class: 'kicker ws-k' }, 'By provider'),
            prov.map(([g, v]) => h('div', { class: 'ws-prov' }, h('b', null, g),
              h('span', { class: 'pbar' }, h('i', { style: { width: `${tot ? Math.max(0.5, (v / tot) * 100) : 0}%`, background: GROUP_COLOR[g] || GROUP_COLOR.Other } })),
              h('span', { class: 'ws-num' }, money(v))))),
          h('div', { class: 'ws-sec' }, shiftBlock(d, c)),
          h('div', { class: 'ws-sec ws-facts' },
            h('div', { class: 'ws-fact' }, h('div', { class: 'ws-fv' }, t.prompts ? money(tot / t.prompts) : f.DASH), h('div', { class: 'ws-fk' }, 'per prompt')),
            h('div', { class: 'ws-fact', title: tokNote }, h('div', { class: 'ws-fv' }, !modelFiltered && t.inTok ? pct0(t.cr / t.inTok) : f.DASH), h('div', { class: 'ws-fk' }, 'input from cache')),
            h('div', { class: 'ws-fact', title: tokNote }, h('div', { class: 'ws-fv' }, !modelFiltered && t.tokSessions ? tokens(t.out) : f.DASH), h('div', { class: 'ws-fk' }, 'tokens written'))));
      },
    };
  },
};
