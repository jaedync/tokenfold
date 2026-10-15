// Today: spend since local midnight against a typical day (median of the last
// 28 days, from the server), the model mix and spend by machine.
import { h, fill, setText, approx, panelHead } from '../dom.js';
import * as f from '../format.js';
import { dataOf, nowFrom } from '../select.js';
import { familyColor, familyOf, FAMILY_ORDER } from '../palette.js';
import { feedLine, staleNote } from '../panel.js';

const rank = (fam) => { const i = FAMILY_ORDER.indexOf(fam); return i < 0 ? FAMILY_ORDER.length : i; };

export default {
  id: 'today',
  label: 'Today',
  mount(el) {
    const head = panelHead('Today');
    const sub = h('span', { class: 'panel-sub' }, 'since midnight');
    head.el.append(sub);
    el.setAttribute('aria-labelledby', head.id);
    const big = h('span', { class: 'today-v' });
    const cmp = h('span', { class: 'today-cmp' });
    const facts = h('div', { class: 'today-facts' });
    const barNow = h('i'), barTypical = h('b'), barDay = h('s');
    const keys = h('div', { class: 'cmp-keys' });
    const mix = h('div', { class: 'mix', role: 'img' });
    const mixLegend = h('ul', { class: 'mix-legend' });
    const machs = h('ul', { class: 'machs' });
    const state = h('div');
    const body = h('div', { class: 'today-body' },
      h('div', { class: 'tb-sum' }, h('div', { class: 'today-big' }, big, cmp), facts,
        h('div', null, h('div', { class: 'cmpbar' }, barNow, barTypical, barDay), keys)),
      h('div', null, h('div', { class: 'kicker sub-k' }, 'Model mix'), mix, mixLegend),
      h('div', null, h('div', { class: 'kicker sub-k' }, 'By machine'), machs));
    el.append(head.el, state, body);

    let seen = null, seenMinute = null, seenScope = null;
    function update(s) {
      const now = nowFrom(s);
      // Redraw only for new data or a new minute: a 5 s poll elsewhere must not reset hover titles.
      if (s.today === seen && Math.floor(now / 60) === seenMinute && s.scope === seenScope) return;
      seen = s.today; seenMinute = Math.floor(now / 60); seenScope = s.scope;
      const t = dataOf(s.today);
      const line = feedLine(s.today, "today's spend");
      fill(state, line, staleNote(s.today, now));
      body.hidden = Boolean(line);
      if (line) return;
      setText(sub, s.scope === 'enterprise' ? 'since midnight · list-price estimate' : 'since midnight');
      fill(big, approx(), f.usd(t.spend_usd, { cents: false }));
      const ref = t.typical_by_now_usd;
      if (!f.missing(ref) && ref > 0) {
        const delta = (t.spend_usd / ref - 1) * 100;
        fill(cmp, h('b', null, `${delta >= 0 ? '+' : '−'}${Math.abs(delta).toFixed(0)}%`), ` vs typical by ${f.clock(now)} (${f.usd(ref, { cents: false })})`);
      } else fill(cmp, 'No typical day yet.');
      setText(facts, `${f.int(t.prompts)} prompts · ${f.dur(t.active_s)} active`);
      const scale = Math.max(t.spend_usd || 0, t.typical_day_usd || 0, ref || 0) * 1.08 || 1;
      barNow.style.width = `${(t.spend_usd || 0) / scale * 100}%`;
      barTypical.hidden = f.missing(ref);
      if (!f.missing(ref)) barTypical.style.left = `${ref / scale * 100}%`;
      barDay.hidden = f.missing(t.typical_day_usd);
      if (!f.missing(t.typical_day_usd)) barDay.style.left = `${t.typical_day_usd / scale * 100}%`;
      fill(keys,
        h('span', null, h('i', { style: { background: 'var(--ink)' } }), 'today'),
        h('span', null, h('i', { style: { background: 'var(--yellow)', 'box-shadow': '0 0 0 1px var(--ink)' } }), 'typical by now (median, 28 days)'),
        f.missing(t.typical_day_usd) ? null : h('span', null, h('i', { style: { 'border-left': '2px dashed var(--ink)', width: '0' } }), `typical full day ${f.usd(t.typical_day_usd, { cents: false })}`));

      const models = (t.by_model || []).map((m) => ({ ...m, family: m.family || familyOf(m.model) }))
        .filter((m) => m.usd > 0).sort((a, b) => rank(a.family) - rank(b.family) || b.usd - a.usd);
      const total = models.reduce((a, m) => a + m.usd, 0) || 1;
      fill(mix, models.map((m) => h('span', { style: { flex: `${m.usd} 1 0`, background: familyColor(m.family) }, title: `${m.model} ${f.usd(m.usd)}` })));
      mix.setAttribute('aria-label', `Model mix: ${models.map((m) => `${m.model} ${Math.round(m.usd / total * 100)} percent`).join(', ') || 'no spend yet'}`);
      fill(mixLegend, models.map((m) => h('li', null,
        h('span', { class: 'swatch', style: { background: familyColor(m.family) } }),
        h('span', { class: 'ml-n', title: m.model }, m.model), h('span', { class: 'ml-v' }, f.usd(m.usd, { cents: false })),
        h('span', { class: 'ml-p' }, `${Math.round(m.usd / total * 100)}%`))));

      const machines = (t.by_machine || []).slice().sort((a, b) => b.usd - a.usd);
      const top = machines.length ? machines[0].usd || 1 : 1;
      fill(machs, machines.length ? machines.map((m) => h('li', null,
        h('span', { class: 'mn', title: m.machine }, m.machine, m.working ? h('span', { class: 'live-dot', title: 'working now', 'aria-label': 'working now' }) : null),
        h('span', { class: 'mb' }, h('i', { style: { width: `${(m.usd || 0) / top * 100}%` } })),
        h('span', { class: 'mv' }, f.usd(m.usd, { cents: false }))))
        : h('li', null, h('span', { class: 'mn' }, 'No spend yet today.')));
    }

    return { update };
  },
};
