// Biggest sessions of the period (after filters), by cost. A row opens the
// session drawer.
import { h, fill, setText, panelHead } from '../dom.js';
import * as f from '../format.js';
import { derive, matched, sessionName, clientName, projectLabel } from './ledger-model.js';
import { colorsFor, swatch, money, plural, openSession, keepFocus } from './ledger-kit.js';

const FEW = 8, MANY = 24;

export function topList(d) {
  return matched(d.listed, d.ui.filters).sort((a, b) => b.cost - a.cost || b.s.start - a.s.start);
}

export default {
  id: 'ledger-top',
  label: 'Biggest sessions',
  mount(el, ctx) {
    const head = panelHead('Biggest sessions');
    const sub = h('span', { class: 'panel-sub' });
    head.el.append(sub);
    el.setAttribute('aria-labelledby', head.id);
    const list = h('ol', { class: 'lt-list' });
    let many = false;
    const more = h('button', { type: 'button', class: 'btn lt-more', hidden: true, onclick: () => { many = !many; render(true); } });
    el.append(head.el, list, more);
    list.addEventListener('click', (ev) => {
      const b = ev.target.closest('[data-sid]');
      if (b) openSession(ctx.store, b.dataset.sid, b);
    });

    let seen = null;
    function render(force) {
      const s = ctx.store.get();
      const d = derive(s, ctx.now());
      if (!force && d === seen) return;
      seen = d;
      if (!d.hasData) { fill(list); setText(sub, ''); more.hidden = true; return; }
      const all = topList(d), shown = all.slice(0, many ? MANY : FEW), c = colorsFor(d);
      const today = d.ui.pkey === 'today';
      setText(sub, all.length ? `${shown.length} of ${f.int(all.length)} · by cost` : '');
      keepFocus(list, () => fill(list, shown.length ? shown.map(({ s: x, cost }) => {
        const subs = x.runs.length;
        const meta = [sessionName(x) === x.project ? null : projectLabel(x.project), x.machine, clientName(x.client),
          today ? f.clock(x.start) : f.dateShort(x.start), f.shortDur(x.end - x.start)].filter(Boolean).join(' · ');
        return h('li', { class: 'lt-item' }, h('button', { type: 'button', class: 'lt-btn', 'data-sid': x.id },
          swatch(c.color(x.model).fill, { class: 'swatch lt-sw' }),
          h('span', { class: 'lt-title' }, sessionName(x), x.live ? h('span', { class: 'live-tag' }, 'Live') : null),
          h('span', { class: 'lt-cost' }, money(cost)),
          h('span', { class: 'lt-meta' }, meta),
          h('span', { class: 'lt-fan' }, subs ? plural(subs, 'subagent') : 'no subagents')));
      }) : h('li', { class: 'lt-empty' }, 'No sessions match.')));
      more.hidden = all.length <= FEW;
      setText(more, many ? 'Show fewer' : `Show ${Math.min(MANY, all.length) - FEW} more`);
    }
    return { update() { render(false); } };
  },
};
