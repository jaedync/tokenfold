// Live bar (sticky, every view): burn now, the waiting queue, working sessions,
// subagents, machines online, data age and the clock. It also drives the
// masthead mark, a two-column gauge of the binding and Claude 5-hour windows.
import { h, setText, setAttr } from '../dom.js';
import * as f from '../format.js';
import { bindingWindow, claudeShortWindow, dataOf, nowFrom } from '../select.js';
import { windowTag } from '../palette.js';
import { feedStatus } from '../panel.js';

export default {
  id: 'livebar',
  label: 'Live status',
  mount(el, ctx) {
    const val = (cls) => h('span', { class: `lb-val ${cls || ''}`.trim() }, f.DASH);
    const burnUsd = h('span', { class: 'b-usd' });
    const burnPct = h('span');
    const burn = h('span', { class: 'lb-val y' }, burnUsd, burnPct);
    const waitN = val();
    const work = val(), subs = val(), machines = val(), dataAge = val();
    const clockEl = h('span', { class: 'lb-clock lb-hide-m' }, f.DASH);
    const wait = h('a', {
      class: 'lb-wait', href: '#runway', 'data-n': '0',
      onclick: (ev) => { ev.preventDefault(); ctx.navigate('runway', 'now'); },
    }, h('span', { class: 'lb-label' }, 'Waiting', h('span', { class: 'long' }, ' on you')), waitN);
    const item = (cls, label, v) => h('span', { class: cls }, h('span', { class: 'lb-label' }, label), v);
    el.append(
      h('span', { class: 'lb-live' }, h('span', { class: 'lb-dot', 'aria-hidden': 'true' }), 'Live'),
      item('lb-burn', 'Burn now', burn), wait,
      item('lb-work', 'Working', work),
      item('lb-hide-m', 'Subagents', subs),
      item('lb-hide-m', 'Machines', machines),
      item('lb-age lb-hide-m', 'Newest data', dataAge),
      clockEl,
    );
    const markBind = document.getElementById('mk-bind');
    const markShort = document.getElementById('mk-short');

    let last = ctx.store.get();
    const tickClock = () => {
      const now = nowFrom(last);
      setText(clockEl, `${f.dow(now)} ${f.clock(now, true)}`);
      const live = dataOf(last.now) && last.now.data.live;
      if (live && typeof live.newest_data_age_s === 'number' && last.now.at) {
        setText(dataAge, f.age(live.newest_data_age_s + Math.max(0, Date.now() / 1000 - last.now.at)));
      }
    };
    const timer = setInterval(tickClock, 1000);

    function update(s) {
      last = s;
      const status = feedStatus(s.now);
      setAttr(el, 'data-conn', status === 'ok' ? 'live' : 'stale');
      const live = dataOf(s.now) && s.now.data.live;
      const bind = bindingWindow(dataOf(s.runway));
      const usdH = live && live.burn_usd_per_h ? live.burn_usd_per_h.m30 : null;
      const pctH = bind && bind.burn_pct_per_h ? bind.burn_pct_per_h.h1 : null;
      setText(burnUsd, f.missing(usdH) ? '' : `≈${f.usd(usdH, { cents: false })}/h${f.missing(pctH) ? '' : ' · '}`);
      setText(burnPct, f.missing(pctH) ? (f.missing(usdH) ? f.DASH : '') : `+${pctH.toFixed(1)}%/h ${windowTag(bind)}`);
      const waiting = live ? live.waiting : null;
      setText(waitN, f.int(waiting));
      setAttr(wait, 'data-n', String(waiting || 0));
      setAttr(wait, 'aria-label', waiting ? `${waiting} sessions waiting on you. Show them.` : 'No session is waiting on you.');
      setText(work, f.int(live ? live.working : null));
      setText(subs, f.int(live ? live.subagents : null));
      setText(machines, live ? `${f.int(live.machines_online)} / ${f.int(live.machines_total)} online` : f.DASH);
      if (!live) setText(dataAge, status === 'unavailable' ? 'feed down' : f.DASH);
      tickClock();
      // The mark: paper covers the unused share of each column.
      const short = claudeShortWindow(dataOf(s.runway));
      const cover = (w) => `${100 - Math.max(0, Math.min(100, w && typeof w.pct === 'number' ? w.pct : 0))}%`;
      if (markBind) markBind.style.height = cover(bind);
      if (markShort) markShort.style.height = cover(short);
    }

    return { update, destroy: () => clearInterval(timer) };
  },
};
