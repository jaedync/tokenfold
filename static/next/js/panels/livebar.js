// Live bar (sticky, every view): burn now, the waiting queue, working sessions,
// subagents, machines online, data age and the clock. Narrow screens drop the
// low-priority cells first (css), so the data age and the clock stay in view. It also drives the
// masthead mark, a two-column gauge of the binding and Claude 5-hour windows.
import { h, patch, setText, setAttr } from '../dom.js';
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
      item('lb-subs lb-hide-m', 'Subagents', subs),
      item('lb-mach lb-hide-m', 'Machines', machines),
      item('lb-age lb-hide-m', 'Newest data', dataAge),
      clockEl,
    );
    const mark = document.getElementById('mark');
    const markCol = (id) => ({
      cover: document.getElementById(`mk-${id}`),
      key: document.getElementById(`mk-${id}-k`),
      value: document.getElementById(`mk-${id}-v`),
    });
    const markBind = markCol('bind');
    const markShort = markCol('short');

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
      const bind = bindingWindow(dataOf(s.runway), s.pin);
      const usdH = live && live.burn_usd_per_h ? live.burn_usd_per_h.m30 : null;
      const pctH = bind && bind.burn_pct_per_h ? bind.burn_pct_per_h.h1 : null;
      // The phone hides the dollar part; with no window rate the dollars are all there is.
      const usdText = f.missing(usdH) ? null : `≈${f.usd(usdH, { cents: false })}/h`;
      setText(burnUsd, usdText && !f.missing(pctH) ? `${usdText} · ` : '');
      setText(burnPct, !f.missing(pctH) ? `+${pctH.toFixed(1)}%/h ${windowTag(bind)}` : usdText || f.DASH);
      const waiting = live ? live.waiting : null;
      setText(waitN, f.int(waiting));
      setAttr(wait, 'data-n', String(waiting || 0));
      setAttr(wait, 'aria-label', waiting ? `${waiting} sessions waiting on you. Show them.` : 'No session is waiting on you.');
      setText(work, f.int(live ? live.working : null));
      setText(subs, f.int(live ? live.subagents : null));
      setText(machines, live ? `${f.int(live.machines_online)} / ${f.int(live.machines_total)} online` : f.DASH);
      if (!live) setText(dataAge, status === 'unavailable' ? 'feed down' : f.DASH);
      tickClock();
      const short = claudeShortWindow(dataOf(s.runway));
      setMark(markBind, bind, bind && bind.kind === 'month' ? 'month' : windowTag(bind) || 'limit');
      setMark(markShort, short, '5h');
      if (mark) {
        mark.title = `Red column: ${bind ? `${bind.label}, ${f.pct(bind.pct)} used` : 'no binding window'}. `
          + `Blue column: ${short ? `${short.label}, ${f.pct(short.pct)} used` : 'no 5-hour window'}.`;
      }
    }

    // The mark: a gauge per column. Paper covers the unused share; the cap
    // names the window and gives the value, so a low value still reads.
    function setMark(col, w, tag) {
      if (!col.cover) return;
      const has = Boolean(w) && typeof w.pct === 'number';
      const used = has ? Math.max(0, Math.min(100, w.pct)) : 0;
      col.cover.style.height = `${100 - used}%`;
      col.cover.parentElement.parentElement.toggleAttribute('data-none', !has);
      setText(col.key, tag);
      // patch(): the 5 s poll changes the number, never the nodes.
      if (col.value) patch(col.value, has ? String(Math.floor(w.pct)) : f.DASH, has ? h('span', { class: 'pc' }, '%') : null);
    }

    return { update, destroy: () => clearInterval(timer) };
  },
};
