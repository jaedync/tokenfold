// Now: live sessions on every machine, waiting first (longest wait on top),
// then working, then idle. Squares are live subagents. A session that starts
// to wait flashes once and is announced to screen readers.
import { h, patch, setText, approx, panelHead } from '../dom.js';
import * as f from '../format.js';
import { dataOf, nowFrom, sortSessions } from '../select.js';
import { familyColor } from '../palette.js';
import { feedLine, staleNote } from '../panel.js';

const GROUPS = [['waiting', 'Waiting on you'], ['working', 'Working'], ['idle', 'Idle']];
const WHY = { question: 'asks a question', permission: 'needs permission', attention: 'needs attention' };
const CHIP = { waiting: ['chip chip-red', 'Waiting'], working: ['chip chip-blue', 'Working'], idle: ['chip chip-line', 'Idle'] };
const MAX_SLOTS = 6;

function slots(n) {
  if (!n) return [];
  const out = Array.from({ length: Math.min(n, MAX_SLOTS) }, () => h('span', { class: 'slot' }));
  if (n > MAX_SLOTS) out.push(h('span', { class: 'subs-more' }, `+${n - MAX_SLOTS}`));
  return out;
}

function row(x, now, flash) {
  const dur = h('span', null, f.shortDur(now - x.state_since));
  const cost = h('span', null, f.usd(x.cost_usd, { cents: true }));
  const [chipCls, chipText] = CHIP[x.state] || ['chip chip-line', x.state];
  const why = x.state === 'waiting' && WHY[x.waiting_reason] ? [`${WHY[x.waiting_reason]} · `, dur] : [dur];
  return h('li', { class: `row${flash ? ' flash' : ''}`, 'data-state': x.state, 'data-id': x.id, 'data-key': x.id, title: x.title || null },
    h('span', { class: 'rbar' }),
    h('span', { class: 'r-id' }, h('span', { class: 'r-mach' }, x.machine), h('span', { class: 'r-proj' }, x.project || '')),
    h('span', { class: 'r-model' }, h('span', { class: 'swatch', style: { background: familyColor(x.family) } }), x.model || f.DASH),
    h('span', { class: 'r-state' }, h('span', { class: chipCls }, chipText), h('span', { class: 'r-why' }, why)),
    h('span', { class: 'r-cost' }, approx(), cost,
      h('small', null, x.state === 'working' && x.burn_usd_per_h ? `≈${f.usd(x.burn_usd_per_h, { cents: false })}/h` : 'this session')),
    h('span', { class: 'r-subs', role: 'img', 'aria-label': `${x.subagents || 0} subagents running`, title: `${x.subagents || 0} subagents running` },
      slots(x.subagents)));
}

export default {
  id: 'now',
  label: 'Now',
  mount(el, ctx) {
    const head = panelHead('Now');
    const sub = h('span', { class: 'panel-sub' });
    head.el.append(sub);
    el.id = 'now';
    el.setAttribute('aria-labelledby', head.id);
    const groups = h('div', { class: 'now-groups' });
    const foot = h('div', { class: 'now-foot' });
    const state = h('div');
    el.append(head.el, state, groups, foot);

    let waitingIds = null;  // null until the first good payload: the first render announces nothing

    // Every update patches the rows in place (keyed by session): a row that
    // did not change keeps its node, so hover and focus stay on it.
    function render(sessions, now, fresh) {
      patch(groups, GROUPS.map(([st, title]) => {
        const rows = sessions.filter((x) => x.state === st);
        if (!rows.length && st !== 'waiting') return null;
        return h('div', { class: 'grp', 'data-state': st, 'data-key': st },
          h('div', { class: 'kicker grp-head' }, h('span', { class: 'sq' }), `${title} · ${rows.length}`),
          rows.length ? h('ul', { class: 'rows' }, rows.map((x) => row(x, now, fresh.has(x.id))))
            : h('div', { class: 'empty-q' }, 'Nobody is waiting on you.'));
      }));
    }

    function update(s) {
      const now = nowFrom(s);
      const data = dataOf(s.now);
      const line = feedLine(s.now, 'the live feed');
      patch(state, line, staleNote(s.now, now));
      if (line) { patch(groups); patch(foot); setText(sub, ''); waitingIds = null; return; }
      const sessions = sortSessions(data.sessions);
      const waiting = sessions.filter((x) => x.state === 'waiting');
      const fresh = new Set(waitingIds ? waiting.filter((x) => !waitingIds.has(x.id)).map((x) => x.id) : []);
      for (const x of waiting) {
        if (fresh.has(x.id)) ctx.announce(`${x.machine} · ${x.project || 'a session'} is waiting on you: ${WHY[x.waiting_reason] || 'it needs you'}.`);
      }
      waitingIds = new Set(waiting.map((x) => x.id));
      render(sessions, now, fresh);
      const machines = new Set(sessions.map((x) => x.machine));
      const subs = data.live ? data.live.subagents : sessions.reduce((a, x) => a + (x.subagents || 0), 0);
      setText(sub, `${sessions.length} sessions on ${machines.size} machines · ${subs} subagents`);
      patch(foot, h('span', null, 'Squares: live subagents under the session.'), h('span', null, 'Time: how long in this state.'),
        h('span', null, s.scope === 'enterprise' ? '≈ $: list-price estimate for this session.' : '≈ $: API-equivalent for this session.'));
    }

    return { update };
  },
};
