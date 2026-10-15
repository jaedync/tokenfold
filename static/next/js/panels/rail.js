// Other windows: every limit window except the binding one, tightest server
// forecast first. Work scope has one meter and no other windows, so the panel
// hides there (the hero says the meter is the limit) and Now takes its place.
import { h, fill, panelHead } from '../dom.js';
import * as f from '../format.js';
import { dataOf, nowFrom, otherWindows } from '../select.js';
import { splitLabel } from '../palette.js';
import { feedLine, staleNote } from '../panel.js';

const clamp = (v) => Math.max(0, Math.min(100, v));

function forecastText(w, now) {
  const p = w.pace || {};
  if (w.verdict === 'runs_out' && !f.missing(p.reaches_100_epoch)) return `100% at ≈${f.when(p.reaches_100_epoch, now)}`;
  if (!f.missing(w.tight_epoch)) return `tight: 100% at ≈${f.when(w.tight_epoch, now)}`;
  if (!f.missing(p.projected_pct)) return `≈${f.pct(p.projected_pct)} at reset`;
  if (p.pace_state === 'too_early') return 'too early to tell';
  return 'no forecast';
}

function row(w, now) {
  const p = w.pace || {};
  const { prov, name } = splitLabel(w);
  const hot = w.verdict === 'runs_out' || w.verdict === 'tight' || (p.projected_pct ?? 0) >= 95;
  const usedPct = f.missing(w.pct) ? 0 : clamp(w.pct);
  const proj = f.missing(p.projected_pct) ? usedPct : clamp(p.projected_pct);
  const resets = w.kind === 'short' ? `in ${f.dur(w.resets_at - now)}` : f.when(w.resets_at, now);
  return h('li', { class: `rw${w.stale ? ' is-stale' : ''}`, 'data-hot': hot ? '' : null, 'data-window': w.id },
    h('span', { class: 'rw-name' }, h('span', { class: 'rw-prov' }, prov), name),
    h('span', { class: 'rw-pct' }, f.pct(w.pct)),
    h('span', { class: 'rw-bar', role: 'img', 'aria-label': `${f.pct(w.pct)} used. Even pace ${f.pct(p.expected_pct)}. ${forecastText(w, now)}.` },
      h('i', { style: { width: `${usedPct}%` } }),
      h('u', { style: { left: `${usedPct}%`, width: `${Math.max(0, proj - usedPct)}%` } }),
      f.missing(p.expected_pct) ? null : h('b', { style: { left: `${clamp(p.expected_pct)}%` } })),
    h('span', { class: 'rw-meta' },
      h('span', { class: hot ? 'hot' : null }, forecastText(w, now),
        w.waitable ? [' · ', h('span', { class: 'chip chip-yellow' }, 'waitable')] : null,
        w.stale ? ` · data ${f.age(w.age_s)}` : null),
      h('span', null, resets)));
}

export default {
  id: 'rail',
  label: 'Other windows',
  mount(el) {
    const head = panelHead('Other windows');
    const sub = h('span', { class: 'panel-sub' }, 'tightest first');
    head.el.append(sub);
    el.setAttribute('aria-labelledby', head.id);
    const list = h('ol', { class: 'rail' });
    const foot = h('div', { class: 'rail-foot' });
    const state = h('div');
    el.append(head.el, state, list, foot);

    function update(s) {
      el.hidden = s.scope === 'enterprise';
      if (el.hidden) return;
      const now = nowFrom(s);
      const runway = dataOf(s.runway);
      const line = feedLine(s.runway, 'the limit feed');
      fill(state, line, staleNote(s.runway, now));
      if (line) { fill(list); fill(foot); sub.hidden = true; return; }
      const rows = otherWindows(runway);
      sub.hidden = !rows.length;
      if (!rows.length) {
        fill(list);
        fill(foot, 'No other window has data.');
        return;
      }
      fill(list, rows.map((w) => row(w, now)));
      fill(foot, h('span', { class: 'tick' }), h('span', null, 'even pace now · hatched: server forecast to reset'));
    }

    return { update };
  },
};
