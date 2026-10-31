// All windows: every limit window, grouped under a heading per provider, in
// a fixed order (select.allWindows) so rows never trade places between polls.
// The hero's row carries a Main chip. A click on another long window pins it
// as the hero (pin.js); while a pin holds, the server's own pick shows an Auto
// chip. Work scope has one meter and no other windows, so the panel hides
// there (the hero says the meter is the limit) and Now takes its place.
import { h, patch, panelHead } from '../dom.js';
import * as f from '../format.js';
import { dataOf, heroId, isPinned, nowFrom, windowsByProvider } from '../select.js';
import { setPin } from '../pin.js';
import { providerName, splitLabel } from '../palette.js';
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

// A long window's name is a button that pins it (its ::after covers the
// row). Short windows reset too soon to be the hero, as in the server rule,
// and the hero's own row has nothing to pick. The provider is the group
// heading, so the row shows only the window name.
function nameCell(w, name, auto, main) {
  const parts = [name,
    main ? h('span', { class: 'chip chip-ink rw-chip', title: 'The window shown large above' }, 'main') : null,
    auto ? h('span', { class: 'chip chip-line rw-chip', title: 'The window the dashboard picks without a pin' }, 'auto') : null];
  if (w.kind === 'short' || main) return h('span', { class: 'rw-name' }, parts);
  return h('button', { class: 'rw-name rw-pick', type: 'button', 'data-label': w.label,
    'aria-label': auto ? `${w.label}, auto pick` : w.label, title: 'Show this window as the main window' }, parts);
}

function row(w, now, auto, main) {
  const p = w.pace || {};
  const { name } = splitLabel(w);
  const hot = w.verdict === 'runs_out' || w.verdict === 'tight' || (p.projected_pct ?? 0) >= 95;
  const usedPct = f.missing(w.pct) ? 0 : clamp(w.pct);
  const proj = f.missing(p.projected_pct) ? usedPct : clamp(p.projected_pct);
  const resets = w.kind === 'short' ? `in ${f.dur(w.resets_at - now)}` : f.when(w.resets_at, now);
  return h('li', { class: `rw${w.stale ? ' is-stale' : ''}`, 'data-hot': hot ? '' : null, 'data-window': w.id, 'data-key': w.id },
    nameCell(w, name, auto, main),
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

// One provider: its name as a heading over its windows.
function group(g, now, autoId, mainId) {
  const headId = `rw-grp-${String(g.provider).replace(/[^a-z0-9-]/gi, '_')}`;
  return h('li', { class: 'rw-grp', 'data-key': `grp:${g.provider}` },
    h('h3', { class: 'rw-grp-h', id: headId }, providerName(g.provider) || 'Other'),
    h('ol', { class: 'rw-list', 'aria-labelledby': headId },
      g.windows.map((w) => row(w, now, w.id === autoId, w.id === mainId))));
}

export default {
  id: 'rail',
  label: 'All windows',
  mount(el, ctx) {
    const head = panelHead('All windows');
    el.setAttribute('aria-labelledby', head.id);
    const list = h('ol', { class: 'rail' });
    const foot = h('div', { class: 'rail-foot' });
    const state = h('div');
    el.append(head.el, state, list, foot);
    // On the row, not the button: a click anywhere on a long window's row pins it.
    list.addEventListener('click', (e) => {
      const li = e.target.closest('.rw[data-window]');
      const pick = li && li.querySelector('.rw-pick');
      if (!pick) return;
      const id = li.dataset.window;
      const runway = dataOf(ctx.store.get().runway);
      // The auto pick's row goes back to auto: a pin on it would hold it after the server pick moves.
      setPin(ctx.store, runway && id === runway.binding ? null : id);
      ctx.announce(`${pick.dataset.label} is now the main window.`);
      // The hero now shows the window; its row here gets the Main chip.
      const hero = document.querySelector('.p-hero .hero-k');
      if (hero) hero.focus();
    });

    function update(s) {
      el.hidden = s.scope === 'enterprise';
      if (el.hidden) return;
      const now = nowFrom(s);
      const runway = dataOf(s.runway);
      const line = feedLine(s.runway, 'the limit feed');
      patch(state, line, staleNote(s.runway, now));
      if (line) { patch(list); patch(foot); return; }
      const groups = windowsByProvider(runway);
      const mainId = heroId(runway, s.pin);
      const autoId = isPinned(runway, s.pin) ? runway.binding : null;
      if (!groups.length) {
        patch(list);
        patch(foot, 'No window has data.');
        return;
      }
      patch(list, groups.map((g) => group(g, now, autoId, mainId)));
      patch(foot, h('span', { class: 'tick' }), h('span', null, 'even pace now · hatched: server forecast to reset'));
    }

    return { update };
  },
};
